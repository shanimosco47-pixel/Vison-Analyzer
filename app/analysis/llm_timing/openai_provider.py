"""OpenAI provider adapter - the second real ``TimingProvider`` implementation.

Per the supervisor's authorization (after two real Gemini gate-1 runs, one
false-confident and one abstaining-but-slow/expensive): a controlled
experiment with a low-cost OpenAI vision model, ``gpt-5-mini`` by default,
via the official ``openai`` Python SDK's **Responses API**
(``client.responses.create(...)``) - not the older Chat Completions API.

This module mirrors ``gemini_provider.py``'s structure and discipline
closely: the same lazy-import pattern (nothing here imports the real
``openai`` SDK at module load time - only :func:`build_default_openai_client`
does, inside its own body), the same injectable-client testability (no
network call anywhere in the test suite), and the same retry-policy shape
(transient vs. permanent, exponential backoff with an injectable
``sleep_fn``). **Nothing calls the real client factory yet** - see
``diagnostics/llm_spike/DESIGN.md``.

Unlike the Gemini adapter's exception-shape caveats (this sandbox's network
access to ``ai.google.dev`` was blocked, so those attribute names were
best-effort), the ``openai`` SDK's exception hierarchy and Responses API
request/response shapes used here were read directly from
``openai/openai-python``'s source on GitHub (``raw.githubusercontent.com``
was reachable even though ``developers.openai.com`` was not) - see the
docstrings below for exactly what was confirmed and where.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from ...errors import ConfigurationError
from .provider import (
    PermanentProviderError,
    ProviderCallError,
    ProviderRequest,
    RawProviderResponse,
    TransientProviderError,
)

DEFAULT_OPENAI_MODEL_ID = "gpt-5-mini"

# Exception types treated as retryable even when a client raises them bare
# (not wrapped in TransientProviderError) - the common shapes of "the
# network hiccuped", which a bounded retry can plausibly fix.
_RETRYABLE_BARE_EXCEPTIONS: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)


@dataclass(frozen=True)
class OpenAICallResult:
    """What an :class:`OpenAIClient` hands back for one call.

    Deliberately minimal and vendor-shape-free, mirroring
    ``gemini_provider.GeminiCallResult`` - a fake test client builds one
    trivially, and nothing outside this module needs to know the real
    Responses API's response object shape.
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class OpenAIClient(Protocol):
    """The seam a fake test client and the real SDK wrapper both implement.

    Kept intentionally narrow (one method, plain-dict parts - the same
    generic shape ``gemini_provider.GeminiClient`` uses) so a test double
    needs nothing beyond stdlib to satisfy it. Raise
    ``provider.TransientProviderError`` for a retryable failure (rate
    limit, server error, timeout/connection) and
    ``provider.PermanentProviderError`` for one that isn't (bad request,
    auth, ...) - see :class:`OpenAITimingProvider`'s retry policy.
    """

    def generate_content(
        self, *, model: str, parts: list[dict], generation_config: dict
    ) -> OpenAICallResult: ...


class OpenAITimingProvider:
    """``TimingProvider`` backed by an OpenAI model (Responses API), via an
    injected client.

    Same injectable-client testability and retry-policy shape as
    ``gemini_provider.GeminiTimingProvider`` - see that class's docstring
    for the full rationale, unchanged here. The real wrapper
    (:func:`build_default_openai_client`) classifies ``RateLimitError``
    (429) and ``InternalServerError`` (5xx) as transient, every other
    ``APIStatusError`` subclass (``BadRequestError`` 400,
    ``AuthenticationError`` 401, ``PermissionDeniedError`` 403,
    ``NotFoundError`` 404, ``ConflictError`` 409,
    ``UnprocessableEntityError`` 422) as permanent, and
    ``APIConnectionError``/``APITimeoutError`` (network-level, not an HTTP
    status) as transient - see ``_classify_status_error``.
    """

    def __init__(
        self,
        client: OpenAIClient,
        *,
        model_id: str = DEFAULT_OPENAI_MODEL_ID,
        max_retries: int = 2,
        temperature: float = 0.0,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 10.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model_id.strip():
            raise ConfigurationError("OpenAITimingProvider needs a non-empty model_id.")
        if max_retries < 0:
            raise ConfigurationError("max_retries must not be negative.")
        if backoff_base_s < 0 or backoff_max_s < 0:
            raise ConfigurationError("backoff_base_s and backoff_max_s must not be negative.")
        self._client = client
        self._model_id = model_id
        self._max_retries = max_retries
        self._temperature = temperature
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._sleep_fn = sleep_fn

    def analyze(self, request: ProviderRequest) -> RawProviderResponse:
        """Send one frame batch, total: any client exception becomes an
        ``error``-carrying :class:`RawProviderResponse`, never an exception
        that escapes this method - identical contract to
        ``gemini_provider.GeminiTimingProvider.analyze``.
        """
        parts = _build_parts(request)
        # pass_name/frame_timestamps_s let the real client (only) constrain
        # the response schema for an end-scan window or a trend-validation
        # request - see _response_schema_for_pass. Harmless additions for
        # any other client (a fake test double just ignores unused dict
        # keys).
        generation_config = {
            "temperature": self._temperature,
            "pass_name": request.pass_name,
            "frame_timestamps_s": [frame.timestamp_s for frame in request.frames],
        }

        retries_so_far = 0
        last_error: Exception | None = None
        started = time.monotonic()
        while True:
            try:
                result = self._client.generate_content(
                    model=self._model_id, parts=parts, generation_config=generation_config
                )
            except Exception as exc:  # the client can raise anything vendor-specific
                last_error = exc
                if not _is_retryable(exc) or retries_so_far >= self._max_retries:
                    return RawProviderResponse(
                        model_id=self._model_id,
                        raw_text="",
                        latency_s=time.monotonic() - started,
                        retries=retries_so_far,
                        error=str(last_error),
                    )
                # A server-suggested delay (e.g. a 429's Retry-After) beats
                # a guessed backoff - use it when the raiser supplied one.
                suggested = getattr(exc, "retry_after_s", None)
                delay = (
                    suggested
                    if suggested is not None
                    else min(self._backoff_base_s * (2**retries_so_far), self._backoff_max_s)
                )
                self._sleep_fn(delay)
                retries_so_far += 1
                continue

            return RawProviderResponse(
                model_id=self._model_id,
                raw_text=result.text,
                latency_s=time.monotonic() - started,
                retries=retries_so_far,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, PermanentProviderError):
        return False
    if isinstance(exc, TransientProviderError):
        return True
    return isinstance(exc, _RETRYABLE_BARE_EXCEPTIONS)


def _build_parts(request: ProviderRequest) -> list[dict]:
    """The same generic ``{"text": ...}``/``{"inline_data": {...}}`` shape
    ``gemini_provider._build_parts`` uses - translated into the Responses
    API's actual part shapes only inside
    :func:`build_default_openai_client`, never here. See that function's
    docstring for why a candidate frame gets a distinguishing label."""
    parts: list[dict] = [{"text": request.prompt_text}]
    for frame in request.frames:
        label = "CANDIDATE frame" if frame.is_candidate else "frame"
        parts.append({"text": f"[{label} at t={frame.timestamp_s:.3f}s]"})
        parts.append(
            {"inline_data": {"mime_type": frame.media_type, "image_bytes": frame.image_bytes}}
        )
    return parts


# Confirmed directly from openai-python's source
# (src/openai/_exceptions.py): every one of these is a subclass of
# APIStatusError with a fixed status_code, except InternalServerError
# (covers the whole 5xx range, no single literal code). RateLimitError
# (429) and InternalServerError are worth retrying; the rest represent a
# request that will not succeed on retry (bad input, auth, a resource that
# doesn't exist, ...).
_RETRYABLE_STATUS_CODES = frozenset({429})


def _classify_status_error(exc: Exception) -> ProviderCallError:
    """Turn an ``openai.APIStatusError`` into the right
    ``ProviderCallError`` subtype, as a standalone function so the policy is
    unit-testable with a fake exception object, independent of whether the
    real SDK is installed - mirrors
    ``gemini_provider._classify_client_error``.

    ``InternalServerError`` (5xx) has no single status code in the SDK's
    type hierarchy (it covers the whole >=500 range), so it's classified by
    exception type directly in :func:`build_default_openai_client` rather
    than by status code here; this function only handles the fixed-code
    subclasses.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code in _RETRYABLE_STATUS_CODES:
        return TransientProviderError(str(exc), retry_after_s=_retry_after_s(exc))
    return PermanentProviderError(str(exc))


def _retry_after_s(exc: Exception) -> float | None:
    """Best-effort extraction of a server-suggested retry delay from a
    429's ``Retry-After`` response header. ``exc.response`` is an
    ``httpx.Response`` (confirmed via ``APIStatusError``'s fields) whose
    ``.headers`` is dict-like. Returns ``None`` (falls back to the caller's
    own exponential backoff) if nothing matches, rather than guessing a
    number."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is None or not hasattr(headers, "get"):
        return None
    raw = headers.get("Retry-After")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


_BASE_RESPONSE_SCHEMA_REQUIRED = (
    "status",
    "start_s",
    "end_s",
    "start_uncertainty_s",
    "end_uncertainty_s",
    "confidence",
    "reason_codes",
    "evidence_frame_timestamps_s",
    "raw_notes",
)


def _response_schema_for_pass(pass_name: str, frame_timestamps_s: list[float]) -> dict:
    """The JSON schema passed as OpenAI's Structured Outputs
    ``text.format.schema``.

    Identical for "coarse"/"fine" to every prior round. For "end_scan" and
    "end_validate", a live gate-1 run against gpt-4.1-mini against a real
    clip returned ``start_s=end_s=5.533`` for a submitted window of
    ``[20.000, 23.000]``s - a value matching none of the frames actually
    shown. ``pipeline._validate_grounding``'s ``out_of_bounds`` check
    safely rejected it (exactly what "never confidently wrong" requires),
    but the call itself was wasted - ~40.7s of provider latency and
    134,152 tokens for an answer the pipeline could never have accepted.
    Prompt wording alone asks the model to copy a shown timestamp; this
    constrains the *schema* so the API can only ever emit one of the
    timestamps actually submitted for this window (or ``null``, for
    abstain) - OpenAI's Structured Outputs (``strict: True``) validates
    this before the response is ever returned, so a value the pipeline
    could never accept anyway becomes structurally impossible to receive,
    not just something the pipeline detects after paying for the call.
    "end_validate" (the trend-validation follow-up request) reuses the
    exact same contract for the same reason: it too must echo back one
    submitted timestamp verbatim (the CANDIDATE frame's own) rather than
    compute one.

    Uses ``anyOf: [{type: number, enum: [...]}, {type: null}]`` for the
    nullable-and-constrained case rather than mixing ``null`` directly into
    one ``enum`` array - both ``anyOf``-for-nullable and
    ``enum``-of-numbers are individually well-documented Structured Outputs
    features; mixing ``null`` into a single ``enum`` alongside numbers is
    not, so this sticks to the confirmed-supported combination.

    Standalone (not inlined in :func:`build_default_openai_client`) so this
    policy is unit-testable without the real SDK or a key - see
    ``tests/test_llm_timing_openai_provider.py``.
    """
    start_end_property: dict = {"type": ["number", "null"]}
    if pass_name in ("end_scan", "end_validate") and frame_timestamps_s:
        allowed = sorted(set(frame_timestamps_s))
        start_end_property = {"anyOf": [{"type": "number", "enum": allowed}, {"type": "null"}]}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(_BASE_RESPONSE_SCHEMA_REQUIRED),
        "properties": {
            "status": {"type": "string", "enum": ["confirmed", "abstain"]},
            "start_s": start_end_property,
            "end_s": start_end_property,
            "start_uncertainty_s": {"type": ["number", "null"]},
            "end_uncertainty_s": {"type": ["number", "null"]},
            "confidence": {"type": "number"},
            "reason_codes": {"type": "array", "items": {"type": "string"}},
            "evidence_frame_timestamps_s": {"type": "array", "items": {"type": "number"}},
            "raw_notes": {"type": "string"},
        },
    }


def build_default_openai_client(api_key_env_var: str = "OPENAI_API_KEY") -> OpenAIClient:
    """Construct the real OpenAI SDK-backed client. Not exercised by tests.

    Imports the SDK lazily, inside this function body, so nothing in this
    package requires it to be installed unless this specific factory is
    actually called - which nothing in this spike does yet. Reads the key
    from the named environment variable only (never a literal), matching
    the "server-side environment/secret store only, never committed"
    requirement.

    Built against the official ``openai`` Python SDK
    (``pip install openai`` - see ``requirements-llm-spike.txt``), using
    the Responses API (``client.responses.create``) with Structured
    Outputs (``text.format.type == "json_schema"``, ``strict=True``) so the
    model's own JSON is schema-validated by the API before it ever reaches
    ``provider.parse_raw_response`` - which still re-validates everything
    itself regardless, per this whole package's "never confidently wrong"
    discipline; a schema-valid response is not automatically a *grounded*
    one (see ``pipeline._validate_grounding``).

    The SDK's own client has built-in retry logic (default
    ``max_retries=2``, an httpx-layer retry independent of anything in
    ``provider.py``); constructed with ``max_retries=0`` here so
    ``OpenAITimingProvider.analyze``'s retry loop is the single source of
    truth for retry counting/backoff/diagnostics - otherwise the two would
    compound and ``RawProviderResponse.retries`` would undercount what
    actually happened.
    """
    import os

    api_key = os.environ.get(api_key_env_var)
    if not api_key:
        raise ConfigurationError(
            f"No OpenAI API key found in the {api_key_env_var} environment variable.",
            detail="set it server-side before constructing a live OpenAI client",
        )

    import base64

    from openai import (
        APIConnectionError,
        APIStatusError,
        InternalServerError,
        OpenAI,
    )

    client = OpenAI(api_key=api_key, max_retries=0)

    def _to_openai_part(part: dict) -> dict:
        if "text" in part:
            return {"type": "input_text", "text": part["text"]}
        inline = part["inline_data"]
        data = base64.b64encode(inline["image_bytes"]).decode("ascii")
        return {
            "type": "input_image",
            "image_url": f"data:{inline['mime_type']};base64,{data}",
            "detail": "auto",
        }

    class _RealOpenAIClient:
        def generate_content(
            self, *, model: str, parts: list[dict], generation_config: dict
        ) -> OpenAICallResult:
            openai_parts = [_to_openai_part(part) for part in parts]
            schema = _response_schema_for_pass(
                generation_config.get("pass_name", ""),
                generation_config.get("frame_timestamps_s", []),
            )
            try:
                response = client.responses.create(
                    model=model,
                    input=[{"role": "user", "content": openai_parts}],
                    temperature=generation_config.get("temperature"),
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "timing_verdict",
                            "schema": schema,
                            "strict": True,
                        }
                    },
                )
            except InternalServerError as exc:
                # 5xx - worth a bounded retry. No fixed status_code on this
                # exception type (it covers the whole >=500 range), so
                # classified by type rather than by _classify_status_error.
                raise TransientProviderError(str(exc)) from exc
            except APIConnectionError as exc:
                # Network-level failure (including APITimeoutError, a
                # subclass of this) - not an HTTP status at all, worth a
                # bounded retry just like a bare ConnectionError/TimeoutError.
                raise TransientProviderError(str(exc)) from exc
            except APIStatusError as exc:
                # Every other 4xx (bad request, auth, not found, ...) minus
                # the 429 rate-limit case _classify_status_error retries.
                raise _classify_status_error(exc) from exc

            usage = response.usage
            return OpenAICallResult(
                text=response.output_text,
                prompt_tokens=usage.input_tokens if usage else None,
                completion_tokens=usage.output_tokens if usage else None,
            )

    return _RealOpenAIClient()
