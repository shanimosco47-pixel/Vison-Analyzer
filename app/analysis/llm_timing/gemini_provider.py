"""Gemini provider adapter - the first real ``TimingProvider`` implementation.

Per the supervisor's authorization: start with ``gemini-2.5-flash-lite`` as
the low-cost image/video engine, keep the exact model identifier
configurable/pinned (never implicit), and unit-test the adapter with an
injected fake client - no network call anywhere in the test suite.

This module never imports the real Gemini SDK at module load time - only
:func:`build_default_gemini_client` does, lazily, inside its own body, so
nothing in this package (or its tests) requires the SDK to be installed
unless that one factory is actually called. **Nothing calls it yet.** No
live request happens anywhere in this spike - see
``diagnostics/llm_spike/DESIGN.md`` - this only makes the adapter
constructible and fully testable behind the existing ``TimingProvider`` seam,
ready to be wired in once a key is supplied and reviewed.

Built against the current GA ``google-genai`` SDK (``from google import
genai``), not the deprecated ``google-generativeai`` package - see
``requirements-llm-spike.txt`` for the (optional, not installed by default)
dependency.
"""

from __future__ import annotations

import base64
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

DEFAULT_GEMINI_MODEL_ID = "gemini-2.5-flash-lite"

# Exception types treated as retryable even when a client raises them bare
# (not wrapped in TransientProviderError) - the common shapes of "the
# network hiccuped", which a bounded retry can plausibly fix.
_RETRYABLE_BARE_EXCEPTIONS: tuple[type[Exception], ...] = (TimeoutError, ConnectionError)


@dataclass(frozen=True)
class GeminiCallResult:
    """What a :class:`GeminiClient` hands back for one call.

    Deliberately minimal and vendor-shape-free: a fake test client builds
    one trivially, and nothing outside this module needs to know the real
    SDK's response object shape.
    """

    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class GeminiClient(Protocol):
    """The seam a fake test client and the real SDK wrapper both implement.

    Kept intentionally narrow (one method, plain-dict parts) so a test
    double needs nothing beyond stdlib to satisfy it. Raise
    ``provider.TransientProviderError`` for a retryable failure (timeout,
    429, 5xx) and ``provider.PermanentProviderError`` for one that isn't
    (auth, validation, payload too large) - see
    :class:`GeminiTimingProvider`'s retry policy. Any other exception is
    treated as permanent by default (see the class docstring for why).
    """

    def generate_content(
        self, *, model: str, parts: list[dict], generation_config: dict
    ) -> GeminiCallResult: ...


class GeminiTimingProvider:
    """``TimingProvider`` backed by a Gemini model, via an injected client.

    The client is injected specifically so this is unit-testable without a
    network call or an installed SDK - see
    :func:`build_default_gemini_client` for the one place that actually
    wraps the real SDK, which this class never imports or references.

    Retry policy: only ``TransientProviderError`` (or a bare
    ``TimeoutError``/``ConnectionError``) is retried, up to ``max_retries``
    times. The real wrapper (``build_default_gemini_client``) raises
    ``TransientProviderError`` for a 408 or 429, ``PermanentProviderError``
    for every other 4xx (auth, validation, payload too large) - see
    ``_client_error_status_code``. The delay before a retry uses a
    server-suggested ``Retry-After`` when the raised error carries one
    (``TransientProviderError.retry_after_s``), otherwise exponential
    backoff (``backoff_base_s * 2**attempt``, capped at ``backoff_max_s``).
    Every other exception - including ``PermanentProviderError`` and
    anything unclassified - is treated as non-retryable: retrying an auth
    failure or a request Gemini already rejected as too large wastes time
    and money without any chance of a different outcome.
    """

    def __init__(
        self,
        client: GeminiClient,
        *,
        model_id: str = DEFAULT_GEMINI_MODEL_ID,
        max_retries: int = 2,
        temperature: float = 0.0,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 10.0,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model_id.strip():
            raise ConfigurationError("GeminiTimingProvider needs a non-empty model_id.")
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
        that escapes this method - ``provider.parse_raw_response`` still
        owns turning that into an ABSTAIN verdict.

        ``retries`` on the returned response counts retries *after* the
        initial attempt, per the contract on
        ``provider.RawProviderResponse.retries`` - a call that fails once
        and is never retried (``max_retries=0``, or a non-retryable error)
        reports ``retries=0``, not 1.
        """
        parts = _build_parts(request)
        generation_config = {
            "temperature": self._temperature,
            "response_mime_type": "application/json",
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
    """The real Gemini content-part shape: text and inline_data entries.

    One text label per frame, immediately before that frame's image, so the
    model can cite a timestamp back at us without having to infer frame
    order from position alone. A frame with ``is_candidate=True`` (only the
    end-scan trend-validation pass sets this) gets a distinguishing
    "CANDIDATE" label instead of the plain one, so the (version-pinned,
    otherwise-static) prompt text can refer to "the CANDIDATE frame" without
    needing to embed a numeric timestamp of its own.
    """
    parts: list[dict] = [{"text": request.prompt_text}]
    for frame in request.frames:
        label = "CANDIDATE frame" if frame.is_candidate else "frame"
        parts.append({"text": f"[{label} at t={frame.timestamp_s:.3f}s]"})
        parts.append(
            {
                "inline_data": {
                    "mime_type": frame.media_type,
                    "data": base64.b64encode(frame.image_bytes).decode("ascii"),
                }
            }
        )
    return parts


def _client_error_status_code(exc: Exception) -> int | None:
    """Best-effort extraction of an HTTP status code from a
    ``google.genai.errors.ClientError``.

    UNVERIFIED against live SDK documentation - this sandbox's network
    access to ai.google.dev is blocked, so the exact attribute name on
    this exception type has not been confirmed against the official
    source. ``code`` and ``status_code`` are both checked since different
    SDK versions/error hierarchies commonly use one or the other; if
    neither is present this returns ``None`` and the caller falls back to
    treating the error as a non-retryable 4xx (the safer default, per
    ``GeminiTimingProvider``'s retry policy). Confirm this against the
    actual installed SDK version before depending on 408/429 retry
    behaviour in a live run.
    """
    for attr in ("code", "status_code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


# Status codes worth retrying even though they arrive as a ClientError
# (4xx): the request may genuinely succeed on a later attempt. Every other
# 4xx (400 bad request, 401/403 auth, 413 payload too large, 422
# validation, ...) is left permanent.
_RETRYABLE_CLIENT_ERROR_STATUS_CODES = frozenset({408, 429})


def _classify_client_error(exc: Exception) -> ProviderCallError:
    """Turn a ``google.genai.errors.ClientError`` into the right
    ``ProviderCallError`` subtype, as a standalone function so the policy
    (which status codes are retryable) is unit-testable with a fake
    exception object, independent of whether the real SDK is installed.
    """
    status_code = _client_error_status_code(exc)
    if status_code in _RETRYABLE_CLIENT_ERROR_STATUS_CODES:
        return TransientProviderError(str(exc), retry_after_s=_client_error_retry_after_s(exc))
    return PermanentProviderError(str(exc))


def _client_error_retry_after_s(exc: Exception) -> float | None:
    """Best-effort extraction of a server-suggested retry delay.

    Same verification caveat as :func:`_client_error_status_code`: the
    attribute/header path checked here is a reasonable guess, not a
    confirmed reading of the SDK's actual exception shape. Returns
    ``None`` (falls back to the caller's own exponential backoff) if
    nothing matches, rather than guessing a number.
    """
    direct = getattr(exc, "retry_after", None)
    if isinstance(direct, int | float):
        return float(direct)
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is not None:
        raw = headers.get("Retry-After") if hasattr(headers, "get") else None
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
    return None


DEFAULT_REQUEST_TIMEOUT_S = 120.0


def _looks_like_a_timeout(exc: Exception) -> bool:
    """Best-effort, class-name-based check for whether ``exc`` represents a
    request that exceeded its configured timeout - not an ``isinstance``
    check against a confirmed exception type, since this sandbox's network
    access to ai.google.dev is blocked and the google-genai SDK's exact
    timeout exception shape has not been verified against live
    documentation, same caveat as every other SDK-shape note in this
    module. Matches ``httpx.TimeoutException``/``ReadTimeout``/
    ``ConnectTimeout`` and the stdlib ``TimeoutError`` alike, which cover
    every plausible shape without depending on which one the installed
    SDK version actually raises."""
    return "timeout" in type(exc).__name__.lower()


def build_default_gemini_client(
    api_key_env_var: str = "GEMINI_API_KEY",
    *,
    api_key: str | None = None,
    timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S,
) -> GeminiClient:
    """Construct the real Gemini SDK-backed client. Not exercised by tests.

    Imports the SDK lazily, inside this function body, so nothing in this
    package requires it to be installed unless this specific factory is
    actually called - which nothing in this spike does yet.

    ``api_key``, if given, is used directly instead of reading
    ``api_key_env_var`` from the environment - the resolved value from an
    OS-protected secret store (see ``app/services/secret_store.py``),
    handed straight to the SDK client constructor and never round-tripped
    through ``os.environ`` (a Codex review of the app-integration slice
    caught the earlier design doing exactly that: it left a saved key
    process-wide for the server's entire lifetime with no cleanup - the
    opposite of the OS-secret-store boundary the key was saved to protect
    in the first place). The environment-variable path is unchanged: when
    ``api_key`` is not given, this still reads ``api_key_env_var`` from
    ``os.environ`` exactly as before, matching the "server-side
    environment/secret store only, never committed" requirement - an
    operator-managed environment-variable reference keeps working exactly
    as it did.

    ``timeout_s`` bounds every request via ``HttpOptions.timeout``
    (milliseconds, per the SDK - UNVERIFIED against live documentation
    like the rest of this function's SDK-shape notes, for the same reason;
    confirm the unit against the installed SDK version before depending on
    it precisely). Without an explicit timeout a stalled request can hang
    indefinitely despite the caller's own staged-progress UI. Whatever
    exception type a timeout actually raises,
    ``GeminiTimingProvider.analyze``'s own catch-all already converges any
    unclassified exception on a safe, non-crashing failed state - see
    :func:`_looks_like_a_timeout` for the best-effort message improvement
    layered on top of that existing guarantee, not a precondition for it.

    Built against the current GA ``google-genai`` SDK
    (``pip install google-genai`` - see ``requirements-llm-spike.txt``),
    not the deprecated ``google-generativeai`` package.
    """
    import os

    resolved_key = api_key if api_key is not None else os.environ.get(api_key_env_var)
    if not resolved_key:
        raise ConfigurationError(
            f"No Gemini API key found in the {api_key_env_var} environment variable.",
            detail="set it server-side before constructing a live Gemini client",
        )

    from google import genai
    from google.genai import errors as genai_errors
    from google.genai import types

    client = genai.Client(
        api_key=resolved_key,
        http_options=types.HttpOptions(timeout=int(timeout_s * 1000)),
    )

    def _to_genai_part(part: dict):
        if "text" in part:
            return types.Part.from_text(text=part["text"])
        inline = part["inline_data"]
        return types.Part.from_bytes(
            data=base64.b64decode(inline["data"]), mime_type=inline["mime_type"]
        )

    class _RealGeminiClient:
        def generate_content(
            self, *, model: str, parts: list[dict], generation_config: dict
        ) -> GeminiCallResult:
            genai_parts = [_to_genai_part(part) for part in parts]
            config = types.GenerateContentConfig(
                temperature=generation_config.get("temperature"),
                response_mime_type=generation_config.get("response_mime_type"),
            )
            try:
                response = client.models.generate_content(
                    model=model, contents=genai_parts, config=config
                )
            except genai_errors.ClientError as exc:
                # Most 4xx (bad request, auth, payload-too-large, schema
                # validation) are not retryable. 408/429 are the
                # exceptions - see _classify_client_error.
                raise _classify_client_error(exc) from exc
            except genai_errors.ServerError as exc:
                # 5xx - worth a bounded retry.
                raise TransientProviderError(str(exc)) from exc
            except Exception as exc:
                # A request that exceeded http_options.timeout above raises
                # from the SDK's own transport layer, not a genai_errors
                # subclass - see _looks_like_a_timeout's own docstring for
                # why this is a best-effort name check rather than a
                # confirmed isinstance check. Non-timeout-shaped exceptions
                # re-raise completely unchanged; GeminiTimingProvider.analyze's
                # own catch-all already handles those safely, exactly as it
                # did before this branch existed.
                if _looks_like_a_timeout(exc):
                    raise TransientProviderError(
                        f"The request to Gemini did not complete within {timeout_s:.0f}s "
                        "and was abandoned."
                    ) from exc
                raise

            usage = getattr(response, "usage_metadata", None)
            return GeminiCallResult(
                text=response.text,
                prompt_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
                completion_tokens=(
                    getattr(usage, "candidates_token_count", None) if usage else None
                ),
            )

    return _RealGeminiClient()
