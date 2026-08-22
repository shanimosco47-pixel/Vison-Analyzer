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
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Protocol

from ...errors import ConfigurationError
from .provider import ProviderRequest, RawProviderResponse

DEFAULT_GEMINI_MODEL_ID = "gemini-2.5-flash-lite"


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
    double needs nothing beyond stdlib to satisfy it.
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
    """

    def __init__(
        self,
        client: GeminiClient,
        *,
        model_id: str = DEFAULT_GEMINI_MODEL_ID,
        max_retries: int = 1,
        temperature: float = 0.0,
    ) -> None:
        if not model_id.strip():
            raise ConfigurationError("GeminiTimingProvider needs a non-empty model_id.")
        if max_retries < 0:
            raise ConfigurationError("max_retries must not be negative.")
        self._client = client
        self._model_id = model_id
        self._max_retries = max_retries
        self._temperature = temperature

    def analyze(self, request: ProviderRequest) -> RawProviderResponse:
        """Send one frame batch, total: any client exception becomes an
        ``error``-carrying :class:`RawProviderResponse`, never an exception
        that escapes this method - ``provider.parse_raw_response`` still
        owns turning that into an ABSTAIN verdict.
        """
        parts = _build_parts(request)
        generation_config = {
            "temperature": self._temperature,
            "response_mime_type": "application/json",
        }

        attempts = 0
        last_error: Exception | None = None
        started = time.monotonic()
        while attempts <= self._max_retries:
            try:
                result = self._client.generate_content(
                    model=self._model_id, parts=parts, generation_config=generation_config
                )
            except Exception as exc:  # the client can raise anything vendor-specific
                last_error = exc
                attempts += 1
                continue
            return RawProviderResponse(
                model_id=self._model_id,
                raw_text=result.text,
                latency_s=time.monotonic() - started,
                retries=attempts,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
            )

        return RawProviderResponse(
            model_id=self._model_id,
            raw_text="",
            latency_s=time.monotonic() - started,
            retries=attempts,
            error=str(last_error) if last_error is not None else "unknown error",
        )


def _build_parts(request: ProviderRequest) -> list[dict]:
    """The real Gemini content-part shape: text and inline_data entries.

    One text label per frame, immediately before that frame's image, so the
    model can cite a timestamp back at us without having to infer frame
    order from position alone.
    """
    parts: list[dict] = [{"text": request.prompt_text}]
    for frame in request.frames:
        parts.append({"text": f"[frame at t={frame.timestamp_s:.3f}s]"})
        parts.append(
            {
                "inline_data": {
                    "mime_type": frame.media_type,
                    "data": base64.b64encode(frame.image_bytes).decode("ascii"),
                }
            }
        )
    return parts


def build_default_gemini_client(api_key_env_var: str = "GEMINI_API_KEY") -> GeminiClient:
    """Construct the real Gemini SDK-backed client. Not exercised by tests.

    Imports the SDK lazily, inside this function body, so nothing in this
    package requires it to be installed unless this specific factory is
    actually called - which nothing in this spike does yet. Reads the key
    from the named environment variable only (never a literal), matching
    the "server-side environment/secret store only, never committed"
    requirement.
    """
    import os

    api_key = os.environ.get(api_key_env_var)
    if not api_key:
        raise ConfigurationError(
            f"No Gemini API key found in the {api_key_env_var} environment variable.",
            detail="set it server-side before constructing a live Gemini client",
        )

    import google.generativeai as genai

    genai.configure(api_key=api_key)

    class _RealGeminiClient:
        def generate_content(
            self, *, model: str, parts: list[dict], generation_config: dict
        ) -> GeminiCallResult:
            gemini_model = genai.GenerativeModel(model)
            response = gemini_model.generate_content(parts, generation_config=generation_config)
            usage = getattr(response, "usage_metadata", None)
            return GeminiCallResult(
                text=response.text,
                prompt_tokens=getattr(usage, "prompt_token_count", None) if usage else None,
                completion_tokens=(
                    getattr(usage, "candidates_token_count", None) if usage else None
                ),
            )

    return _RealGeminiClient()
