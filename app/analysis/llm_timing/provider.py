"""The provider boundary: everything network/vendor-specific stops here.

``TimingProvider`` is the single seam a real vendor integration (OpenAI,
Anthropic, Google, ...) plugs into. Nothing above this module knows or cares
which vendor is behind it - see ``diagnostics/llm_spike/DESIGN.md`` for why
that matters (the supervisor's authorization requires provider/model
selection to stay replaceable).

No concrete network-calling provider is implemented in this spike yet: that
needs an API key, which is explicitly deferred. ``StubTimingProvider`` is the
only implementation here, used for tests and for the evaluation harness's
plumbing until a real key is supplied.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from .schema import TimingStatus, TimingVerdict


@dataclass(frozen=True)
class TimedFrame:
    """One deterministically-extracted frame, ready to hand to a provider.

    ``image_bytes`` is an already-encoded image (JPEG - see
    ``video.reader.encode_jpeg``), not a raw ndarray, so this dataclass is
    safe to log, hash and replay without an OpenCV dependency at the call
    site.
    """

    timestamp_s: float
    image_bytes: bytes
    media_type: str = "image/jpeg"


@dataclass(frozen=True)
class ProviderRequest:
    """One call's worth of work: a deterministically-extracted frame batch.

    Frame extraction is owned by ``pipeline.py``, not by the provider - the
    supervisor's authorization calls for "deterministic frame
    extraction/cropping" as a pipeline responsibility, not something left to
    a vendor's own (unaudited, potentially non-deterministic) video
    ingestion. ``pass_name`` records which stage of the coarse-to-fine
    strategy this batch belongs to, purely for logging.
    """

    prompt_version: str
    prompt_text: str
    frames: tuple[TimedFrame, ...]
    pass_name: str  # "coarse" | "fine"


@dataclass(frozen=True)
class RawProviderResponse:
    """What came back from the provider call, before any trust is extended.

    ``raw_text`` is exactly what the provider returned - never read
    programmatically outside of :func:`parse_raw_response`. ``error`` is set
    when the call itself failed (timeout, 5xx, network); a set ``error``
    always parses to an ABSTAIN verdict regardless of ``raw_text``.
    """

    model_id: str
    raw_text: str
    latency_s: float
    retries: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class TimingProvider(Protocol):
    """A production-callable timing analysis backend.

    Deliberately *not* "an interactive chat session" - the supervisor's
    authorization is explicit that the runtime design must not depend on one.
    A concrete implementation wraps a real vendor SDK call; it must not block
    on anything a human would need to answer interactively.
    """

    def analyze(self, request: ProviderRequest) -> RawProviderResponse: ...


class StubTimingProvider:
    """Deterministic, offline provider for tests and harness plumbing.

    Takes a callable so tests can script exactly what "the model said" for a
    given request without touching a network. No default canned answer is
    provided - a test that doesn't specify one is a test that hasn't decided
    what it's checking.
    """

    def __init__(self, respond: Callable[[ProviderRequest], RawProviderResponse]) -> None:
        self._respond = respond
        self.calls: list[ProviderRequest] = []

    def analyze(self, request: ProviderRequest) -> RawProviderResponse:
        self.calls.append(request)
        return self._respond(request)


def canned_json_response(
    *,
    model_id: str = "stub-model",
    start_s: float,
    end_s: float,
    start_uncertainty_s: float = 0.1,
    end_uncertainty_s: float = 0.1,
    confidence: float = 0.9,
    reason_codes: tuple[str, ...] = (),
    evidence_frame_timestamps_s: tuple[float, ...] = (),
    latency_s: float = 0.01,
) -> RawProviderResponse:
    """Build a well-formed CONFIRMED raw response - the common test case."""
    payload = {
        "status": "confirmed",
        "start_s": start_s,
        "end_s": end_s,
        "start_uncertainty_s": start_uncertainty_s,
        "end_uncertainty_s": end_uncertainty_s,
        "confidence": confidence,
        "reason_codes": list(reason_codes),
        "evidence_frame_timestamps_s": list(evidence_frame_timestamps_s),
    }
    return RawProviderResponse(
        model_id=model_id, raw_text=json.dumps(payload), latency_s=latency_s
    )


def _abstain(
    *, reason_code: str, model_id: str, prompt_version: str, raw_notes: str = ""
) -> TimingVerdict:
    return TimingVerdict.abstain(
        reason_codes=(reason_code,),
        model_id=model_id,
        prompt_version=prompt_version,
        raw_notes=raw_notes,
    )


def parse_raw_response(
    response: RawProviderResponse, *, prompt_version: str, min_confidence: float
) -> TimingVerdict:
    """Turn an untrusted raw response into a validated verdict.

    Every failure mode - a provider-level error, unparseable JSON, a missing
    field, a value that fails :class:`TimingVerdict`'s own invariants, or
    confidence below ``min_confidence`` - converges on ABSTAIN. Nothing here
    ever fabricates a start/end time to paper over a gap.
    """
    if response.error is not None:
        return _abstain(
            reason_code="provider_error",
            model_id=response.model_id,
            prompt_version=prompt_version,
            raw_notes=response.error,
        )

    try:
        payload = json.loads(response.raw_text)
    except json.JSONDecodeError as exc:
        return _abstain(
            reason_code="malformed_output",
            model_id=response.model_id,
            prompt_version=prompt_version,
            raw_notes=f"not valid JSON: {exc}",
        )

    if not isinstance(payload, dict):
        return _abstain(
            reason_code="malformed_output",
            model_id=response.model_id,
            prompt_version=prompt_version,
            raw_notes="response JSON was not an object",
        )

    status_raw = payload.get("status")
    if status_raw not in ("confirmed", "abstain"):
        return _abstain(
            reason_code="malformed_output",
            model_id=response.model_id,
            prompt_version=prompt_version,
            raw_notes=f"unrecognised status: {status_raw!r}",
        )

    reason_codes = tuple(payload.get("reason_codes") or ())
    if not all(isinstance(code, str) for code in reason_codes):
        return _abstain(
            reason_code="malformed_output",
            model_id=response.model_id,
            prompt_version=prompt_version,
            raw_notes="reason_codes contained a non-string entry",
        )

    if status_raw == "abstain":
        codes = reason_codes or ("no_continuous_stream_found",)
        return TimingVerdict.abstain(
            reason_codes=codes,
            model_id=response.model_id,
            prompt_version=prompt_version,
            confidence=float(payload.get("confidence", 0.0) or 0.0),
            raw_notes=str(payload.get("raw_notes", "")),
        )

    try:
        evidence = tuple(float(t) for t in (payload.get("evidence_frame_timestamps_s") or ()))
        verdict = TimingVerdict(
            status=TimingStatus.CONFIRMED,
            start_s=float(payload["start_s"]),
            end_s=float(payload["end_s"]),
            start_uncertainty_s=float(payload.get("start_uncertainty_s", 0.0)),
            end_uncertainty_s=float(payload.get("end_uncertainty_s", 0.0)),
            confidence=float(payload["confidence"]),
            reason_codes=reason_codes,
            evidence_frame_timestamps_s=evidence,
            model_id=response.model_id,
            prompt_version=prompt_version,
            raw_notes=str(payload.get("raw_notes", "")),
        )
    except (KeyError, TypeError, ValueError) as exc:
        return _abstain(
            reason_code="malformed_output",
            model_id=response.model_id,
            prompt_version=prompt_version,
            raw_notes=f"schema violation: {exc}",
        )
    except Exception as exc:  # TimingVerdict.__post_init__'s ConfigurationError
        return _abstain(
            reason_code="invalid_invariant",
            model_id=response.model_id,
            prompt_version=prompt_version,
            raw_notes=str(exc),
        )

    if verdict.confidence < min_confidence:
        return _abstain(
            reason_code="low_confidence",
            model_id=response.model_id,
            prompt_version=prompt_version,
            raw_notes=(
                f"confidence {verdict.confidence:.3f} below floor {min_confidence:.3f} "
                f"(would-have-been start_s={verdict.start_s} end_s={verdict.end_s})"
            ),
        )

    return verdict
