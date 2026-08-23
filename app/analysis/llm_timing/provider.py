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

from .redaction import sanitize_untrusted_text
from .schema import TimingStatus, TimingVerdict


class ProviderCallError(Exception):
    """Base for exceptions a ``TimingProvider`` implementation may raise to
    signal that one call failed.

    Not required - ``pipeline.run_llm_timing`` and any adapter's own retry
    logic catch and convert *any* exception, so a provider that never raises
    one of these still works. Raising one of the two subtypes below instead
    of a bare exception lets retry logic distinguish "try again" from "don't
    bother" - see ``gemini_provider.GeminiTimingProvider`` for the consumer.
    """


class TransientProviderError(ProviderCallError):
    """A failure worth retrying: timeout (408), rate limit (429), 5xx, a
    network blip. The same request might succeed on a later attempt.

    ``retry_after_s``, when a raiser can supply it (e.g. a 429's
    ``Retry-After`` header), overrides the retry loop's own computed
    backoff delay for the next attempt - honoring what the server actually
    asked for beats guessing. Leave it ``None`` when there's nothing to go
    on; the caller falls back to its own exponential backoff.
    """

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class PermanentProviderError(ProviderCallError):
    """A failure that will not go away on retry: bad credentials, a 4xx
    validation error, a request that's simply too large. Retrying wastes
    time, money, and (for a rate limit that looks like a 4xx on some
    vendors) can make things worse."""


@dataclass(frozen=True)
class TimedFrame:
    """One deterministically-extracted frame, ready to hand to a provider.

    ``image_bytes`` is an already-encoded image (JPEG - see
    ``video.reader.encode_jpeg``), not a raw ndarray, so this dataclass is
    safe to log, hash and replay without an OpenCV dependency at the call
    site.

    ``is_candidate`` distinguishes one specific frame within a batch as the
    subject of the request - currently only the end-scan trend-validation
    pass (``pipeline._build_validation_frames``) uses this, to identify
    which submitted frame is the break a prior pass proposed, without
    putting a numeric value into the (version-pinned, otherwise-static)
    prompt text itself. See ``gemini_provider._build_parts``/
    ``openai_provider._build_parts`` for the resulting frame label.

    ``is_trend_checkpoint`` marks a frame the pipeline is suggesting as a
    *trend-confirmation* anchor - roughly a second past the candidate onset,
    then a second past that, and so on (see
    ``pipeline._suggested_trend_checkpoints``) - distinct from the dense
    frames around the candidate itself, which only localize *where* the
    onset is. A label only: the model may still cite any grounded
    timestamp in ``trend_checkpoint_timestamps_s``, and
    ``pipeline._validate_trend_checkpoints`` is what actually enforces the
    spacing contract, independent of which frames carry this label.

    ``is_contact_sheet`` marks a frame whose ``image_bytes`` is not a
    single video frame but a deterministically-composed grid of several
    frames (a "contact sheet" - see ``contact_sheet.py`` and
    ``pipeline._run_end_validation_pass``'s candidate-centred cascade),
    each panel self-labelled with its own timestamp baked into the image
    itself. ``timestamp_s`` on a contact-sheet frame is only a
    representative anchor (the candidate the sheet is centred on, or the
    look-back point it ends at) - the actual per-panel timestamps a
    provider may cite live in ``ProviderRequest.grounding_timestamps_s``,
    not in any single ``TimedFrame``.
    """

    timestamp_s: float
    image_bytes: bytes
    media_type: str = "image/jpeg"
    is_candidate: bool = False
    is_trend_checkpoint: bool = False
    is_contact_sheet: bool = False


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
    grounding_timestamps_s: tuple[float, ...] | None = None
    """Explicit override of "which timestamps a provider may legitimately
    cite back" for this request, used only when that set differs from
    ``[frame.timestamp_s for frame in frames]`` - currently only the
    contact-sheet cascade (a single composite ``TimedFrame`` standing in
    for several logical panel timestamps - see
    ``TimedFrame.is_contact_sheet``). ``None`` (the default, and every
    non-cascade request) means "derive it from ``frames`` as before" -
    both ``openai_provider``'s Structured-Outputs enum constraint and
    ``pipeline.py``'s own grounding-region construction fall back to that
    when this is unset, so every existing call site is unaffected."""


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
    evidence_frame_timestamps_s: tuple[float, ...] | None = None,
    trend_checkpoint_timestamps_s: tuple[float, ...] = (),
    possible_collapse_s: float | None = None,
    latency_s: float = 0.01,
) -> RawProviderResponse:
    """Build a well-formed CONFIRMED raw response - the common test case.

    ``evidence_frame_timestamps_s`` defaults to ``(start_s, end_s)`` rather
    than empty - a confirmed answer with no cited evidence would (correctly)
    fail ``pipeline._validate_grounding``'s grounding checks, and most
    callers of this helper are testing something else and don't want to
    think about evidence timestamps. Pass an explicit value (including
    ``()``) to test grounding failures themselves.

    ``trend_checkpoint_timestamps_s`` has no such convenience default - it
    is only meaningful for an end-validate CONFIRMED verdict, and a caller
    testing that path must decide deliberately what to cite (see
    :func:`spaced_trend_checkpoints`) rather than get a default that could
    silently satisfy or silently fail
    ``pipeline._validate_trend_checkpoints``.
    """
    if evidence_frame_timestamps_s is None:
        evidence_frame_timestamps_s = (start_s, end_s)
    payload = {
        "status": "confirmed",
        "start_s": start_s,
        "end_s": end_s,
        "start_uncertainty_s": start_uncertainty_s,
        "end_uncertainty_s": end_uncertainty_s,
        "confidence": confidence,
        "reason_codes": list(reason_codes),
        "evidence_frame_timestamps_s": list(evidence_frame_timestamps_s),
        "trend_checkpoint_timestamps_s": list(trend_checkpoint_timestamps_s),
        "possible_collapse_s": possible_collapse_s,
    }
    return RawProviderResponse(model_id=model_id, raw_text=json.dumps(payload), latency_s=latency_s)


def spaced_trend_checkpoints(
    frames: tuple[TimedFrame, ...],
    onset_ts: float,
    *,
    min_gap_s: float = 0.75,
    count: int = 2,
) -> tuple[float, ...]:
    """Test helper: pick ``count`` real, already-submitted frame timestamps
    from ``frames``, each at least ``min_gap_s`` after ``onset_ts`` and at
    least ``min_gap_s`` after the previously picked one - i.e. a
    deliberately compliant trend-checkpoint citation, for a stub provider
    that wants an end-validate response to actually reach CONFIRMED under
    ``pipeline._validate_trend_checkpoints``. Returns fewer than ``count``
    entries (possibly none) if ``frames`` doesn't have enough room; callers
    testing the gate itself should pass a deliberately non-compliant tuple
    by hand instead of using this helper.
    """
    available = sorted({frame.timestamp_s for frame in frames if frame.timestamp_s > onset_ts})
    picked: list[float] = []
    floor = onset_ts + min_gap_s
    for ts in available:
        if ts >= floor - 1e-9:
            picked.append(ts)
            floor = ts + min_gap_s
            if len(picked) >= count:
                break
    return tuple(picked)


def cascade_confirmed_response(
    request: ProviderRequest,
    *,
    model_id: str = "stub-model",
    confidence: float = 0.9,
    onset_index: int | None = None,
) -> RawProviderResponse:
    """Test helper: a compliant CONFIRMED response for a contact-sheet
    cascade request (see ``pipeline._run_end_validation_pass`` and
    ``contact_sheet.py``) - the composite-image analogue of
    :func:`canned_json_response`/:func:`spaced_trend_checkpoints` for the
    older dense-frame shape.

    Picks ``request.grounding_timestamps_s`` (the panel timestamps the
    contact sheet actually shows - never ``request.frames``, which for a
    cascade request holds only the one composite image) rather than
    inventing timestamps, so a stub scripted this way can never
    accidentally cite something ungrounded. Defaults to the middle panel
    as the onset (matching the coarse sheet's own candidate-centring) and
    every later panel as a trend checkpoint; pass ``onset_index`` to pick a
    different panel (e.g. an earlier one on the refine sheet).
    """
    panels = sorted(request.grounding_timestamps_s or ())
    if not panels:
        raise ValueError(
            "cascade_confirmed_response needs request.grounding_timestamps_s - "
            "this stub was called with a non-cascade request"
        )
    index = onset_index if onset_index is not None else len(panels) // 2
    onset = panels[index]
    checkpoints = tuple(ts for ts in panels if ts > onset)
    return canned_json_response(
        model_id=model_id,
        start_s=onset,
        end_s=onset,
        confidence=confidence,
        evidence_frame_timestamps_s=(onset, *checkpoints[:2]),
        trend_checkpoint_timestamps_s=checkpoints,
    )


def cascade_abstain_response(
    request: ProviderRequest,
    *,
    model_id: str = "stub-model",
    reason_codes: tuple[str, ...] = ("ambiguous_trend",),
    possible_collapse_index: int | None = None,
    latency_s: float = 0.01,
) -> RawProviderResponse:
    """Test helper: a compliant ABSTAIN response for a contact-sheet cascade
    request - optionally citing one of the sheet's own panels as
    ``possible_collapse_s`` (the coarse-sheet escape hatch that seeds the
    denser refine look-back - see ``TimingVerdict.possible_collapse_s``).
    ``possible_collapse_index=None`` (the default) means "no possible
    collapse to point at", i.e. a plain, terminal abstain.
    """
    panels = sorted(request.grounding_timestamps_s or ())
    possible_collapse_s = (
        panels[possible_collapse_index] if possible_collapse_index is not None else None
    )
    payload = {
        "status": "abstain",
        "start_s": None,
        "end_s": None,
        "start_uncertainty_s": None,
        "end_uncertainty_s": None,
        "confidence": 0.0,
        "reason_codes": list(reason_codes),
        "evidence_frame_timestamps_s": [],
        "trend_checkpoint_timestamps_s": [],
        "possible_collapse_s": possible_collapse_s,
        "raw_notes": "",
    }
    return RawProviderResponse(model_id=model_id, raw_text=json.dumps(payload), latency_s=latency_s)


def _abstain(
    *, reason_code: str, model_id: str, prompt_version: str, raw_notes: str = ""
) -> TimingVerdict:
    return TimingVerdict.abstain(
        reason_codes=(reason_code,),
        model_id=model_id,
        prompt_version=prompt_version,
        raw_notes=sanitize_untrusted_text(raw_notes),
    )


def _parse_confirmed(
    payload: dict,
    reason_codes: tuple[str, ...],
    response: RawProviderResponse,
    prompt_version: str,
) -> TimingVerdict:
    """Build the CONFIRMED verdict, or raise so the caller converges on ABSTAIN.

    Every conversion from an untrusted payload value happens here, inside
    one narrow scope, so the caller only needs two except clauses to make
    the whole thing total (see :func:`parse_raw_response`).
    """
    evidence = tuple(float(t) for t in (payload.get("evidence_frame_timestamps_s") or ()))
    trend_checkpoints = tuple(
        float(t) for t in (payload.get("trend_checkpoint_timestamps_s") or ())
    )
    possible_collapse_raw = payload.get("possible_collapse_s")
    possible_collapse = float(possible_collapse_raw) if possible_collapse_raw is not None else None
    return TimingVerdict(
        status=TimingStatus.CONFIRMED,
        start_s=float(payload["start_s"]),
        end_s=float(payload["end_s"]),
        start_uncertainty_s=float(payload.get("start_uncertainty_s", 0.0)),
        end_uncertainty_s=float(payload.get("end_uncertainty_s", 0.0)),
        confidence=float(payload["confidence"]),
        reason_codes=reason_codes,
        evidence_frame_timestamps_s=evidence,
        trend_checkpoint_timestamps_s=trend_checkpoints,
        possible_collapse_s=possible_collapse,
        model_id=response.model_id,
        prompt_version=prompt_version,
        raw_notes=sanitize_untrusted_text(str(payload.get("raw_notes", ""))),
    )


def _parse_abstain(
    payload: dict,
    reason_codes: tuple[str, ...],
    response: RawProviderResponse,
    prompt_version: str,
) -> TimingVerdict:
    """Build the model-requested ABSTAIN verdict, or raise (see :func:`_parse_confirmed`)."""
    codes = reason_codes or ("no_continuous_stream_found",)
    possible_collapse_raw = payload.get("possible_collapse_s")
    possible_collapse = float(possible_collapse_raw) if possible_collapse_raw is not None else None
    return TimingVerdict.abstain(
        reason_codes=codes,
        model_id=response.model_id,
        prompt_version=prompt_version,
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        raw_notes=sanitize_untrusted_text(str(payload.get("raw_notes", ""))),
        possible_collapse_s=possible_collapse,
    )


def parse_raw_response(
    response: RawProviderResponse, *, prompt_version: str, min_confidence: float
) -> TimingVerdict:
    """Turn an untrusted raw response into a validated verdict.

    Every failure mode - a provider-level error, unparseable JSON, a missing
    field, a non-numeric or NaN/Infinity numeric field (in either a
    "confirmed" or an "abstain" payload), a value that fails
    :class:`TimingVerdict`'s own invariants, or confidence below
    ``min_confidence`` - converges on ABSTAIN. This function must be total:
    no shape or content of ``response.raw_text`` may raise out of it.
    Nothing here ever fabricates a start/end time to paper over a gap.

    This only validates the verdict's own internal shape (finite numbers,
    ``end_s >= start_s``, etc.) - it does not know whether ``start_s`` falls
    inside the video, or whether ``evidence_frame_timestamps_s`` corresponds
    to frames actually sent. That context-aware check happens in
    ``pipeline.py``'s ``_validate_verdict_against_request``, which has the
    video duration and the exact frames of this request.
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

    reason_codes_raw = payload.get("reason_codes")
    if reason_codes_raw is None:
        reason_codes_raw = []
    if not isinstance(reason_codes_raw, list) or not all(
        isinstance(code, str) for code in reason_codes_raw
    ):
        return _abstain(
            reason_code="malformed_output",
            model_id=response.model_id,
            prompt_version=prompt_version,
            raw_notes="reason_codes was not a list of strings",
        )
    reason_codes = tuple(reason_codes_raw)

    # Both branches below convert untrusted values (float(), TimingVerdict's
    # own invariants) and must therefore be wrapped identically: a
    # non-numeric/missing field is a schema violation (malformed_output); a
    # numeric value that is finite-but-invalid (NaN, Infinity, out of [0,1],
    # end before start) fails TimingVerdict.__post_init__'s ConfigurationError
    # (invalid_invariant). Neither may escape this function.
    try:
        if status_raw == "abstain":
            verdict = _parse_abstain(payload, reason_codes, response, prompt_version)
        else:
            verdict = _parse_confirmed(payload, reason_codes, response, prompt_version)
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

    if verdict.status is TimingStatus.ABSTAIN:
        return verdict

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
