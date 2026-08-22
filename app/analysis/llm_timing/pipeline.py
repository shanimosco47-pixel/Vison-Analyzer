"""Coarse-to-fine orchestration: the one place that decides which frames a
provider gets to see, and what happens when it abstains.

This reproduces, as a deterministic and auditable pipeline, the two-pass
strategy that worked by hand against two real clips before this spike
existed (see ``diagnostics/llm_spike/DESIGN.md``): sample sparsely across
the whole clip first, then re-extract every frame inside the short windows
that pass suggested, and ask again for a precise answer.

Nothing here fabricates a result. A provider abstention, a parse failure, or
a coarse pass that finds no plausible window all converge on the same
outcome: no :class:`~app.analysis.base_detector.Event`, and the caller falls
back to the existing assisted/manual workflow - the "never confidently
wrong" contract this whole application is built under.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from ...errors import ConfigurationError
from ...video.metadata import VideoInfo
from ...video.reader import VideoReader, encode_jpeg
from ..base_detector import Event, EventStatus
from .pricing import PRICING_TABLE_VERSION, ModelPricing, estimate_cost_usd
from .prompts import (
    PROMPT_END_COARSE_V2,
    PROMPT_END_COARSE_V2_ID,
    PROMPT_END_VALIDATE_V2,
    PROMPT_END_VALIDATE_V2_ID,
    PROMPT_START_REFINE_V1,
    PROMPT_START_REFINE_V1_ID,
)
from .provider import (
    ProviderRequest,
    RawProviderResponse,
    TimedFrame,
    TimingProvider,
    parse_raw_response,
)
from .schema import TimingStatus, TimingVerdict


@dataclass(frozen=True)
class PipelineConfig:
    """Tunables for the coarse-to-fine scan. All are in seconds unless noted.

    Attributes:
        coarse_step_s: spacing between frames in the coarse pass.
        fine_margin_s: how far past the coarse pass's start/end estimate the
            fine pass extends, each direction, to make sure the true
            transition is inside the fine window even if the coarse estimate
            was off.
        fine_max_span_s: a fine window wider than this is refused rather
            than silently sending an enormous frame batch - a coarse pass
            answer that is wildly uncertain should abstain, not balloon the
            next request.
        min_confidence: verdicts below this confidence become ABSTAIN (see
            ``provider.parse_raw_response``). Applies to both passes.
        review_confidence: a CONFIRMED verdict at or above ``min_confidence``
            but below this becomes an :class:`Event` with
            ``EventStatus.REVIEW`` instead of ``CONFIRMED`` - plausible, but
            a human should look, per the same vocabulary the classical
            detectors already use.
        max_uncertainty_s: cap on a CONFIRMED verdict's
            ``start_uncertainty_s``/``end_uncertainty_s``. A verdict can be
            internally consistent (``schema.TimingVerdict``'s own
            invariants) and still be too unsure of itself to trust as
            CONFIRMED - a provider that says "start_s=4.0 +/- 12s" has not
            actually located the boundary. Exceeding this cap converges on
            ABSTAIN (reason code "uncertainty_exceeds_cap"), same as every
            other grounding failure - see ``_validate_grounding``. Default
            is deliberately tight relative to the ±0.75s acceptance target
            (gate 1) - an uncertainty near that bound is not "probably
            fine", it is "probably outside the gate" - and is itself
            uncalibrated until the blinded real-clip set (gate 3) says
            otherwise.
        target_tolerance_s: the accuracy this pipeline is trying to hit
            (matches the supervisor's own gate-1 bound, ±0.75s). Used to
            derive how sparse the fine pass's frame sampling is allowed to
            get under the request-size budget below - see
            ``_min_frames_for_window`` - not to change the grounding checks
            themselves.
        max_frame_dimension_px: every extracted frame is downscaled (never
            upscaled) so its longer side is at most this many pixels,
            before JPEG encoding - deterministic, applied identically
            regardless of provider.
        jpeg_quality: passed straight to ``video.reader.encode_jpeg``.
        max_request_bytes: an explicit ceiling on one request's estimated
            serialized size (image bytes, base64-inflated, plus the prompt
            text) - see ``_estimated_request_bytes``. Default leaves
            headroom under Gemini's ~20MB inline-request limit; set this to
            match whichever provider is actually wired in. Frames are
            thinned deterministically (``_fit_frames_to_budget``) to fit,
            never silently below the precision floor
            ``_min_frames_for_window`` derives from ``target_tolerance_s`` -
            a request that still can't fit at that floor aborts with
            ``request_too_large`` rather than sending fewer frames than the
            gate needs, or a request the provider would reject anyway.
        end_validation_baseline_s: how much clip, immediately before the
            end-coarse pass's nominated candidate, gets sent along with a
            trend-validation follow-up request as the "already established,
            continuous stream" baseline the model checks the candidate
            against - see ``_build_validation_frames`` and ``run_llm_timing``'s
            trend-validation follow-up request below. Supervisor-specified
            floor is "~1s"; this is that default.
        end_validation_horizon_s: how much clip, immediately after the
            end-coarse pass's nominated candidate, gets sent along with the
            same follow-up request as the evidence a sustained shortening
            trend must hold across before the candidate is accepted.
            Supervisor-specified floor is "~2s"; this is that default.
    """

    coarse_step_s: float = 0.5
    fine_margin_s: float = 1.5
    fine_max_span_s: float = 6.0
    min_confidence: float = 0.5
    review_confidence: float = 0.75
    max_uncertainty_s: float = 0.5
    target_tolerance_s: float = 0.75
    max_frame_dimension_px: int = 768
    jpeg_quality: int = 80
    max_request_bytes: int = 18_000_000
    end_validation_baseline_s: float = 1.0
    end_validation_horizon_s: float = 2.0

    def validate(self) -> None:
        if self.coarse_step_s <= 0:
            raise ConfigurationError("coarse_step_s must be positive.")
        if self.fine_margin_s < 0:
            raise ConfigurationError("fine_margin_s must not be negative.")
        if self.fine_max_span_s <= 0:
            raise ConfigurationError("fine_max_span_s must be positive.")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigurationError("min_confidence must be between 0 and 1.")
        if not self.min_confidence <= self.review_confidence <= 1.0:
            raise ConfigurationError("review_confidence must be between min_confidence and 1.")
        if self.max_uncertainty_s <= 0:
            raise ConfigurationError("max_uncertainty_s must be positive.")
        if self.target_tolerance_s <= 0:
            raise ConfigurationError("target_tolerance_s must be positive.")
        if self.max_frame_dimension_px <= 0:
            raise ConfigurationError("max_frame_dimension_px must be positive.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigurationError("jpeg_quality must be between 1 and 100.")
        if self.end_validation_baseline_s <= 0:
            raise ConfigurationError("end_validation_baseline_s must be positive.")
        if self.end_validation_horizon_s <= 0:
            raise ConfigurationError("end_validation_horizon_s must be positive.")
        if self.max_request_bytes <= 0:
            raise ConfigurationError("max_request_bytes must be positive.")


@dataclass
class PipelineOutcome:
    """Everything the caller and the audit log need from one run.

    ``event`` is ``None`` whenever ``verdict.status`` is ABSTAIN - the
    absence of an event *is* the abstention; callers must not infer an
    abstention from field values, only from this being unset.
    """

    verdict: TimingVerdict
    event: Event | None
    coarse_response: RawProviderResponse
    fine_response: RawProviderResponse | None
    end_coarse_response: RawProviderResponse | None = None
    """The single whole-clip (post-start), sparsely-sampled call that
    nominates one end candidate - see ``PROMPT_END_COARSE_V2``. ``None``
    for a run that never reached this phase (an abstain before it). This
    replaced a chronological multi-window scan (Codex/supervisor
    experiment, see ``diagnostics/llm_spike/DESIGN.md``): a real gate-1
    rerun showed that decomposing the end search into many narrow,
    isolated windows lost the temporal context needed to judge a
    *sustained* trend and produced a false rejection of the true break,
    while one coherent request with the same trend contract succeeded.
    Kept separate from ``fine_response`` so every pass's cost/latency/
    retries is individually inspectable, not collapsed into one number."""
    end_validation_response: RawProviderResponse | None = None
    """The single dense, bounded trend-validation follow-up call that
    confirms or rejects ``end_coarse_response``'s candidate - see
    ``_build_validation_frames``. ``None`` for a run that never reached
    this phase. A rejected or unvalidatable candidate now converges
    directly on ABSTAIN (never confidently wrong) rather than searching
    for another candidate - see ``run_llm_timing``."""

    @property
    def total_retries(self) -> int:
        return (
            self.coarse_response.retries
            + (self.fine_response.retries if self.fine_response else 0)
            + (self.end_coarse_response.retries if self.end_coarse_response else 0)
            + (self.end_validation_response.retries if self.end_validation_response else 0)
        )

    @property
    def total_latency_s(self) -> float:
        return (
            self.coarse_response.latency_s
            + (self.fine_response.latency_s if self.fine_response else 0.0)
            + (self.end_coarse_response.latency_s if self.end_coarse_response else 0.0)
            + (self.end_validation_response.latency_s if self.end_validation_response else 0.0)
        )

    @property
    def total_tokens(self) -> int | None:
        """``None`` (unknown) unless every pass that ran reported usage."""
        parts = [
            self.coarse_response,
            self.fine_response,
            self.end_coarse_response,
            self.end_validation_response,
        ]
        counts = []
        for response in parts:
            if response is None:
                continue
            if response.prompt_tokens is None or response.completion_tokens is None:
                return None
            counts.append(response.prompt_tokens + response.completion_tokens)
        return sum(counts) if counts else None

    def estimated_cost_usd(self, *, table: dict[str, ModelPricing] | None = None) -> float | None:
        """Sum of each pass's estimated cost, or ``None`` if any is unknown.

        Deliberately not "sum the known ones and ignore the rest" - a
        partial total would understate cost silently. See ``pricing.py``.
        """
        responses = [
            self.coarse_response,
            self.fine_response,
            self.end_coarse_response,
            self.end_validation_response,
        ]
        total = 0.0
        for response in responses:
            if response is None:
                continue
            cost = estimate_cost_usd(
                response.model_id,
                response.prompt_tokens,
                response.completion_tokens,
                table=table,
            )
            if cost is None:
                return None
            total += cost
        return total

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.to_dict(),
            "event": self.event.to_dict() if self.event is not None else None,
            "coarse_model_id": self.coarse_response.model_id,
            "coarse_latency_s": self.coarse_response.latency_s,
            "coarse_retries": self.coarse_response.retries,
            "coarse_prompt_tokens": self.coarse_response.prompt_tokens,
            "coarse_completion_tokens": self.coarse_response.completion_tokens,
            "fine_model_id": self.fine_response.model_id if self.fine_response else None,
            "fine_latency_s": self.fine_response.latency_s if self.fine_response else None,
            "fine_retries": self.fine_response.retries if self.fine_response else None,
            "fine_prompt_tokens": self.fine_response.prompt_tokens if self.fine_response else None,
            "fine_completion_tokens": (
                self.fine_response.completion_tokens if self.fine_response else None
            ),
            "end_coarse_model_id": (
                self.end_coarse_response.model_id if self.end_coarse_response else None
            ),
            "end_coarse_latency_s": (
                self.end_coarse_response.latency_s if self.end_coarse_response else None
            ),
            "end_coarse_retries": (
                self.end_coarse_response.retries if self.end_coarse_response else None
            ),
            "end_validation_model_id": (
                self.end_validation_response.model_id if self.end_validation_response else None
            ),
            "end_validation_latency_s": (
                self.end_validation_response.latency_s if self.end_validation_response else None
            ),
            "end_validation_retries": (
                self.end_validation_response.retries if self.end_validation_response else None
            ),
            "total_latency_s": self.total_latency_s,
            "total_retries": self.total_retries,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd(),
            "pricing_table_version": PRICING_TABLE_VERSION,
        }


def _resize_for_encoding(image: np.ndarray, max_dimension_px: int) -> np.ndarray:
    """Downscale (never upscale) so the longer side is at most this many
    pixels - deterministic, applied before every frame is encoded, so a
    request's size never depends on hidden per-provider behaviour."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension_px:
        return image
    scale = max_dimension_px / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _extract_frames(
    reader: VideoReader,
    timestamps_s: list[float],
    *,
    max_dimension_px: int,
    jpeg_quality: int,
) -> list[TimedFrame]:
    frames = []
    for ts in timestamps_s:
        image = _resize_for_encoding(reader.frame_at(ts), max_dimension_px)
        frames.append(
            TimedFrame(timestamp_s=ts, image_bytes=encode_jpeg(image, quality=jpeg_quality))
        )
    return frames


# Roughly what base64 costs on top of raw bytes (RFC 4648: 4 output bytes
# per 3 input bytes) - used only to estimate a request's serialized size
# before sending it, not to actually encode anything.
_BASE64_INFLATION = 4 / 3


def _estimated_request_bytes(frames: list[TimedFrame], prompt_text: str) -> int:
    image_bytes = sum(len(frame.image_bytes) for frame in frames)
    return int(image_bytes * _BASE64_INFLATION) + len(prompt_text.encode("utf-8"))


def _min_frames_for_window(span_s: float, target_tolerance_s: float) -> int:
    """The sparsest sampling still "sufficient for the ±0.75s gate": at
    least one sample per half-tolerance, so consecutive samples are never
    farther apart than half of what the gate allows as an error - a
    deliberate safety margin, not sampling at exactly the tolerance itself.
    """
    if span_s <= 0:
        return 2
    step = max(target_tolerance_s / 2, 1e-6)
    return max(3, math.ceil(span_s / step) + 1)


def _min_coarse_frames(duration_s: float, fine_margin_s: float) -> int:
    """The sparsest coarse sampling that still guarantees *some* coarse
    sample lands within ``fine_margin_s`` of the true transition, wherever
    it actually is - so the fine window later built around the coarse
    estimate (+/- ``fine_margin_s``) is still guaranteed to cover the true
    boundary, even in the worst case.

    A flat floor (the previous ``min_frames=4``) can't make that guarantee
    for a long clip: with only 4 samples spread across a 60s video,
    consecutive coarse samples can be ~20s apart - far past any reasonable
    ``fine_margin_s`` - so the coarse estimate could be nowhere near a
    sample, and the resulting fine window could miss the true boundary
    entirely (Codex re-review round 3, finding 1).
    """
    if fine_margin_s <= 0:
        return 4
    return max(4, math.ceil(duration_s / fine_margin_s) + 1)


def _dense_timestamps(lo: float, hi: float, step_s: float) -> list[float]:
    """Every native-fps timestamp in ``[lo, hi]``, deterministically."""
    return sorted(
        {
            round(lo + i * step_s, 6)
            for i in range(int((hi - lo) / step_s) + 2)
            if lo + i * step_s <= hi
        }
    )


def _merge_frames_sorted(*groups: list[TimedFrame]) -> list[TimedFrame]:
    """Combine independently-thinned window frame lists into the one batch
    actually sent to the provider - sorted, and de-duplicated by timestamp
    in case two windows happen to overlap (a short true efflux time can put
    the start and end fine windows close enough to share native-fps
    timestamps)."""
    by_timestamp: dict[float, TimedFrame] = {}
    for group in groups:
        for frame in group:
            by_timestamp.setdefault(frame.timestamp_s, frame)
    return [by_timestamp[ts] for ts in sorted(by_timestamp)]


def _fit_fine_windows_to_budget(
    start_frames: list[TimedFrame],
    end_frames: list[TimedFrame],
    *,
    prompt_text: str,
    max_request_bytes: int,
    start_min_frames: int,
    end_min_frames: int,
) -> tuple[list[TimedFrame] | None, list[TimedFrame] | None]:
    """Thin the start and end fine-pass windows independently, each against
    its own half of the shared request byte budget.

    Independent, not "merge both windows then thin the combined list" (the
    previous approach): merging first lets whichever window happens to have
    larger or more numerous frames crowd out the other window's density
    during thinning, and - separately - is what made
    ``_effective_tolerance_s`` blow up to the empty space *between* the two
    windows rather than either window's own achieved sampling density (see
    ``_validate_grounding``/``_GroundingRegion``). Splitting the budget in
    half up front keeps each boundary's achieved precision fully
    determined by its own frames, regardless of the other window's size
    (Codex re-review round 3, finding 1).

    Each element of the returned pair is ``None`` independently if that
    window's floor still doesn't fit in its half of the budget - the caller
    decides how to report a fine-pass abstain from that, per-window.
    """
    window_budget = max_request_bytes // 2
    start_fit = _fit_frames_to_budget(
        start_frames,
        prompt_text=prompt_text,
        max_request_bytes=window_budget,
        min_frames=start_min_frames,
    )
    end_fit = _fit_frames_to_budget(
        end_frames,
        prompt_text=prompt_text,
        max_request_bytes=window_budget,
        min_frames=end_min_frames,
    )
    return start_fit, end_fit


def _fit_frames_to_budget(
    frames: list[TimedFrame], *, prompt_text: str, max_request_bytes: int, min_frames: int
) -> list[TimedFrame] | None:
    """Deterministically thin an already-extracted frame list until the
    estimated serialized request fits the budget, never dropping below
    ``min_frames``.

    Returns ``None`` - never a silent partial selection below that floor -
    if even the sparsest allowed selection still doesn't fit; the caller
    turns that into an abstain rather than sending an undersized or
    oversized request.
    """
    candidate = frames
    while _estimated_request_bytes(candidate, prompt_text) > max_request_bytes:
        if len(candidate) <= min_frames:
            return None
        target_count = max(min_frames, (len(candidate) + 1) // 2)
        if target_count >= len(candidate):
            target_count = len(candidate) - 1
        step = len(candidate) / target_count
        indices = sorted({min(len(candidate) - 1, int(i * step)) for i in range(target_count)})
        if len(indices) < 2:
            return None
        candidate = [candidate[i] for i in indices]
    return candidate


def _effective_tolerance_s(timestamps_s: list[float], fallback_step_s: float) -> float:
    """Grounding tolerance derived from the density actually achieved after
    budget-fitting, not the nominal (pre-thinning) sampling step - if the
    frame set had to be thinned, "near a submitted frame" and "near the
    claimed boundary" must widen to match, or a legitimately-cited frame
    would be rejected purely because thinning made it farther from its
    neighbours than the original dense step assumed.
    """
    if len(timestamps_s) < 2:
        return 2.0 * fallback_step_s
    ordered = sorted(timestamps_s)
    largest_gap = max(b - a for a, b in zip(ordered, ordered[1:], strict=False))
    return max(2.0 * fallback_step_s, largest_gap)


def _end_scan_windows(
    scan_from_s: float, duration_s: float, window_width_s: float, step_s: float
) -> list[tuple[float, float]]:
    """Bounded, overlapping windows scanning chronologically forward from
    ``scan_from_s`` to ``duration_s``.

    Overlap (``step_s < window_width_s``) so a break sitting near one
    window's boundary still falls fully inside the *next* window too,
    rather than being split across a boundary and visible nowhere in full.
    """
    windows = []
    lo = scan_from_s
    while lo < duration_s:
        hi = min(duration_s, lo + window_width_s)
        windows.append((lo, hi))
        if hi >= duration_s:
            break
        lo += step_s
    return windows


def _build_validation_frames(
    reader: VideoReader,
    *,
    candidate_ts: float,
    validation_lo: float,
    validation_hi: float,
    fine_step_s: float,
    max_dimension_px: int,
    jpeg_quality: int,
    prompt_text: str,
    max_request_bytes: int,
    target_tolerance_s: float,
) -> list[TimedFrame] | None:
    """Dense frames spanning one end-scan candidate's trend-validation
    window - baseline before it, evidence after it - with the candidate's
    own frame always present and marked (see ``TimedFrame.is_candidate``),
    even if budget-fitting would otherwise have thinned it away: a
    validation request with no labelled CANDIDATE frame at all is useless,
    not merely imprecise, since the prompt has nothing else to point the
    model at.

    Returns ``None`` - same contract as ``_fit_frames_to_budget`` - if even
    the sparsest allowed selection still doesn't fit the budget.
    """
    dense_times = sorted(
        set(_dense_timestamps(validation_lo, validation_hi, fine_step_s)) | {candidate_ts}
    )
    frames_dense = _extract_frames(
        reader, dense_times, max_dimension_px=max_dimension_px, jpeg_quality=jpeg_quality
    )
    min_frames = _min_frames_for_window(validation_hi - validation_lo, target_tolerance_s)
    frames = _fit_frames_to_budget(
        frames_dense,
        prompt_text=prompt_text,
        max_request_bytes=max_request_bytes,
        min_frames=min_frames,
    )
    if frames is None:
        return None
    if not any(frame.timestamp_s == candidate_ts for frame in frames):
        candidate_frame = next(f for f in frames_dense if f.timestamp_s == candidate_ts)
        frames = _merge_frames_sorted(frames, [candidate_frame])
    return [
        replace(frame, is_candidate=True) if frame.timestamp_s == candidate_ts else frame
        for frame in frames
    ]


def _coarse_timestamps(duration_s: float, step_s: float) -> list[float]:
    if duration_s <= 0:
        raise ConfigurationError(
            "Cannot run the coarse pass: the video has no known positive duration."
        )
    count = max(2, int(duration_s / step_s) + 1)
    return [min(i * step_s, duration_s) for i in range(count)]


# Pure float-rounding slack for "is this value inside the window we sized
# for it" - not a tolerance for the model being approximately right.
_BOUNDS_EPSILON_S = 1e-6


def _grounding_abstain(verdict: TimingVerdict, reason_code: str, detail: str) -> TimingVerdict:
    return TimingVerdict.abstain(
        reason_codes=(reason_code,),
        model_id=verdict.model_id,
        prompt_version=verdict.prompt_version,
        confidence=verdict.confidence,
        raw_notes=detail,
    )


@dataclass(frozen=True)
class _GroundingRegion:
    """One claimed boundary's own submitted-frame context.

    ``bounds`` is the window that boundary's claim must fall inside;
    ``submitted_timestamps_s``/``time_tolerance_s`` describe *only the
    frames actually sent for this boundary* and the tolerance derived from
    their own achieved density.

    Kept separate per boundary (rather than one shared set/tolerance across
    both) because the start and end fine windows can be many seconds apart:
    merging them first (as an earlier version of this pipeline did) lets
    the empty space *between* the two windows dominate
    ``_effective_tolerance_s``, which in turn makes "evidence corresponds
    to a frame actually sent" and "evidence is near the claimed boundary"
    far too permissive - a timestamp that was never sent to the provider at
    all (it falls in the dead zone between the windows) can end up
    "matching" a real frame purely because the inter-window gap inflated
    the tolerance, and can even "ground" *both* the start and end claims at
    once. Keeping each boundary's region separate closes that gap (Codex
    re-review round 3, finding 1) and also prevents evidence legitimately
    drawn from one window from grounding the other window's claim.
    """

    bounds: tuple[float, float]
    submitted_timestamps_s: tuple[float, ...]
    time_tolerance_s: float


def _validate_grounding(
    verdict: TimingVerdict,
    *,
    start_region: _GroundingRegion,
    end_region: _GroundingRegion,
    max_uncertainty_s: float,
) -> TimingVerdict:
    """Re-check a parsed CONFIRMED verdict against the request it answers.

    ``provider.parse_raw_response`` only validates a verdict's own internal
    shape - it has no idea what video or which frames were actually sent, so
    a well-formed, internally consistent, *fabricated* answer (a timestamp
    nowhere near the submitted frames, evidence that doesn't correspond to
    anything actually sent) parses cleanly. This is the check that catches
    that: every failure converges on ABSTAIN, same discipline as
    ``parse_raw_response``, just with request-level context it doesn't have.

    ``start_region``/``end_region`` are independent - see
    :class:`_GroundingRegion`. For the coarse pass (one shared frame batch,
    not yet split into a start/end window), pass the same region for both;
    its distinct-regions behaviour only matters once the fine pass has
    separate windows.

    A no-op for an already-ABSTAIN verdict (nothing to ground).
    """
    if verdict.status is not TimingStatus.CONFIRMED:
        return verdict
    assert verdict.start_s is not None and verdict.end_s is not None

    start_lo, start_hi = start_region.bounds
    end_lo, end_hi = end_region.bounds

    if not (start_lo - _BOUNDS_EPSILON_S <= verdict.start_s <= start_hi + _BOUNDS_EPSILON_S):
        return _grounding_abstain(
            verdict,
            "out_of_bounds",
            f"start_s={verdict.start_s} outside the submitted window "
            f"[{start_lo:.3f}, {start_hi:.3f}]",
        )
    if not (end_lo - _BOUNDS_EPSILON_S <= verdict.end_s <= end_hi + _BOUNDS_EPSILON_S):
        return _grounding_abstain(
            verdict,
            "out_of_bounds",
            f"end_s={verdict.end_s} outside the submitted window [{end_lo:.3f}, {end_hi:.3f}]",
        )

    if (
        verdict.start_uncertainty_s > max_uncertainty_s
        or verdict.end_uncertainty_s > max_uncertainty_s
    ):
        return _grounding_abstain(
            verdict,
            "uncertainty_exceeds_cap",
            f"start_uncertainty_s={verdict.start_uncertainty_s} "
            f"end_uncertainty_s={verdict.end_uncertainty_s} exceeds cap {max_uncertainty_s}",
        )

    if not verdict.evidence_frame_timestamps_s:
        return _grounding_abstain(
            verdict, "ungrounded_evidence", "no evidence_frame_timestamps_s given"
        )

    def _matches_region(ts: float, region: _GroundingRegion) -> bool:
        return any(
            abs(ts - sent) <= region.time_tolerance_s for sent in region.submitted_timestamps_s
        )

    ungrounded = [
        ts
        for ts in verdict.evidence_frame_timestamps_s
        if not _matches_region(ts, start_region) and not _matches_region(ts, end_region)
    ]
    if ungrounded:
        return _grounding_abstain(
            verdict,
            "ungrounded_evidence",
            f"evidence timestamps {ungrounded} do not correspond to any frame "
            f"actually submitted in this request",
        )

    # Evidence grounds a claim only if it both (a) is within that claim's
    # own region's tolerance of the claim, and (b) actually matches a frame
    # submitted *for that region* - so evidence legitimately drawn from the
    # end window (however close it happens to land, numerically, to
    # start_s) can never ground the start claim, and vice versa.
    near_start = any(
        abs(ts - verdict.start_s) <= start_region.time_tolerance_s
        and _matches_region(ts, start_region)
        for ts in verdict.evidence_frame_timestamps_s
    )
    near_end = any(
        abs(ts - verdict.end_s) <= end_region.time_tolerance_s and _matches_region(ts, end_region)
        for ts in verdict.evidence_frame_timestamps_s
    )
    if not (near_start and near_end):
        missing = "start_s" if not near_start else "end_s"
        claim = verdict.start_s if not near_start else verdict.end_s
        return _grounding_abstain(
            verdict,
            "evidence_far_from_claim",
            f"evidence_frame_timestamps_s={list(verdict.evidence_frame_timestamps_s)} has "
            f"nothing within its own window's tolerance of {missing}={claim}",
        )

    return verdict


def _abstain_outcome(
    verdict: TimingVerdict, coarse_response: RawProviderResponse
) -> PipelineOutcome:
    return PipelineOutcome(
        verdict=verdict, event=None, coarse_response=coarse_response, fine_response=None
    )


def _unsent_response(reason: str) -> RawProviderResponse:
    """A placeholder for a pass that was never sent to the provider at all.

    Used only when frame-budget fitting fails before any network call was
    attempted (see ``request_too_large`` below) - ``PipelineOutcome``
    always needs a ``RawProviderResponse`` to report against, even when the
    honest answer is "we refused to send this."
    """
    return RawProviderResponse(model_id="", raw_text="", latency_s=0.0, error=reason)


def _oversized_abstain(prompt_version: str, pass_name: str, detail: str) -> TimingVerdict:
    return TimingVerdict.abstain(
        reason_codes=("request_too_large",),
        model_id="",
        prompt_version=prompt_version,
        raw_notes=f"{pass_name} pass: {detail}; never sent to the provider",
    )


def run_llm_timing(
    video_path: Path,
    provider: TimingProvider,
    *,
    prompt_version: str,
    prompt_text: str,
    config: PipelineConfig | None = None,
    video_info: VideoInfo | None = None,
) -> PipelineOutcome:
    """Run the coarse-then-fine timing pass and return a full outcome.

    Every branch that cannot produce a trustworthy answer returns an
    ABSTAIN verdict with ``event=None`` rather than raising - the caller is
    expected to treat that exactly like "the detector found nothing" and
    fall back to the assisted/manual workflow. Genuine programming errors
    (a bad config, an unreadable video) still raise, per the rest of this
    codebase's error conventions.
    """
    cfg = config or PipelineConfig()
    cfg.validate()

    with VideoReader(video_path, video_info).open() as reader:
        duration_s = reader.info.duration_s
        if duration_s is None:
            raise ConfigurationError(
                "Cannot run the LLM timing pipeline on a video with unknown duration."
            )

        coarse_times = _coarse_timestamps(duration_s, cfg.coarse_step_s)
        coarse_frames_dense = _extract_frames(
            reader,
            coarse_times,
            max_dimension_px=cfg.max_frame_dimension_px,
            jpeg_quality=cfg.jpeg_quality,
        )
        # The coarse pass only has to locate an approximate region, not hit
        # gate-level precision - but it must be dense enough that *some*
        # sample lands within fine_margin_s of wherever the true transition
        # actually is, or the fine window built around the coarse estimate
        # could miss it entirely. See _min_coarse_frames.
        coarse_min_frames = _min_coarse_frames(duration_s, cfg.fine_margin_s)
        coarse_frames = _fit_frames_to_budget(
            coarse_frames_dense,
            prompt_text=prompt_text,
            max_request_bytes=cfg.max_request_bytes,
            min_frames=coarse_min_frames,
        )
        if coarse_frames is None:
            abstain = _oversized_abstain(
                prompt_version,
                "coarse",
                f"{len(coarse_frames_dense)} frames still exceed the request byte budget "
                f"even at the sparsest sampling ({coarse_min_frames} frames) that "
                f"guarantees coverage within fine_margin_s of the true transition",
            )
            return _abstain_outcome(abstain, _unsent_response(abstain.raw_notes))

        coarse_request = ProviderRequest(
            prompt_version=prompt_version,
            prompt_text=prompt_text,
            frames=tuple(coarse_frames),
            pass_name="coarse",
        )
        coarse_response = provider.analyze(coarse_request)
        coarse_verdict = parse_raw_response(
            coarse_response, prompt_version=prompt_version, min_confidence=cfg.min_confidence
        )
        coarse_submitted = [frame.timestamp_s for frame in coarse_frames]
        # A single shared frame batch, not yet split into per-boundary
        # windows - the same region serves both start_s and end_s here.
        coarse_region = _GroundingRegion(
            bounds=(0.0, duration_s),
            submitted_timestamps_s=tuple(coarse_submitted),
            time_tolerance_s=_effective_tolerance_s(coarse_submitted, cfg.coarse_step_s),
        )
        coarse_verdict = _validate_grounding(
            coarse_verdict,
            start_region=coarse_region,
            end_region=coarse_region,
            max_uncertainty_s=cfg.max_uncertainty_s,
        )

        if coarse_verdict.status is not TimingStatus.CONFIRMED:
            return _abstain_outcome(coarse_verdict, coarse_response)

        assert coarse_verdict.start_s is not None and coarse_verdict.end_s is not None
        # The fine pass now asks about the start ONLY - a single window,
        # widened past fine_margin_s when the coarse pass itself reported
        # more start uncertainty than that. It used to also build and send
        # an end window in the same request (to refine both boundaries at
        # once), but that end answer was already fully discarded once the
        # chronological end-scan below was introduced - and a live gate-1
        # run showed why sending it anyway is actively unsafe: the fine
        # pass's own end_s can fall outside its (irrelevant) end window,
        # and _validate_grounding requires *both* claimed boundaries to
        # ground before confirming anything - so an invalid answer to a
        # question nobody needed could abstain the whole run before the
        # end-scan ever got to start (supervisor-directed fix, see
        # diagnostics/llm_spike/DESIGN.md). See PROMPT_START_REFINE_V1.
        start_margin = max(cfg.fine_margin_s, coarse_verdict.start_uncertainty_s)
        start_lo = max(0.0, coarse_verdict.start_s - start_margin)
        start_hi = min(duration_s, coarse_verdict.start_s + start_margin)

        if (start_hi - start_lo) > cfg.fine_max_span_s:
            abstain = TimingVerdict.abstain(
                reason_codes=("ambiguous_evidence",),
                model_id=coarse_response.model_id,
                prompt_version=prompt_version,
                raw_notes=(
                    f"coarse pass's own start uncertainty makes the fine start window "
                    f"too wide to scan densely ({start_hi - start_lo:.2f}s, "
                    f"cap={cfg.fine_max_span_s:.2f}s); coarse estimate was "
                    f"start_s={coarse_verdict.start_s}"
                ),
            )
            return _abstain_outcome(abstain, coarse_response)

        fps = reader.info.fps
        fine_step_s = 1.0 / fps
        start_times = _dense_timestamps(start_lo, start_hi, fine_step_s)
        start_frames_dense = _extract_frames(
            reader,
            start_times,
            max_dimension_px=cfg.max_frame_dimension_px,
            jpeg_quality=cfg.jpeg_quality,
        )
        start_min_frames = _min_frames_for_window(start_hi - start_lo, cfg.target_tolerance_s)
        start_frames = _fit_frames_to_budget(
            start_frames_dense,
            prompt_text=PROMPT_START_REFINE_V1,
            max_request_bytes=cfg.max_request_bytes,
            min_frames=start_min_frames,
        )
        if start_frames is None:
            abstain = _oversized_abstain(
                PROMPT_START_REFINE_V1_ID,
                "fine",
                f"start window: {len(start_frames_dense)} frames still exceed the "
                f"request byte budget even at the sparsest sampling "
                f"({start_min_frames} frames) that meets the target tolerance",
            )
            return PipelineOutcome(
                verdict=abstain,
                event=None,
                coarse_response=coarse_response,
                fine_response=_unsent_response(abstain.raw_notes),
            )

        fine_request = ProviderRequest(
            prompt_version=PROMPT_START_REFINE_V1_ID,
            prompt_text=PROMPT_START_REFINE_V1,
            frames=tuple(start_frames),
            pass_name="fine",
        )
        fine_response = provider.analyze(fine_request)
        fine_verdict = parse_raw_response(
            fine_response,
            prompt_version=PROMPT_START_REFINE_V1_ID,
            min_confidence=cfg.min_confidence,
        )
        start_submitted = [frame.timestamp_s for frame in start_frames]
        start_region = _GroundingRegion(
            bounds=(start_lo, start_hi),
            submitted_timestamps_s=tuple(start_submitted),
            time_tolerance_s=_effective_tolerance_s(start_submitted, fine_step_s),
        )
        # Single shared region for both arguments - this call only ever
        # claims one boundary (start_s == end_s, per PROMPT_START_REFINE_V1),
        # same pattern as the coarse pass and each end-scan window.
        fine_verdict = _validate_grounding(
            fine_verdict,
            start_region=start_region,
            end_region=start_region,
            max_uncertainty_s=cfg.max_uncertainty_s,
        )

        if fine_verdict.status is not TimingStatus.CONFIRMED:
            return PipelineOutcome(
                verdict=fine_verdict,
                event=None,
                coarse_response=coarse_response,
                fine_response=fine_response,
            )

        # The end is nominated by ONE whole-clip (post-start), sparsely
        # sampled coarse request, then confirmed or rejected by ONE dense
        # trend-validation follow-up - a bounded two-stage strategy
        # (2 calls, not N chronological scan windows). This replaced an
        # earlier chronological end-scan entirely: a real gate-1 experiment
        # found that decomposing the search into many narrow, isolated
        # windows lost the temporal context a model needs to judge a
        # *sustained* trend - a 13-window scan found the true break's
        # window but then wrongly rejected it, while one coherent request
        # covering the same span with the same trend contract correctly
        # confirmed it (787,182 tokens/~7 minutes for the 13-window scan,
        # versus ~16,000 tokens/~22s for the isolated single-request
        # equivalent of stage 2 below) - supervisor-directed replacement,
        # see diagnostics/llm_spike/DESIGN.md and PROMPT_END_COARSE_V2/
        # PROMPT_END_VALIDATE_V2. Failed validation now falls back directly
        # to ABSTAIN (assisted/manual workflow) rather than searching for
        # another candidate - single-shot, not a search.
        #
        # The coarse candidate search must not start at locked_start_s
        # itself: a real gate-1 rerun (against the old chronological scan)
        # showed the first end-scan window built that way still contained
        # the onset transition (nothing visible -> stream visible), and a
        # model can misread that transition as a "break" - a false-
        # confirmed end just a fraction of a second after the true start.
        # The fix is structural, not a prompt-wording request, and applies
        # equally here: sampling begins no earlier than start_hi, the far
        # edge of the grounded start-refinement window, so no onset/
        # pre-flow frame is ever eligible to be submitted as a candidate
        # end timestamp in the first place (supervisor-directed fix, see
        # diagnostics/llm_spike/DESIGN.md).
        assert fine_verdict.start_s is not None
        locked_start_s: float = fine_verdict.start_s
        locked_start_uncertainty_s = fine_verdict.start_uncertainty_s
        start_side_evidence = fine_verdict.evidence_frame_timestamps_s

        scan_from_s = min(max(start_hi, locked_start_s), duration_s)

        end_coarse_times = _dense_timestamps(scan_from_s, duration_s, cfg.coarse_step_s)
        end_coarse_frames_dense = _extract_frames(
            reader,
            end_coarse_times,
            max_dimension_px=cfg.max_frame_dimension_px,
            jpeg_quality=cfg.jpeg_quality,
        )
        # Same coverage guarantee _min_coarse_frames already gives the
        # start-side coarse pass, applied to the (shorter) post-start span.
        end_coarse_min_frames = _min_coarse_frames(duration_s - scan_from_s, cfg.fine_margin_s)
        end_coarse_frames = _fit_frames_to_budget(
            end_coarse_frames_dense,
            prompt_text=PROMPT_END_COARSE_V2,
            max_request_bytes=cfg.max_request_bytes,
            min_frames=end_coarse_min_frames,
        )
        if end_coarse_frames is None:
            abstain = _oversized_abstain(
                PROMPT_END_COARSE_V2_ID,
                "end_coarse",
                f"{len(end_coarse_frames_dense)} frames still exceed the request byte "
                f"budget even at the sparsest sampling ({end_coarse_min_frames} frames) "
                f"that guarantees coverage within fine_margin_s of the true transition",
            )
            return PipelineOutcome(
                verdict=abstain,
                event=None,
                coarse_response=coarse_response,
                fine_response=fine_response,
            )

        end_coarse_request = ProviderRequest(
            prompt_version=PROMPT_END_COARSE_V2_ID,
            prompt_text=PROMPT_END_COARSE_V2,
            frames=tuple(end_coarse_frames),
            pass_name="end_coarse",
        )
        end_coarse_response = provider.analyze(end_coarse_request)
        end_coarse_verdict = parse_raw_response(
            end_coarse_response,
            prompt_version=PROMPT_END_COARSE_V2_ID,
            min_confidence=cfg.min_confidence,
        )
        end_coarse_submitted = [frame.timestamp_s for frame in end_coarse_frames]
        end_coarse_region = _GroundingRegion(
            bounds=(scan_from_s, duration_s),
            submitted_timestamps_s=tuple(end_coarse_submitted),
            time_tolerance_s=_effective_tolerance_s(end_coarse_submitted, cfg.coarse_step_s),
        )
        end_coarse_verdict = _validate_grounding(
            end_coarse_verdict,
            start_region=end_coarse_region,
            end_region=end_coarse_region,
            max_uncertainty_s=cfg.max_uncertainty_s,
        )
        if end_coarse_verdict.status is not TimingStatus.CONFIRMED:
            return PipelineOutcome(
                verdict=end_coarse_verdict,
                event=None,
                coarse_response=coarse_response,
                fine_response=fine_response,
                end_coarse_response=end_coarse_response,
            )

        # A sparse coarse candidate is never trusted on its own - it must
        # still pass a dense, bounded trend-validation check before being
        # accepted (supervisor-directed generalization: a real break is a
        # *sustained* shortening trend, not a single frame that happens to
        # look shorter - see PROMPT_END_VALIDATE_V2 and
        # diagnostics/llm_spike/DESIGN.md).
        assert end_coarse_verdict.end_s is not None
        candidate_ts = end_coarse_verdict.end_s

        # The "at least ~2s future context" rule is enforced structurally
        # here, not left to the model noticing a silently-shortened horizon
        # and self-abstaining per the prompt: a candidate this close to the
        # end of the clip can never receive the full mandatory validation
        # horizon, so the pipeline refuses to even attempt the follow-up
        # (supervisor-directed, see diagnostics/llm_spike/DESIGN.md).
        if candidate_ts + cfg.end_validation_horizon_s > duration_s:
            abstain = TimingVerdict.abstain(
                reason_codes=("insufficient_future_context",),
                model_id=end_coarse_verdict.model_id,
                prompt_version=PROMPT_END_COARSE_V2_ID,
                raw_notes=(
                    f"candidate at t={candidate_ts:.2f}s needs "
                    f"{cfg.end_validation_horizon_s:.2f}s of future context to validate, "
                    f"but the clip ends at t={duration_s:.2f}s "
                    f"({duration_s - candidate_ts:.2f}s available)"
                ),
            )
            return PipelineOutcome(
                verdict=abstain,
                event=None,
                coarse_response=coarse_response,
                fine_response=fine_response,
                end_coarse_response=end_coarse_response,
            )

        validation_lo = max(locked_start_s, candidate_ts - cfg.end_validation_baseline_s)
        # Never clamped by duration_s here - the check above already
        # guarantees the full horizon fits, so every validation request
        # gets the mandatory horizon in full, never a silently-shortened one.
        validation_hi = candidate_ts + cfg.end_validation_horizon_s
        validation_frames = _build_validation_frames(
            reader,
            candidate_ts=candidate_ts,
            validation_lo=validation_lo,
            validation_hi=validation_hi,
            fine_step_s=fine_step_s,
            max_dimension_px=cfg.max_frame_dimension_px,
            jpeg_quality=cfg.jpeg_quality,
            prompt_text=PROMPT_END_VALIDATE_V2,
            max_request_bytes=cfg.max_request_bytes,
            target_tolerance_s=cfg.target_tolerance_s,
        )
        if validation_frames is None:
            abstain = _oversized_abstain(
                PROMPT_END_VALIDATE_V2_ID,
                "end_validate",
                f"candidate at t={candidate_ts:.2f}s: validation window "
                f"[{validation_lo:.2f}, {validation_hi:.2f}]s still exceeds the request "
                f"byte budget even at the sparsest sampling that meets the target "
                f"tolerance",
            )
            return PipelineOutcome(
                verdict=abstain,
                event=None,
                coarse_response=coarse_response,
                fine_response=fine_response,
                end_coarse_response=end_coarse_response,
            )

        validation_request = ProviderRequest(
            prompt_version=PROMPT_END_VALIDATE_V2_ID,
            prompt_text=PROMPT_END_VALIDATE_V2,
            frames=tuple(validation_frames),
            pass_name="end_validate",
        )
        validation_response = provider.analyze(validation_request)
        validation_verdict = parse_raw_response(
            validation_response,
            prompt_version=PROMPT_END_VALIDATE_V2_ID,
            min_confidence=cfg.min_confidence,
        )
        validation_submitted = [frame.timestamp_s for frame in validation_frames]
        validation_region = _GroundingRegion(
            bounds=(validation_lo, validation_hi),
            submitted_timestamps_s=tuple(validation_submitted),
            time_tolerance_s=_effective_tolerance_s(validation_submitted, fine_step_s),
        )
        validation_verdict = _validate_grounding(
            validation_verdict,
            start_region=validation_region,
            end_region=validation_region,
            max_uncertainty_s=cfg.max_uncertainty_s,
        )
        if validation_verdict.status is TimingStatus.CONFIRMED:
            # The validation pass may refine the onset to a different
            # timestamp than the sparse candidate (see PROMPT_END_VALIDATE_V2)
            # - so the "full future horizon" guarantee established above for
            # candidate_ts must be re-checked against whatever timestamp was
            # actually reported: refining forward, toward the window's own
            # edge, can leave less than the mandatory horizon of *submitted*
            # evidence after it, which the pipeline must catch structurally
            # rather than trust the model's own compliance for
            # (supervisor-directed, see diagnostics/llm_spike/DESIGN.md).
            assert validation_verdict.end_s is not None
            if validation_verdict.end_s + cfg.end_validation_horizon_s > (
                validation_hi + _BOUNDS_EPSILON_S
            ):
                validation_verdict = _grounding_abstain(
                    validation_verdict,
                    "insufficient_future_context",
                    f"refined onset at t={validation_verdict.end_s:.3f}s needs "
                    f"{cfg.end_validation_horizon_s:.2f}s of future context within this "
                    f"window, but only {validation_hi - validation_verdict.end_s:.2f}s of "
                    f"submitted evidence follows it (window ends at "
                    f"t={validation_hi:.2f}s)",
                )
        if validation_verdict.status is not TimingStatus.CONFIRMED:
            # Rejected (trend didn't hold) or couldn't be validated at all
            # (insufficient future context, ambiguous, malformed, or
            # ungrounded) - single-shot: fall back to the assisted/manual
            # workflow rather than hunting for another candidate
            # (supervisor-directed, see diagnostics/llm_spike/DESIGN.md).
            return PipelineOutcome(
                verdict=validation_verdict,
                event=None,
                coarse_response=coarse_response,
                fine_response=fine_response,
                end_coarse_response=end_coarse_response,
                end_validation_response=validation_response,
            )

        # Report the dense validation pass's own (possibly refined) onset,
        # not the sparse end-coarse candidate that nominated the window it
        # searched: PROMPT_END_VALIDATE_V2 is deliberately allowed to
        # localize the true onset anywhere within its own grounded window,
        # since the sparse pass only has to localize roughly - a real
        # rerun showed the dense pass's own baseline frames already
        # contained the true break the sparse candidate overshot by ~1s
        # (supervisor-directed, see diagnostics/llm_spike/DESIGN.md). The
        # future-context guarantee for this exact timestamp was already
        # re-checked above.
        assert validation_verdict.end_s is not None
        final_confidence = min(
            fine_verdict.confidence, end_coarse_verdict.confidence, validation_verdict.confidence
        )
        final_verdict = TimingVerdict(
            status=TimingStatus.CONFIRMED,
            start_s=locked_start_s,
            end_s=validation_verdict.end_s,
            start_uncertainty_s=locked_start_uncertainty_s,
            end_uncertainty_s=validation_verdict.end_uncertainty_s,
            confidence=final_confidence,
            reason_codes=tuple(
                sorted(
                    set(fine_verdict.reason_codes)
                    | set(end_coarse_verdict.reason_codes)
                    | set(validation_verdict.reason_codes)
                )
            ),
            evidence_frame_timestamps_s=tuple(
                sorted(
                    set(start_side_evidence)
                    | set(end_coarse_verdict.evidence_frame_timestamps_s)
                    | set(validation_verdict.evidence_frame_timestamps_s)
                )
            ),
            model_id=validation_verdict.model_id,
            prompt_version=PROMPT_END_VALIDATE_V2_ID,
            raw_notes=(
                f"start confirmed via {PROMPT_START_REFINE_V1_ID}; candidate break "
                f"nominated via {PROMPT_END_COARSE_V2_ID} at t={candidate_ts:.3f}s; "
                f"refined and confirmed as a sustained trend via "
                f"{PROMPT_END_VALIDATE_V2_ID} at t={validation_verdict.end_s:.3f}s"
            ),
        )

        status = (
            EventStatus.CONFIRMED
            if final_verdict.confidence >= cfg.review_confidence
            else EventStatus.REVIEW
        )
        event = Event(
            label="Efflux (LLM spike)",
            start_s=locked_start_s,
            end_s=validation_verdict.end_s,
            confidence=final_verdict.confidence,
            detector="llm_timing_spike",
            status=status,
            notes=final_verdict.reason_codes,
            details={
                "start_uncertainty_s": final_verdict.start_uncertainty_s,
                "end_uncertainty_s": final_verdict.end_uncertainty_s,
                "evidence_frame_timestamps_s": list(final_verdict.evidence_frame_timestamps_s),
                "model_id": final_verdict.model_id,
                "start_prompt_version": PROMPT_START_REFINE_V1_ID,
                "end_coarse_prompt_version": PROMPT_END_COARSE_V2_ID,
                "end_validate_prompt_version": PROMPT_END_VALIDATE_V2_ID,
                "coarse_start_s": coarse_verdict.start_s,
                "coarse_end_s": coarse_verdict.end_s,
                "end_coarse_candidate_s": candidate_ts,
            },
        )
        return PipelineOutcome(
            verdict=final_verdict,
            event=event,
            coarse_response=coarse_response,
            fine_response=fine_response,
            end_coarse_response=end_coarse_response,
            end_validation_response=validation_response,
        )
