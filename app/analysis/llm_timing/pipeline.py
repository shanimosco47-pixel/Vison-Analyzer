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
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ...errors import ConfigurationError
from ...video.metadata import VideoInfo
from ...video.reader import VideoReader, encode_jpeg
from ..base_detector import Event, EventStatus
from .pricing import PRICING_TABLE_VERSION, ModelPricing, estimate_cost_usd
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

    @property
    def total_retries(self) -> int:
        return self.coarse_response.retries + (
            self.fine_response.retries if self.fine_response else 0
        )

    @property
    def total_latency_s(self) -> float:
        return self.coarse_response.latency_s + (
            self.fine_response.latency_s if self.fine_response else 0.0
        )

    @property
    def total_tokens(self) -> int | None:
        """``None`` (unknown) unless every pass that ran reported usage."""
        parts = [self.coarse_response, self.fine_response]
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
        responses = [self.coarse_response, self.fine_response]
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


def _validate_grounding(
    verdict: TimingVerdict,
    *,
    start_bounds: tuple[float, float],
    end_bounds: tuple[float, float],
    submitted_timestamps_s: tuple[float, ...],
    time_tolerance_s: float,
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

    A no-op for an already-ABSTAIN verdict (nothing to ground).
    """
    if verdict.status is not TimingStatus.CONFIRMED:
        return verdict
    assert verdict.start_s is not None and verdict.end_s is not None

    start_lo, start_hi = start_bounds
    end_lo, end_hi = end_bounds

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

    def _matches_a_submitted_frame(ts: float) -> bool:
        return any(abs(ts - sent) <= time_tolerance_s for sent in submitted_timestamps_s)

    ungrounded = [
        ts for ts in verdict.evidence_frame_timestamps_s if not _matches_a_submitted_frame(ts)
    ]
    if ungrounded:
        return _grounding_abstain(
            verdict,
            "ungrounded_evidence",
            f"evidence timestamps {ungrounded} do not correspond to any frame "
            f"actually submitted in this request",
        )

    near_start = any(
        abs(ts - verdict.start_s) <= time_tolerance_s for ts in verdict.evidence_frame_timestamps_s
    )
    near_end = any(
        abs(ts - verdict.end_s) <= time_tolerance_s for ts in verdict.evidence_frame_timestamps_s
    )
    if not (near_start and near_end):
        missing = "start_s" if not near_start else "end_s"
        claim = verdict.start_s if not near_start else verdict.end_s
        return _grounding_abstain(
            verdict,
            "evidence_far_from_claim",
            f"evidence_frame_timestamps_s={list(verdict.evidence_frame_timestamps_s)} has "
            f"nothing within {time_tolerance_s:.3f}s of {missing}={claim}",
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


def _oversized_abstain(prompt_version: str, pass_name: str, frame_count: int) -> TimingVerdict:
    return TimingVerdict.abstain(
        reason_codes=("request_too_large",),
        model_id="",
        prompt_version=prompt_version,
        raw_notes=(
            f"{pass_name} pass: {frame_count} frames still exceed the request byte "
            f"budget even at the sparsest sampling that meets the target tolerance; "
            f"never sent to the provider"
        ),
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
        # gate-level precision - 4 frames is enough to bracket "roughly
        # where", so its floor doesn't scale off target_tolerance_s the way
        # the fine pass's does.
        coarse_frames = _fit_frames_to_budget(
            coarse_frames_dense,
            prompt_text=prompt_text,
            max_request_bytes=cfg.max_request_bytes,
            min_frames=4,
        )
        if coarse_frames is None:
            abstain = _oversized_abstain(prompt_version, "coarse", len(coarse_frames_dense))
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
        coarse_verdict = _validate_grounding(
            coarse_verdict,
            start_bounds=(0.0, duration_s),
            end_bounds=(0.0, duration_s),
            submitted_timestamps_s=tuple(coarse_submitted),
            time_tolerance_s=_effective_tolerance_s(coarse_submitted, cfg.coarse_step_s),
            max_uncertainty_s=cfg.max_uncertainty_s,
        )

        if coarse_verdict.status is not TimingStatus.CONFIRMED:
            return _abstain_outcome(coarse_verdict, coarse_response)

        assert coarse_verdict.start_s is not None and coarse_verdict.end_s is not None
        # Each fine window is sized independently around its own boundary, and
        # widened past fine_margin_s when the coarse pass itself reported more
        # uncertainty than that - a wide window is only "too ambiguous to
        # fine-scan" relative to how uncertain *that one boundary* is, not
        # relative to how far apart start and end naturally are (a real
        # efflux time is routinely tens of seconds).
        start_margin = max(cfg.fine_margin_s, coarse_verdict.start_uncertainty_s)
        end_margin = max(cfg.fine_margin_s, coarse_verdict.end_uncertainty_s)
        start_lo = max(0.0, coarse_verdict.start_s - start_margin)
        start_hi = min(duration_s, coarse_verdict.start_s + start_margin)
        end_lo = max(0.0, coarse_verdict.end_s - end_margin)
        end_hi = min(duration_s, coarse_verdict.end_s + end_margin)

        oversized = [
            (name, lo, hi)
            for name, lo, hi in (("start", start_lo, start_hi), ("end", end_lo, end_hi))
            if (hi - lo) > cfg.fine_max_span_s
        ]
        if oversized:
            detail = "; ".join(f"{name} window {hi - lo:.2f}s" for name, lo, hi in oversized)
            abstain = TimingVerdict.abstain(
                reason_codes=("ambiguous_evidence",),
                model_id=coarse_response.model_id,
                prompt_version=prompt_version,
                raw_notes=(
                    f"coarse pass's own uncertainty makes at least one fine window too "
                    f"wide to scan densely ({detail}, cap={cfg.fine_max_span_s:.2f}s); "
                    f"coarse estimate was start_s={coarse_verdict.start_s} "
                    f"end_s={coarse_verdict.end_s}"
                ),
            )
            return _abstain_outcome(abstain, coarse_response)

        fps = reader.info.fps
        fine_step_s = 1.0 / fps
        fine_times = sorted(
            {
                round(start_lo + i * fine_step_s, 6)
                for i in range(int((start_hi - start_lo) / fine_step_s) + 2)
                if start_lo + i * fine_step_s <= start_hi
            }
            | {
                round(end_lo + i * fine_step_s, 6)
                for i in range(int((end_hi - end_lo) / fine_step_s) + 2)
                if end_lo + i * fine_step_s <= end_hi
            }
        )
        fine_frames_dense = _extract_frames(
            reader,
            fine_times,
            max_dimension_px=cfg.max_frame_dimension_px,
            jpeg_quality=cfg.jpeg_quality,
        )
        # The floor is the sparsest sampling that can still resolve *each*
        # boundary to within target_tolerance_s - summed across both
        # windows, since they share one request's byte budget.
        fine_min_frames = _min_frames_for_window(
            start_hi - start_lo, cfg.target_tolerance_s
        ) + _min_frames_for_window(end_hi - end_lo, cfg.target_tolerance_s)
        fine_frames = _fit_frames_to_budget(
            fine_frames_dense,
            prompt_text=prompt_text,
            max_request_bytes=cfg.max_request_bytes,
            min_frames=fine_min_frames,
        )
        if fine_frames is None:
            abstain = _oversized_abstain(prompt_version, "fine", len(fine_frames_dense))
            return PipelineOutcome(
                verdict=abstain,
                event=None,
                coarse_response=coarse_response,
                fine_response=_unsent_response(abstain.raw_notes),
            )

        fine_request = ProviderRequest(
            prompt_version=prompt_version,
            prompt_text=prompt_text,
            frames=tuple(fine_frames),
            pass_name="fine",
        )
        fine_response = provider.analyze(fine_request)
        fine_verdict = parse_raw_response(
            fine_response, prompt_version=prompt_version, min_confidence=cfg.min_confidence
        )
        fine_submitted = [frame.timestamp_s for frame in fine_frames]
        fine_verdict = _validate_grounding(
            fine_verdict,
            start_bounds=(start_lo, start_hi),
            end_bounds=(end_lo, end_hi),
            submitted_timestamps_s=tuple(fine_submitted),
            time_tolerance_s=_effective_tolerance_s(fine_submitted, fine_step_s),
            max_uncertainty_s=cfg.max_uncertainty_s,
        )

        if fine_verdict.status is not TimingStatus.CONFIRMED:
            return PipelineOutcome(
                verdict=fine_verdict,
                event=None,
                coarse_response=coarse_response,
                fine_response=fine_response,
            )

        status = (
            EventStatus.CONFIRMED
            if fine_verdict.confidence >= cfg.review_confidence
            else EventStatus.REVIEW
        )
        assert fine_verdict.start_s is not None and fine_verdict.end_s is not None
        event = Event(
            label="Efflux (LLM spike)",
            start_s=fine_verdict.start_s,
            end_s=fine_verdict.end_s,
            confidence=fine_verdict.confidence,
            detector="llm_timing_spike",
            status=status,
            notes=fine_verdict.reason_codes,
            details={
                "start_uncertainty_s": fine_verdict.start_uncertainty_s,
                "end_uncertainty_s": fine_verdict.end_uncertainty_s,
                "evidence_frame_timestamps_s": list(fine_verdict.evidence_frame_timestamps_s),
                "model_id": fine_verdict.model_id,
                "prompt_version": fine_verdict.prompt_version,
                "coarse_start_s": coarse_verdict.start_s,
                "coarse_end_s": coarse_verdict.end_s,
            },
        )
        return PipelineOutcome(
            verdict=fine_verdict,
            event=event,
            coarse_response=coarse_response,
            fine_response=fine_response,
        )
