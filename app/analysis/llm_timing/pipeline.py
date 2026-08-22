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

from dataclasses import dataclass
from pathlib import Path

from ...errors import ConfigurationError
from ...video.metadata import VideoInfo
from ...video.reader import VideoReader, encode_jpeg
from ..base_detector import Event, EventStatus
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
    """

    coarse_step_s: float = 0.5
    fine_margin_s: float = 1.5
    fine_max_span_s: float = 6.0
    min_confidence: float = 0.5
    review_confidence: float = 0.75

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
            raise ConfigurationError(
                "review_confidence must be between min_confidence and 1."
            )


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

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.to_dict(),
            "event": self.event.to_dict() if self.event is not None else None,
            "coarse_model_id": self.coarse_response.model_id,
            "coarse_latency_s": self.coarse_response.latency_s,
            "fine_latency_s": self.fine_response.latency_s if self.fine_response else None,
        }


def _extract_frames(
    reader: VideoReader, timestamps_s: list[float], *, jpeg_quality: int = 85
) -> tuple[TimedFrame, ...]:
    frames = []
    for ts in timestamps_s:
        image = reader.frame_at(ts)
        frames.append(
            TimedFrame(timestamp_s=ts, image_bytes=encode_jpeg(image, quality=jpeg_quality))
        )
    return tuple(frames)


def _coarse_timestamps(duration_s: float, step_s: float) -> list[float]:
    if duration_s <= 0:
        raise ConfigurationError(
            "Cannot run the coarse pass: the video has no known positive duration."
        )
    count = max(2, int(duration_s / step_s) + 1)
    return [min(i * step_s, duration_s) for i in range(count)]


def _abstain_outcome(
    verdict: TimingVerdict, coarse_response: RawProviderResponse
) -> PipelineOutcome:
    return PipelineOutcome(
        verdict=verdict, event=None, coarse_response=coarse_response, fine_response=None
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
        coarse_frames = _extract_frames(reader, coarse_times)
        coarse_request = ProviderRequest(
            prompt_version=prompt_version,
            prompt_text=prompt_text,
            frames=coarse_frames,
            pass_name="coarse",
        )
        coarse_response = provider.analyze(coarse_request)
        coarse_verdict = parse_raw_response(
            coarse_response, prompt_version=prompt_version, min_confidence=cfg.min_confidence
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
        fine_frames = _extract_frames(reader, fine_times)
        fine_request = ProviderRequest(
            prompt_version=prompt_version,
            prompt_text=prompt_text,
            frames=fine_frames,
            pass_name="fine",
        )
        fine_response = provider.analyze(fine_request)
        fine_verdict = parse_raw_response(
            fine_response, prompt_version=prompt_version, min_confidence=cfg.min_confidence
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
