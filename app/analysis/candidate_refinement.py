"""Stage B - dense re-analysis around each candidate window.

Stage A knows roughly *where* something happened; its boundaries are only as
accurate as its sampling interval.  Stage B seeks directly to
``[start - pre_roll, end + post_roll]`` and analyses that short span densely
(by default every frame, at a higher resolution) to place the boundaries
properly.

The whole video is never re-decoded: only the candidate neighbourhoods are,
and each one is read in a single forward pass after one seek.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..config import ROI, RefinementConfig
from ..logging_setup import get_logger
from ..video.reader import VideoReader
from ..video.sampling import build_sampling_plan, scale_factor_for_width
from .base_detector import ActivityScorer, ActivityTrace, ProgressReporter, null_progress
from .temporal import (
    Interval,
    adaptive_thresholds,
    bridge_gaps,
    clamp,
    drop_short,
    find_active_intervals,
    safe_ratio,
)

logger = get_logger(__name__)

ScorerFactory = Callable[[], ActivityScorer]


@dataclass
class RefinedEvent:
    """A candidate turned into an accurate interval, with its evidence."""

    interval: Interval
    peak_score: float
    mean_score: float
    enter_threshold: float
    sample_count: int
    active_sample_count: int
    disturbed_ratio: float
    clipped_start: bool
    clipped_end: bool
    confidence: float
    window: Interval

    def evidence(self) -> dict[str, Any]:
        return {
            "peak_score": round(self.peak_score, 6),
            "mean_score": round(self.mean_score, 6),
            "enter_threshold": round(self.enter_threshold, 6),
            "samples_in_window": self.sample_count,
            "active_samples": self.active_sample_count,
            "disturbed_ratio": round(self.disturbed_ratio, 4),
            "boundary_clipped": self.clipped_start or self.clipped_end,
            "refined_window_s": [round(self.window.start_s, 3), round(self.window.end_s, 3)],
        }


@dataclass
class RefinementOutcome:
    events: list[RefinedEvent] = field(default_factory=list)
    elapsed_s: float = 0.0
    windows_analysed: int = 0


def refine_candidates(
    reader: VideoReader,
    scorer_factory: ScorerFactory,
    candidates: list[Interval],
    config: RefinementConfig,
    *,
    coarse_interval_s: float,
    roi: ROI | None = None,
    progress: ProgressReporter = null_progress,
    blur_kernel: int = 3,
) -> RefinementOutcome:
    """Refine every candidate window and return accurate events."""
    config.validate()
    video = reader.info
    fallback_end = candidates[-1].end_s + config.post_roll_s if candidates else 0.0
    duration_s = video.duration_s or fallback_end

    # The coarse boundary can be wrong by up to one sampling interval, so the
    # roll-back must be at least that long or the true start stays outside the
    # window we refine.
    pre_roll = max(config.pre_roll_s, coarse_interval_s)
    post_roll = max(config.post_roll_s, coarse_interval_s)

    analysed_width = roi.width if roi is not None else video.width
    scale = scale_factor_for_width(analysed_width, config.refine_width_px)
    dense_interval_s = config.dense_step_s if config.dense_step_s > 0 else 1.0 / video.fps

    started = time.monotonic()
    outcome = RefinementOutcome()

    for position, candidate in enumerate(candidates, start=1):
        window = candidate.expanded(pre_roll, post_roll, lower=0.0, upper=duration_s)
        progress(
            stage="refining",
            fraction=(position - 1) / max(1, len(candidates)),
            message=f"Refining candidate {position} of {len(candidates)}",
        )
        events = _refine_one(
            reader,
            scorer_factory(),
            candidate=candidate,
            window=window,
            config=config,
            scale=scale,
            dense_interval_s=dense_interval_s,
            roi=roi,
            blur_kernel=blur_kernel,
        )
        outcome.events.extend(events)
        outcome.windows_analysed += 1
        logger.debug(
            "Candidate %d/%d [%.2f-%.2f]s refined into %d event(s)",
            position,
            len(candidates),
            candidate.start_s,
            candidate.end_s,
            len(events),
        )

    outcome.elapsed_s = time.monotonic() - started
    outcome.events = _merge_overlapping(outcome.events)
    logger.info(
        "Refinement complete: %d window(s) -> %d event(s) in %.1fs",
        outcome.windows_analysed,
        len(outcome.events),
        outcome.elapsed_s,
    )
    return outcome


def _refine_one(
    reader: VideoReader,
    scorer: ActivityScorer,
    *,
    candidate: Interval,
    window: Interval,
    config: RefinementConfig,
    scale: float,
    dense_interval_s: float,
    roi: ROI | None,
    blur_kernel: int,
) -> list[RefinedEvent]:
    plan = build_sampling_plan(
        fps=reader.info.fps,
        duration_s=window.end_s,
        interval_s=dense_interval_s,
        scale=scale,
        start_s=window.start_s,
        end_s=window.end_s,
    )
    trace = ActivityTrace()
    for sample in reader.iter_samples(plan, roi=roi, grayscale=True, blur_kernel=blur_kernel):
        result = scorer.score(sample)
        trace.add(sample.timestamp_s, result.value, result.disturbed)

    if len(trace) < 3:
        logger.warning(
            "Refinement window %.2f-%.2fs yielded only %d samples; keeping the coarse bounds",
            window.start_s,
            window.end_s,
            len(trace),
        )
        return [
            RefinedEvent(
                interval=candidate,
                peak_score=max(trace.scores, default=0.0),
                mean_score=(sum(trace.scores) / len(trace.scores)) if trace.scores else 0.0,
                enter_threshold=0.0,
                sample_count=len(trace),
                active_sample_count=0,
                disturbed_ratio=trace.disturbed_ratio,
                clipped_start=True,
                clipped_end=True,
                confidence=0.35,  # explicitly low: the window could not be measured
                window=window,
            )
        ]

    # A refinement window is mostly event, so "quiet" is a low quantile of it
    # rather than its median.  The pre/post roll guarantees quiet samples exist.
    enter, exit_ = adaptive_thresholds(
        trace.scores,
        enter_sigma=4.0,
        exit_sigma=2.0,
        floor=0.0015,
        sensitivity=0.5,
        baseline_quantile=0.25,
    )

    intervals = find_active_intervals(
        trace.times,
        trace.scores,
        enter_threshold=enter,
        exit_threshold=exit_,
        disturbed=trace.disturbed,
        conservative_boundaries=False,  # dense pass: accurate to one sample
    )
    intervals = drop_short(bridge_gaps(intervals, config.merge_gap_s), config.min_event_duration_s)
    # Discard anything that belongs to a *neighbouring* event that happened to
    # fall inside the pre/post roll; that neighbour has its own candidate.
    intervals = [i for i in intervals if i.overlaps(candidate)]

    return [_build_event(interval, trace, window, enter) for interval in intervals]


def _build_event(
    interval: Interval,
    trace: ActivityTrace,
    window: Interval,
    enter: float,
) -> RefinedEvent:
    inside = [
        (t, score)
        for t, score in zip(trace.times, trace.scores, strict=True)
        if interval.start_s <= t <= interval.end_s
    ]
    scores = [score for _, score in inside] or [0.0]
    peak = max(scores)
    mean = sum(scores) / len(scores)
    active = sum(1 for score in scores if score >= enter)

    sample_spacing = _median_spacing(trace.times)
    clipped_start = interval.start_s - window.start_s <= sample_spacing
    clipped_end = window.end_s - interval.end_s <= sample_spacing

    confidence = _confidence(
        peak=peak,
        enter=enter,
        active_ratio=safe_ratio(active, len(scores)),
        disturbed_ratio=trace.disturbed_ratio,
        clipped=clipped_start or clipped_end,
    )
    return RefinedEvent(
        interval=interval,
        peak_score=peak,
        mean_score=mean,
        enter_threshold=enter,
        sample_count=len(trace),
        active_sample_count=active,
        disturbed_ratio=trace.disturbed_ratio,
        clipped_start=clipped_start,
        clipped_end=clipped_end,
        confidence=confidence,
        window=window,
    )


def _confidence(
    *,
    peak: float,
    enter: float,
    active_ratio: float,
    disturbed_ratio: float,
    clipped: bool,
) -> float:
    """Confidence from measurable evidence, not from wishful thinking.

    Contributions:

    *   0.45 base for "an event was detected at all";
    *   up to +0.30 for how far the peak exceeded the threshold (an event that
        barely crosses the line is not the same as one ten times above it);
    *   up to +0.25 for how much of the event's own span was above threshold
        (a solid block of activity beats a flickering one);
    *   -0.15 if a boundary sits at the edge of the refined window, because the
        true boundary may lie outside what was analysed;
    *   -0.30 x the fraction of disturbed frames in the window.

    The result is clamped to [0.05, 0.99]: this software never claims certainty.
    """
    margin = clamp(safe_ratio(peak - enter, max(enter, 1e-6)), 0.0, 1.0)
    score = 0.45 + 0.30 * margin + 0.25 * clamp(active_ratio, 0.0, 1.0)
    if clipped:
        score -= 0.15
    score -= 0.30 * clamp(disturbed_ratio, 0.0, 1.0)
    return clamp(score, 0.05, 0.99)


def _median_spacing(times: list[float]) -> float:
    if len(times) < 2:
        return 0.0
    gaps = sorted(b - a for a, b in zip(times, times[1:], strict=False))
    return gaps[len(gaps) // 2]


def _merge_overlapping(events: list[RefinedEvent]) -> list[RefinedEvent]:
    """Two candidates can refine into the same real event; keep it once."""
    ordered = sorted(events, key=lambda e: e.interval.start_s)
    merged: list[RefinedEvent] = []
    for event in ordered:
        if merged and event.interval.overlaps(merged[-1].interval):
            previous = merged.pop()
            keeper = previous if previous.confidence >= event.confidence else event
            merged.append(
                RefinedEvent(
                    interval=Interval(
                        min(previous.interval.start_s, event.interval.start_s),
                        max(previous.interval.end_s, event.interval.end_s),
                    ),
                    peak_score=max(previous.peak_score, event.peak_score),
                    mean_score=(previous.mean_score + event.mean_score) / 2,
                    enter_threshold=keeper.enter_threshold,
                    sample_count=previous.sample_count + event.sample_count,
                    active_sample_count=previous.active_sample_count + event.active_sample_count,
                    disturbed_ratio=max(previous.disturbed_ratio, event.disturbed_ratio),
                    clipped_start=previous.clipped_start,
                    clipped_end=event.clipped_end,
                    confidence=max(previous.confidence, event.confidence),
                    window=Interval(
                        min(previous.window.start_s, event.window.start_s),
                        max(previous.window.end_s, event.window.end_s),
                    ),
                )
            )
        else:
            merged.append(event)
    return merged
