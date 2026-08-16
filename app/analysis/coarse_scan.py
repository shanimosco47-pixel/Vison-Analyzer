"""Stage A - the cheap pass over the whole recording.

The goal is *not* to measure anything accurately.  It is to answer, as fast as
possible, "which parts of these twelve hours are worth a closer look?", while
guaranteeing that an event of at least ``shortest_event_s`` cannot slip
between two samples.

Two independent optimisations are applied, and the module keeps them
separate because they trade off differently:

*   **frame skipping** - decode one frame every ``interval_s``.  This is the
    big win (a 12 h 25 FPS recording is ~1.1 M frames; sampling every 5 s
    leaves 8 640) but it costs temporal resolution, so the interval is derived
    from the shortest event that must not be missed, never hard-coded.
*   **resolution reduction** - analyse a downscaled frame.  This keeps every
    sampled instant but makes each one ~20x cheaper at 1080p.  It costs
    detail, not time coverage.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from ..config import ROI, CoarseScanConfig
from ..logging_setup import get_logger
from ..video.metadata import VideoInfo
from ..video.reader import VideoReader
from ..video.sampling import (
    SamplingPlan,
    build_sampling_plan,
    resolve_sample_interval,
    scale_factor_for_width,
)
from .base_detector import ActivityScorer, ActivityTrace, ProgressReporter, null_progress
from .temporal import Interval, adaptive_thresholds, bridge_gaps, drop_short

logger = get_logger(__name__)

# Progress is reported at most this often to avoid flooding the job state.
PROGRESS_UPDATE_INTERVAL_S = 0.5


@dataclass
class CoarseScanResult:
    """Everything Stage A learned, including *why* it flagged what it flagged."""

    plan: SamplingPlan
    trace: ActivityTrace
    candidates: list[Interval]
    enter_threshold: float
    exit_threshold: float
    truncated: bool = False
    elapsed_s: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sampling": {
                "interval_s": round(self.plan.effective_interval_s, 4),
                "step_frames": self.plan.step_frames,
                "scale": round(self.plan.scale, 4),
                "samples": len(self.trace),
            },
            "enter_threshold": round(self.enter_threshold, 6),
            "exit_threshold": round(self.exit_threshold, 6),
            "candidate_count": len(self.candidates),
            "truncated": self.truncated,
            "elapsed_s": round(self.elapsed_s, 2),
            "disturbed_ratio": round(self.trace.disturbed_ratio, 4),
            **self.stats,
        }


def plan_coarse_scan(
    video: VideoInfo,
    config: CoarseScanConfig,
    *,
    roi: ROI | None = None,
) -> SamplingPlan:
    """Decide the sampling interval and scan resolution for ``video``."""
    config.validate()
    interval_s = resolve_sample_interval(
        config.shortest_event_s,
        config.safety_factor,
        fps=video.fps,
        min_interval_s=config.min_sample_interval_s,
        max_interval_s=config.max_sample_interval_s,
    )
    analysed_width = roi.width if roi is not None else video.width
    scale = scale_factor_for_width(analysed_width, config.scan_width_px)
    duration_s = video.duration_s if video.duration_s is not None else 0.0
    return build_sampling_plan(
        fps=video.fps,
        duration_s=duration_s,
        interval_s=interval_s,
        scale=scale,
    )


def run_coarse_scan(
    reader: VideoReader,
    scorer: ActivityScorer,
    config: CoarseScanConfig,
    *,
    roi: ROI | None = None,
    plan: SamplingPlan | None = None,
    progress: ProgressReporter = null_progress,
    blur_kernel: int = 3,
) -> CoarseScanResult:
    """Scan the whole recording cheaply and return candidate windows."""
    config.validate()
    video = reader.info
    plan = plan or plan_coarse_scan(video, config, roi=roi)
    scorer.reset()

    logger.info("Coarse scan of %s: %s", video.path.name, plan.describe())
    started = time.monotonic()
    trace = ActivityTrace()
    last_report = 0.0
    total_span = max(plan.end_s - plan.start_s, 1e-6)

    for sample in reader.iter_samples(plan, roi=roi, grayscale=True, blur_kernel=blur_kernel):
        result = scorer.score(sample)
        trace.add(sample.timestamp_s, result.value, result.disturbed)

        now = time.monotonic()
        if now - last_report >= PROGRESS_UPDATE_INTERVAL_S:
            last_report = now
            fraction = min(1.0, (sample.timestamp_s - plan.start_s) / total_span)
            progress(
                stage="scanning",
                fraction=fraction,
                message=f"Scanning video ({_format_clock(sample.timestamp_s)} of "
                f"{_format_clock(plan.end_s)})",
            )

    elapsed = time.monotonic() - started
    enter, exit_ = adaptive_thresholds(
        trace.scores,
        enter_sigma=config.enter_sigma,
        exit_sigma=config.exit_sigma,
        floor=config.min_changed_area_ratio,
        sensitivity=config.sensitivity,
        baseline_quantile=0.5,
    )

    intervals = _extract_candidates(trace, config, enter, exit_)
    truncated = len(intervals) > config.max_candidates
    if truncated:
        logger.warning(
            "Coarse scan produced %d candidates; keeping the %d strongest",
            len(intervals),
            config.max_candidates,
        )
        intervals = _strongest(intervals, trace, config.max_candidates)

    scan_result = CoarseScanResult(
        plan=plan,
        trace=trace,
        candidates=intervals,
        enter_threshold=enter,
        exit_threshold=exit_,
        truncated=truncated,
        elapsed_s=elapsed,
        stats={
            "frames_decoded": reader.stats.frames_decoded,
            "frames_skipped": reader.stats.frames_skipped,
            "read_failures": reader.stats.read_failures,
            "seeks": reader.stats.seeks,
        },
    )
    logger.info(
        "Coarse scan complete: %d samples in %.1fs (%.0f samples/s), "
        "thresholds enter=%.5f exit=%.5f, %d candidate window(s)",
        len(trace),
        elapsed,
        len(trace) / elapsed if elapsed > 0 else 0.0,
        enter,
        exit_,
        len(intervals),
    )
    return scan_result


def _extract_candidates(
    trace: ActivityTrace,
    config: CoarseScanConfig,
    enter: float,
    exit_: float,
) -> list[Interval]:
    from .temporal import find_active_intervals  # local import keeps the API surface obvious

    intervals = find_active_intervals(
        trace.times,
        trace.scores,
        enter_threshold=enter,
        exit_threshold=exit_,
        disturbed=trace.disturbed,
        # Conservative: the true edge is guaranteed to be inside the window,
        # which is what Stage B needs in order to find it.
        conservative_boundaries=True,
    )
    intervals = bridge_gaps(intervals, config.bridge_gap_s)
    return drop_short(intervals, config.min_candidate_duration_s)


def _strongest(intervals: list[Interval], trace: ActivityTrace, limit: int) -> list[Interval]:
    """Keep the ``limit`` windows with the highest peak score, in time order."""

    def peak(interval: Interval) -> float:
        values = [
            score
            for t, score in zip(trace.times, trace.scores, strict=True)
            if interval.start_s <= t <= interval.end_s
        ]
        return max(values) if values else 0.0

    ranked = sorted(intervals, key=peak, reverse=True)[:limit]
    return sorted(ranked, key=lambda i: i.start_s)


def _format_clock(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:d}:{(total % 3600) // 60:02d}:{total % 60:02d}"
