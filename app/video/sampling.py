"""Time/frame conversion and sampling plans.

All timing logic in the application funnels through this module.  It is pure
(no OpenCV, no I/O) so it can be exhaustively unit-tested, and it is the only
place allowed to turn seconds into frame indices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..errors import ConfigurationError

# A frame rate outside this range is almost certainly a metadata error rather
# than a real recording.
MIN_PLAUSIBLE_FPS = 0.5
MAX_PLAUSIBLE_FPS = 1000.0


def validate_fps(fps: float) -> float:
    """Return ``fps`` if it is plausible, otherwise raise."""
    if not math.isfinite(fps) or not MIN_PLAUSIBLE_FPS <= fps <= MAX_PLAUSIBLE_FPS:
        raise ConfigurationError(
            "The video frame rate could not be determined reliably.",
            detail=f"fps={fps!r}",
        )
    return float(fps)


def seconds_to_frames(seconds: float, fps: float, *, minimum: int = 0) -> int:
    """Convert a duration in seconds to a whole number of frames.

    Rounds to nearest so that, for example, 0.5 s at 30 FPS is 15 frames and
    not 14.  ``minimum`` clamps the result upward, which callers use to keep a
    step of at least one frame.
    """
    validate_fps(fps)
    if seconds < 0:
        raise ConfigurationError("A duration in seconds must not be negative.")
    return max(minimum, int(round(seconds * fps)))


def frames_to_seconds(frames: float, fps: float) -> float:
    """Convert a frame count (or frame index) to seconds."""
    validate_fps(fps)
    return float(frames) / fps


def frame_index_at(timestamp_s: float, fps: float, frame_count: int | None = None) -> int:
    """Frame index whose presentation time contains ``timestamp_s``."""
    validate_fps(fps)
    index = max(0, int(math.floor(max(0.0, timestamp_s) * fps)))
    if frame_count is not None and frame_count > 0:
        index = min(index, frame_count - 1)
    return index


def resolve_sample_interval(
    shortest_event_s: float,
    safety_factor: float,
    *,
    fps: float,
    min_interval_s: float = 0.1,
    max_interval_s: float = 5.0,
) -> float:
    """Derive the coarse sampling interval from the shortest event of interest.

    The interval is ``shortest_event_s / safety_factor`` so that at least
    ``safety_factor`` samples fall inside the shortest event that must not be
    missed.  Sampling exactly once per event length would allow an event to
    fall between two samples, which is the classic mistake this guards
    against.  The result is clamped to the configured bounds and can never be
    shorter than one frame.
    """
    validate_fps(fps)
    if shortest_event_s <= 0:
        raise ConfigurationError("The shortest event duration must be greater than zero.")
    if safety_factor < 2.0:
        raise ConfigurationError(
            "The sampling safety factor must be at least 2, otherwise an event can "
            "fall between two samples."
        )
    if min_interval_s > max_interval_s:
        raise ConfigurationError("The minimum sampling interval exceeds the maximum.")

    interval = shortest_event_s / safety_factor
    interval = min(max(interval, min_interval_s), max_interval_s)
    # Never ask for a step finer than the recording itself provides.
    return max(interval, 1.0 / fps)


@dataclass(frozen=True)
class SamplingPlan:
    """A concrete decision about which frames a stage will look at."""

    fps: float
    start_s: float
    end_s: float
    interval_s: float
    step_frames: int
    scale: float  # 1.0 = source resolution, 0.25 = quarter width/height

    @property
    def effective_interval_s(self) -> float:
        """The interval actually achieved once rounded to whole frames."""
        return frames_to_seconds(self.step_frames, self.fps)

    @property
    def estimated_samples(self) -> int:
        span = max(0.0, self.end_s - self.start_s)
        return int(span / self.effective_interval_s) + 1

    def describe(self) -> str:
        return (
            f"{self.start_s:.2f}-{self.end_s:.2f}s every "
            f"{self.effective_interval_s:.3f}s ({self.step_frames} frames) "
            f"at scale {self.scale:.3f} (~{self.estimated_samples} samples)"
        )


def build_sampling_plan(
    *,
    fps: float,
    duration_s: float,
    interval_s: float,
    scale: float = 1.0,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> SamplingPlan:
    """Create a :class:`SamplingPlan`, rounding the interval to whole frames."""
    validate_fps(fps)
    if duration_s < 0:
        raise ConfigurationError("The video duration must not be negative.")
    if not 0 < scale <= 1.0:
        raise ConfigurationError("The scale factor must be in the range (0, 1].")

    stop = duration_s if end_s is None else min(end_s, duration_s)
    begin = max(0.0, min(start_s, stop))
    step_frames = seconds_to_frames(interval_s, fps, minimum=1)
    return SamplingPlan(
        fps=fps,
        start_s=begin,
        end_s=stop,
        interval_s=interval_s,
        step_frames=step_frames,
        scale=scale,
    )


def scale_factor_for_width(source_width: int, target_width: int) -> float:
    """Downscale factor that maps ``source_width`` to ``target_width``.

    Never upscales: analysing an enlarged frame costs more and adds nothing.
    """
    if source_width <= 0 or target_width <= 0:
        raise ConfigurationError("Frame widths must be positive.")
    return min(1.0, target_width / source_width)
