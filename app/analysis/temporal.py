"""Turning a time series of activity scores into time intervals.

This is the shared temporal core of every detector, and it is deliberately
free of OpenCV so it can be tested with synthetic sequences: no motion, brief
false motion, sustained motion, motion with gaps, and so on.

Two ideas run through all of it:

*   **Hysteresis** - a higher threshold to enter an event than to leave it, so
    a score hovering around one threshold does not produce a burst of events.
*   **Time-based persistence** - every duration is in seconds and compared
    against wall-clock spacing of the samples, so behaviour does not change
    with the frame rate or the sampling interval.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..errors import ConfigurationError

# Scale factor turning a median-absolute-deviation into a standard-deviation
# estimate for normally distributed data.  MAD is used instead of the plain
# standard deviation because a few large events would otherwise inflate the
# noise estimate and hide themselves.
MAD_TO_SIGMA = 1.4826


@dataclass(frozen=True)
class Interval:
    """A closed time span in video-relative seconds."""

    start_s: float
    end_s: float

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ConfigurationError(
                "An interval cannot end before it starts.",
                detail=f"{self.start_s}..{self.end_s}",
            )

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def expanded(self, pre_s: float, post_s: float, *, lower: float, upper: float) -> Interval:
        """Grow by pre/post roll, clamped to ``[lower, upper]``."""
        return Interval(
            max(lower, self.start_s - pre_s),
            min(upper, self.end_s + post_s),
        )

    def overlaps(self, other: Interval) -> bool:
        return self.start_s <= other.end_s and other.start_s <= self.end_s


def median(values: Sequence[float]) -> float:
    if not values:
        raise ConfigurationError("Cannot take the median of an empty sequence.")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile (``q`` in [0, 1])."""
    if not values:
        raise ConfigurationError("Cannot take a quantile of an empty sequence.")
    if not 0.0 <= q <= 1.0:
        raise ConfigurationError("A quantile must be between 0 and 1.")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = q * (len(ordered) - 1)
    lower_index = int(math.floor(position))
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight


def robust_noise_sigma(values: Sequence[float], *, baseline: float | None = None) -> float:
    """Noise estimate that ignores the events themselves (MAD based).

    Only samples at or below ``baseline`` contribute, so a window that is
    mostly *active* (which is the normal case during refinement) still yields
    the noise level of its quiet part rather than of its event.
    """
    if len(values) < 2:
        return 0.0
    centre = median(values) if baseline is None else baseline
    quiet = [v for v in values if v <= centre] or list(values)
    deviations = [abs(v - centre) for v in quiet]
    return MAD_TO_SIGMA * median(deviations)


def adaptive_thresholds(
    scores: Sequence[float],
    *,
    enter_sigma: float,
    exit_sigma: float,
    floor: float,
    sensitivity: float = 0.5,
    baseline_quantile: float = 0.5,
) -> tuple[float, float]:
    """Derive (enter, exit) thresholds from the statistics of the trace itself.

    A fixed threshold cannot serve both a clean indoor camera and a noisy
    compressed night-time stream.  The thresholds are therefore
    ``median + k * sigma`` of the observed score distribution, with ``floor``
    as an absolute minimum so a perfectly static scene cannot fire on noise.

    ``sensitivity`` in [0, 1] scales the sigma multipliers: 0.5 leaves them as
    configured, 1.0 halves them (more sensitive, more candidates), 0.0 doubles
    them (fewer, stronger candidates).

    ``baseline_quantile`` selects what counts as "quiet".  The coarse pass uses
    the median because most of a long recording is uneventful; the dense pass
    uses a lower quantile because a refinement window is mostly event.
    """
    if not 0.0 <= sensitivity <= 1.0:
        raise ConfigurationError("Sensitivity must be between 0 and 1.")
    if not scores:
        return floor, floor

    gain = 2.0 ** (1.0 - 2.0 * sensitivity)  # 2.0 at sensitivity 0, 0.5 at 1.0
    baseline = quantile(scores, baseline_quantile)
    sigma = robust_noise_sigma(scores, baseline=baseline)

    enter = max(floor, baseline + enter_sigma * gain * sigma)
    exit_ = max(floor * 0.5, baseline + exit_sigma * gain * sigma)
    return enter, min(exit_, enter)


def find_active_intervals(
    times: Sequence[float],
    scores: Sequence[float],
    *,
    enter_threshold: float,
    exit_threshold: float,
    disturbed: Sequence[bool] | None = None,
    conservative_boundaries: bool = True,
) -> list[Interval]:
    """Hysteresis thresholding of a score trace.

    Args:
        times: sample timestamps in seconds, ascending.
        scores: one score per timestamp.
        enter_threshold: score at or above which an event starts.
        exit_threshold: score below which a running event ends.
        disturbed: optional per-sample flag; disturbed samples can neither
            start nor end an event, they are simply carried through.  This is
            how camera shake and people crossing the frame are prevented from
            creating or truncating events.
        conservative_boundaries: when True (the coarse pass), the boundary is
            placed at the neighbouring sample so the true edge is guaranteed
            to lie inside the interval.  When False (the dense pass), the
            crossing sample itself is used, which is accurate to one frame.

    Returns:
        Intervals in ascending order.  An event still active at the end of the
        trace is closed at the last sample time.
    """
    if len(times) != len(scores):
        raise ConfigurationError("Times and scores must have the same length.")
    if disturbed is not None and len(disturbed) != len(times):
        raise ConfigurationError("The disturbance flags must match the trace length.")
    if exit_threshold > enter_threshold:
        raise ConfigurationError("The exit threshold must not exceed the enter threshold.")

    intervals: list[Interval] = []
    active = False
    start_s = 0.0

    for i, (t, score) in enumerate(zip(times, scores, strict=True)):
        if disturbed is not None and disturbed[i]:
            continue
        if not active:
            if score >= enter_threshold:
                active = True
                start_s = times[i - 1] if (conservative_boundaries and i > 0) else t
        else:
            if score < exit_threshold:
                end_s = t if conservative_boundaries else times[max(0, i - 1)]
                intervals.append(Interval(start_s, max(end_s, start_s)))
                active = False

    if active and times:
        intervals.append(Interval(start_s, max(times[-1], start_s)))
    return intervals


def bridge_gaps(intervals: Sequence[Interval], max_gap_s: float) -> list[Interval]:
    """Merge intervals separated by less than ``max_gap_s``."""
    if max_gap_s < 0:
        raise ConfigurationError("The bridging gap must not be negative.")
    merged: list[Interval] = []
    for interval in sorted(intervals, key=lambda i: i.start_s):
        if merged and interval.start_s - merged[-1].end_s <= max_gap_s:
            previous = merged.pop()
            merged.append(Interval(previous.start_s, max(previous.end_s, interval.end_s)))
        else:
            merged.append(interval)
    return merged


def drop_short(intervals: Sequence[Interval], min_duration_s: float) -> list[Interval]:
    """Discard intervals shorter than ``min_duration_s``."""
    if min_duration_s < 0:
        raise ConfigurationError("The minimum duration must not be negative.")
    return [i for i in intervals if i.duration_s >= min_duration_s]


@dataclass
class PersistenceTimer:
    """Requires a condition to hold for a period of *time* before firing.

    Frame counting is deliberately avoided: "15 frames" means half a second on
    one camera and a quarter of a second on another.  The timer instead
    remembers when the current run of true observations began and fires once
    that run has lasted ``required_s``.

    Typical use is one instance for "liquid is flowing" and one for "liquid has
    stopped", both fed by the same per-frame observation.
    """

    required_s: float
    _run_start_s: float | None = None
    _last_true_s: float | None = None
    fired: bool = False

    def __post_init__(self) -> None:
        if self.required_s < 0:
            raise ConfigurationError("A persistence period must not be negative.")

    def reset(self) -> None:
        self._run_start_s = None
        self._last_true_s = None
        self.fired = False

    @property
    def run_start_s(self) -> float | None:
        """Timestamp at which the current (or firing) run began."""
        return self._run_start_s

    @property
    def elapsed_s(self) -> float:
        if self._run_start_s is None or self._last_true_s is None:
            return 0.0
        return self._last_true_s - self._run_start_s

    def update(self, timestamp_s: float, condition: bool) -> bool:
        """Feed one observation. Returns True once the run reaches its length.

        The return value stays True for every further observation while the
        condition holds, so callers can treat it as a level rather than an
        edge; use :attr:`run_start_s` to recover *when* the run started.
        """
        if not condition:
            self.reset()
            return False
        if self._run_start_s is None:
            self._run_start_s = timestamp_s
        self._last_true_s = timestamp_s
        # A zero-length requirement fires on the first observation.
        self.fired = (timestamp_s - self._run_start_s) >= self.required_s or self.required_s == 0.0
        return self.fired


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that returns ``default`` instead of raising on a zero divisor."""
    if denominator == 0 or not math.isfinite(denominator):
        return default
    return numerator / denominator
