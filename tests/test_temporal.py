"""Unit tests for the shared temporal logic.

These use synthetic score sequences rather than video: no motion, brief false
motion, sustained motion, motion with gaps, and so on.
"""

from __future__ import annotations

import pytest

from app.analysis.temporal import (
    Interval,
    PersistenceTimer,
    adaptive_thresholds,
    bridge_gaps,
    clamp,
    drop_short,
    find_active_intervals,
    median,
    quantile,
    robust_noise_sigma,
    safe_ratio,
)
from app.errors import ConfigurationError


def times(count: int, step: float = 1.0) -> list[float]:
    return [i * step for i in range(count)]


class TestInterval:
    def test_duration(self):
        assert Interval(2.0, 5.5).duration_s == pytest.approx(3.5)

    def test_end_before_start_rejected(self):
        with pytest.raises(ConfigurationError):
            Interval(5.0, 2.0)

    def test_expansion_is_clamped(self):
        interval = Interval(2.0, 5.0).expanded(5.0, 5.0, lower=0.0, upper=8.0)
        assert (interval.start_s, interval.end_s) == (0.0, 8.0)

    def test_overlap(self):
        assert Interval(0, 5).overlaps(Interval(4, 9))
        assert not Interval(0, 5).overlaps(Interval(5.1, 9))


class TestStatistics:
    def test_median_even_and_odd(self):
        assert median([3, 1, 2]) == 2
        assert median([4, 1, 2, 3]) == 2.5

    def test_quantile_interpolates(self):
        assert quantile([0, 10], 0.5) == pytest.approx(5.0)
        assert quantile([0, 1, 2, 3, 4], 0.25) == pytest.approx(1.0)

    def test_noise_sigma_ignores_events(self):
        """A few large spikes must not inflate the noise estimate."""
        quiet = [0.01, 0.012, 0.009, 0.011] * 20
        with_events = quiet + [0.9, 0.95, 0.88]
        assert robust_noise_sigma(with_events) == pytest.approx(
            robust_noise_sigma(quiet), abs=0.002
        )

    def test_empty_sequence_rejected(self):
        with pytest.raises(ConfigurationError):
            median([])


class TestAdaptiveThresholds:
    def test_floor_applies_to_a_static_scene(self):
        """Pure noise must not produce a threshold below the absolute floor."""
        scores = [0.0005, 0.0004, 0.0006] * 50
        enter, exit_ = adaptive_thresholds(scores, enter_sigma=6.0, exit_sigma=3.0, floor=0.004)
        assert enter == pytest.approx(0.004)
        assert exit_ <= enter

    def test_threshold_rises_with_a_noisy_trace(self):
        quiet = [0.001, 0.002, 0.0015] * 40
        noisy = [0.02, 0.05, 0.01, 0.04] * 30
        quiet_enter, _ = adaptive_thresholds(quiet, enter_sigma=6.0, exit_sigma=3.0, floor=0.001)
        noisy_enter, _ = adaptive_thresholds(noisy, enter_sigma=6.0, exit_sigma=3.0, floor=0.001)
        assert noisy_enter > quiet_enter

    def test_sensitivity_lowers_the_threshold(self):
        scores = [0.01, 0.012, 0.011, 0.05] * 20
        low, _ = adaptive_thresholds(
            scores, enter_sigma=6.0, exit_sigma=3.0, floor=0.001, sensitivity=0.0
        )
        high, _ = adaptive_thresholds(
            scores, enter_sigma=6.0, exit_sigma=3.0, floor=0.001, sensitivity=1.0
        )
        assert high < low

    def test_exit_never_exceeds_enter(self):
        scores = [0.1] * 10
        enter, exit_ = adaptive_thresholds(scores, enter_sigma=1.0, exit_sigma=9.0, floor=0.0)
        assert exit_ <= enter

    def test_invalid_sensitivity_rejected(self):
        with pytest.raises(ConfigurationError):
            adaptive_thresholds([0.1], enter_sigma=1, exit_sigma=1, floor=0, sensitivity=2.0)


class TestFindActiveIntervals:
    def test_no_motion_yields_no_intervals(self):
        scores = [0.001] * 30
        assert (
            find_active_intervals(times(30), scores, enter_threshold=0.1, exit_threshold=0.05) == []
        )

    def test_continuous_motion_is_one_interval(self):
        scores = [0.001] * 5 + [0.5] * 10 + [0.001] * 5
        intervals = find_active_intervals(
            times(20),
            scores,
            enter_threshold=0.1,
            exit_threshold=0.05,
            conservative_boundaries=False,
        )
        assert len(intervals) == 1
        assert (intervals[0].start_s, intervals[0].end_s) == (5.0, 14.0)

    def test_conservative_boundaries_widen_the_interval(self):
        """The coarse pass must guarantee the true edge is inside the window."""
        scores = [0.001] * 5 + [0.5] * 10 + [0.001] * 5
        conservative = find_active_intervals(
            times(20),
            scores,
            enter_threshold=0.1,
            exit_threshold=0.05,
            conservative_boundaries=True,
        )[0]
        assert conservative.start_s == 4.0  # one sample earlier
        assert conservative.end_s == 15.0  # one sample later

    def test_hysteresis_keeps_a_wobbling_signal_as_one_event(self):
        # Values dip below the enter threshold but stay above the exit one.
        scores = [0.0] * 3 + [0.5, 0.08, 0.5, 0.07, 0.5] + [0.0] * 3
        intervals = find_active_intervals(
            times(11),
            scores,
            enter_threshold=0.1,
            exit_threshold=0.05,
            conservative_boundaries=False,
        )
        assert len(intervals) == 1

    def test_event_still_active_at_end_is_closed(self):
        scores = [0.0] * 3 + [0.5] * 4
        intervals = find_active_intervals(
            times(7), scores, enter_threshold=0.1, exit_threshold=0.05
        )
        assert intervals[-1].end_s == 6.0

    def test_disturbed_samples_cannot_start_an_event(self):
        scores = [0.0, 0.0, 0.9, 0.9, 0.0]
        disturbed = [False, False, True, True, False]
        assert (
            find_active_intervals(
                times(5), scores, enter_threshold=0.1, exit_threshold=0.05, disturbed=disturbed
            )
            == []
        )

    def test_mismatched_lengths_rejected(self):
        with pytest.raises(ConfigurationError):
            find_active_intervals([0.0, 1.0], [0.5], enter_threshold=0.1, exit_threshold=0.0)

    def test_exit_above_enter_rejected(self):
        with pytest.raises(ConfigurationError):
            find_active_intervals(times(3), [0, 0, 0], enter_threshold=0.1, exit_threshold=0.5)


class TestBridgeAndFilter:
    def test_short_gaps_are_bridged(self):
        merged = bridge_gaps([Interval(0, 5), Interval(6, 9)], max_gap_s=2.0)
        assert merged == [Interval(0, 9)]

    def test_long_gaps_are_kept_separate(self):
        merged = bridge_gaps([Interval(0, 5), Interval(20, 25)], max_gap_s=2.0)
        assert len(merged) == 2

    def test_short_intervals_dropped(self):
        kept = drop_short([Interval(0, 0.4), Interval(2, 8)], min_duration_s=1.0)
        assert kept == [Interval(2, 8)]

    def test_brief_false_motion_is_removed(self):
        """A single noisy sample must not become an event."""
        scores = [0.0] * 10 + [0.9] + [0.0] * 10
        intervals = find_active_intervals(
            times(21),
            scores,
            enter_threshold=0.1,
            exit_threshold=0.05,
            conservative_boundaries=False,
        )
        assert drop_short(intervals, min_duration_s=1.0) == []


class TestPersistenceTimer:
    def test_fires_only_after_the_required_time(self):
        timer = PersistenceTimer(required_s=0.5)
        assert not timer.update(0.0, True)
        assert not timer.update(0.2, True)
        assert not timer.update(0.4, True)
        assert timer.update(0.5, True)

    def test_reports_when_the_run_started_not_when_it_fired(self):
        """The reported flow start must be the first frame of the run."""
        timer = PersistenceTimer(required_s=0.5)
        for t in (2.0, 2.2, 2.4, 2.6):
            timer.update(t, True)
        assert timer.fired
        assert timer.run_start_s == 2.0

    def test_interruption_resets_the_run(self):
        timer = PersistenceTimer(required_s=0.5)
        timer.update(0.0, True)
        timer.update(0.3, True)
        timer.update(0.4, False)  # one bad frame
        assert not timer.update(0.6, True)
        assert timer.run_start_s == 0.6

    def test_is_frame_rate_independent(self):
        """Half a second is half a second at 15 FPS and at 60 FPS."""
        for fps in (15.0, 30.0, 60.0):
            timer = PersistenceTimer(required_s=0.5)
            fired_at = None
            for index in range(int(fps)):
                timestamp = index / fps
                if timer.update(timestamp, True) and fired_at is None:
                    fired_at = timestamp
            assert fired_at == pytest.approx(0.5, abs=1.0 / fps)

    def test_zero_requirement_fires_immediately(self):
        assert PersistenceTimer(required_s=0.0).update(1.0, True)

    def test_negative_requirement_rejected(self):
        with pytest.raises(ConfigurationError):
            PersistenceTimer(required_s=-1.0)


class TestSmallHelpers:
    def test_clamp(self):
        assert clamp(5, 0, 1) == 1
        assert clamp(-5, 0, 1) == 0

    def test_safe_ratio_handles_zero(self):
        assert safe_ratio(1, 0, default=0.5) == 0.5
        assert safe_ratio(1, 4) == 0.25
