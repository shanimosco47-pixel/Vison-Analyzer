"""Unit tests for time/frame conversion and sampling plans."""

from __future__ import annotations

import pytest

from app.errors import ConfigurationError
from app.video.sampling import (
    build_sampling_plan,
    frame_index_at,
    frames_to_seconds,
    resolve_sample_interval,
    scale_factor_for_width,
    seconds_to_frames,
    validate_fps,
)


class TestSecondsToFrames:
    @pytest.mark.parametrize(
        ("seconds", "fps", "expected"),
        [
            (0.5, 30.0, 15),
            (0.5, 60.0, 30),
            (0.5, 15.0, 8),  # 7.5 rounds to 8
            (1.0, 25.0, 25),
            (0.0, 25.0, 0),
            (2.0, 29.97, 60),
        ],
    )
    def test_conversion_depends_on_fps(self, seconds, fps, expected):
        """The same half second is a different number of frames per camera."""
        assert seconds_to_frames(seconds, fps) == expected

    def test_minimum_clamps_upward(self):
        assert seconds_to_frames(0.001, 25.0, minimum=1) == 1

    def test_negative_duration_rejected(self):
        with pytest.raises(ConfigurationError):
            seconds_to_frames(-1.0, 25.0)

    def test_round_trip(self):
        assert frames_to_seconds(seconds_to_frames(2.0, 30.0), 30.0) == pytest.approx(2.0)


class TestValidateFps:
    @pytest.mark.parametrize("fps", [0.0, -25.0, 100000.0, float("nan"), float("inf")])
    def test_implausible_values_rejected(self, fps):
        with pytest.raises(ConfigurationError):
            validate_fps(fps)

    def test_plausible_value_returned(self):
        assert validate_fps(23.976) == pytest.approx(23.976)


class TestFrameIndexAt:
    def test_floor_semantics(self):
        # 1.999 s at 25 fps is still frame 49 (frame 50 starts at 2.0 s).
        assert frame_index_at(1.999, 25.0) == 49
        assert frame_index_at(2.0, 25.0) == 50

    def test_clamped_to_frame_count(self):
        assert frame_index_at(1000.0, 25.0, frame_count=100) == 99

    def test_negative_time_is_zero(self):
        assert frame_index_at(-5.0, 25.0) == 0


class TestResolveSampleInterval:
    def test_derived_from_shortest_event(self):
        """15 s events with a safety factor of 3 means sampling every 5 s."""
        assert resolve_sample_interval(15.0, 3.0, fps=25.0) == pytest.approx(5.0)

    def test_never_samples_only_once_per_event(self):
        """The classic mistake: one sample per event length can miss the event."""
        interval = resolve_sample_interval(15.0, 3.0, fps=25.0)
        assert interval < 15.0

    def test_safety_factor_below_two_rejected(self):
        with pytest.raises(ConfigurationError):
            resolve_sample_interval(15.0, 1.5, fps=25.0)

    def test_clamped_to_maximum(self):
        assert resolve_sample_interval(600.0, 3.0, fps=25.0, max_interval_s=5.0) == 5.0

    def test_clamped_to_minimum(self):
        assert resolve_sample_interval(0.1, 3.0, fps=25.0, min_interval_s=0.2) == 0.2

    def test_never_finer_than_one_frame(self):
        """A 2 FPS camera cannot be sampled every 0.1 s."""
        assert resolve_sample_interval(0.2, 3.0, fps=2.0, min_interval_s=0.01) == pytest.approx(0.5)

    def test_zero_length_event_rejected(self):
        with pytest.raises(ConfigurationError):
            resolve_sample_interval(0.0, 3.0, fps=25.0)


class TestSamplingPlan:
    def test_step_frames_rounded_from_seconds(self):
        plan = build_sampling_plan(fps=30.0, duration_s=60.0, interval_s=0.5)
        assert plan.step_frames == 15
        assert plan.effective_interval_s == pytest.approx(0.5)

    def test_estimated_samples_for_a_long_recording(self):
        """A 12 hour recording sampled every 5 s is 8 641 samples, not a million."""
        plan = build_sampling_plan(fps=25.0, duration_s=12 * 3600, interval_s=5.0)
        assert plan.estimated_samples == 8641
        assert plan.step_frames == 125

    def test_range_is_clamped_to_duration(self):
        plan = build_sampling_plan(fps=25.0, duration_s=10.0, interval_s=1.0, end_s=99.0)
        assert plan.end_s == 10.0

    def test_start_cannot_exceed_end(self):
        plan = build_sampling_plan(fps=25.0, duration_s=10.0, interval_s=1.0, start_s=50.0)
        assert plan.start_s == plan.end_s == 10.0

    def test_step_is_at_least_one_frame(self):
        plan = build_sampling_plan(fps=25.0, duration_s=10.0, interval_s=0.0001)
        assert plan.step_frames == 1

    def test_invalid_scale_rejected(self):
        with pytest.raises(ConfigurationError):
            build_sampling_plan(fps=25.0, duration_s=10.0, interval_s=1.0, scale=1.5)

    def test_describe_is_human_readable(self):
        plan = build_sampling_plan(fps=25.0, duration_s=60.0, interval_s=2.0, scale=0.25)
        text = plan.describe()
        assert "2.000s" in text and "50 frames" in text


class TestScaleFactor:
    def test_downscale(self):
        assert scale_factor_for_width(1920, 320) == pytest.approx(320 / 1920)

    def test_never_upscales(self):
        assert scale_factor_for_width(160, 320) == 1.0

    def test_invalid_widths_rejected(self):
        with pytest.raises(ConfigurationError):
            scale_factor_for_width(0, 320)
