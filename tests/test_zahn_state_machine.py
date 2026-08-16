"""Zahn flow state machine, driven by synthetic observation sequences.

Each test feeds the state machine a scripted sequence of frames - continuous
stream, flicker, drops, disturbance - and asserts the timestamps it reports.
No video decoding is involved, so these run in milliseconds and pin down the
timing rules exactly.
"""

from __future__ import annotations

import pytest

from app.analysis.base_detector import ScoreSample
from app.analysis.zahn_detector import FlowStateMachine, score_confidence
from app.config import ZahnConfig

FPS = 25.0
FRAME = 1.0 / FPS


def liquid(value: float = 0.6, outlet: float | None = None, contrast: float = 40.0) -> ScoreSample:
    """A frame in which liquid is visible."""
    return ScoreSample(
        value=value,
        extras={
            "outlet_score": value if outlet is None else outlet,
            "stream_contrast": contrast,
            "noise_sigma": 3.0,
        },
    )


def empty() -> ScoreSample:
    return ScoreSample(value=0.0, extras={"outlet_score": 0.0, "noise_sigma": 3.0})


def disturbed() -> ScoreSample:
    return ScoreSample(value=0.0, disturbed=True, extras={"outlet_score": 0.9, "noise_sigma": 3.0})


def feed(machine: FlowStateMachine, samples: list[ScoreSample], start_s: float = 0.0) -> float:
    """Feed frames at a constant frame rate; returns the last timestamp."""
    timestamp = start_s
    for index, sample in enumerate(samples):
        timestamp = start_s + index * FRAME
        machine.update(timestamp, sample)
    return timestamp


def config(**overrides) -> ZahnConfig:
    base = ZahnConfig()
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


class TestFlowStart:
    def test_start_requires_persistence(self):
        """A single bright frame is not the start of flow."""
        machine = FlowStateMachine(config(flow_start_persistence_s=0.3))
        feed(machine, [empty()] * 10 + [liquid()] + [empty()] * 10)
        assert machine.measurement.start_s is None

    def test_start_is_the_first_frame_of_the_run(self):
        """Not the frame at which persistence was satisfied."""
        machine = FlowStateMachine(config(flow_start_persistence_s=0.3))
        feed(machine, [empty()] * 25 + [liquid()] * 50)
        assert machine.measurement.start_s == pytest.approx(1.0, abs=FRAME)

    def test_flicker_shorter_than_persistence_is_ignored(self):
        machine = FlowStateMachine(config(flow_start_persistence_s=0.5))
        # Five separate 0.2 s flickers, none long enough to be real.
        sequence: list[ScoreSample] = []
        for _ in range(5):
            sequence += [liquid()] * 5 + [empty()] * 10
        feed(machine, sequence)
        assert machine.measurement.start_s is None

    def test_liquid_below_the_outlet_alone_does_not_start_the_clock(self):
        """Something moving lower in the region is not flow from the orifice."""
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2))
        feed(machine, [liquid(value=0.8, outlet=0.0)] * 50)
        assert machine.measurement.start_s is None

    def test_disturbed_frames_cannot_start_the_clock(self):
        """Camera shake or a hand crossing the cup must not trigger a start."""
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2))
        feed(machine, [disturbed()] * 50)
        assert machine.measurement.start_s is None
        assert machine.measurement.frames_disturbed == 50


class TestFlowEnd:
    def test_end_is_not_triggered_by_a_single_broken_frame(self):
        """The stream flickers; timing must continue."""
        machine = FlowStateMachine(config(flow_end_persistence_s=0.5))
        sequence = [liquid()] * 50 + [empty()] * 3 + [liquid()] * 50
        feed(machine, sequence)
        assert not machine.finished

    def test_intermittent_drops_keep_the_measurement_open(self):
        """stream -> weak stream -> drops must be timed to the last drop."""
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        sequence = [liquid()] * 100  # 4 s of continuous stream
        # Six drops, each one frame long, 0.3 s apart - shorter than the
        # 0.5 s persistence, so none of them may end the measurement.
        for _ in range(6):
            sequence += [liquid()] + [empty()] * 7
        last_drop_index = len(sequence) - 8
        sequence += [empty()] * 40  # 1.6 s of nothing: this ends it

        feed(machine, sequence)
        assert machine.finished
        assert machine.measurement.end_confirmed
        assert machine.measurement.end_s == pytest.approx(last_drop_index * FRAME, abs=FRAME)

    def test_end_is_the_last_activity_not_the_confirmation_moment(self):
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        sequence = [liquid()] * 100 + [empty()] * 40
        feed(machine, sequence)
        assert machine.measurement.end_s == pytest.approx(99 * FRAME, abs=FRAME)
        # Confirmation happened 0.5 s later, and must not be reported as the end.
        assert machine.measurement.end_s < 100 * FRAME

    def test_persistence_is_time_based_not_frame_based(self):
        """0.5 s must mean 0.5 s at any frame rate - never '15 frames'."""
        results = {}
        for fps in (10.0, 25.0, 50.0):
            machine = FlowStateMachine(
                config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5)
            )
            frame = 1.0 / fps
            timestamp = 0.0
            for index in range(int(3 * fps)):  # 3 s of stream
                timestamp = index * frame
                machine.update(timestamp, liquid())
            index = int(3 * fps)
            while not machine.finished and index < int(6 * fps):
                timestamp = index * frame
                machine.update(timestamp, empty())
                index += 1
            results[fps] = machine.measurement.end_s
        # Every frame rate must report the same end time (within one frame).
        assert max(results.values()) - min(results.values()) < 0.11

    def test_video_ending_mid_flow_is_not_confirmed(self):
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2))
        last = feed(machine, [liquid()] * 100)
        measurement = machine.finalize(last)
        assert measurement.end_s is not None
        assert measurement.end_confirmed is False

    def test_efflux_time_matches_the_scripted_run(self):
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        # 1 s of nothing, 10 s of stream, then silence.
        sequence = [empty()] * 25 + [liquid()] * 250 + [empty()] * 40
        last = feed(machine, sequence)
        measurement = machine.finalize(last)
        assert measurement.efflux_s == pytest.approx(10.0, abs=2 * FRAME)


class TestBreaksAndContinuity:
    def test_long_breaks_are_recorded(self):
        machine = FlowStateMachine(
            config(
                flow_start_persistence_s=0.2,
                flow_end_persistence_s=1.0,
                break_report_threshold_s=0.2,
            )
        )
        sequence = [liquid()] * 50 + [empty()] * 10 + [liquid()] * 50 + [empty()] * 40
        last = feed(machine, sequence)
        measurement = machine.finalize(last)
        assert len(measurement.breaks) == 1

    def test_continuity_excludes_the_confirmation_tail(self):
        """The 0.5 s spent proving absence must not count as 'no liquid'."""
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        feed(machine, [liquid()] * 100 + [empty()] * 40)
        measurement = machine.finalize(6.0)
        assert measurement.continuity == pytest.approx(1.0, abs=0.02)


class TestConfidence:
    def test_no_start_means_zero_confidence(self):
        machine = FlowStateMachine(config())
        measurement = machine.finalize(5.0)
        confidence, reasons = score_confidence(measurement, config())
        assert confidence == 0.0
        assert reasons

    def test_clean_run_is_confident(self):
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        feed(machine, [empty()] * 25 + [liquid()] * 750 + [empty()] * 40)
        measurement = machine.finalize(35.0)
        confidence, _ = score_confidence(measurement, config())
        assert confidence >= 0.70

    def test_unconfirmed_end_is_capped(self):
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2))
        last = feed(machine, [liquid()] * 750)
        measurement = machine.finalize(last)
        confidence, reasons = score_confidence(measurement, config())
        assert confidence <= 0.5
        assert any("lower bound" in reason for reason in reasons)

    def test_disturbance_reduces_confidence(self):
        settings = config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5)
        clean = FlowStateMachine(settings)
        feed(clean, [liquid()] * 250 + [empty()] * 40)
        clean_confidence, _ = score_confidence(clean.finalize(12.0), settings)

        noisy = FlowStateMachine(settings)
        # Flow starts cleanly, then someone keeps knocking the camera.
        sequence: list[ScoreSample] = [liquid()] * 20
        for _ in range(46):
            sequence += [liquid()] * 4 + [disturbed()]
        sequence += [empty()] * 40
        feed(noisy, sequence)
        noisy_confidence, reasons = score_confidence(noisy.finalize(12.0), settings)

        assert noisy_confidence < clean_confidence
        assert any("movement around the cup" in reason for reason in reasons)

    def test_implausibly_short_measurement_is_capped(self):
        settings = config(
            flow_start_persistence_s=0.1, flow_end_persistence_s=0.3, min_plausible_efflux_s=2.0
        )
        machine = FlowStateMachine(settings)
        feed(machine, [liquid()] * 20 + [empty()] * 20)
        measurement = machine.finalize(2.0)
        confidence, reasons = score_confidence(measurement, settings)
        assert confidence <= 0.45
        assert any("very short" in reason for reason in reasons)
