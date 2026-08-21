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


def feed_trust(
    machine: FlowStateMachine, entries: list[tuple[ScoreSample, bool]], start_s: float = 0.0
) -> float:
    """Like ``feed``, but each frame carries its own ``trusted`` flag."""
    timestamp = start_s
    for index, (sample, trusted) in enumerate(entries):
        timestamp = start_s + index * FRAME
        machine.update(timestamp, sample, trusted=trusted)
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

    def test_disturbance_does_not_count_toward_the_end_persistence(self):
        """Frames we could not trust must not be credited as observed absence.

        Regression: the disturbed branch reset only the start timer, so the
        absence run kept its pre-disturbance start time. One clean frame
        arriving after a long disturbance then satisfied the 0.5 s rule on
        0.16 s of real evidence.
        """
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        sequence = [liquid()] * 100  # stream; last liquid at 3.96 s
        sequence += [empty()] * 3  # 0.12 s of genuine absence
        sequence += [disturbed()] * 25  # 1.0 s proving nothing either way
        sequence += [empty()] * 1  # one clean frame: must NOT confirm the end
        feed(machine, sequence)
        assert not machine.finished

        # Only a fresh, full 0.5 s run of trusted absence may end the measurement.
        feed(machine, [empty()] * 14, start_s=len(sequence) * FRAME)
        assert machine.finished
        assert machine.measurement.end_confirmed
        # The reported end is still the last frame that showed liquid.
        assert machine.measurement.end_s == pytest.approx(99 * FRAME, abs=FRAME)

    def test_disturbance_between_final_drops_does_not_end_the_measurement(self):
        """The realistic case: a hand crosses the cup while the last drops fall."""
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        sequence = [liquid()] * 100  # continuous stream
        for _ in range(3):  # three drops, 0.2 s apart
            sequence += [liquid()] + [empty()] * 5
        sequence += [empty()] * 2 + [disturbed()] * 25 + [empty()] * 1
        for _ in range(2):  # two more drops after the disturbance
            sequence += [liquid()] + [empty()] * 5
        last_drop_index = max(i for i, sample in enumerate(sequence) if sample.value > 0)
        sequence += [empty()] * 20  # sustained absence: now it may end

        feed(machine, sequence)
        assert machine.finished
        assert machine.measurement.end_confirmed
        # Timing must run to the last drop, not to the drop before the disturbance.
        assert machine.measurement.end_s == pytest.approx(last_drop_index * FRAME, abs=FRAME)

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


class TestUntrackedGapsDuringFlow:
    """Codex review: a gap forgotten once trusted liquid returns is a bug.

    ``_gap_after_activity`` alone is cleared by ``_update_flowing`` the
    moment fresh trusted liquid arrives - a gap in the *middle* of an
    otherwise clean run left no trace by the time a much later, ordinary
    end was confirmed. The acceptance criterion is that tracking lost
    longer than ``zahn_max_endpoint_uncertainty_s`` must make the
    measurement unconfirmed wherever in the run it happens, not only right
    at the reported end.

    Second review round: a first fix reported such a gap's own (start, end)
    span as ``end_uncertainty_bounds`` - but trusted liquid seen *after* the
    gap closed proves the true end is not "somewhere in that gap"; folding
    it in fabricated a bound that could exclude the actual later end
    entirely. The tests below pin the corrected contract: the measurement
    becomes unconfirmed (``end_gap_unresolved``), but no bound is invented
    for where the true end sits.
    """

    def test_a_long_gap_in_the_middle_of_flow_still_poisons_a_later_clean_end(self):
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        entries: list[tuple[ScoreSample, bool]] = []
        entries += [(liquid(), True)] * 30  # 1.2 s of clean, trusted flow
        entries += [(empty(), False)] * 15  # 0.6 s untracked gap (> the 0.5 s default)
        entries += [(liquid(), True)] * 50  # flow visibly, trustedly continues
        last_liquid_index = len(entries) - 1
        entries += [(empty(), True)] * 20  # 0.8 s of trusted absence: ends cleanly

        feed_trust(machine, entries)

        assert machine.finished
        # The reported end itself is still exactly where the last liquid was
        # - trusted liquid after the gap proves flow continued past it, so
        # nothing about the end's own timing is actually in question.
        assert machine.measurement.end_s == pytest.approx(last_liquid_index * FRAME, abs=FRAME)
        # But it must NOT be reported confirmed and precise: a >0.5 s
        # untracked gap happened earlier in the run, so continuity through
        # it could not be verified.
        assert machine.measurement.end_confirmed is False
        assert machine.measurement.end_uncertain is True
        assert machine.measurement.end_gap_unresolved is True
        # No fabricated bound: the gap does not tell us where the true end
        # is (liquid after it proves that story wrong), so there is nothing
        # honest to report as end_uncertainty_bounds.
        assert machine.measurement.end_uncertainty_bounds is None

    def test_a_mid_flow_gap_plus_a_separate_adjacent_end_gap_keeps_the_adjacent_bounds(self):
        """A second, genuinely adjacent gap right before the end is unaffected.

        The mid-flow gap only forces the measurement unconfirmed; it must
        not blank out an unrelated, honestly-computed bound from a
        different gap that really is adjacent to the reported end.
        """
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        entries: list[tuple[ScoreSample, bool]] = []
        entries += [(liquid(), True)] * 30  # clean, trusted flow
        entries += [(empty(), False)] * 15  # mid-flow gap: continuity unverified
        entries += [(liquid(), True)] * 50  # flow visibly, trustedly continues
        last_activity_idx = len(entries) - 1
        entries += [(empty(), False)] * 15  # a *second*, separate gap...
        first_trusted_after_idx = len(entries)
        entries += [
            (empty(), True)
        ] * 20  # ...with only trusted absence after it: adjacent to the end

        feed_trust(machine, entries)

        assert machine.finished
        assert machine.measurement.end_confirmed is False
        assert machine.measurement.end_gap_unresolved is True
        # This bound comes from the second gap alone, and it still honestly
        # brackets the end: no trusted liquid was seen between it closing
        # and the end being confirmed.
        bounds = machine.measurement.end_uncertainty_bounds
        assert bounds is not None
        lo, hi = bounds
        assert lo == pytest.approx(last_activity_idx * FRAME, abs=FRAME)
        assert hi == pytest.approx(first_trusted_after_idx * FRAME, abs=FRAME)

    def test_a_short_gap_in_the_middle_of_flow_does_not_taint_a_later_end(self):
        """Sanity check: this is about the gap's *width*, not its position."""
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        entries: list[tuple[ScoreSample, bool]] = []
        entries += [(liquid(), True)] * 30
        entries += [(empty(), False)] * 3  # 0.12 s: well under the 0.5 s default
        entries += [(liquid(), True)] * 50
        entries += [(empty(), True)] * 20

        feed_trust(machine, entries)

        assert machine.finished
        assert machine.measurement.end_confirmed is True
        assert machine.measurement.end_uncertain is False
        assert machine.measurement.end_uncertainty_bounds is None
        assert machine.measurement.end_gap_unresolved is False

    def test_a_gap_immediately_adjacent_to_the_end_is_not_mislabelled_mid_flow(self):
        """Third Codex review round: a promotion-timing regression.

        A first fix for this class set end_gap_unresolved the instant a
        qualifying gap *closed*, before knowing whether liquid or absence
        followed - so this ordinary adjacent-to-the-end case (liquid never
        resumes; trusted absence follows straight through to a confirmed
        end) was wrongly labelled the same as genuine mid-flow resumption.
        Promotion must wait for liquid to actually be observed again.
        """
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        entries: list[tuple[ScoreSample, bool]] = []
        entries += [(liquid(), True)] * 30  # clean, trusted flow
        entries += [(empty(), False)] * 15  # untracked gap (> the 0.5 s default)...
        entries += [(empty(), True)] * 20  # ...followed only by trusted absence: ends cleanly

        feed_trust(machine, entries)

        assert machine.finished
        assert machine.measurement.end_confirmed is False
        assert machine.measurement.end_uncertain is True
        assert machine.measurement.end_uncertainty_bounds is not None
        # The bug: this must stay False. Liquid was never seen again after
        # the gap, so nothing here demonstrates resumption - it is solely
        # the ordinary adjacent-gap case end_uncertainty_bounds exists for.
        assert machine.measurement.end_gap_unresolved is False


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

    def test_a_mid_flow_gap_and_a_separate_adjacent_gap_both_get_a_reason(self):
        """Third Codex review round: compose, don't mask, distinct concerns.

        Two different gaps in the same run - one genuinely mid-flow (liquid
        resumes after it), one genuinely adjacent to the end (liquid never
        resumes) - are two independent reasons a viewer should see, not one
        hiding the other.
        """
        machine = FlowStateMachine(config(flow_start_persistence_s=0.2, flow_end_persistence_s=0.5))
        entries: list[tuple[ScoreSample, bool]] = []
        entries += [(liquid(), True)] * 30  # clean, trusted flow
        entries += [(empty(), False)] * 15  # mid-flow gap: liquid resumes after it
        entries += [(liquid(), True)] * 50
        entries += [(empty(), False)] * 15  # a second, separate gap...
        entries += [(empty(), True)] * 20  # ...adjacent to the end: liquid never resumes

        feed_trust(machine, entries)
        measurement = machine.measurement
        assert measurement.end_gap_unresolved is True
        assert measurement.end_uncertainty_bounds is not None

        _, reasons = score_confidence(measurement, config())
        joined = " ".join(reasons).lower()
        assert "seen again afterward" in joined  # the mid-flow reason
        assert "could have occurred anywhere in that span" in joined  # the adjacent reason

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
