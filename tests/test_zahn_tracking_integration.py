"""Outlet tracking against generated video: the regression and acceptance tests.

These are the Stage 1 integration tests over real (synthetic) video decoding,
complementing the pure-numpy unit tests in ``test_outlet_tracker.py`` and
``test_zahn_state_machine.py``:

* a regression test that reproduces the original white-on-white, hand-held
  "efflux several seconds too long" bug on the fixed-ROI path and confirms
  tracking (the default) measures it within the synthetic tolerance;
* the true-break-during-a-tracking-gap case, confirming the endpoint comes
  back unconfirmed with explicit bounds rather than falsely precise;
* a runtime/memory sanity check, since tracking changes what gets decoded
  each frame.

See ``diagnostics/stage0/STAGE0_REPORT.md`` for the real-footage version of
the same reproduction and the ablation that isolated the cause.
"""

from __future__ import annotations

import resource
import time

from app.analysis.zahn_detector import ZahnCupDetector
from app.video.metadata import probe_video
from app.video.reader import VideoReader

# Stage 0's synthetic acceptance criterion (README "Acceptance criteria"):
# endpoint error within +/-0.5s on synthetic tests with known ground truth.
SYNTHETIC_TOLERANCE_S = 0.5


def _run(video_path, outlet_xy, **params):
    info = probe_video(video_path)
    detector = ZahnCupDetector(
        info, {"outlet": {"x": round(outlet_xy[0]), "y": round(outlet_xy[1])}, **params}
    )
    with VideoReader(video_path, info) as reader:
        return detector.run(reader)


class TestRegressionOriginalBug:
    """The bug Stage 0 diagnosed and this stage fixes."""

    def test_fixed_roi_reproduces_the_original_long_timing_bug(self, handheld_zahn_video):
        """With tracking off, the same failure Stage 0 found in the field."""
        clip = handheld_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, zahn_track_outlet=False)
        summary = result.summary
        assert summary["efflux_seconds"] is not None
        error = summary["efflux_seconds"] - clip.efflux_s
        # The whole point of the regression: it must be measurably, not
        # marginally, too long - matching Stage 0's real-footage finding.
        assert error > SYNTHETIC_TOLERANCE_S
        assert summary["frames_untracked"] == 0  # the fixed path never even notices

    def test_tracking_measures_the_same_clip_within_tolerance(self, handheld_zahn_video):
        """The fix: tracking on (the default) recovers the true timing."""
        clip = handheld_zahn_video
        result = _run(clip.path, clip.outlet_at_reference)  # zahn_track_outlet defaults True
        summary = result.summary
        assert summary["efflux_seconds"] is not None
        error = abs(summary["efflux_seconds"] - clip.efflux_s)
        assert error <= SYNTHETIC_TOLERANCE_S
        assert summary["track_outlet"] is True

    def test_tracking_start_time_matches_ground_truth(self, handheld_zahn_video):
        clip = handheld_zahn_video
        result = _run(clip.path, clip.outlet_at_reference)
        assert abs(result.summary["flow_start_s"] - clip.flow_start_s) <= SYNTHETIC_TOLERANCE_S


class TestConfidenceReflectsMotion:
    def test_hand_held_confidence_is_not_higher_than_a_steady_run(
        self, zahn_video, handheld_zahn_video
    ):
        """Confidence must be visibly lower/flagged for heavy motion, never higher."""
        steady = _run(zahn_video.path, (zahn_video.truth["outlet_x"], zahn_video.truth["outlet_y"]))
        moving = _run(handheld_zahn_video.path, handheld_zahn_video.outlet_at_reference)
        assert moving.summary["confidence"] <= steady.summary["confidence"]


class TestBreakDuringATrackingGap:
    """The case the endpoint-uncertainty machinery exists for."""

    def test_endpoint_is_unconfirmed_with_bounds_not_falsely_precise(self, handheld_gap_zahn_video):
        clip = handheld_gap_zahn_video
        result = _run(clip.path, clip.outlet_at_reference)
        summary = result.summary

        assert summary["end_confirmed"] is False
        assert summary["end_uncertain"] is True
        assert summary["end_uncertainty_bounds"] is not None
        lo, hi = summary["end_uncertainty_bounds"]
        # The bound must actually bracket where the true break could be -
        # not just exist as a number. The true break and the true end both
        # fell inside the occlusion window.
        assert lo <= clip.stream_break_s <= hi
        assert lo <= clip.flow_end_s <= hi
        assert summary["status"] == "review"
        assert summary["confidence"] <= 0.5

    def test_fixed_roi_cannot_see_through_the_same_gap_either(self, handheld_gap_zahn_video):
        """Sanity check: the gap is real footage, not a tracking-only artefact."""
        clip = handheld_gap_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, zahn_track_outlet=False)
        # A fixed ROI has no notion of "untracked" at all - it has nothing to
        # honestly flag the gap with. This is exactly the risk the report's
        # "falsely precise" language describes.
        assert result.summary["frames_untracked"] == 0


class TestPerformance:
    """Streaming and memory-bounded: tracking must not change that."""

    def test_runtime_and_memory_are_bounded(self, handheld_zahn_video):
        clip = handheld_zahn_video
        before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        started = time.perf_counter()
        result = _run(clip.path, clip.outlet_at_reference)
        elapsed = time.perf_counter() - started
        after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        assert elapsed < clip.duration_s  # comfortably faster than real time
        # Peak RSS growth for one short clip's worth of small crops must stay
        # in the tens of MB, not scale with the video - one frame at a time.
        assert (after_kb - before_kb) < 200_000  # KB on Linux
        assert result.summary["frames_analysed"] > 0
