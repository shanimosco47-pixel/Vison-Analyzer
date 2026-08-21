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
from pathlib import Path

import cv2
import numpy as np
import pytest

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


@pytest.fixture(scope="session")
def blank_video_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A perfectly flat clip: nothing above any outlet point has texture."""
    path = tmp_path_factory.mktemp("videos") / "blank.mp4"
    width, height, fps, duration = 200, 200, 20.0, 3.0
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - depends on the OpenCV build
        pytest.skip("This OpenCV build cannot write MP4 files")
    frame = np.full((height, width, 3), 200, dtype=np.uint8)
    for _ in range(int(duration * fps)):
        writer.write(frame)
    writer.release()
    return path


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

        # No falsely precise duration: the authoritative number is
        # suppressed, and a bound is given in its place instead.
        assert summary["efflux_seconds"] is None
        assert summary["efflux_seconds_bounds"] is not None
        bounds_lo, bounds_hi = summary["efflux_seconds_bounds"]
        assert bounds_lo <= clip.efflux_s <= bounds_hi

    def test_fixed_roi_cannot_see_through_the_same_gap_either(self, handheld_gap_zahn_video):
        """Sanity check: the gap is real footage, not a tracking-only artefact."""
        clip = handheld_gap_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, zahn_track_outlet=False)
        # A fixed ROI has no notion of "untracked" at all - it has nothing to
        # honestly flag the gap with. This is exactly the risk the report's
        # "falsely precise" language describes.
        assert result.summary["frames_untracked"] == 0


class TestBreakDuringATrackingGapAtStart:
    """Symmetric case: the true *start*, not the end, falls inside a gap.

    A late-but-precise start is exactly as dishonest as a falsely late end -
    the reported duration comes out shorter than true, with full confidence.
    """

    def test_start_is_unconfirmed_with_bounds_not_falsely_precise(
        self, handheld_gap_at_start_zahn_video
    ):
        clip = handheld_gap_at_start_zahn_video
        result = _run(clip.path, clip.outlet_at_reference)
        summary = result.summary

        assert summary["start_uncertain"] is True
        assert summary["start_uncertainty_bounds"] is not None
        lo, hi = summary["start_uncertainty_bounds"]
        assert lo <= clip.flow_start_s <= hi

        # No falsely precise duration: the authoritative number is
        # suppressed, and a bound is given in its place, bracketing the true
        # efflux time.
        assert summary["efflux_seconds"] is None
        assert summary["efflux_seconds_bounds"] is not None
        bounds_lo, bounds_hi = summary["efflux_seconds_bounds"]
        assert bounds_lo <= clip.efflux_s <= bounds_hi

        assert summary["status"] == "review"
        assert summary["confidence"] <= 0.5

    def test_the_event_is_preserved_for_review_without_a_precise_duration(
        self, handheld_gap_at_start_zahn_video
    ):
        """Codex review: preserve a candidate, but never present it as precise."""
        clip = handheld_gap_at_start_zahn_video
        result = _run(clip.path, clip.outlet_at_reference)
        assert len(result.events) == 1
        event = result.events[0]
        assert event.status.value == "review"
        assert event.details["efflux_seconds"] is None
        assert event.details["efflux_seconds_bounds"] is not None
        assert event.details["start_uncertain"] is True


class TestTrackingInitFailure:
    """Init failure must fail loudly, never silently fall back while tracking is on.

    zahn_track_outlet is the *only* sanctioned rollback to the fixed-ROI
    path. If tracking is requested and cannot even start, guessing on stale,
    fixed geometry and reporting it confirmed would be exactly the kind of
    invented precision the rest of this stage exists to refuse.
    """

    def test_no_texture_above_outlet_fails_loudly_when_tracking_is_on(self, blank_video_path):
        result = _run(blank_video_path, (100, 100))  # zahn_track_outlet defaults True
        summary = result.summary
        assert summary["status"] == "failed"
        assert summary["efflux_seconds"] is None
        assert summary["flow_start_s"] is None
        assert summary["frames_analysed"] == 0
        assert result.events == []
        assert result.warnings
        assert "tracking" in result.warnings[0].lower()

    def test_the_same_clip_with_tracking_explicitly_off_does_not_fail(self, blank_video_path):
        """The only sanctioned rollback: explicit zahn_track_outlet=False."""
        result = _run(blank_video_path, (100, 100), zahn_track_outlet=False)
        # A blank clip still has no liquid to find, so this is not a
        # "confirmed" measurement either - the point is that it goes through
        # the ordinary fixed-ROI failure path (no stream found) rather than
        # being blocked before a single frame is even decoded.
        assert result.summary["track_outlet"] is False
        assert result.summary["frames_analysed"] > 0


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


class TestTrackingFramesLog:
    """Codex review: bounded/streamed per-frame evidence, not just transitions.

    State-transition JSON alone does not show the tracked outlet position,
    search window, scoring ROI and state for every analysed frame. This is
    the file that does - written one JSON line at a time as the run
    progresses, so inspecting it never requires loading the whole video (or
    a per-frame list) into memory.
    """

    def test_a_frame_evidence_file_is_streamed_when_a_diagnostics_dir_is_given(
        self, handheld_zahn_video, tmp_path
    ):
        import json

        clip = handheld_zahn_video
        diagnostics_dir = tmp_path / "diag"
        result = _run(clip.path, clip.outlet_at_reference, diagnostics_dir=str(diagnostics_dir))

        log_path = result.diagnostics["tracking_frames_log"]
        assert log_path is not None
        path = Path(log_path)
        assert path.parent == diagnostics_dir
        assert path.exists()

        lines = path.read_text().splitlines()
        assert len(lines) == result.summary["frames_analysed"]

        records = [json.loads(line) for line in lines]
        first = records[0]
        assert set(first) == {
            "timestamp_s",
            "state",
            "trusted",
            "reacquired",
            "search_roi",
            "roi",
            "guard_roi",
        }
        assert first["state"] in {"tracked", "predicted", "lost"}
        assert isinstance(first["trusted"], bool)
        assert set(first["search_roi"]) == {"x", "y", "width", "height"}
        # Timestamps are monotonic - one line per frame, in decode order.
        timestamps = [record["timestamp_s"] for record in records]
        assert timestamps == sorted(timestamps)

    def test_no_file_is_written_without_a_diagnostics_dir(self, handheld_zahn_video):
        """The default: no diagnostics_dir param means nothing hits disk."""
        clip = handheld_zahn_video
        result = _run(clip.path, clip.outlet_at_reference)
        assert result.diagnostics["tracking_frames_log"] is None
