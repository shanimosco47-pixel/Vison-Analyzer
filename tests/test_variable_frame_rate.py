"""Frame-timing verification: multi-window sampling, and rejection.

Every timing in this application comes from a single frame rate. A recording
whose cadence varies would be reported with confidently wrong timestamps, so
such files are refused rather than measured.

The evidence must describe the *whole* recording. An earlier version sampled
only the opening 120 frames, which let a file that starts steady and changes
later pass and then be mis-timed everywhere — the original defect surviving the
fix. Windows are now spread across the duration and compared both internally
and against each other.

Where a real file cannot express the pattern under test, a `FakeCapture`
supplies scripted presentation timestamps to the *real* sampling code, so the
seeking and windowing logic is exercised rather than stubbed out. A genuine VFR
container is still never used: OpenCV's writer emits constant-rate files only,
and the ffmpeg build available here is stripped of both `lavfi` and the
`concat` demuxer. Validating against real VFR footage remains outstanding.
"""

from __future__ import annotations

import io

import cv2
import pytest

from app.errors import VariableFrameRateError
from app.video import metadata
from app.video.metadata import (
    FrameTimingEvidence,
    assess_frame_timing,
    collect_interval_windows,
    probe_video,
)


def constant(interval_ms: float, count: int) -> list[float]:
    return [interval_ms] * count


def evidence(windows: list[list[float]], frame_count: int | None = 100_000):
    return FrameTimingEvidence(
        windows=windows, frame_count=frame_count, windows_requested=len(windows) or 1
    )


class FakeCapture:
    """A capture whose presentation timestamps follow a script.

    Only the three calls the sampler makes are implemented: seek by frame
    index, read the current position, and grab the next frame.
    """

    def __init__(self, stamps_ms: list[float], *, honour_seeks: bool = True):
        self.stamps = stamps_ms
        self.honour_seeks = honour_seeks
        self.position = 0

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES and self.honour_seeks:
            self.position = int(value)
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position)
        if prop == cv2.CAP_PROP_POS_MSEC:
            index = min(max(self.position - 1, 0), len(self.stamps) - 1)
            return self.stamps[index]
        return 0.0

    def grab(self) -> bool:
        if self.position >= len(self.stamps):
            return False
        self.position += 1
        return True


def stamps_from_intervals(*segments: tuple[float, int]) -> list[float]:
    """Build a timestamp track from (interval_ms, frame_count) segments."""
    stamps = [0.0]
    for interval, count in segments:
        for _ in range(count):
            stamps.append(stamps[-1] + interval)
    return stamps


class TestAssessFrameTiming:
    def test_a_clean_constant_rate_is_accepted(self):
        verdict = assess_frame_timing(evidence([constant(40.0, 39)] * 5))
        assert verdict.reliable
        assert verdict.reason == "constant"

    def test_ntsc_millisecond_quantisation_is_accepted(self):
        """29.97 FPS lands on 33/34 ms alternately once quantised to whole ms."""
        window = [33.0 if index % 2 else 34.0 for index in range(39)]
        assert assess_frame_timing(evidence([window] * 5)).reliable

    def test_an_occasional_dropped_frame_is_accepted(self):
        window = constant(40.0, 39)
        window[20] = 80.0  # one doubled interval
        assert assess_frame_timing(evidence([window] * 5)).reliable

    def test_an_irregular_window_is_refused(self):
        windows = [constant(33.3, 39)] * 4 + [constant(33.3, 20) + constant(200.0, 19)]
        verdict = assess_frame_timing(evidence(windows))
        assert not verdict.reliable
        assert verdict.reason == "irregular_within_window"

    def test_a_later_window_at_a_different_steady_rate_is_refused(self):
        """Each window is internally perfect, but the rate changes part-way."""
        windows = [constant(33.3, 39)] * 3 + [constant(200.0, 39)] * 2
        verdict = assess_frame_timing(evidence(windows))
        assert not verdict.reliable
        assert verdict.reason == "rate_changes_between_windows"

    def test_a_modest_rate_change_is_still_refused(self):
        """30 -> 25 FPS is only a 20% drift, but it still corrupts timings."""
        windows = [constant(33.3, 39)] * 3 + [constant(40.0, 39)] * 2
        assert not assess_frame_timing(evidence(windows)).reliable

    def test_missing_timestamps_are_refused_not_assumed_constant(self):
        """Unverifiable is a refusal: an unchecked assumption is the defect."""
        verdict = assess_frame_timing(evidence([]))
        assert not verdict.reliable
        assert verdict.reason == "no_usable_timestamps"

    def test_a_single_opening_window_on_a_long_file_is_refused(self):
        """The exact blind spot of the first fix: opening-only evidence."""
        verdict = assess_frame_timing(evidence([constant(40.0, 39)], frame_count=100_000))
        assert not verdict.reliable
        assert verdict.reason == "unrepresentative_sample"

    def test_a_single_window_covering_a_short_file_is_enough(self):
        verdict = assess_frame_timing(evidence([constant(40.0, 39)], frame_count=60))
        assert verdict.reliable

    def test_a_very_short_recording_is_accepted_without_evidence(self):
        """Drift cannot accumulate in under a second, so do not refuse it."""
        verdict = assess_frame_timing(evidence([], frame_count=15))
        assert verdict.reliable
        assert verdict.reason == "too_short_to_matter"


class TestCollectIntervalWindows:
    def test_windows_are_spread_across_the_recording(self):
        capture = FakeCapture(stamps_from_intervals((40.0, 999)))
        result = collect_interval_windows(capture, frame_count=1000)
        assert result.windows_requested == metadata.VFR_WINDOW_COUNT
        assert len(result.usable_windows) == metadata.VFR_WINDOW_COUNT

    def test_a_late_rate_change_is_caught_end_to_end(self):
        """Constant opening, variable later — the reviewer's scenario.

        The opening 120 frames are perfectly steady, so the previous
        opening-only sampler saw nothing wrong.
        """
        stamps = stamps_from_intervals((33.3, 500), (200.0, 500))
        capture = FakeCapture(stamps)
        result = collect_interval_windows(capture, frame_count=len(stamps))
        verdict = assess_frame_timing(result)
        assert not verdict.reliable
        assert verdict.reason in {"rate_changes_between_windows", "irregular_within_window"}

        # And the opening alone really does look fine, which is why the old
        # sampler passed this file.
        opening = FakeCapture(stamps)
        opening_only = collect_interval_windows(
            opening, frame_count=len(stamps), window_count=1, window_frames=120
        )
        assert assess_frame_timing(evidence(opening_only.windows, frame_count=200)).reliable

    def test_a_backend_that_ignores_seeks_yields_no_trustworthy_windows(self):
        """Unhonoured seeks would resample the opening and fake steadiness."""
        stamps = stamps_from_intervals((33.3, 500), (200.0, 500))
        capture = FakeCapture(stamps, honour_seeks=False)
        result = collect_interval_windows(capture, frame_count=len(stamps))
        # Only the window that legitimately starts at 0 is kept.
        assert len(result.usable_windows) <= 1
        assert not assess_frame_timing(result).reliable

    def test_a_real_constant_rate_file_is_sampled_and_accepted(self, motion_video):
        capture = cv2.VideoCapture(str(motion_video.path))
        try:
            result = collect_interval_windows(
                capture, frame_count=int(motion_video.duration_s * motion_video.fps)
            )
        finally:
            capture.release()

        assert len(result.usable_windows) >= 2
        verdict = assess_frame_timing(result)
        assert verdict.reliable, verdict.detail
        expected_ms = 1000.0 / motion_video.fps
        assert max(max(window) for window in result.usable_windows) == pytest.approx(
            expected_ms, abs=1.0
        )


class TestProbeRejectsUnreliableTiming:
    def test_probe_raises_for_a_late_rate_change(self, motion_video, monkeypatch):
        stamps = stamps_from_intervals((33.3, 500), (200.0, 500))
        monkeypatch.setattr(
            metadata,
            "collect_interval_windows",
            lambda capture, **kwargs: collect_interval_windows(
                FakeCapture(stamps), frame_count=len(stamps)
            ),
        )
        with pytest.raises(VariableFrameRateError) as excinfo:
            probe_video(motion_video.path)

        assert "variable frame rate" in excinfo.value.user_message
        assert "constant frame rate" in excinfo.value.user_message
        # Either rule may fire first: the window straddling the transition is
        # internally irregular, and the windows after it sit at a different
        # median. Both are correct refusals of the same file.
        assert any(
            reason in (excinfo.value.detail or "")
            for reason in ("rate_changes_between_windows", "irregular_within_window")
        )

    def test_probe_raises_a_distinct_message_when_timing_cannot_be_checked(
        self, motion_video, monkeypatch
    ):
        monkeypatch.setattr(
            metadata,
            "collect_interval_windows",
            lambda capture, **kwargs: FrameTimingEvidence(
                windows=[], frame_count=100_000, windows_requested=5
            ),
        )
        with pytest.raises(VariableFrameRateError) as excinfo:
            probe_video(motion_video.path)

        assert "could not be verified" in excinfo.value.user_message
        assert "no_usable_timestamps" in (excinfo.value.detail or "")

    def test_a_constant_rate_recording_is_still_accepted(self, motion_video):
        info = probe_video(motion_video.path)
        assert info.fps == pytest.approx(motion_video.fps, abs=0.01)

    def test_every_generated_fixture_still_probes(self, zahn_video, quiet_video):
        """The guard must not reject the ordinary constant-rate cases."""
        for video in (zahn_video, quiet_video):
            assert probe_video(video.path).fps == pytest.approx(video.fps, abs=0.01)


class TestUploadRejectsUnreliableTiming:
    def test_upload_is_refused_and_leaves_no_file_behind(
        self, client, app, motion_video, monkeypatch
    ):
        stamps = stamps_from_intervals((33.3, 500), (200.0, 500))
        monkeypatch.setattr(
            metadata,
            "collect_interval_windows",
            lambda capture, **kwargs: collect_interval_windows(
                FakeCapture(stamps), frame_count=len(stamps)
            ),
        )
        response = client.post(
            "/api/videos",
            data={"file": (io.BytesIO(motion_video.path.read_bytes()), "phone_clip.mp4")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        assert "variable frame rate" in response.get_json()["error"]
        assert list(app.extensions["app_config"].upload_dir.glob("*")) == []
