"""Variable-frame-rate detection and rejection.

A VFR recording declares a plausible *average* frame rate, so it passes every
other check and is then timed as if it were constant - the error accumulating
through the file. This version refuses such recordings instead of reporting
confidently wrong timestamps.

The decision function is pure, so the patterns below are scripted timestamp
sequences rather than video files. That is deliberate and it is also a
limitation: no genuine VFR container is exercised anywhere in this suite,
because OpenCV's writer emits constant-rate files only and the ffmpeg build
available here is stripped of both `lavfi` and the `concat` demuxer. See the
PR discussion; validating against a real VFR file remains outstanding.
"""

from __future__ import annotations

import pytest

from app.errors import VariableFrameRateError
from app.video import metadata
from app.video.metadata import is_variable_frame_rate, probe_video


def constant(interval_ms: float, count: int) -> list[float]:
    return [interval_ms] * count


class TestIsVariableFrameRate:
    def test_a_clean_constant_rate_is_accepted(self):
        assert not is_variable_frame_rate(constant(40.0, 60))  # 25 FPS

    def test_ntsc_millisecond_quantisation_is_accepted(self):
        """29.97 FPS lands on 33/34 ms alternately once quantised to whole ms."""
        intervals = [33.0 if index % 2 else 34.0 for index in range(60)]
        assert not is_variable_frame_rate(intervals)

    def test_an_occasional_dropped_frame_is_accepted(self):
        """One doubled interval in a long run is a dropped frame, not VFR."""
        intervals = constant(40.0, 59)
        intervals[30] = 80.0
        assert not is_variable_frame_rate(intervals)

    def test_a_genuine_rate_change_is_detected(self):
        """30 FPS for a while, then 5 FPS: index/fps would drift by seconds."""
        intervals = constant(33.3, 30) + constant(200.0, 30)
        assert is_variable_frame_rate(intervals)

    def test_sparse_irregularity_is_detected(self):
        """A screen recording that stalls whenever the picture is static."""
        intervals = constant(33.0, 50) + constant(200.0, 10)
        assert is_variable_frame_rate(intervals)

    def test_too_few_intervals_is_not_an_accusation(self):
        """A very short clip must not be rejected for lack of evidence."""
        assert not is_variable_frame_rate([33.0, 200.0, 33.0])

    def test_missing_timestamps_are_not_treated_as_variable(self):
        """Some backends report 0 for every frame; that is missing data."""
        assert not is_variable_frame_rate(constant(0.0, 60))

    def test_no_intervals_at_all(self):
        assert not is_variable_frame_rate([])


class TestMeasureFrameIntervals:
    def test_a_constant_rate_file_yields_even_intervals(self, motion_video):
        import cv2

        capture = cv2.VideoCapture(str(motion_video.path))
        try:
            intervals = metadata.measure_frame_intervals(capture, limit=40)
        finally:
            capture.release()

        assert len(intervals) >= 20
        expected_ms = 1000.0 / motion_video.fps
        assert max(intervals) == pytest.approx(expected_ms, abs=1.0)
        assert not is_variable_frame_rate(intervals)


class TestProbeRejectsVariableFrameRate:
    def test_probe_raises_for_a_variable_rate_recording(self, motion_video, monkeypatch):
        """The whole probe must refuse the file, not merely warn about it."""
        monkeypatch.setattr(
            metadata,
            "measure_frame_intervals",
            lambda capture, limit=metadata.VFR_SAMPLE_FRAMES: (
                constant(33.3, 30) + constant(200.0, 30)
            ),
        )
        with pytest.raises(VariableFrameRateError) as excinfo:
            probe_video(motion_video.path)

        message = excinfo.value.user_message
        assert "variable frame rate" in message
        assert "constant frame rate" in message  # tells the user what to do
        assert "intervals sampled" in (excinfo.value.detail or "")  # evidence for the log

    def test_a_constant_rate_recording_is_still_accepted(self, motion_video):
        """The guard must not reject the ordinary case."""
        info = probe_video(motion_video.path)
        assert info.fps == pytest.approx(motion_video.fps, abs=0.01)


class TestUploadRejectsVariableFrameRate:
    def test_upload_is_refused_and_leaves_no_file_behind(
        self, client, app, motion_video, monkeypatch
    ):
        import io

        monkeypatch.setattr(
            metadata,
            "measure_frame_intervals",
            lambda capture, limit=metadata.VFR_SAMPLE_FRAMES: (
                constant(33.3, 30) + constant(200.0, 30)
            ),
        )
        response = client.post(
            "/api/videos",
            data={"file": (io.BytesIO(motion_video.path.read_bytes()), "phone_clip.mp4")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        assert "variable frame rate" in response.get_json()["error"]
        # A rejected upload must not be left on disk.
        assert list(app.extensions["app_config"].upload_dir.glob("*")) == []
