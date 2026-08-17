"""Frame-timing verification: multi-window sampling, and rejection.

Every timing in this application comes from a single frame rate. A recording
whose cadence varies would be reported with confidently wrong timestamps, so
such files are refused rather than measured.

The evidence must describe the *whole* recording. An earlier version sampled
only the opening 120 frames, which let a file that starts steady and changes
later pass and then be mis-timed everywhere — the original defect surviving the
fix. Windows are now spread across the duration and compared both internally
and against each other.

Spreading the windows only helps if the seeks actually happened, so each seek
must be positively verified before its samples count. A backend that ignores
seeks, or that cannot report where it landed, would otherwise resample the
opening frames and manufacture the very steadiness the check exists to
disprove.

Shape rules are not enough on their own. The windows cover roughly a third of a
long recording, so an irregular burst falling *between* two of them leaves every
window internally perfect and every median identical — no threshold catches
that. The last rule therefore measures the harm rather than its shape: at each
window start the real presentation timestamp is compared against `index / fps`,
which is the only timing this application reports, and the *spread* of those
offsets is charged against a time budget. A constant offset shifts every
timestamp equally and cancels out of every duration, so it is not drift.

Genuine variable-rate containers now live in `tests/fixtures/` (see the README
there). They matter: every earlier round was validated against scripted
timestamps alone, and three successive versions of the detector shipped with the
same defect intact. `FakeCapture` is kept for the backend misbehaviour a real
file cannot express — ignored seeks, unreportable positions, failing `set()`.
"""

from __future__ import annotations

import io
from dataclasses import replace
from pathlib import Path

import cv2
import pytest

from app.config import AppConfig
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


def evidence(
    windows: list[list[float]],
    frame_count: int | None = 100_000,
    offsets_ms: list[float] | None = None,
):
    return FrameTimingEvidence(
        windows=windows,
        frame_count=frame_count,
        windows_requested=len(windows) or 1,
        offsets_ms=offsets_ms or [],
    )


FIXTURES = Path(__file__).parent / "fixtures"


class FakeCapture:
    """A capture whose presentation timestamps follow a script.

    Only the three calls the sampler makes are implemented: seek by frame
    index, read the current position, and grab the next frame.

    The three flags model the ways a real backend can fail to seek, each of
    which must stop the resulting samples from being trusted:

    ``honour_seeks``
        ``False`` means the seek is silently ignored and decoding continues
        from wherever it already was.
    ``report_position``
        ``False`` means ``CAP_PROP_POS_FRAMES`` comes back as ``NaN`` - the
        backend cannot say where it is, so nothing about the seek is provable.
    ``seek_succeeds``
        ``False`` means ``set()`` itself reports failure.
    """

    def __init__(
        self,
        stamps_ms: list[float],
        *,
        honour_seeks: bool = True,
        report_position: bool = True,
        seek_succeeds: bool = True,
    ):
        self.stamps = stamps_ms
        self.honour_seeks = honour_seeks
        self.report_position = report_position
        self.seek_succeeds = seek_succeeds
        self.position = 0

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES and self.honour_seeks:
            self.position = int(value)
        return self.seek_succeeds

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_POS_FRAMES:
            return float(self.position) if self.report_position else float("nan")
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


class TestTimingDriftRule:
    """The rule that measures reported-timestamp error rather than its shape."""

    STEADY = [constant(33.3, 39)] * 5

    def test_drift_beyond_the_budget_is_refused(self):
        """Windows all perfectly steady, yet the timeline has slipped."""
        verdict = assess_frame_timing(
            evidence(self.STEADY, offsets_ms=[0.0, 475.0, -60.0, -30.0, -30.0])
        )
        assert not verdict.reliable
        assert verdict.reason == "timing_drift"
        assert "535 ms" in verdict.detail
        assert "50 ms budget" in verdict.detail

    def test_drift_inside_the_budget_is_accepted(self):
        """Bounded accuracy, not constant-rate purity: 30 ms is under budget."""
        verdict = assess_frame_timing(
            evidence(self.STEADY, offsets_ms=[0.0, 0.2, 0.4, -29.4, -29.2])
        )
        assert verdict.reliable
        assert verdict.reason == "constant"
        assert "timing drift within 29.8 ms" in verdict.detail

    def test_a_constant_non_zero_offset_is_not_drift(self):
        """A recording that merely starts late must not be refused.

        A constant offset shifts every reported timestamp by the same amount,
        so it cancels out of every duration the application measures.
        """
        verdict = assess_frame_timing(evidence(self.STEADY, offsets_ms=[5000.0] * 5))
        assert verdict.reliable
        assert verdict.reason == "constant"

    def test_the_budget_is_configurable(self):
        offsets = [0.0, 120.0]
        assert not assess_frame_timing(
            evidence(self.STEADY, offsets_ms=offsets), max_timing_error_s=0.05
        ).reliable
        assert assess_frame_timing(
            evidence(self.STEADY, offsets_ms=offsets), max_timing_error_s=0.20
        ).reliable

    def test_a_single_offset_cannot_establish_drift(self):
        """One sample has nothing to be spread against; do not invent a verdict."""
        verdict = assess_frame_timing(evidence(self.STEADY, offsets_ms=[900.0]))
        assert verdict.reliable
        assert "timing drift" not in verdict.detail

    def test_the_shape_rules_still_fire_first(self):
        """Drift is the last rule, not a replacement for the others."""
        windows = [constant(33.3, 39)] * 3 + [constant(200.0, 39)] * 2
        verdict = assess_frame_timing(evidence(windows, offsets_ms=[0.0, 0.0]))
        assert not verdict.reliable
        assert verdict.reason == "rate_changes_between_windows"


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

    def test_a_backend_that_cannot_report_its_position_is_refused(self):
        """NaN position: the seek is unprovable, so no window may be trusted.

        This is the false negative the previous guard had. It skipped a window
        only when the landed position was finite *and* far away, so a backend
        reporting ``NaN`` fell straight through and its samples were kept. With
        seeks also ignored, those samples are the opening frames over and over -
        five apparently steady windows from a recording that changes cadence at
        the halfway point, passed as constant and then mis-timed throughout.
        """
        stamps = stamps_from_intervals((33.3, 500), (200.0, 500))
        capture = FakeCapture(stamps, honour_seeks=False, report_position=False)
        result = collect_interval_windows(capture, frame_count=len(stamps))

        assert result.windows == []
        verdict = assess_frame_timing(result)
        assert not verdict.reliable
        assert verdict.reason in {"no_usable_timestamps", "unrepresentative_sample"}

    def test_position_is_unprovable_even_when_the_seek_was_honoured(self):
        """Correct seeks with unreadable positions are still not evidence.

        Conservative by design: the samples here happen to be genuine, but
        nothing observable distinguishes them from the case above, and guessing
        is what produces confidently wrong timestamps.
        """
        capture = FakeCapture(stamps_from_intervals((40.0, 999)), report_position=False)
        result = collect_interval_windows(capture, frame_count=1000)

        assert result.windows == []
        assert not assess_frame_timing(result).reliable

    def test_a_seek_the_backend_reports_as_failed_is_not_trusted(self):
        """``set()`` returning False is a refusal even if the position looks right."""
        stamps = stamps_from_intervals((33.3, 500), (200.0, 500))
        capture = FakeCapture(stamps, honour_seeks=False, seek_succeeds=False)
        result = collect_interval_windows(capture, frame_count=len(stamps))

        assert result.windows == []
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


class TestGenuineVariableRateContainers:
    """Real H.264/MP4 files with genuinely non-uniform presentation timestamps.

    These are the tests the earlier rounds lacked. Everything here goes through
    real OpenCV seeks and real ``CAP_PROP_POS_MSEC`` values; nothing is scripted.
    """

    def sample(self, name: str):
        path = FIXTURES / name
        capture = cv2.VideoCapture(str(path))
        try:
            fps = capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            result = collect_interval_windows(capture, frame_count=frame_count, fps=fps)
        finally:
            capture.release()
        return fps, frame_count, result

    def test_the_fixtures_really_do_carry_non_uniform_timestamps(self):
        """Guard the guard: if a fixture were re-encoded to CFR, say so here."""
        capture = cv2.VideoCapture(str(FIXTURES / "vfr_distributed.mp4"))
        stamps = []
        while capture.grab():
            stamps.append(capture.get(cv2.CAP_PROP_POS_MSEC))
        capture.release()

        intervals = [round(b - a, 3) for a, b in zip(stamps, stamps[1:], strict=False)]
        assert len(stamps) == 639
        assert intervals.count(31.667) == 604
        assert intervals.count(63.333) == 34

    def test_distributed_doubled_intervals_are_accepted_within_budget(self):
        """Bounded accuracy: real VFR, but under one frame period of error.

        Refusing this file would reject the ordinary case of a constant-rate
        export that dropped a few frames, so the policy accepts it and the
        drift it carries is measured rather than assumed.
        """
        fps, frame_count, result = self.sample("vfr_distributed.mp4")
        assert fps == pytest.approx(29.981, abs=0.01)
        assert len(result.offsets_ms) == metadata.VFR_WINDOW_COUNT

        spread = max(result.offsets_ms) - min(result.offsets_ms)
        assert spread == pytest.approx(29.8, abs=1.0)
        assert spread < metadata.VFR_MAX_TIMING_ERROR_S * 1000

        verdict = assess_frame_timing(result)
        assert verdict.reliable, verdict.detail
        assert probe_video(FIXTURES / "vfr_distributed.mp4").fps == pytest.approx(fps)

    def test_a_burst_between_the_windows_is_refused(self):
        """The case no threshold can catch, and the reason drift is measured.

        Every sampled window is internally perfect and every median identical,
        because the irregularity sits in frames 60-126 and the windows examine
        frames 0-39, 150-189, 300-339, 449-488 and 599-638.
        """
        _, _, result = self.sample("vfr_clustered_burst.mp4")

        for window in result.usable_windows:
            median = metadata._median(window)
            irregular = sum(1 for value in window if abs(value - median) > 0.5 * median)
            assert irregular == 0, "the burst is supposed to be invisible to the shape rules"

        verdict = assess_frame_timing(result)
        assert not verdict.reliable
        assert verdict.reason == "timing_drift"
        assert max(result.offsets_ms) - min(result.offsets_ms) > 500

    def test_the_burst_file_is_refused_by_probe_with_its_own_message(self):
        with pytest.raises(VariableFrameRateError) as excinfo:
            probe_video(FIXTURES / "vfr_clustered_burst.mp4")

        # Distinct from both "the rate varies" and "we could not check".
        assert "drifts too far" in excinfo.value.user_message
        assert "ffmpeg -vsync cfr" in excinfo.value.user_message
        assert "timing_drift" in (excinfo.value.detail or "")

    def test_probe_video_accepts_an_explicit_wider_budget(self):
        """The scalar override on probe_video(), not just the internal rule.

        A caller with looser accuracy needs (e.g. AppConfig configured for it)
        can widen the budget without editing app/video/metadata.py.
        """
        path = FIXTURES / "vfr_clustered_burst.mp4"
        with pytest.raises(VariableFrameRateError):
            probe_video(path)  # default 0.05 s budget still refuses it

        info = probe_video(path, max_timing_error_s=1.0)  # 1 s comfortably covers it
        assert info.frame_count == 639

    def test_probe_video_defaults_to_the_module_constant(self):
        """Direct callers with no config object see unchanged behaviour."""
        with pytest.raises(VariableFrameRateError):
            probe_video(FIXTURES / "vfr_clustered_burst.mp4")

    def test_a_recording_that_merely_starts_late_is_accepted(self):
        """A non-zero container start time is not drift."""
        _, _, result = self.sample("cfr_late_start.mp4")
        spread = max(result.offsets_ms) - min(result.offsets_ms)
        assert spread < metadata.VFR_MAX_TIMING_ERROR_S * 1000
        assert assess_frame_timing(result).reliable
        assert probe_video(FIXTURES / "cfr_late_start.mp4").frame_count == 639

    def test_the_generated_cfr_fixtures_show_no_drift_at_all(self, zahn_video, motion_video):
        """The separation is three orders of magnitude, not a delicate margin."""
        for video in (zahn_video, motion_video):
            capture = cv2.VideoCapture(str(video.path))
            try:
                result = collect_interval_windows(
                    capture,
                    frame_count=int(video.duration_s * video.fps),
                    fps=video.fps,
                )
            finally:
                capture.release()
            spread = max(result.offsets_ms) - min(result.offsets_ms)
            assert spread == pytest.approx(0.0, abs=0.5), f"{video.path.name}: {spread} ms"


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

    def test_a_genuine_drifting_container_is_refused_by_the_endpoint(self, client, app):
        """The whole path on a real file: nothing monkeypatched, nothing kept."""
        payload = (FIXTURES / "vfr_clustered_burst.mp4").read_bytes()
        response = client.post(
            "/api/videos",
            data={"file": (io.BytesIO(payload), "shift_export.mp4")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        message = response.get_json()["error"]
        assert "drifts too far" in message
        assert "ffmpeg -vsync cfr" in message  # actionable, not just a refusal
        assert "Traceback" not in message
        assert list(app.extensions["app_config"].upload_dir.glob("*")) == []

    def test_the_endpoint_honours_a_configured_wider_budget(self, tmp_path):
        """AppConfig.max_timing_error_s reaches the upload path, not just probe_video().

        The default 0.05 s budget refuses this file (asserted above); a server
        configured with a wider tolerance must accept the same bytes.
        """
        from app.web.routes import create_app

        config = replace(
            AppConfig(),
            data_dir=tmp_path / "data",
            log_level="WARNING",
            max_timing_error_s=1.0,
        )
        application = create_app(config)
        application.config.update(TESTING=True)
        try:
            response = application.test_client().post(
                "/api/videos",
                data={
                    "file": (
                        io.BytesIO((FIXTURES / "vfr_clustered_burst.mp4").read_bytes()),
                        "shift_export.mp4",
                    )
                },
                content_type="multipart/form-data",
            )
            assert response.status_code == 200
            assert response.get_json()["frame_count"] == 639
        finally:
            application.extensions["analysis_service"].shutdown()

    def test_a_genuine_container_within_budget_is_accepted_by_the_endpoint(self, client):
        payload = (FIXTURES / "vfr_distributed.mp4").read_bytes()
        response = client.post(
            "/api/videos",
            data={"file": (io.BytesIO(payload), "handheld_clip.mp4")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 200
        assert response.get_json()["frame_count"] == 639

    def test_upload_is_refused_when_seeks_cannot_be_verified(
        self, client, app, motion_video, monkeypatch
    ):
        """A backend reporting NaN positions must not get a measurement."""
        stamps = stamps_from_intervals((33.3, 500), (200.0, 500))
        monkeypatch.setattr(
            metadata,
            "collect_interval_windows",
            lambda capture, **kwargs: collect_interval_windows(
                FakeCapture(stamps, honour_seeks=False, report_position=False),
                frame_count=len(stamps),
            ),
        )
        response = client.post(
            "/api/videos",
            data={"file": (io.BytesIO(motion_video.path.read_bytes()), "phone_clip.mp4")},
            content_type="multipart/form-data",
        )

        assert response.status_code == 400
        assert "could not be verified" in response.get_json()["error"]
        assert list(app.extensions["app_config"].upload_dir.glob("*")) == []
