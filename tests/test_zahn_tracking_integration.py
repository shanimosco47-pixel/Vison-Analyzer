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

import time
from pathlib import Path

try:
    import resource
except ImportError:  # pragma: no cover - resource is POSIX-only (no Windows build)
    resource = None  # type: ignore[assignment]

import cv2
import numpy as np
import pytest

from app.analysis.zahn_detector import MAX_TRACKING_RECENTRES, ZahnCupDetector
from app.errors import InvalidROIError
from app.services.event_log import build_event_log, summarise
from app.video.metadata import probe_video
from app.video.reader import VideoReader

from ._synthetic_handheld import MARGIN, HandheldClip, _camera_offset, _hand_offset
from .conftest import PORTRAIT_TIMELINE

# Stage 0's synthetic acceptance criterion (README "Acceptance criteria"):
# endpoint error within +/-0.5s on synthetic tests with known ground truth.
SYNTHETIC_TOLERANCE_S = 0.5


def _run(
    video_path,
    outlet_xy,
    *,
    clip=None,
    outlet_end_xy=None,
    outlet_end_reference_s=None,
    **params,
):
    """Run detection, satisfying the two-anchor contract by default.

    Every outlet-click configuration now requires both an early and a late
    anchor (Codex review, fourth round). Pass ``clip`` - a ``HandheldClip``
    - to derive the late anchor automatically from its own
    ``outlet_at_late_reference``/``late_reference_s``: the common case,
    since most tests only care that a *valid* late anchor exists, not its
    exact position. ``outlet_end_xy``/``outlet_end_reference_s`` override
    that explicitly, for tests that need to control the late anchor
    themselves (a false correlation between anchors, a duplicate/reversed
    timestamp, a missing anchor, ...) - pass ``outlet_end_xy=None`` with no
    ``clip`` either to omit ``outlet_end`` entirely and test the
    missing-anchor validation path.
    """
    info = probe_video(video_path)
    outlet_params: dict = {"outlet": {"x": round(outlet_xy[0]), "y": round(outlet_xy[1])}}
    if outlet_end_xy is not None:
        outlet_params["outlet_end"] = {
            "x": round(outlet_end_xy[0]),
            "y": round(outlet_end_xy[1]),
        }
        if outlet_end_reference_s is not None:
            outlet_params["outlet_end_reference_s"] = outlet_end_reference_s
    elif clip is not None:
        outlet_params["outlet_end"] = {
            "x": round(clip.outlet_at_late_reference[0]),
            "y": round(clip.outlet_at_late_reference[1]),
        }
        outlet_params["outlet_end_reference_s"] = clip.late_reference_s
    if "outlet_end" in outlet_params:
        # outlet_at_reference is always captured at t=0 (see
        # build_handheld_clip) - the natural default early-anchor
        # timestamp, overridden by an explicit outlet_reference_s param.
        params.setdefault("outlet_reference_s", 0.0)
    detector = ZahnCupDetector(info, {**outlet_params, **params})
    with VideoReader(video_path, info) as reader:
        return detector.run(reader)


def _assert_no_precise_duration_leaks(result) -> None:
    """Codex review: Event.duration_s is always end_s - start_s, computed by
    the shared Event/EventLog contract regardless of what a detail dict
    says. An uncertain measurement must not produce an Event at all -
    checked here at every layer a UI, API or CSV export could read a
    duration from, not only the events list itself.
    """
    assert result.events == []
    assert result.to_dict()["events"] == []
    log = build_event_log(result.events, video=None)
    assert log.rows() == []
    csv_lines = [line for line in log.to_csv().splitlines() if line]
    assert len(csv_lines) == 1  # header only - no data row carrying a duration
    counts = summarise(result.events)
    assert counts["total"] == 0
    assert counts["total_active_seconds"] == 0


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
        result = _run(clip.path, clip.outlet_at_reference, clip=clip, zahn_track_outlet=False)
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
        result = _run(
            clip.path, clip.outlet_at_reference, clip=clip
        )  # zahn_track_outlet defaults True
        summary = result.summary
        assert summary["efflux_seconds"] is not None
        error = abs(summary["efflux_seconds"] - clip.efflux_s)
        assert error <= SYNTHETIC_TOLERANCE_S
        assert summary["track_outlet"] is True

    def test_tracking_start_time_matches_ground_truth(self, handheld_zahn_video):
        clip = handheld_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        assert abs(result.summary["flow_start_s"] - clip.flow_start_s) <= SYNTHETIC_TOLERANCE_S


class TestConfidenceReflectsMotion:
    def test_hand_held_confidence_is_not_higher_than_a_steady_run(
        self, zahn_video, handheld_zahn_video
    ):
        """Confidence must be visibly lower/flagged for heavy motion, never higher."""
        steady_outlet = (zahn_video.truth["outlet_x"], zahn_video.truth["outlet_y"])
        # zahn_video's cup never moves - a late anchor at the same position,
        # comfortably before the clip ends, is exactly correct here.
        steady = _run(
            zahn_video.path,
            steady_outlet,
            outlet_end_xy=steady_outlet,
            outlet_end_reference_s=zahn_video.duration_s - 1.0,
        )
        moving = _run(
            handheld_zahn_video.path,
            handheld_zahn_video.outlet_at_reference,
            clip=handheld_zahn_video,
        )
        assert moving.summary["confidence"] <= steady.summary["confidence"]


class TestOutletReferenceFrame:
    """Second Codex review round: a real clip's outlet is only markable
    after the camera has already panned well past the true start.

    ``portrait_reference_frame_zahn_video`` is resolution-scaled to the
    real clip's own geometry (1080x1920, large early reframing); its true
    flow start (3.9s) is well before the 4.5s frame these tests mark the
    outlet on. Wiring ``outlet_reference_s`` through correctly must recover
    the true start, not merely convert the failure into an honestly
    unmeasurable one - see diagnostics/stage1/STAGE1_REPORT.md §17.3.
    """

    def _outlet_at_reference(self, clip: HandheldClip, reference_s: float) -> tuple[float, float]:
        """Where the outlet truly is at ``reference_s`` - what a user would
        actually click there, computed the same way the fixture itself
        placed the cup (see build_handheld_clip)."""
        cx, cy = _camera_offset(
            reference_s, PORTRAIT_TIMELINE["camera_drift_px"], PORTRAIT_TIMELINE["tremor_px"]
        )
        hx, hy = _hand_offset(reference_s, PORTRAIT_TIMELINE["hand_drift_px"])
        base_x = MARGIN + clip.width / 2.0
        base_y = MARGIN + clip.height * 0.34
        return (base_x + hx) - (MARGIN + cx), (base_y + hy) - (MARGIN + cy)

    def test_marking_a_later_frame_without_its_timestamp_is_rejected(
        self, portrait_reference_frame_zahn_video
    ):
        """Fourth Codex review round: the two-anchor contract now refuses
        this configuration outright, rather than silently misdating the
        outlet mark to frame zero - a real improvement over the previous
        round's silent-wrong-answer failure mode this test used to pin
        down. A valid outlet_end is supplied (required to even reach this
        check) but its own required timestamp is not, matching the
        specific gap this test targets: a click made on a later frame,
        given no timestamp for it at all.
        """
        clip = portrait_reference_frame_zahn_video
        outlet_at_4_5 = self._outlet_at_reference(clip, 4.5)
        late_at_26 = self._outlet_at_reference(clip, 26.0)
        with pytest.raises(InvalidROIError):
            _run(
                clip.path,
                outlet_at_4_5,
                outlet_end_xy=late_at_26,
                analysis_start_s=4.5,  # the user's natural attempted workaround
            )

    def test_outlet_reference_s_recovers_the_true_start(self, portrait_reference_frame_zahn_video):
        """The fix: mark on a later frame, but tell the detector which one."""
        clip = portrait_reference_frame_zahn_video
        outlet_at_4_5 = self._outlet_at_reference(clip, 4.5)
        late_at_26 = self._outlet_at_reference(clip, 26.0)
        result = _run(
            clip.path,
            outlet_at_4_5,
            outlet_reference_s=4.5,  # analysis_start_s stays at its default, 0
            outlet_end_xy=late_at_26,
            outlet_end_reference_s=26.0,
        )
        summary = result.summary
        assert summary["status"] == "confirmed"
        assert abs(summary["flow_start_s"] - clip.flow_start_s) <= SYNTHETIC_TOLERANCE_S
        assert abs(summary["efflux_seconds"] - clip.efflux_s) <= SYNTHETIC_TOLERANCE_S

    def test_a_reference_frame_matching_analysis_start_s_is_unaffected(
        self, portrait_reference_frame_zahn_video
    ):
        """outlet_reference_s == analysis_start_s must behave exactly as before."""
        clip = portrait_reference_frame_zahn_video
        result = _run(
            clip.path,
            clip.outlet_at_reference,
            clip=clip,
            outlet_reference_s=0.0,
            analysis_start_s=0.0,
        )
        summary = result.summary
        assert summary["status"] == "confirmed"
        assert abs(summary["flow_start_s"] - clip.flow_start_s) <= SYNTHETIC_TOLERANCE_S


class TestTwoAnchorValidation:
    """Fourth Codex review round: the two-anchor contract's own input
    validation - missing, out-of-bounds, misordered or duplicate anchors -
    exercised directly, independent of any specific tracking scenario.
    ``zahn_video``'s static cup is enough here: every case below is
    rejected (or accepted) in ``configure()``, before a single frame of
    tracking runs.
    """

    def test_missing_late_anchor_is_rejected(self, zahn_video):
        outlet = (zahn_video.truth["outlet_x"], zahn_video.truth["outlet_y"])
        with pytest.raises(InvalidROIError):
            _run(zahn_video.path, outlet)  # no clip, no outlet_end_xy: omits outlet_end entirely

    def test_late_anchor_outside_the_frame_is_rejected(self, zahn_video):
        outlet = (zahn_video.truth["outlet_x"], zahn_video.truth["outlet_y"])
        with pytest.raises(InvalidROIError):
            _run(
                zahn_video.path,
                outlet,
                outlet_end_xy=(zahn_video.width + 50, outlet[1]),
                outlet_end_reference_s=10.0,
            )

    def test_a_reference_timestamp_outside_the_analysed_range_is_rejected(self, zahn_video):
        outlet = (zahn_video.truth["outlet_x"], zahn_video.truth["outlet_y"])
        with pytest.raises(InvalidROIError):
            _run(
                zahn_video.path,
                outlet,
                outlet_end_xy=outlet,
                outlet_end_reference_s=zahn_video.duration_s + 5.0,
            )

    def test_a_late_anchor_before_the_early_one_is_rejected(self, zahn_video):
        outlet = (zahn_video.truth["outlet_x"], zahn_video.truth["outlet_y"])
        with pytest.raises(InvalidROIError):
            _run(
                zahn_video.path,
                outlet,
                outlet_reference_s=10.0,
                outlet_end_xy=outlet,
                outlet_end_reference_s=5.0,
            )

    def test_duplicate_anchor_timestamps_are_rejected(self, zahn_video):
        """Sharing a timestamp is exactly as invalid as inverting one - two
        marks made "at the same instant" carry no independent evidence."""
        outlet = (zahn_video.truth["outlet_x"], zahn_video.truth["outlet_y"])
        with pytest.raises(InvalidROIError):
            _run(
                zahn_video.path,
                outlet,
                outlet_reference_s=5.0,
                outlet_end_xy=outlet,
                outlet_end_reference_s=5.0,
            )

    def test_anchors_at_the_same_position_with_distinct_timestamps_are_accepted(self, zahn_video):
        """Duplicate *coordinates* are not duplicate anchors: a cup that
        genuinely has not moved between two distinct, validly-ordered marks
        is a legitimate (if uninformative) two-anchor run, not an error."""
        outlet = (zahn_video.truth["outlet_x"], zahn_video.truth["outlet_y"])
        result = _run(
            zahn_video.path,
            outlet,
            outlet_end_xy=outlet,
            outlet_end_reference_s=zahn_video.duration_s - 1.0,
        )
        assert result.summary["status"] in ("confirmed", "review")


class TestPortraitGuardGeometry:
    """Second Codex review round: roi_height_fraction on a tall portrait
    frame can leave the guard/capture geometry no room to follow the
    tracked outlet drifting downward before it clips against the bottom of
    the frame - marking otherwise-correctly-tracked frames untrusted for a
    reason that has nothing to do with tracking quality.
    """

    def test_the_guard_region_does_not_clip_against_the_frame_edge(
        self, portrait_reference_frame_zahn_video
    ):
        clip = portrait_reference_frame_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        summary = result.summary
        # Not zero untracked - hand-held motion is still hand-held motion -
        # but nowhere near the near-total failure geometry clipping causes;
        # the overwhelming majority of frames must track cleanly.
        untracked_ratio = summary["frames_untracked"] / max(1, summary["frames_analysed"])
        assert untracked_ratio < 0.1
        assert summary["status"] == "confirmed"


class TestTrackingRefusesAStrongNearbyDistractor:
    """Second Codex review round: "the tracker is following the wrong
    structure through/around the translucent cup, not merely running out
    of search width."

    A genuinely-initialised cup (anchor feature is real cup texture, not a
    distractor) that later drifts near a strong, stationary patch is the
    case this stage's per-frame patch-correlation check (§17, "why
    transforms are accepted") can address: it must not let the fitted
    transform get pulled onto the distractor and reported as confidently
    tracked. This is a narrower, and Stage-1-solvable, claim than "the
    tracker always finds the cup regardless of what else is in frame" -
    see diagnostics/stage1/STAGE1_REPORT.md for why a distractor that
    already overlaps the cup at initialisation (the true translucent-
    material case) is not solvable at this layer, and what would be
    needed (Stage 3's deferred camera-motion compensation).
    """

    def test_the_result_is_not_confidently_wrong(self, translucent_cup_near_distractor_zahn_video):
        clip, _distractor_center = translucent_cup_near_distractor_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        summary = result.summary
        # The pre-fix failure mode: RANSAC/inlier/displacement checks alone
        # accept the distractor's own smooth, self-consistent motion,
        # reporting status=confirmed with high confidence on a wrong
        # number. The safety property this stage can deliver is that this
        # must not happen - either the run stays honestly uncertain
        # (review/failed, capped confidence), or a genuinely correct
        # measurement is reported; never a confident, wrong one.
        if summary["status"] == "confirmed":
            assert summary["efflux_seconds"] is not None
            assert abs(summary["efflux_seconds"] - clip.efflux_s) <= SYNTHETIC_TOLERANCE_S
        else:
            assert summary["confidence"] <= 0.5

    def test_a_false_correlation_between_anchors_is_never_marked_trusted(
        self, translucent_cup_near_distractor_zahn_video, tmp_path
    ):
        """Supervisor decision (two-anchor round): "disagreement/loss
        remains untrusted and must feed the existing uncertainty logic" -
        checked directly against per-frame evidence, not just the summary.

        The distractor sits close enough to the cup's drift path that a
        single direction's independent track could plausibly lock onto it
        and still clear the correlation/displacement checks on its own.
        The two-anchor contract's safeguard is that a frame is only ever
        marked ``trusted`` when forward and backward tracking - accumulated
        independently, from opposite ends - agree, so a false lock in one
        direction cannot promote to trusted just because it lies between
        the anchors. This asserts the stronger, frame-level claim: every
        ``trusted`` frame's tracked position is close to the true cup, not
        merely that the final summary happens not to be confidently wrong.
        """
        import json

        from ._synthetic_handheld import MARGIN, _camera_offset, outlet_position_at

        clip, distractor_center = translucent_cup_near_distractor_zahn_video
        diagnostics_dir = tmp_path / "diag"
        result = _run(
            clip.path, clip.outlet_at_reference, clip=clip, diagnostics_dir=str(diagnostics_dir)
        )
        log_path = Path(result.diagnostics["tracking_frames_log"])
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        trusted = [record for record in records if record["trusted"]]
        assert trusted, "expected at least some frames to reconcile and be trusted"

        # Same placement the fixture itself used (translucent_cup_near_
        # distractor_zahn_video, tests/conftest.py): base_x/base_y for the
        # cup's own motion, and the world-to-frame conversion (subtracting
        # MARGIN plus the current camera offset) that the distractor's
        # fixed *world* coordinates need before they are comparable to a
        # per-frame, source-frame tracked position.
        camera_drift_px, hand_drift_px, tremor_px = 30.0, 26.0, 2.0
        base_x = MARGIN + clip.width / 2.0
        base_y = MARGIN + clip.height * 0.34
        for record in trusted:
            timestamp_s = record["timestamp_s"]
            roi = record["roi"]
            assert roi is not None
            tracked_x = roi["x"] + roi["width"] / 2.0
            tracked_y = roi["y"] + roi["height"] / 2.0
            true_x, true_y = outlet_position_at(
                timestamp_s,
                base_x=base_x,
                base_y=base_y,
                camera_drift_px=camera_drift_px,
                hand_drift_px=hand_drift_px,
                tremor_px=tremor_px,
            )
            cx, cy = _camera_offset(timestamp_s, camera_drift_px, tremor_px)
            distractor_x = distractor_center[0] - MARGIN - cx
            distractor_y = distractor_center[1] - MARGIN - cy
            distance_to_truth = ((tracked_x - true_x) ** 2 + (tracked_y - true_y) ** 2) ** 0.5
            distance_to_distractor = (
                (tracked_x - distractor_x) ** 2 + (tracked_y - distractor_y) ** 2
            ) ** 0.5
            assert distance_to_truth < distance_to_distractor


class TestCameraMotionCompensation:
    """Stage 3: camera-motion compensation, authorised after the two-anchor
    design (Stage 1's own architecture) proved unable to bridge the real
    clip's translucent, low-texture interval - diagnostics/stage1/
    STAGE1_REPORT.md §27. ``translucent_cup_overlapping_distractor_zahn_
    video`` is the worst case named there and in §18.3/§24.2: a strongly-
    textured distractor already sitting under the cup at t=0, so the
    reference patch itself would be extracted from it - no patch-
    correlation check can distinguish that from the genuine cup, since the
    check is circular by construction. Camera-motion compensation is a
    different kind of evidence entirely - not "does this still look like
    the reference," but "does this move independently of the background" -
    which is exactly what this fixture needs to be handled safely.
    """

    def test_background_through_cup_is_never_confidently_wrong(
        self, translucent_cup_overlapping_distractor_zahn_video
    ):
        clip, _distractor_center = translucent_cup_overlapping_distractor_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        summary = result.summary
        # The same safety property TestTrackingRefusesAStrongNearbyDistractor
        # checks for the near-but-not-overlapping case: either the result is
        # genuinely correct, or the run stays honestly uncertain (capped
        # confidence) - never a confident, wrong number.
        if summary["status"] == "confirmed":
            assert summary["efflux_seconds"] is not None
            assert abs(summary["efflux_seconds"] - clip.efflux_s) <= SYNTHETIC_TOLERANCE_S
        else:
            assert summary["confidence"] <= 0.5

    def test_the_veto_actually_engages_and_bounds_untracked_frames(
        self, translucent_cup_overlapping_distractor_zahn_video, tmp_path
    ):
        """The mechanism claim, not just the summary outcome: on this
        worst-case fixture, ``OutletTracker`` must actually reject a
        meaningful share of frames as ``background_consistent`` - proof the
        veto engages, not merely that the final result happens not to be
        wrong - while ``MAX_TRACKING_RECENTRES`` and the existing
        persistence gates still bound the run (a pathological clip cannot
        recentre or reacquire without limit just because Stage 3 adds a new
        way to lose trust). A frame-by-frame "every rejection was
        genuinely near the distractor" claim is not checked here: a
        rejected frame's logged position is wherever the tracker was last
        frozen (``OutletTracker._declare_lost`` does not move
        ``self._outlet``), not the vetoed candidate itself, and on this
        particular fixture - built so the distractor coincides with the
        cup's own t=0 position - "trusted" frames themselves cluster
        exactly where truth and distractor are hard to tell apart, which
        is the expected shape of the known circular-verification
        limitation (§24.2), not something a per-frame position check can
        cleanly separate out.
        """
        import json

        clip, _distractor_center = translucent_cup_overlapping_distractor_zahn_video
        diagnostics_dir = tmp_path / "diag"
        result = _run(
            clip.path, clip.outlet_at_reference, clip=clip, diagnostics_dir=str(diagnostics_dir)
        )
        summary = result.summary
        log_path = Path(result.diagnostics["tracking_frames_log"])
        records = [json.loads(line) for line in log_path.read_text().splitlines()]
        rejected_count = sum(1 for r in records if r["rejection_reason"] == "background_consistent")
        assert rejected_count, "expected the veto to actually engage on this fixture"

        recentre_events = result.diagnostics["recentre_events"]
        assert len(recentre_events) <= MAX_TRACKING_RECENTRES
        # The tool's job on a case this hard is to say "unmeasurable," not
        # invent a number confidently - status alone does not prove that
        # (checked separately), but a run that rejects a real share of
        # frames and still reports every one of them (none silently
        # dropped) is at least behaving as designed.
        assert summary["frames_untracked"] > 0
        assert summary["frames_analysed"] == len(records)

    def test_compensation_recovers_trust_and_trackability_through_the_flow_window(
        self,
        translucent_cup_near_distractor_genuinely_translucent_zahn_video,
        monkeypatch,
        tmp_path,
    ):
        """The mechanism claim behind this round's actual-compensation work
        (STAGE1_REPORT.md): a veto alone cannot help when the base
        LK/RANSAC tracker never finds a plausible cup candidate to begin
        with (the real clip's own failure mode, diagnosed after the
        previous, veto-only round) - only compensated *feature selection*
        can. This checks that causal claim directly, running the same clip
        with ``_foreground_residual_mask`` disabled (falls back to the
        pre-compensation behaviour) and enabled, rather than trusting a
        single run's outcome:

        * far fewer frames go untracked overall;
        * within the clip's *true* flow window specifically - the interval
          that determines the measured efflux - the fraction of frames
          actually TRACKED (not bridged/lost) rises dramatically, since a
          window nobody could track through cannot possibly be timed
          correctly regardless of what the state machine does with it;
        * the positions reported for those window-tracked frames stay
          within a bound loose enough for the fixture's own drift/tremor
          but tight enough that locking onto the distractor (over 200px
          away) would fail it - "trusted" that is actually wrong is worse
          than untracked.

        This fixture's own liquid-visibility rendering is not calibrated
        to make the full summary (``flow_start_s``/``efflux_seconds``) land
        on the true value - that depends on a separate classification
        threshold this round does not touch - so those fields are
        deliberately not asserted here; trackability through the window is
        the claim Stage 3 compensation actually makes, and is what is
        checked.
        """
        import json
        import statistics

        import app.analysis.outlet_tracker as ot_mod

        from ._synthetic_handheld import MARGIN, outlet_position_at

        clip, _distractor_center = translucent_cup_near_distractor_genuinely_translucent_zahn_video
        base_x = MARGIN + clip.width / 2.0
        base_y = MARGIN + clip.height * 0.34
        above_px = max(2, int(0.02 * clip.height))

        def run(*, disabled: bool, diagnostics_dir: Path):
            if disabled:
                monkeypatch.setattr(
                    ot_mod.OutletTracker, "_foreground_residual_mask", lambda self, gray: None
                )
            result = _run(
                clip.path,
                clip.outlet_at_reference,
                clip=clip,
                diagnostics_dir=str(diagnostics_dir),
            )
            if disabled:
                monkeypatch.undo()
            records = [
                json.loads(line)
                for line in Path(result.diagnostics["tracking_frames_log"]).read_text().splitlines()
            ]
            window = [
                r for r in records if clip.flow_start_s <= r["timestamp_s"] <= clip.flow_end_s
            ]
            tracked_in_window = [r for r in window if r["state"] == "tracked"]
            errors = []
            for record in tracked_in_window:
                roi = record["roi"]
                if roi is None:
                    continue
                tracked_x = roi["x"] + roi["width"] / 2.0
                tracked_y = roi["y"] + above_px
                true_x, true_y = outlet_position_at(
                    record["timestamp_s"],
                    base_x=base_x,
                    base_y=base_y,
                    camera_drift_px=15.0,
                    hand_drift_px=13.0,
                    tremor_px=1.0,
                )
                errors.append(float(np.hypot(tracked_x - true_x, tracked_y - true_y)))
            return result.summary["frames_untracked"], len(window), len(tracked_in_window), errors

        untracked_off, window_len, tracked_off, _errors_off = run(
            disabled=True, diagnostics_dir=tmp_path / "disabled"
        )
        untracked_on, window_len_on, tracked_on, errors_on = run(
            disabled=False, diagnostics_dir=tmp_path / "enabled"
        )
        assert window_len == window_len_on

        assert untracked_on < untracked_off, "compensation should trust more frames overall"
        coverage_off = tracked_off / window_len
        coverage_on = tracked_on / window_len
        assert coverage_on > coverage_off * 2, (
            "compensation should track through much more of the true flow window - "
            f"{coverage_off:.0%} -> {coverage_on:.0%}"
        )
        assert coverage_on >= 0.5, "compensation should track through most of the flow window"
        assert errors_on, "expected at least some window-tracked frames to check position against"
        assert statistics.median(errors_on) < 60.0


class TestSearchWindowRecentres:
    """Third Codex review round, against real footage: "the real cup later
    moves outside the current x=460..760 search window" - a fixed
    click-anchored capture/search window cannot decode pixels the outlet
    has actually drifted beyond, no matter how good the tracker or its
    reacquisition search is. ``wide_pan_zahn_video``'s outlet pans well
    outside that fixed budget; only a search window that recentres on the
    tracker's last credible estimate (``ZahnCupDetector._run_segment``) can
    keep up.
    """

    def test_recentring_keeps_the_outlet_trackable_through_a_wide_pan(self, wide_pan_zahn_video):
        clip = wide_pan_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        summary = result.summary

        # The mechanism under test actually engaged - not just "the result
        # happens to be fine regardless." Bounded by MAX_TRACKING_RECENTRES
        # so a pathological run cannot recentre without limit.
        recentre_events = result.diagnostics["recentre_events"]
        assert recentre_events, "expected at least one recentre on this wide a swing"
        assert len(recentre_events) <= MAX_TRACKING_RECENTRES
        for event in recentre_events:
            assert event["timestamp_s"] > 0.0

        # A fixed window would have left the tracker `lost` for the entire
        # remainder of the run once the outlet first left it - most frames
        # untracked, not a handful. Recentring must keep that a small
        # minority.
        untracked_ratio = summary["frames_untracked"] / max(1, summary["frames_analysed"])
        assert untracked_ratio < 0.3

        if summary["status"] == "confirmed":
            assert summary["efflux_seconds"] is not None
            assert abs(summary["efflux_seconds"] - clip.efflux_s) <= SYNTHETIC_TOLERANCE_S
        else:
            assert summary["confidence"] <= 0.5


class TestBreakDuringATrackingGap:
    """The case the endpoint-uncertainty machinery exists for."""

    def test_endpoint_is_unconfirmed_with_bounds_not_falsely_precise(self, handheld_gap_zahn_video):
        clip = handheld_gap_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
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

    def test_no_event_is_emitted_for_an_uncertain_measurement(self, handheld_gap_zahn_video):
        """Codex review: Event.duration_s is precise regardless of `details`.

        The bounded candidate stays visible in the summary; it must not also
        reach the events list, DetectorResult.to_dict(), the CSV export, or
        the review-count totals as a precise, confirmed-shaped occurrence.
        """
        clip = handheld_gap_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        _assert_no_precise_duration_leaks(result)

    def test_this_adjacent_gap_is_not_mislabelled_as_a_resumed_mid_flow_gap(
        self, handheld_gap_zahn_video
    ):
        """Third Codex review round: a promotion-timing regression.

        Liquid never returns after this gap - the occlusion covers both the
        true break and the true end, so trusted observation resumes
        straight into absence, confirming the end via the ordinary
        adjacent-gap path. A gap being wide enough must not, on its own,
        mark the measurement end_gap_unresolved: that flag is reserved for
        a gap trusted liquid was actually seen to resume *after*, which
        this case is not.
        """
        clip = handheld_gap_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        summary = result.summary

        assert summary["end_gap_unresolved"] is False
        assert result.warnings
        warning_text = " ".join(result.warnings).lower()
        assert "could have occurred earlier" in warning_text
        assert "seen again" not in warning_text
        assert "resumed" not in warning_text

    def test_fixed_roi_cannot_see_through_the_same_gap_either(self, handheld_gap_zahn_video):
        """Sanity check: the gap is real footage, not a tracking-only artefact."""
        clip = handheld_gap_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip, zahn_track_outlet=False)
        # A fixed ROI has no notion of "untracked" at all - it has nothing to
        # honestly flag the gap with. This is exactly the risk the report's
        # "falsely precise" language describes.
        assert result.summary["frames_untracked"] == 0


class TestUnresolvedGapInTheMiddleOfFlow:
    """Second Codex review round: a gap with trusted liquid on both sides.

    Unlike TestBreakDuringATrackingGap (where nothing trusted is seen again
    before the end is confirmed), here liquid is trustedly visible right
    before *and* right after the gap, and flow continues normally to an
    ordinary, cleanly-confirmed end well afterward. That later liquid
    proves the true end is not "somewhere in the gap" - so, unlike the
    adjacent-gap case, no bound may be reported that claims to know where
    the true end is. The measurement must still come back unconfirmed
    (continuity through the gap could not be verified), with no Event and
    no precise duration - but with honest (here: absent) bounds, and a
    warning that does not claim the end could have occurred earlier.
    """

    def test_the_measurement_is_unconfirmed_without_a_fabricated_bound(
        self, handheld_gap_mid_flow_zahn_video
    ):
        clip = handheld_gap_mid_flow_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        summary = result.summary

        assert summary["end_confirmed"] is False
        assert summary["end_uncertain"] is True
        assert summary["end_gap_unresolved"] is True
        assert summary["status"] == "review"
        assert summary["confidence"] <= 0.5

        # The reported end itself is still close to the true one: liquid
        # was trustedly seen right up to the real break, well after the gap
        # closed. Nothing about its own timing was actually in question.
        assert summary["flow_end_s"] == pytest.approx(clip.flow_end_s, abs=SYNTHETIC_TOLERANCE_S)

        # No fabricated bound: the gap does not honestly bound the end
        # (trusted liquid afterward disproves that story), and there is no
        # separate, genuinely adjacent gap here to bound it either - so
        # there is nothing to report as end_uncertainty_bounds or
        # efflux_seconds_bounds, per the "or be null" half of the contract.
        assert summary["end_uncertainty_bounds"] is None
        assert summary["efflux_seconds"] is None
        assert summary["efflux_seconds_bounds"] is None

    def test_no_event_is_emitted_and_the_warning_does_not_claim_an_earlier_end(
        self, handheld_gap_mid_flow_zahn_video
    ):
        clip = handheld_gap_mid_flow_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        _assert_no_precise_duration_leaks(result)

        assert result.warnings
        warning_text = " ".join(result.warnings).lower()
        # The specific thing Codex flagged: liquid was seen again after the
        # gap, so claiming the true end "could have occurred earlier" would
        # contradict the evidence rather than honestly describe it.
        assert "could have occurred earlier" not in warning_text


class TestBreakDuringATrackingGapAtStart:
    """Symmetric case: the true *start*, not the end, falls inside a gap.

    A late-but-precise start is exactly as dishonest as a falsely late end -
    the reported duration comes out shorter than true, with full confidence.
    """

    def test_start_is_unconfirmed_with_bounds_not_falsely_precise(
        self, handheld_gap_at_start_zahn_video
    ):
        clip = handheld_gap_at_start_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
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

    def test_no_event_is_emitted_for_an_uncertain_measurement(
        self, handheld_gap_at_start_zahn_video
    ):
        """Codex review: Event.duration_s is precise regardless of `details`.

        A first fix suppressed only ``details["efflux_seconds"]`` on the
        Event - but Event.duration_s is always end_s - start_s, computed by
        the shared Event/EventLog contract, and reaches the CSV export and
        review-count totals independently of `details`. The candidate stays
        visible in the summary; it must not also reach the events list.
        """
        clip = handheld_gap_at_start_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        _assert_no_precise_duration_leaks(result)


class TestTrackingInitFailure:
    """Init failure must fail loudly, never silently fall back while tracking is on.

    zahn_track_outlet is the *only* sanctioned rollback to the fixed-ROI
    path. If tracking is requested and cannot even start, guessing on stale,
    fixed geometry and reporting it confirmed would be exactly the kind of
    invented precision the rest of this stage exists to refuse.
    """

    def test_no_texture_above_outlet_fails_loudly_when_tracking_is_on(self, blank_video_path):
        # zahn_track_outlet defaults True, so a valid late anchor is
        # required to even reach the texture-init failure this test
        # targets - the blank clip has no texture anywhere, so it fails
        # there regardless of where the late anchor itself is marked.
        result = _run(
            blank_video_path, (100, 100), outlet_end_xy=(100, 100), outlet_end_reference_s=2.0
        )
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

    @pytest.mark.skipif(
        resource is None, reason="resource (RSS measurement) is POSIX-only; unavailable on Windows"
    )
    def test_runtime_and_memory_are_bounded(self, handheld_zahn_video):
        clip = handheld_zahn_video
        before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        started = time.perf_counter()
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        elapsed = time.perf_counter() - started
        after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        assert elapsed < clip.duration_s  # comfortably faster than real time
        # Peak RSS growth for one short clip's worth of small crops must stay
        # in the tens of MB, not scale with the video - one frame at a time.
        assert (after_kb - before_kb) < 200_000  # KB on Linux
        assert result.summary["frames_analysed"] > 0

    def test_runtime_is_bounded_without_the_platform_specific_rss_check(self, handheld_zahn_video):
        """The runtime half of the check above, portable to Windows.

        Only the peak-RSS assertion needs ``resource`` (POSIX-only); elapsed
        time comfortably under the clip's own duration is checkable
        everywhere, so it must not be skipped along with that platform-
        specific measurement.
        """
        clip = handheld_zahn_video
        started = time.perf_counter()
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        elapsed = time.perf_counter() - started

        assert elapsed < clip.duration_s  # comfortably faster than real time
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
        result = _run(
            clip.path, clip.outlet_at_reference, clip=clip, diagnostics_dir=str(diagnostics_dir)
        )

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
            # Stage 3 camera-motion compensation - see OutletTracker's
            # module docstring and _BackgroundMotionEstimator.
            "background_available",
            "background_dx",
            "background_dy",
            "residual_px",
            "rejection_reason",
            "used_residual",
        }
        assert first["state"] in {"tracked", "predicted", "lost"}
        assert isinstance(first["trusted"], bool)
        assert isinstance(first["background_available"], bool)
        assert isinstance(first["used_residual"], bool)
        assert set(first["search_roi"]) == {"x", "y", "width", "height"}
        # Timestamps are monotonic - one line per frame, in decode order.
        timestamps = [record["timestamp_s"] for record in records]
        assert timestamps == sorted(timestamps)

    def test_no_file_is_written_without_a_diagnostics_dir(self, handheld_zahn_video):
        """The default: no diagnostics_dir param means nothing hits disk."""
        clip = handheld_zahn_video
        result = _run(clip.path, clip.outlet_at_reference, clip=clip)
        assert result.diagnostics["tracking_frames_log"] is None
