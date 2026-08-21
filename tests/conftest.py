"""Shared fixtures.

Real Zahn cup and factory recordings cannot be committed to a repository, so
the integration tests build *synthetic* videos with known ground truth: we know
exactly when the stream starts, when the last drop falls, and when the machine
moves.  A detector that cannot get these right has no chance on real footage,
and a regression in the timing logic shows up immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.config import AppConfig
from app.web.routes import create_app

from ._synthetic_handheld import HandheldClip, build_handheld_clip

RANDOM_SEED = 20260316

# Parameters validated to reproduce the Stage 0 field bug reliably: a fixed
# ROI measures this timeline several seconds long (frames_untracked=0, the
# fixed path never even notices anything is wrong), while tracking measures
# it within the +/-0.5s synthetic tolerance. See
# diagnostics/stage0/STAGE0_REPORT.md for the real-footage version of this.
HANDHELD_TIMELINE = {
    "width": 480,
    "height": 360,
    "fps": 25.0,
    "duration_s": 22.0,
    "flow_start_s": 3.0,
    "stream_break_s": 16.0,
    "flow_end_s": 17.5,
}


@dataclass(frozen=True)
class SyntheticVideo:
    """A generated video and the ground truth used to assert against it."""

    path: Path
    fps: float
    duration_s: float
    width: int
    height: int
    truth: dict[str, float]


def _writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():  # pragma: no cover - depends on the OpenCV build
        pytest.skip("This OpenCV build cannot write MP4 files")
    return writer


def _noisy_background(
    rng: np.random.Generator, width: int, height: int, level: int = 180
) -> np.ndarray:
    """A plain background with light sensor noise, as any real camera has."""
    base = np.full((height, width, 3), level, dtype=np.float32)
    base += rng.normal(0.0, 2.5, base.shape).astype(np.float32)
    return np.clip(base, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def zahn_video(tmp_path_factory: pytest.TempPathFactory) -> SyntheticVideo:
    """A side-on Zahn cup run with a known efflux time.

    Timeline (25 FPS, 26 s):
        0.0 -  4.0 s   cup full, nothing leaving the outlet
        4.0 - 20.0 s   continuous stream from the outlet downward
       20.0 - 21.6 s   the stream breaks up into intermittent drops
       21.6 - 26.0 s   nothing at all

    Ground-truth efflux time: 4.0 s to the last drop at ~21.5 s.
    """
    rng = np.random.default_rng(RANDOM_SEED)
    width, height, fps, duration = 480, 480, 25.0, 26.0
    path = tmp_path_factory.mktemp("videos") / "zahn.mp4"
    writer = _writer(path, fps, (width, height))

    outlet = (240, 180)  # x, y of the orifice
    stream_start, stream_end = 4.0, 20.0
    drops_end = 21.5

    total_frames = int(duration * fps)
    for index in range(total_frames):
        timestamp = index / fps
        frame = _noisy_background(rng, width, height)
        # The cup body: a static dark shape above the outlet. A rim and a
        # handle give it more than a bare rectangle's four corners - real
        # Zahn cups are not flat-sided, and outlet tracking (Stage 1) needs
        # genuine texture to find above the outlet the same way it would on
        # an actual cup.
        cv2.rectangle(frame, (170, 60), (310, outlet[1]), (90, 90, 90), -1)
        cv2.ellipse(frame, (240, 60), (70, 14), 0, 0, 360, (78, 78, 78), -1)
        cv2.rectangle(frame, (310, 84), (334, 100), (82, 82, 82), -1)

        if stream_start <= timestamp < stream_end:
            # A continuous, slightly wobbling stream below the orifice.
            wobble = int(2 * np.sin(timestamp * 6.0))
            cv2.line(
                frame,
                (outlet[0], outlet[1]),
                (outlet[0] + wobble, height - 1),
                (40, 40, 40),
                4,
            )
        elif stream_end <= timestamp < drops_end:
            # Intermittent drops: one every ~0.3 s, falling from the orifice.
            phase = (timestamp - stream_end) % 0.3
            if phase < 0.12:
                fall = int(600 * phase)
                centre_y = min(height - 6, outlet[1] + 20 + fall)
                cv2.circle(frame, (outlet[0], centre_y), 4, (40, 40, 40), -1)

        writer.write(frame)
    writer.release()

    return SyntheticVideo(
        path=path,
        fps=fps,
        duration_s=duration,
        width=width,
        height=height,
        truth={
            "outlet_x": float(outlet[0]),
            "outlet_y": float(outlet[1]),
            "flow_start_s": stream_start,
            "flow_end_s": drops_end,
            "efflux_s": drops_end - stream_start,
        },
    )


@pytest.fixture(scope="session")
def motion_video(tmp_path_factory: pytest.TempPathFactory) -> SyntheticVideo:
    """A static scene with two separate periods of movement.

    Timeline (20 FPS, 30 s): movement in 5.0-9.0 s and 18.0-23.0 s.
    """
    rng = np.random.default_rng(RANDOM_SEED + 1)
    width, height, fps, duration = 320, 240, 20.0, 30.0
    path = tmp_path_factory.mktemp("videos") / "motion.mp4"
    writer = _writer(path, fps, (width, height))

    intervals = [(5.0, 9.0), (18.0, 23.0)]
    for index in range(int(duration * fps)):
        timestamp = index / fps
        frame = _noisy_background(rng, width, height, level=120)
        for start, end in intervals:
            if start <= timestamp < end:
                progress = (timestamp - start) / (end - start)
                x = int(40 + progress * 200)
                cv2.rectangle(frame, (x, 90), (x + 45, 150), (240, 240, 240), -1)
        writer.write(frame)
    writer.release()

    return SyntheticVideo(
        path=path,
        fps=fps,
        duration_s=duration,
        width=width,
        height=height,
        truth={
            "first_start_s": 5.0,
            "first_end_s": 9.0,
            "second_start_s": 18.0,
            "second_end_s": 23.0,
        },
    )


@pytest.fixture(scope="session")
def quiet_video(tmp_path_factory: pytest.TempPathFactory) -> SyntheticVideo:
    """A recording in which nothing whatsoever happens (noise only)."""
    rng = np.random.default_rng(RANDOM_SEED + 2)
    width, height, fps, duration = 320, 240, 15.0, 12.0
    path = tmp_path_factory.mktemp("videos") / "quiet.mp4"
    writer = _writer(path, fps, (width, height))
    for _ in range(int(duration * fps)):
        writer.write(_noisy_background(rng, width, height, level=100))
    writer.release()
    return SyntheticVideo(
        path=path, fps=fps, duration_s=duration, width=width, height=height, truth={}
    )


@pytest.fixture(scope="session")
def handheld_zahn_video(tmp_path_factory: pytest.TempPathFactory) -> HandheldClip:
    """Hand-held camera *and* hand-held cup, white-on-white - the field report.

    Same timeline as diagnostics/stage0/STAGE0_REPORT.md's `handheld_white`:
    13 s of continuous stream then a sparse drop tail, with the outlet
    drifting off a fixed ROI's default width over the run. Regression
    fixture for the original "several seconds too long" bug and the
    synthetic-tolerance test for its fix.
    """
    path = tmp_path_factory.mktemp("videos") / "handheld_zahn.mp4"
    return build_handheld_clip(path, **HANDHELD_TIMELINE)


@pytest.fixture(scope="session")
def handheld_gap_zahn_video(tmp_path_factory: pytest.TempPathFactory) -> HandheldClip:
    """The same clip, with the outlet swung out of frame across the true break.

    Occlusion covers 15.4-18.2s: the true stream break (16.0s) and the true
    end (17.5s) both happen while the outlet cannot be seen at all. Nothing
    can honestly measure a precise end from this footage - the fixture exists
    to prove the detector says so instead of inventing one.
    """
    path = tmp_path_factory.mktemp("videos") / "handheld_gap_zahn.mp4"
    return build_handheld_clip(path, occlusion_s=(15.4, 18.2), **HANDHELD_TIMELINE)


@pytest.fixture(scope="session")
def handheld_gap_mid_flow_zahn_video(tmp_path_factory: pytest.TempPathFactory) -> HandheldClip:
    """The outlet swings out of frame in the *middle* of an otherwise clean run.

    Occlusion covers 8.0-9.6s - well inside the continuous-stream window
    (3.0-16.0s), nowhere near the true start or the true break/end. Liquid
    is trustedly visible both immediately before and immediately after the
    gap, and the run then continues to an ordinary, cleanly-confirmed end.
    Second Codex review round: this is the case FlowStateMachine's
    end_gap_unresolved handling exists for - trusted liquid returning after
    the gap proves the true end is not "somewhere in that gap", so the
    measurement must be unconfirmed without inventing a bound that could
    exclude the actual, later true end.
    """
    path = tmp_path_factory.mktemp("videos") / "handheld_gap_mid_flow_zahn.mp4"
    return build_handheld_clip(path, occlusion_s=(8.0, 9.6), **HANDHELD_TIMELINE)


@pytest.fixture(scope="session")
def handheld_gap_at_start_zahn_video(tmp_path_factory: pytest.TempPathFactory) -> HandheldClip:
    """The outlet is swung out of frame across the true *start*, not the end.

    Occlusion covers 2.6-4.2s, straddling the true flow start at 3.0s: a
    late-but-precise start is exactly as falsely precise as a falsely late
    end, since the reported duration comes out too short with full
    confidence. See FlowStateMachine's start_uncertain handling.
    """
    path = tmp_path_factory.mktemp("videos") / "handheld_gap_at_start_zahn.mp4"
    return build_handheld_clip(path, occlusion_s=(2.6, 4.2), **HANDHELD_TIMELINE)


# Second Codex review round on the real-footage validation failure: a
# portrait, resolution-scaled timeline approximating the real clip's own
# geometry (1080x1920, large early reframing). camera_drift_px/hand_drift_px
# are scaled ~2.25x vs HANDHELD_TIMELINE's 480px-wide baseline - the same
# relative motion, more raw pixels at higher resolution.
PORTRAIT_TIMELINE = {
    "width": 1080,
    "height": 1920,
    "fps": 30.0,
    "duration_s": 27.0,
    "flow_start_s": 3.9,
    "stream_break_s": 20.0,
    "flow_end_s": 20.5,
    "camera_drift_px": 14.0 * (1080 / 480),
    "hand_drift_px": 12.0 * (1080 / 480),
    "tremor_px": 1.0 * (1080 / 480),
}


@pytest.fixture(scope="session")
def portrait_reference_frame_zahn_video(tmp_path_factory: pytest.TempPathFactory) -> HandheldClip:
    """The outlet is only markable well after the true start - the exact
    real-footage bug (Codex review, reopened after a real clip was
    supplied): the camera has already panned well past frame 0 by the time
    the outlet becomes clearly visible. ``outlet_at_reference`` is this
    clip's t=0 position; ``PORTRAIT_TIMELINE``'s own flow_start_s (3.9s) is
    well before the 4.5s reference frame the tests mark the outlet on, so
    the true start can only be recovered - not just honestly flagged
    unmeasurable - if outlet_reference_s is wired through and used
    correctly. See ZahnCupDetector.outlet_reference_s.
    """
    path = tmp_path_factory.mktemp("videos") / "portrait_reference_frame_zahn.mp4"
    return build_handheld_clip(path, **PORTRAIT_TIMELINE)


@pytest.fixture(scope="session")
def wide_pan_zahn_video(tmp_path_factory: pytest.TempPathFactory) -> HandheldClip:
    """The outlet pans steadily, in one direction, far enough that a fixed,
    click-anchored search window cannot contain it for the whole run - only
    a search window that recentres on the tracker's last credible estimate
    can. Third Codex review round, against real footage: "the real cup
    later moves outside the current x=460..760 search window."

    A *steady, one-directional* pan (not ``build_handheld_clip``'s
    oscillating hand/camera drift, whose sinusoidal reversals stress
    velocity extrapolation in a way a real single reframe usually does not)
    - 380px of total drift, guard half-width (~70px) + track_search_margin_px
    (80px default) bounds a fixed window to ~150px from the click, so this
    comfortably needs at least one recentre, at a per-frame speed (<1px)
    ordinary optical flow tracks easily between them. See
    diagnostics/stage1/STAGE1_REPORT.md and
    ZahnCupDetector._run_segment/_build_capture_roi's recentring.

    A *flat* background, deliberately, unlike most other fixtures here: with
    the camera held still and only the cup panning, ``_world_background``'s
    spatial texture (and the distractor shapes baked into it) would slide
    past the tracked analysis window as it follows the cup across 380px of
    otherwise-static scene, and StreamActivityScorer's own background model
    - built for a *stable* view - would read that sliding texture as
    spurious activity. That is a real, separate scoring-side limitation
    (background stability assumes a window that does not itself sweep
    across a textured static scene) worth its own coverage another time;
    this fixture isolates the one thing it exists to test - the search
    window recentring - by removing that confound rather than compounding
    two different failure modes in one fixture. The cup's own drawn shading
    (rim/handle/nose edges) still gives the tracker plenty to key on.
    """
    import numpy as np

    from ._synthetic_handheld import MARGIN, _draw_cup, _draw_liquid

    width, height, fps, duration = 640, 480, 25.0, 16.0
    background_level, sensor_noise = 214, 1.8
    cup_delta, stream_delta = 14, 18
    flow_start_s, stream_break_s, flow_end_s = 2.0, 11.0, 12.0
    base_x, base_y, total_dx = 150.0, 250.0, 380.0
    rng = np.random.default_rng(20260821)
    world = np.full(
        (height + 2 * MARGIN, width + 2 * MARGIN), float(background_level), dtype=np.float32
    )

    path = tmp_path_factory.mktemp("videos") / "wide_pan_zahn.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - depends on the OpenCV build
        pytest.skip("This OpenCV build cannot write MP4 files")

    # Deliberately early, unlike the "near the end" default elsewhere: most
    # of the 380px pan must fall *after* the late anchor (in segment C, the
    # continuation past outlet_end_reference_s) for this fixture to still
    # exercise search-window recentring - a late anchor near duration - 1
    # would place nearly the whole pan inside segment B (between the two
    # anchors), which never recentres by design.
    late_s = 4.0
    reference_outlet: tuple[float, float] | None = None
    total_frames = int(round(duration * fps))
    for index in range(total_frames):
        t = index / fps
        frame = world[MARGIN : MARGIN + height, MARGIN : MARGIN + width].copy()
        ox, oy = base_x + (total_dx / duration) * t, base_y
        if reference_outlet is None:
            reference_outlet = (ox, oy)
        _draw_cup(frame, ox, oy, background_level - cup_delta)
        _draw_liquid(
            frame,
            t,
            ox,
            oy,
            start_s=flow_start_s,
            break_s=stream_break_s,
            end_s=flow_end_s,
            level=background_level - stream_delta,
        )
        frame += rng.normal(0.0, sensor_noise, frame.shape).astype(np.float32)
        bgr = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        writer.write(bgr)
    writer.release()

    assert reference_outlet is not None
    late_outlet = (base_x + (total_dx / duration) * late_s, base_y)
    return HandheldClip(
        path=path,
        fps=fps,
        duration_s=duration,
        width=width,
        height=height,
        flow_start_s=flow_start_s,
        stream_break_s=stream_break_s,
        flow_end_s=flow_end_s,
        outlet_at_reference=reference_outlet,
        occlusion_s=None,
        outlet_at_late_reference=late_outlet,
        late_reference_s=late_s,
    )


@pytest.fixture(scope="session")
def translucent_cup_near_distractor_zahn_video(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[HandheldClip, tuple[float, float]]:
    """A genuinely-initialised cup that drifts near a strong, stationary
    distractor patch later in the run - not overlapping it at t=0, unlike
    ``build_translucent_cup_clip``'s worst case (see diagnostics/stage1/
    STAGE1_REPORT.md for why that case is not Stage-1-solvable). Tests here
    check the safety property this stage *can* deliver: when a nearby
    distractor is strong enough to threaten pulling the fitted transform
    off the cup, the tracker must become honestly uncertain rather than
    confidently wrong (Codex review: "why transforms are accepted").
    """
    import numpy as np

    from ._synthetic_handheld import (
        MARGIN,
        _camera_offset,
        _checkerboard_patch,
        _draw_liquid,
        _world_background,
        outlet_position_at,
    )

    def _draw_cup_with_rim_and_handle(
        frame: np.ndarray, ox: float, oy: float, level: float
    ) -> None:
        x, y = int(round(ox)), int(round(oy))
        cv2.rectangle(frame, (x - 55, y - 95), (x + 55, y - 5), level, -1)
        cv2.ellipse(frame, (x, y - 5), (55, 13), 0, 0, 180, level, -1)
        cv2.ellipse(frame, (x, y - 95), (55, 13), 0, 0, 360, level - 4, -1)
        cv2.rectangle(frame, (x + 55, y - 76), (x + 82, y - 66), level - 10, -1)
        cv2.rectangle(frame, (x + 72, y - 66), (x + 82, y - 32), level - 10, -1)

    width, height, fps, duration = 1080, 1920, 30.0, 16.0
    background_level, sensor_noise = 214, 1.8
    camera_drift_px, hand_drift_px, tremor_px = 30.0, 26.0, 2.0
    flow_start_s, stream_break_s, flow_end_s = 3.0, 12.0, 12.6
    rng = np.random.default_rng(9)
    base_x = MARGIN + width / 2.0
    base_y = MARGIN + height * 0.34
    world = _world_background(width, height, background_level, 6.0)
    # Offset from the cup's t=0 position, so initialisation locks onto the
    # genuine cup - the drift toward this distractor happens over the run.
    distractor_center = (base_x + 140, base_y - 40)
    _checkerboard_patch(world, distractor_center, background_level)

    path = tmp_path_factory.mktemp("videos") / "translucent_near_distractor.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - depends on the OpenCV build
        pytest.skip("This OpenCV build cannot write MP4 files")

    late_s = max(0.0, duration - 1.0)
    reference_outlet: tuple[float, float] | None = None
    for index in range(int(duration * fps)):
        t = index / fps
        cx, cy = _camera_offset(t, camera_drift_px, tremor_px)
        x0, y0 = int(round(MARGIN + cx)), int(round(MARGIN + cy))
        frame = world[y0 : y0 + height, x0 : x0 + width].copy()
        ox, oy = outlet_position_at(
            t,
            base_x=base_x,
            base_y=base_y,
            camera_drift_px=camera_drift_px,
            hand_drift_px=hand_drift_px,
            tremor_px=tremor_px,
        )
        if reference_outlet is None:
            reference_outlet = (ox, oy)
        _draw_cup_with_rim_and_handle(frame, ox, oy, float(background_level - 14))
        _draw_liquid(
            frame,
            t,
            ox,
            oy,
            start_s=flow_start_s,
            break_s=stream_break_s,
            end_s=flow_end_s,
            level=background_level - 18,
        )
        frame += rng.normal(0.0, sensor_noise, frame.shape).astype(np.float32)
        bgr = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        writer.write(bgr)
    writer.release()

    assert reference_outlet is not None
    late_outlet = outlet_position_at(
        late_s,
        base_x=base_x,
        base_y=base_y,
        camera_drift_px=camera_drift_px,
        hand_drift_px=hand_drift_px,
        tremor_px=tremor_px,
    )
    clip = HandheldClip(
        path=path,
        fps=fps,
        duration_s=duration,
        width=width,
        height=height,
        flow_start_s=flow_start_s,
        stream_break_s=stream_break_s,
        flow_end_s=flow_end_s,
        outlet_at_reference=reference_outlet,
        occlusion_s=None,
        outlet_at_late_reference=late_outlet,
        late_reference_s=late_s,
    )
    return clip, distractor_center


@pytest.fixture(scope="session")
def translucent_cup_overlapping_distractor_zahn_video(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[HandheldClip, tuple[float, float]]:
    """Stage 3's own regression case: a strongly-textured distractor patch
    that already sits *under* the translucent cup at t=0, unlike
    ``translucent_cup_near_distractor_zahn_video``'s "drifts near it later"
    case - the worst case ``build_translucent_cup_clip`` exists for, and
    the one diagnostics/stage1/STAGE1_REPORT.md's earlier rounds (§18.3,
    §24.2) named as *not solvable* by patch-correlation verification alone,
    since the reference patch itself would be extracted from the
    distractor - "more of the same background, wherever it later appears"
    (the checkerboard, still visible through the translucent cup at every
    later frame, since it only moves with camera pan) answers that
    verification correctly. Camera-motion compensation (Stage 3) is a
    different kind of evidence - not "does this still look like the
    reference," but "does this move independently of the background" - so
    this is the fixture that actually exercises it, not merely a case
    Stage 1's own checks already handled.
    """
    from ._synthetic_handheld import MARGIN, build_translucent_cup_clip

    path = tmp_path_factory.mktemp("videos") / "translucent_overlapping_distractor.mp4"
    width, height = 1080, 1920
    clip = build_translucent_cup_clip(path, width=width, height=height)
    distractor_center = (MARGIN + width / 2.0, MARGIN + height * 0.34)
    return clip, distractor_center


@pytest.fixture(scope="session")
def translucent_cup_near_distractor_genuinely_translucent_zahn_video(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[HandheldClip, tuple[float, float]]:
    """The achievable case Stage 3 compensation is meant for, distinct from
    ``translucent_cup_overlapping_distractor_zahn_video``'s worst case: a
    genuinely translucent (alpha-blended - ``_draw_translucent_cup``, not
    ``translucent_cup_near_distractor_zahn_video``'s solid-filled cup)
    outlet, initialised cleanly (the distractor sits away from the cup's
    own t=0 position, so the reference patch is drawn from real, if weak,
    cup texture) with a strong distractor nearby it could later drift
    toward. Motion-compensated feature selection has a genuine chance here
    - unlike the overlapping case, where even the reference patch itself
    is already contaminated.

    ``cup_opacity=0.5`` - the default 0.22 is too faint for any feature
    selection strategy (raw or compensated) to find much to work with at
    all; a fully opaque cup is Stage 1's already-solved case. 0.5 is
    genuinely translucent (background still visibly shows through) while
    leaving enough real contrast for compensated evidence to exist -
    measured empirically, not a physical constant, same caveat as the
    other Stage 3 thresholds. A smaller/shorter clip than the worst-case
    fixture's 1080x1920/20s: this one exists to be run in an A/B
    comparison (see the integration test using it), not to reproduce the
    real clip's own resolution.
    """
    from ._synthetic_handheld import MARGIN, build_translucent_cup_clip

    path = tmp_path_factory.mktemp("videos") / "translucent_near_distractor_real.mp4"
    width, height, duration_s = 480, 640, 10.0
    offset = (80.0, -25.0)
    clip = build_translucent_cup_clip(
        path,
        width=width,
        height=height,
        duration_s=duration_s,
        flow_start_s=1.5,
        stream_break_s=7.0,
        flow_end_s=7.3,
        distractor_offset=offset,
        cup_opacity=0.5,
        camera_drift_px=15.0,
        hand_drift_px=13.0,
        tremor_px=1.0,
    )
    distractor_center = (MARGIN + width / 2.0 + offset[0], MARGIN + height * 0.34 + offset[1])
    return clip, distractor_center


@pytest.fixture(scope="session")
def translucent_cup_boundary_only_zahn_video(
    tmp_path_factory: pytest.TempPathFactory,
) -> HandheldClip:
    """Stage 3, round two's own regression case: sparse residual-filtered
    *corner* features fail, but the cup's rim/boundary silhouette remains
    visible - the exact gap the corner/LK path's own compensation (§29)
    could not close, per the supervisor's real-clip diagnosis of `3208924`
    (`used_residual` fired on 805/811 frames, yet only 17 ended up trusted:
    residual-filtered corner features simply do not exist in enough
    density on a real translucent cup).

    ``cup_opacity=0.12`` - low enough that the cup's alpha-blended *body*
    offers essentially no exploitable corner texture (measured: only 15/300
    frames redetect any usable residual-filtered corner set at all, and
    window-trackability without the contour path is 7.4%), while
    ``_draw_translucent_cup``'s rim ellipse is drawn at a fixed, opacity-
    independent contrast (always ~8 grey levels darker than the
    background, regardless of body opacity) - a genuine, matchable
    boundary that never disappears just because the body does. This is
    the calibrated split between "corner features fail" and "the boundary
    remains visible" the authorising review asked for, found empirically
    (see diagnostics/stage1/STAGE1_REPORT.md §30) by sweeping opacity
    while watching both the corner path's own trust and the contour
    path's own match outcomes independently.

    ``distractor_offset=(250, -25)`` - far enough that the checkerboard
    distractor (``_checkerboard_patch``'s own 70px span) never enters the
    contour search/round-trip windows around the cup; a closer offset
    (this fixture's own calibration first tried the existing 80px offset
    other translucent-cup fixtures use) let the checkerboard's repeating
    edges produce several near-identical match peaks inside the search
    window, correctly triggering the score-margin safeguard - genuine
    ambiguity, not a bug, but a different failure mode than the one this
    fixture exists to isolate (a distractor's own robustness is already
    covered by the other translucent-cup fixtures above).
    """
    from ._synthetic_handheld import build_translucent_cup_clip

    path = tmp_path_factory.mktemp("videos") / "translucent_boundary_only.mp4"
    width, height, duration_s = 480, 640, 10.0
    return build_translucent_cup_clip(
        path,
        width=width,
        height=height,
        duration_s=duration_s,
        flow_start_s=1.5,
        stream_break_s=7.0,
        flow_end_s=7.3,
        distractor_offset=(250.0, -25.0),
        cup_opacity=0.12,
        camera_drift_px=15.0,
        hand_drift_px=13.0,
        tremor_px=1.0,
    )


@pytest.fixture
def broken_video(tmp_path: Path) -> Path:
    """A file with a video extension that is not a video at all."""
    path = tmp_path / "not_really.mp4"
    path.write_bytes(b"this is not a video file, it is a trap" * 32)
    return path


@pytest.fixture
def app(tmp_path: Path):
    """The real Flask application, with a temporary data directory."""
    config = replace(
        AppConfig(),
        data_dir=tmp_path / "data",
        max_upload_mb=64,
        log_level="WARNING",
    )
    application = create_app(config)
    application.config.update(TESTING=True)
    yield application
    application.extensions["analysis_service"].shutdown()


@pytest.fixture
def client(app):
    return app.test_client()
