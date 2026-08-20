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
        # The cup body: a static dark shape above the outlet.
        cv2.rectangle(frame, (170, 60), (310, outlet[1]), (90, 90, 90), -1)

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
