"""Hand-held Zahn cup clip generator for tests.

Builds the scene in *world* coordinates and crops it with a moving camera
window, so the two motions Stage 0 diagnosed - a hand-held phone (background,
cup and stream all drift together) and a hand-held cup (cup and stream drift,
camera does not) - are genuinely independent, exactly as in
``diagnostics/stage0/make_handheld_clip.py``. Ground truth is what is drawn,
never anything a detector reports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytest

MARGIN = 60  # world padding around the camera window, in pixels


@dataclass(frozen=True)
class HandheldClip:
    """A generated clip and the ground truth used to assert against it."""

    path: Path
    fps: float
    duration_s: float
    width: int
    height: int
    flow_start_s: float
    stream_break_s: float
    flow_end_s: float
    outlet_at_reference: tuple[float, float]
    occlusion_s: tuple[float, float] | None

    @property
    def efflux_s(self) -> float:
        return self.flow_end_s - self.flow_start_s


def _world_background(width: int, height: int, level: int, texture: float) -> np.ndarray:
    h, w = height + 2 * MARGIN, width + 2 * MARGIN
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.full((h, w), float(level), dtype=np.float32)
    field -= texture * (yy / h)
    field += 0.5 * texture * np.sin(xx / 47.0)
    field += 0.4 * texture * np.sin((xx + yy) / 91.0)
    cv2.circle(field, (int(0.22 * w), int(0.30 * h)), 26, float(level - 9), -1)
    cv2.rectangle(
        field,
        (int(0.70 * w), int(0.62 * h)),
        (int(0.82 * w), int(0.74 * h)),
        float(level - 7),
        -1,
    )
    return cv2.GaussianBlur(field, (0, 0), 6.0)


def _camera_offset(t: float, drift_px: float, tremor_px: float) -> tuple[float, float]:
    if drift_px <= 0 and tremor_px <= 0:
        return 0.0, 0.0
    dx = drift_px * math.sin(2 * math.pi * t / 17.0)
    dy = 0.6 * drift_px * math.sin(2 * math.pi * t / 11.0 + 1.1)
    dx += tremor_px * math.sin(2 * math.pi * t * 3.7)
    dy += tremor_px * math.sin(2 * math.pi * t * 4.3 + 0.5)
    return dx, dy


def _hand_offset(t: float, drift_px: float) -> tuple[float, float]:
    if drift_px <= 0:
        return 0.0, 0.0
    dx = drift_px * math.sin(2 * math.pi * t / 13.0 + 0.4)
    dy = 0.5 * drift_px * math.sin(2 * math.pi * t / 9.0 + 2.0)
    return dx, dy


def _draw_cup(frame: np.ndarray, ox: float, oy: float, level: int) -> None:
    x, y = int(round(ox)), int(round(oy))
    cv2.rectangle(frame, (x - 55, y - 95), (x + 55, y - 5), float(level), -1)
    cv2.ellipse(frame, (x, y - 5), (55, 13), 0, 0, 180, float(level), -1)
    cv2.ellipse(frame, (x, y - 95), (55, 13), 0, 0, 360, float(level - 4), -1)
    nose = np.array([[x - 13, y - 10], [x + 13, y - 10], [x + 4, y], [x - 4, y]], dtype=np.int32)
    cv2.fillPoly(frame, [nose], float(level - 2))
    cv2.rectangle(frame, (x + 55, y - 76), (x + 82, y - 66), float(level - 10), -1)
    cv2.rectangle(frame, (x + 72, y - 66), (x + 82, y - 32), float(level - 10), -1)


def _draw_liquid(
    frame: np.ndarray,
    t: float,
    ox: float,
    oy: float,
    *,
    start_s: float,
    break_s: float,
    end_s: float,
    level: int,
    drop_period_s: float = 0.3,
) -> None:
    x, y = int(round(ox)), int(round(oy))
    height = frame.shape[0]
    if start_s <= t < break_s:
        wobble = int(round(2.0 * math.sin(t * 6.0)))
        cv2.line(frame, (x, y), (x + wobble, height - 1), float(level), 3)
        return
    if break_s <= t <= end_s:
        phase = (t - break_s) % drop_period_s
        if phase < 0.4 * drop_period_s:
            fall = int(round(700.0 * phase))
            cv2.circle(frame, (x, min(height - 4, y + 16 + fall)), 3, float(level), -1)


def build_handheld_clip(
    path: Path,
    *,
    width: int = 480,
    height: int = 360,
    fps: float = 25.0,
    duration_s: float = 14.0,
    flow_start_s: float = 2.0,
    stream_break_s: float = 9.0,
    flow_end_s: float = 10.2,
    background_level: int = 214,
    texture_strength: float = 6.0,
    sensor_noise: float = 1.8,
    cup_delta: int = 14,
    stream_delta: int = 18,
    camera_drift_px: float = 14.0,
    hand_drift_px: float = 12.0,
    tremor_px: float = 1.0,
    occlusion_s: tuple[float, float] | None = None,
    seed: int = 20260820,
) -> HandheldClip:
    """Render one hand-held clip and return its (drawn, not inferred) ground truth."""
    rng = np.random.default_rng(seed)
    world = _world_background(width, height, background_level, texture_strength)

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():  # pragma: no cover - depends on the OpenCV build
        pytest.skip("This OpenCV build cannot write MP4 files")

    base_x = MARGIN + width / 2.0
    base_y = MARGIN + height * 0.34
    reference_outlet: tuple[float, float] | None = None
    total_frames = int(round(duration_s * fps))

    for index in range(total_frames):
        t = index / fps
        cx, cy = _camera_offset(t, camera_drift_px, tremor_px)
        x0, y0 = int(round(MARGIN + cx)), int(round(MARGIN + cy))
        frame = world[y0 : y0 + height, x0 : x0 + width].copy()

        hx, hy = _hand_offset(t, hand_drift_px)
        ox = (base_x + hx) - (MARGIN + cx)
        oy = (base_y + hy) - (MARGIN + cy)
        if reference_outlet is None:
            reference_outlet = (ox, oy)

        hidden = occlusion_s is not None and occlusion_s[0] <= t < occlusion_s[1]
        if not hidden:
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
    return HandheldClip(
        path=path,
        fps=fps,
        duration_s=duration_s,
        width=width,
        height=height,
        flow_start_s=flow_start_s,
        stream_break_s=stream_break_s,
        flow_end_s=flow_end_s,
        outlet_at_reference=reference_outlet,
        occlusion_s=occlusion_s,
    )
