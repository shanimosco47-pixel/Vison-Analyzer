"""Generate controlled hand-held Zahn cup clips with exact known ground truth.

Stage 0 diagnostic tool.  No production code is imported, and nothing here is
used by the application; the clips exist so each candidate cause of the
"efflux several seconds too long" report can be switched on and off
independently and the resulting timing error attributed.

The scene is built in *world* coordinates and then cropped by a moving camera
window, so the two motions the field report describes are genuinely separate:

    camera drift  - moves background, cup and stream together (hand-held phone)
    hand drift    - moves cup and stream only (hand-held cup)

Ground truth is what is *drawn*, never what a detector reports.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

MARGIN = 60  # world padding around the camera window, in pixels


@dataclass(frozen=True)
class ClipSpec:
    """Everything that distinguishes one ablation clip from another."""

    name: str
    width: int = 640
    height: int = 480
    fps: float = 30.0
    duration_s: float = 26.0

    # Timeline (seconds).  These four numbers *are* the ground truth.
    flow_start_s: float = 3.0
    stream_break_s: float = 19.0   # continuous stream ends, drops begin
    last_drop_s: float = 20.5      # final drop leaves the outlet
    drop_period_s: float = 0.3

    # Appearance.  Background is near-white; the deltas are how much darker
    # the cup and the liquid are than it.  Small deltas = white-on-white.
    background_level: int = 214
    texture_strength: float = 6.0
    sensor_noise: float = 2.2
    cup_delta: int = 14
    stream_delta: int = 18
    stream_width_px: int = 3
    drop_radius_px: int = 3

    # Motion amplitudes in pixels (peak excursion from the reference frame).
    camera_drift_px: float = 14.0
    hand_drift_px: float = 12.0
    tremor_px: float = 1.2

    # Optional loss-of-visibility window: the outlet leaves the camera window
    # entirely (operator re-frames).  Used for the tracking-gap case.
    occlusion_s: tuple[float, float] | None = None

    @property
    def total_frames(self) -> int:
        return int(round(self.duration_s * self.fps))


def _world_background(spec: ClipSpec, rng: np.random.Generator) -> np.ndarray:
    """A near-white wall: flat, but with the faint shading any real wall has.

    Without some texture a translating camera would produce no image
    difference at all, which would make the motion tests vacuously easy.
    """
    h, w = spec.height + 2 * MARGIN, spec.width + 2 * MARGIN
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.full((h, w), float(spec.background_level), dtype=np.float32)
    field -= spec.texture_strength * (yy / h)                       # top-lit wall
    field += 0.5 * spec.texture_strength * np.sin(xx / 47.0)        # faint banding
    field += 0.4 * spec.texture_strength * np.sin((xx + yy) / 91.0)
    # A couple of very faint marks so translation is locally observable.
    cv2.circle(field, (int(0.22 * w), int(0.30 * h)), 26, float(spec.background_level - 9), -1)
    cv2.rectangle(
        field,
        (int(0.70 * w), int(0.62 * h)),
        (int(0.82 * w), int(0.74 * h)),
        float(spec.background_level - 7),
        -1,
    )
    field = cv2.GaussianBlur(field, (0, 0), 6.0)
    _ = rng  # noise is added per frame, not here
    return field


def camera_offset(spec: ClipSpec, t: float) -> tuple[float, float]:
    """Slow hand-held drift of the phone, plus a small tremor."""
    if spec.camera_drift_px <= 0 and spec.tremor_px <= 0:
        return 0.0, 0.0
    dx = spec.camera_drift_px * math.sin(2 * math.pi * t / 17.0)
    dy = 0.6 * spec.camera_drift_px * math.sin(2 * math.pi * t / 11.0 + 1.1)
    dx += spec.tremor_px * math.sin(2 * math.pi * t * 3.7)
    dy += spec.tremor_px * math.sin(2 * math.pi * t * 4.3 + 0.5)
    return dx, dy


def hand_offset(spec: ClipSpec, t: float) -> tuple[float, float]:
    """Drift of the cup within the scene, independent of the camera."""
    if spec.hand_drift_px <= 0:
        return 0.0, 0.0
    dx = spec.hand_drift_px * math.sin(2 * math.pi * t / 13.0 + 0.4)
    dy = 0.5 * spec.hand_drift_px * math.sin(2 * math.pi * t / 9.0 + 2.0)
    return dx, dy


def outlet_at(spec: ClipSpec, t: float) -> tuple[float, float]:
    """Ground-truth outlet position in *frame* coordinates at time ``t``."""
    base_x = MARGIN + spec.width / 2.0
    base_y = MARGIN + spec.height * 0.34
    hx, hy = hand_offset(spec, t)
    cx, cy = camera_offset(spec, t)
    return (base_x + hx) - (MARGIN + cx), (base_y + hy) - (MARGIN + cy)


def _draw_cup(frame: np.ndarray, spec: ClipSpec, ox: float, oy: float) -> None:
    """A pale metal cup body sitting immediately above the outlet."""
    level = float(spec.background_level - spec.cup_delta)
    x, y = int(round(ox)), int(round(oy))
    cv2.rectangle(frame, (x - 70, y - 120), (x + 70, y - 6), level, -1)
    cv2.ellipse(frame, (x, y - 6), (70, 16), 0, 0, 180, level, -1)
    cv2.ellipse(frame, (x, y - 120), (70, 16), 0, 0, 360, level - 4, -1)
    # The tapered nose down to the orifice.
    nose = np.array([[x - 16, y - 12], [x + 16, y - 12], [x + 5, y], [x - 5, y]], dtype=np.int32)
    cv2.fillPoly(frame, [nose], level - 2)
    # The handle: the one genuinely textured feature on the cup.
    cv2.rectangle(frame, (x + 70, y - 96), (x + 104, y - 84), level - 10, -1)
    cv2.rectangle(frame, (x + 92, y - 84), (x + 104, y - 40), level - 10, -1)


def _draw_liquid(frame: np.ndarray, spec: ClipSpec, t: float, ox: float, oy: float) -> None:
    """Stream while flowing, then intermittent drops until the last one."""
    level = float(spec.background_level - spec.stream_delta)
    x, y = int(round(ox)), int(round(oy))
    height = frame.shape[0]

    if spec.flow_start_s <= t < spec.stream_break_s:
        wobble = int(round(2.0 * math.sin(t * 6.0)))
        cv2.line(frame, (x, y), (x + wobble, height - 1), level, spec.stream_width_px)
        return

    if spec.stream_break_s <= t <= spec.last_drop_s:
        # One drop every drop_period_s, each visible for the first 40% of its
        # period as it falls away from the orifice.
        phase = (t - spec.stream_break_s) % spec.drop_period_s
        if phase < 0.4 * spec.drop_period_s:
            fall = int(round(700.0 * phase))
            cv2.circle(
                frame,
                (x, min(height - spec.drop_radius_px - 1, y + 18 + fall)),
                spec.drop_radius_px,
                level,
                -1,
            )


def render(spec: ClipSpec, path: Path) -> dict:
    """Write the clip and return its ground truth."""
    rng = np.random.default_rng(20260820)
    world = _world_background(spec, rng)

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), spec.fps, (spec.width, spec.height)
    )
    if not writer.isOpened():
        raise RuntimeError("this OpenCV build cannot write MP4")

    outlet_track: list[list[float]] = []
    for index in range(spec.total_frames):
        t = index / spec.fps
        cx, cy = camera_offset(spec, t)
        x0, y0 = int(round(MARGIN + cx)), int(round(MARGIN + cy))
        frame = world[y0 : y0 + spec.height, x0 : x0 + spec.width].copy()

        ox, oy = outlet_at(spec, t)
        hidden = bool(
            spec.occlusion_s is not None and spec.occlusion_s[0] <= t < spec.occlusion_s[1]
        )
        if hidden:
            # The operator swings the phone away: the whole cup leaves frame.
            ox += spec.width
        else:
            _draw_cup(frame, spec, ox, oy)
            _draw_liquid(frame, spec, t, ox, oy)

        frame += rng.normal(0.0, spec.sensor_noise, frame.shape).astype(np.float32)
        bgr = cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        writer.write(bgr)
        outlet_track.append([round(t, 4), round(ox, 2), round(oy, 2), float(hidden)])
    writer.release()

    truth = {
        "clip": path.name,
        "spec": dict(asdict(spec)),
        "flow_start_s": spec.flow_start_s,
        "stream_break_s": spec.stream_break_s,
        "flow_end_s": spec.last_drop_s,
        "efflux_s": round(spec.last_drop_s - spec.flow_start_s, 4),
        "reference_frame_s": 0.0,
        "outlet_at_reference": [round(v, 2) for v in outlet_at(spec, 0.0)],
        "outlet_track": outlet_track,
    }
    path.with_suffix(".truth.json").write_text(json.dumps(truth, indent=2))
    return truth


# --------------------------------------------------------------------------- #
# The ablation set
# --------------------------------------------------------------------------- #

CLIPS: dict[str, ClipSpec] = {
    # The reported field condition: everything moves, everything is white.
    "handheld_white": ClipSpec(name="handheld_white"),
    # One variable removed at a time, against the same timeline.
    "static_white": ClipSpec(name="static_white", camera_drift_px=0.0, hand_drift_px=0.0,
                             tremor_px=0.0),
    "handheld_contrast": ClipSpec(name="handheld_contrast", cup_delta=90, stream_delta=110),
    "static_contrast": ClipSpec(name="static_contrast", camera_drift_px=0.0, hand_drift_px=0.0,
                                tremor_px=0.0, cup_delta=90, stream_delta=110),
    # Camera moves but the cup is on a stand: isolates global motion.
    "camera_only_white": ClipSpec(name="camera_only_white", hand_drift_px=0.0),
    # Cup moves but the camera is on a tripod: isolates ROI drift.
    "hand_only_white": ClipSpec(name="hand_only_white", camera_drift_px=0.0, tremor_px=0.0),
    # No tail drops at all: the stream simply stops.  Isolates the tail rule.
    "handheld_no_drops": ClipSpec(name="handheld_no_drops", stream_break_s=19.0,
                                  last_drop_s=19.0),
    # The true break happens while the outlet is out of frame (Stage 1 case).
    "handheld_gap_at_break": ClipSpec(name="handheld_gap_at_break",
                                      occlusion_s=(18.4, 21.2)),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "clips")
    parser.add_argument("--only", nargs="*", help="render only these clips")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    names = args.only or list(CLIPS)
    for name in names:
        spec = CLIPS[name]
        truth = render(spec, args.out / f"{name}.mp4")
        print(
            f"{name:22s} efflux={truth['efflux_s']:.2f}s "
            f"start={truth['flow_start_s']:.2f} end={truth['flow_end_s']:.2f} "
            f"outlet@0s={truth['outlet_at_reference']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
