"""Stage 0 ablation: what does each candidate cause actually cost, in seconds?

Every variant below reuses the *production* scorer and state machine
(:class:`StreamActivityScorer`, :class:`FlowStateMachine`) unmodified.  Only
what is fed to them changes, so any difference in the reported timing is
attributable to that one change.

Variants
--------
fixed        the production path: one ROI, fixed at the reference frame.
tracked      an *oracle* tracker: the guard crop is re-cut around the
             ground-truth outlet position every frame.  This is the upper
             bound on what a perfect Stage 1 tracker can recover, and nothing
             else about the pipeline is altered.
tracked+bg   oracle tracking, plus the background model refreshed on a fixed
             cadence rather than only while the region is quiet.  Isolates the
             "frozen background goes stale under motion" effect from the
             "ROI loses the outlet" effect.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.analysis.zahn_detector import (
    ANALYSIS_WIDTH_PX,
    FlowStateMachine,
    StreamActivityScorer,
    ZahnCupDetector,
)
from app.config import ROI
from app.video.metadata import probe_video
from app.video.reader import FrameSample
from app.video.sampling import frames_to_seconds, scale_factor_for_width

HERE = Path(__file__).parent


def _truth(clip: Path) -> dict[str, Any]:
    return json.loads(clip.with_suffix(".truth.json").read_text())


def _prepare(frame: np.ndarray, roi: ROI, scale: float) -> np.ndarray:
    crop = frame[roi.y : roi.y2, roi.x : roi.x2]
    if scale < 1.0:
        h, w = crop.shape[:2]
        crop = cv2.resize(
            crop, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA
        )
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(crop, (3, 3), 0)


def run_variant(clip: Path, variant: str) -> dict[str, Any]:
    truth = _truth(clip)
    info = probe_video(clip)
    ox0, oy0 = truth["outlet_at_reference"]
    detector = ZahnCupDetector(info, {"outlet": {"x": round(ox0), "y": round(oy0)}})
    config, roi0, guard0 = detector.config, detector.roi, detector.guard_roi
    scale = scale_factor_for_width(guard0.width, ANALYSIS_WIDTH_PX)
    inner = detector._inner_rect(scale)
    scorer = StreamActivityScorer(config, inner, keep_mask=False)
    machine = FlowStateMachine(config)

    track = {round(row[0], 4): (row[1], row[2], bool(row[3])) for row in truth["outlet_track"]}
    tracking = variant in {"tracked", "tracked+bg"}
    refresh_bg = variant == "tracked+bg"

    capture = cv2.VideoCapture(str(clip))
    index, last_t, lost_frames = 0, 0.0, 0
    max_offset = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = frames_to_seconds(index, info.fps)
            gx, gy, hidden = track.get(round(t, 4), (ox0, oy0, False))
            dx, dy = (gx - ox0, gy - oy0) if tracking else (0.0, 0.0)
            max_offset = max(max_offset, float(np.hypot(gx - ox0, gy - oy0)))

            if tracking and hidden:
                # A perfect tracker still cannot see an outlet that is not in
                # frame.  Such frames are simply not scored - which is exactly
                # the `lost` state Stage 1 has to define.
                lost_frames += 1
                index += 1
                continue

            guard = ROI(
                x=int(round(guard0.x + dx)),
                y=int(round(guard0.y + dy)),
                width=guard0.width,
                height=guard0.height,
            ).clipped_to(info.width, info.height)
            if guard.width != guard0.width or guard.height != guard0.height:
                lost_frames += 1
                index += 1
                continue

            image = _prepare(frame, guard, scale)
            if refresh_bg and scorer._background is not None and index % 30 == 0:
                cv2.accumulateWeighted(image.astype(np.float32), scorer._background, 0.35)
            scored = scorer.score(FrameSample(index=index, timestamp_s=t, image=image, scale=scale))
            machine.update(t, scored)
            last_t = t
            index += 1
            if machine.finished:
                break
    finally:
        capture.release()

    m = machine.finalize(last_t)
    return {
        "clip": clip.stem,
        "variant": variant,
        "truth_start_s": truth["flow_start_s"],
        "truth_end_s": truth["flow_end_s"],
        "truth_efflux_s": truth["efflux_s"],
        "start_s": None if m.start_s is None else round(m.start_s, 3),
        "end_s": None if m.end_s is None else round(m.end_s, 3),
        "efflux_s": None if m.efflux_s is None else round(m.efflux_s, 3),
        "end_confirmed": m.end_confirmed,
        "error_s": None if m.efflux_s is None else round(m.efflux_s - truth["efflux_s"], 3),
        "frames_disturbed": m.frames_disturbed,
        "frames_analysed": m.frames_analysed,
        "frames_skipped_lost": lost_frames,
        "max_outlet_offset_px": round(max_offset, 1),
        "roi": roi0.to_dict(),
    }


def _fmt(value: float | None) -> str:
    return "   None" if value is None else f"{value:7.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=HERE / "clips")
    parser.add_argument("--out", type=Path, default=HERE / "ablation.json")
    args = parser.parse_args()

    results = []
    print(
        f"{'clip':24s} {'variant':11s} {'start':>7s} {'end':>7s} {'efflux':>7s} "
        f"{'error':>7s}  conf  lost  maxoff"
    )
    for clip in sorted(args.clips.glob("*.mp4")):
        for variant in ("fixed", "tracked", "tracked+bg"):
            row = run_variant(clip, variant)
            results.append(row)
            print(
                f"{row['clip']:24s} {variant:11s} {_fmt(row['start_s'])} {_fmt(row['end_s'])} "
                f"{_fmt(row['efflux_s'])} {_fmt(row['error_s'])}  "
                f"{str(row['end_confirmed'])[:5]:5s} {row['frames_skipped_lost']:4d} "
                f"{row['max_outlet_offset_px']:6.1f}"
            )
    args.out.write_text(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
