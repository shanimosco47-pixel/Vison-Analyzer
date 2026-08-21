"""Stage 0: geometry statistics and annotated diagnostic frames.

For each clip this reports how often the ground-truth outlet has drifted out
of the fixed ROI and out of the outlet band, and writes annotated frames for
the four situations the diagnosis has to show: continuous flow, the true
stream break, a false positive after the break, and a frame at maximum
cup/camera displacement.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.analysis.zahn_detector import (
    ANALYSIS_WIDTH_PX,
    StreamActivityScorer,
    ZahnCupDetector,
)
from app.video.metadata import probe_video
from app.video.reader import FrameSample
from app.video.sampling import frames_to_seconds, scale_factor_for_width

HERE = Path(__file__).parent
GREEN, RED, BLUE, YELLOW, MAGENTA = (
    (0, 200, 0),
    (0, 0, 255),
    (255, 120, 0),
    (0, 200, 255),
    (255, 0, 255),
)


def _trace(name: str) -> dict[float, dict[str, float]]:
    path = HERE / f"trace_{name}.csv"
    if not path.exists():
        return {}
    with path.open() as handle:
        return {
            round(float(r["t"]), 4): {k: float(v) for k, v in r.items()}
            for r in csv.DictReader(handle)
        }


def analyse(clip: Path, out_dir: Path) -> dict[str, Any]:
    truth = json.loads(clip.with_suffix(".truth.json").read_text())
    info = probe_video(clip)
    ox0, oy0 = truth["outlet_at_reference"]
    detector = ZahnCupDetector(info, {"outlet": {"x": round(ox0), "y": round(oy0)}})
    roi, guard, config = detector.roi, detector.guard_roi, detector.config
    scale = scale_factor_for_width(guard.width, ANALYSIS_WIDTH_PX)
    scorer = StreamActivityScorer(config, detector._inner_rect(scale), keep_mask=True)

    band_px = max(1, round(roi.height * config.outlet_band_fraction))
    trace = _trace(clip.stem)
    track = {round(r[0], 4): (r[1], r[2], bool(r[3])) for r in truth["outlet_track"]}

    outside_roi = outside_band = total = 0
    offsets: list[float] = []
    wanted: dict[str, float] = {
        "continuous_flow": (truth["flow_start_s"] + truth["stream_break_s"]) / 2,
        "true_break": truth["stream_break_s"],
        "true_end": truth["flow_end_s"],
    }
    # A false positive after the true end, and the frame of biggest displacement.
    fps_after = [
        t for t, r in sorted(trace.items()) if t > truth["flow_end_s"] + 0.05 and r["liquid"] >= 0.5
    ]
    if fps_after:
        wanted["false_positive_after_break"] = fps_after[len(fps_after) // 2]
        wanted["last_false_positive"] = fps_after[-1]
    best_t, best_off = 0.0, -1.0
    for t, (gx, gy, hidden) in track.items():
        if hidden:
            continue
        off = float(np.hypot(gx - ox0, gy - oy0))
        if off > best_off:
            best_t, best_off = t, off
    wanted["max_displacement"] = best_t

    targets = {round(v, 4): k for k, v in wanted.items()}
    capture = cv2.VideoCapture(str(clip))
    index = 0
    written: list[str] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            t = round(frames_to_seconds(index, info.fps), 4)
            gx, gy, hidden = track.get(t, (ox0, oy0, False))
            if not hidden:
                total += 1
                offsets.append(float(np.hypot(gx - ox0, gy - oy0)))
                if not (roi.x <= gx < roi.x2 and roi.y <= gy < roi.y2):
                    outside_roi += 1
                elif gy >= roi.y + band_px:
                    outside_band += 1

            # Score every frame so the scorer state matches the production run
            # exactly at the moment a diagnostic frame is written.
            crop = frame[guard.y : guard.y2, guard.x : guard.x2]
            h, w = crop.shape[:2]
            small = cv2.resize(
                crop,
                (max(1, round(w * scale)), max(1, round(h * scale))),
                interpolation=cv2.INTER_AREA,
            )
            small = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
            scored = scorer.score(FrameSample(index=index, timestamp_s=t, image=small, scale=scale))

            label = targets.get(t)
            if label:
                written.append(
                    _write(
                        out_dir,
                        clip.stem,
                        label,
                        t,
                        index,
                        frame,
                        roi,
                        guard,
                        band_px,
                        gx,
                        gy,
                        hidden,
                        scored,
                        trace.get(t, {}),
                    )
                )
            index += 1
    finally:
        capture.release()

    return {
        "clip": clip.stem,
        "roi": roi.to_dict(),
        "guard_roi": guard.to_dict(),
        "outlet_band_px": band_px,
        "outlet_at_reference": [ox0, oy0],
        "frames_visible": total,
        "max_outlet_offset_px": round(max(offsets), 1) if offsets else 0.0,
        "mean_outlet_offset_px": round(sum(offsets) / len(offsets), 1) if offsets else 0.0,
        "pct_outlet_outside_roi": round(100 * outside_roi / total, 1) if total else 0.0,
        "pct_outlet_below_outlet_band": round(100 * outside_band / total, 1) if total else 0.0,
        "frames_written": written,
    }


def _write(out_dir, stem, label, t, index, frame, roi, guard, band_px, gx, gy, hidden, scored, row):
    canvas = frame.copy()
    cv2.rectangle(canvas, (guard.x, guard.y), (guard.x2, guard.y2), BLUE, 1)
    cv2.rectangle(canvas, (roi.x, roi.y), (roi.x2, roi.y2), GREEN, 2)
    cv2.rectangle(canvas, (roi.x, roi.y), (roi.x2, roi.y + band_px), YELLOW, 1)
    if not hidden:
        cv2.drawMarker(canvas, (int(round(gx)), int(round(gy))), RED, cv2.MARKER_CROSS, 18, 2)
    cv2.drawMarker(canvas, (roi.x + roi.width // 2, roi.y), MAGENTA, cv2.MARKER_TILTED_CROSS, 14, 1)

    if scored.mask is not None and scored.mask.any():
        mask = cv2.resize(
            scored.mask * 255, (roi.width, roi.height), interpolation=cv2.INTER_NEAREST
        )
        patch = canvas[roi.y : roi.y2, roi.x : roi.x2]
        patch[mask > 0] = (0, 0, 255)

    lines = [
        f"{stem}  {label}",
        f"t={t:.3f}s  frame={index}",
        "green=fixed ROI  yellow=outlet band  blue=guard",
        "red cross=true outlet  magenta=marked outlet",
        f"outlet drift="
        f"{0.0 if hidden else np.hypot(gx - (roi.x + roi.width / 2), gy - roi.y):.1f}px"
        + ("  OUTLET OUT OF FRAME" if hidden else ""),
        f"activity={row.get('activity', float('nan')):.3f}"
        f" outlet={row.get('outlet', float('nan')):.3f}"
        f" liquid={int(row.get('liquid', 0))} disturbed={int(row.get('disturbed', 0))}",
    ]
    for i, text in enumerate(lines):
        y = 18 + 16 * i
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3)
        cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

    name = f"{stem}__{label}__t{t:07.3f}.png"
    cv2.imwrite(str(out_dir / name), canvas)
    return name


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=HERE / "clips")
    parser.add_argument("--out", type=Path, default=HERE / "frames")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    report = []
    print(f"{'clip':24s} {'maxoff':>7s} {'meanoff':>8s} {'%outROI':>8s} {'%belowBand':>11s}")
    for clip in sorted(args.clips.glob("*.mp4")):
        stats = analyse(clip, args.out)
        report.append(stats)
        print(
            f"{stats['clip']:24s} {stats['max_outlet_offset_px']:7.1f} "
            f"{stats['mean_outlet_offset_px']:8.1f} {stats['pct_outlet_outside_roi']:8.1f} "
            f"{stats['pct_outlet_below_outlet_band']:11.1f}"
        )
    (HERE / "geometry.json").write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
