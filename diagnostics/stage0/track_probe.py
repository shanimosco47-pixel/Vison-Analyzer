"""Stage 0 feasibility probe for Stage 1 outlet tracking.

`opencv-python-headless` (this project's pinned dependency) ships no CSRT,
KCF or MOSSE tracker - only MIL, plus the classical building blocks.  This
probe measures whether the classical route is good enough before Stage 1
commits to it:

    goodFeaturesToTrack over the cup  ->  pyramidal Lucas-Kanade
    ->  estimateAffinePartial2D(RANSAC)  ->  outlet carried as a rigid offset

Error is the Euclidean distance between the predicted outlet and the
ground-truth outlet that the clip generator drew.  Nothing here is production
code; it exists to justify (or rule out) the Stage 1 method.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).parent
LK = {
    "winSize": (21, 21),
    "maxLevel": 3,
    "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
}


def cup_box(ox: float, oy: float, w: int, h: int) -> tuple[int, int, int, int]:
    """Where the cup is, relative to the clicked outlet: mostly above it."""
    x0 = max(0, int(ox - 0.28 * w))
    x1 = min(w, int(ox + 0.28 * w))
    y0 = max(0, int(oy - 0.34 * h))
    y1 = min(h, int(oy + 0.04 * h))
    return x0, y0, x1, y1


def probe(clip: Path, min_features: int = 24) -> dict:
    truth = json.loads(clip.with_suffix(".truth.json").read_text())
    track = {round(r[0], 4): (r[1], r[2], bool(r[3])) for r in truth["outlet_track"]}
    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("empty clip")

    prev = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = prev.shape
    ox, oy = truth["outlet_at_reference"]
    outlet = np.array([ox, oy], dtype=np.float64)

    def detect(gray, at):
        x0, y0, x1, y1 = cup_box(at[0], at[1], w, h)
        mask = np.zeros_like(gray)
        mask[y0:y1, x0:x1] = 255
        return cv2.goodFeaturesToTrack(
            gray, maxCorners=120, qualityLevel=0.01, minDistance=5, mask=mask, blockSize=5
        )

    pts = detect(prev, outlet)
    errors, states, redetects = [], [], 0
    index = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        t = round(index / fps, 4)
        gx, gy, hidden = track.get(t, (None, None, False))

        state = "lost"
        if pts is not None and len(pts) >= 6:
            nxt, status, err = cv2.calcOpticalFlowPyrLK(prev, gray, pts, None, **LK)
            back, status2, _ = cv2.calcOpticalFlowPyrLK(gray, prev, nxt, None, **LK)
            good = (status.ravel() == 1) & (status2.ravel() == 1)
            good &= np.linalg.norm(back - pts, axis=2).ravel() < 1.0   # forward-backward check
            a, b = pts[good], nxt[good]
            if len(a) >= 6:
                matrix, inliers = cv2.estimateAffinePartial2D(
                    a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000
                )
                if matrix is not None and inliers is not None and int(inliers.sum()) >= 6:
                    outlet = (matrix @ np.array([outlet[0], outlet[1], 1.0]))[:2]
                    pts = b[inliers.ravel() == 1].reshape(-1, 1, 2)
                    state = "tracked"
        if state == "tracked" and len(pts) < min_features:
            fresh = detect(gray, outlet)
            if fresh is not None and len(fresh) >= min_features:
                pts, redetects = fresh, redetects + 1
        prev = gray
        states.append(state)
        if gx is not None and not hidden and state == "tracked":
            errors.append(float(np.hypot(outlet[0] - gx, outlet[1] - gy)))
        index += 1
    cap.release()

    arr = np.array(errors) if errors else np.array([0.0])
    return {
        "clip": clip.stem,
        "frames": len(states),
        "tracked_pct": round(100 * states.count("tracked") / max(1, len(states)), 1),
        "redetections": redetects,
        "err_mean_px": round(float(arr.mean()), 2),
        "err_p95_px": round(float(np.percentile(arr, 95)), 2),
        "err_max_px": round(float(arr.max()), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=HERE / "clips")
    args = parser.parse_args()
    rows = []
    print(f"{'clip':24s} {'frames':>7s} {'tracked%':>9s} {'redet':>6s} "
          f"{'mean_px':>8s} {'p95_px':>7s} {'max_px':>7s}")
    for clip in sorted(args.clips.glob("*.mp4")):
        r = probe(clip)
        rows.append(r)
        print(f"{r['clip']:24s} {r['frames']:7d} {r['tracked_pct']:9.1f} {r['redetections']:6d} "
              f"{r['err_mean_px']:8.2f} {r['err_p95_px']:7.2f} {r['err_max_px']:7.2f}")
    (HERE / "track_probe.json").write_text(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
