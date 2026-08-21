"""Stage 0 separating experiment: fixed-but-wider ROI.

The `tracked` ablation re-cuts the ROI around the outlet every frame, which
changes two things at once - the region regains the outlet, *and* its contents
stop moving, so the scorer's frozen background model stays valid.  This script
changes only the first: the ROI stays fixed but is enlarged to contain the
whole drift envelope.  If that alone restores the timing, ROI coverage is the
cause; if it does not, the background model going stale under motion is.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.analysis.zahn_detector import ZahnCupDetector
from app.config import ROI
from app.video.metadata import probe_video
from app.video.reader import VideoReader

HERE = Path(__file__).parent


def _fmt(value: float | None) -> str:
    return "   None" if value is None else f"{value:7.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=HERE / "clips")
    parser.add_argument("--pad", type=int, default=40, help="pixels of slack around the drift")
    args = parser.parse_args()

    out = []
    print(f"{'clip':24s} {'roi':>22s} {'start':>7s} {'end':>7s} {'efflux':>7s} {'error':>7s} conf")
    for clip in sorted(args.clips.glob("*.mp4")):
        truth = json.loads(clip.with_suffix(".truth.json").read_text())
        info = probe_video(clip)
        ox, oy = truth["outlet_at_reference"]
        base = ZahnCupDetector(info, {"outlet": {"x": round(ox), "y": round(oy)}}).roi
        wide = ROI(
            x=base.x - args.pad,
            y=base.y - args.pad,
            width=base.width + 2 * args.pad,
            height=base.height + args.pad,
        ).clipped_to(info.width, info.height)

        detector = ZahnCupDetector(info, {"roi": wide.to_dict()})
        with VideoReader(clip, info) as reader:
            summary = detector.run(reader).summary
        err = (
            None
            if summary["efflux_seconds"] is None
            else round(summary["efflux_seconds"] - truth["efflux_s"], 3)
        )
        out.append({"clip": clip.stem, "roi": wide.to_dict(), **summary, "error_s": err})
        shape = f"{wide.width}x{wide.height}"
        print(
            f"{clip.stem:24s} {shape:>22s} "
            f"{_fmt(summary['flow_start_s'])} {_fmt(summary['flow_end_s'])} "
            f"{_fmt(summary['efflux_seconds'])} {_fmt(err)} {summary['status']}"
        )
    (HERE / "widen_roi.json").write_text(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
