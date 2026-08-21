"""Stage 0: reproduce and attribute the Zahn "efflux too long" error.

Runs the *unmodified* :class:`app.analysis.zahn_detector.ZahnCupDetector` over
each ablation clip and reports detected vs. ground-truth start/end.  A second,
instrumented replay re-uses the same scorer and state-machine classes to
capture per-frame evidence; the replay's start/end are asserted to match the
unmodified run, so the instrumentation is proven not to change any decision.

Nothing in this file is imported by the application.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from app.analysis.zahn_detector import (
    ANALYSIS_WIDTH_PX,
    FlowStateMachine,
    StreamActivityScorer,
    ZahnCupDetector,
)
from app.video.metadata import probe_video
from app.video.reader import VideoReader
from app.video.sampling import build_sampling_plan, scale_factor_for_width

HERE = Path(__file__).parent


def _truth_for(clip: Path) -> dict[str, Any]:
    return json.loads(clip.with_suffix(".truth.json").read_text())


def run_detector(clip: Path, truth: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """The production path, untouched: what the app reports today."""
    info = probe_video(clip)
    ox, oy = truth["outlet_at_reference"]
    params: dict[str, Any] = {"outlet": {"x": int(round(ox)), "y": int(round(oy))}}
    params.update(overrides)
    detector = ZahnCupDetector(info, params)
    started = time.perf_counter()
    with VideoReader(clip, info) as reader:
        result = detector.run(reader)
    wall = time.perf_counter() - started
    summary = dict(result.summary)
    summary["wall_seconds"] = round(wall, 3)
    summary["_roi"] = detector.roi.to_dict()
    summary["_guard"] = detector.guard_roi.to_dict()
    return summary


def replay_instrumented(
    clip: Path, truth: dict[str, Any], overrides: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Same scorer, same state machine, plus a per-frame record."""
    info = probe_video(clip)
    ox, oy = truth["outlet_at_reference"]
    params = {"outlet": {"x": int(round(ox)), "y": int(round(oy))}, **overrides}
    detector = ZahnCupDetector(info, params)
    config = detector.config
    roi, guard = detector.roi, detector.guard_roi

    scale = scale_factor_for_width(guard.width, ANALYSIS_WIDTH_PX)
    plan = build_sampling_plan(
        fps=info.fps, duration_s=info.duration_s or 0.0, interval_s=1.0 / info.fps, scale=scale
    )
    inner = detector._inner_rect(scale)
    scorer = StreamActivityScorer(config, inner, keep_mask=False)
    machine = FlowStateMachine(config)

    rows: list[dict[str, Any]] = []
    last_t = 0.0
    with VideoReader(clip, info) as reader:
        for sample in reader.iter_samples(plan, roi=guard, grayscale=True, blur_kernel=3):
            scored = scorer.score(sample)
            was_flowing, was_done = machine.flowing, machine.finished
            machine.update(sample.timestamp_s, scored)
            rows.append(
                {
                    "frame": sample.index,
                    "t": round(sample.timestamp_s, 4),
                    "activity": round(float(scored.value), 5),
                    "outlet": round(float(scored.extras.get("outlet_score", 0.0)), 5),
                    "liquid": int(float(scored.extras.get("liquid_present", 0.0)) >= 0.5),
                    "disturbed": int(scored.disturbed),
                    "outside_ratio": round(
                        float(scored.extras.get("outside_changed_ratio", 0.0)), 5
                    ),
                    "blobs": round(float(scored.extras.get("blob_count", 0.0)), 1),
                    "contrast": round(float(scored.extras.get("stream_contrast", 0.0)), 3),
                    "noise_sigma": round(float(scored.extras.get("noise_sigma", 0.0)), 3),
                    "threshold": round(float(scored.extras.get("threshold", 0.0)), 3),
                    "flowing": int(machine.flowing),
                    "became_flowing": int(machine.flowing and not was_flowing),
                    "finished": int(machine.finished and not was_done),
                }
            )
            last_t = sample.timestamp_s
            if machine.finished:
                break
    measurement = machine.finalize(last_t)
    out = {
        "flow_start_s": measurement.start_s,
        "flow_end_s": measurement.end_s,
        "end_confirmed": measurement.end_confirmed,
        "efflux_seconds": measurement.efflux_s,
        "frames_disturbed": measurement.frames_disturbed,
        "frames_analysed": measurement.frames_analysed,
        "roi": roi.to_dict(),
        "guard_roi": guard.to_dict(),
    }
    return out, rows


def _near(a: float | None, b: float | None, tol: float = 1e-6) -> bool:
    if a is None or b is None:
        return a is b
    return abs(a - b) <= tol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clips", type=Path, default=HERE / "clips")
    parser.add_argument("--out", type=Path, default=HERE)
    parser.add_argument("--only", nargs="*")
    args = parser.parse_args()

    clips = sorted(args.clips.glob("*.mp4"))
    if args.only:
        clips = [c for c in clips if c.stem in set(args.only)]

    table: list[dict[str, Any]] = []
    for clip in clips:
        truth = _truth_for(clip)
        summary = run_detector(clip, truth, {})
        replay, rows = replay_instrumented(clip, truth, {})

        matches = _near(summary["flow_start_s"], replay["flow_start_s"]) and _near(
            summary["flow_end_s"], replay["flow_end_s"]
        )
        trace_path = args.out / f"trace_{clip.stem}.csv"
        with trace_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        row = {
            "clip": clip.stem,
            "truth_start_s": truth["flow_start_s"],
            "truth_break_s": truth["stream_break_s"],
            "truth_end_s": truth["flow_end_s"],
            "truth_efflux_s": truth["efflux_s"],
            "det_start_s": summary["flow_start_s"],
            "det_end_s": summary["flow_end_s"],
            "det_efflux_s": summary["efflux_seconds"],
            "end_confirmed": summary["end_confirmed"],
            "status": summary["status"],
            "confidence": summary["confidence"],
            "frames_disturbed": summary["frames_disturbed"],
            "frames_analysed": summary["frames_analysed"],
            "wall_seconds": summary["wall_seconds"],
            "instrumentation_identical": matches,
            "roi": summary["_roi"],
            "guard_roi": summary["_guard"],
        }
        if summary["efflux_seconds"] is not None:
            row["efflux_error_s"] = round(summary["efflux_seconds"] - truth["efflux_s"], 3)
            row["end_error_s"] = round((summary["flow_end_s"] or 0.0) - truth["flow_end_s"], 3)
            row["start_error_s"] = round(
                (summary["flow_start_s"] or 0.0) - truth["flow_start_s"], 3
            )
        table.append(row)

        print(
            f"{clip.stem:22s} truth {truth['flow_start_s']:6.2f}->{truth['flow_end_s']:6.2f} "
            f"({truth['efflux_s']:5.2f}s)   detected "
            f"{_fmt(summary['flow_start_s'])}->{_fmt(summary['flow_end_s'])} "
            f"({_fmt(summary['efflux_seconds'])}s)  "
            f"err={_fmt(row.get('efflux_error_s'))}s  "
            f"conf={summary['confidence']:.2f} {summary['status']:9s} "
            f"dist={summary['frames_disturbed']:4d}/{summary['frames_analysed']:4d} "
            f"instr_ok={matches}"
        )

    (args.out / "results.json").write_text(json.dumps(table, indent=2))
    return 0


def _fmt(value: float | None) -> str:
    return "  None" if value is None else f"{value:6.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
