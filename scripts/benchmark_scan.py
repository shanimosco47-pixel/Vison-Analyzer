"""Measure coarse-scan throughput and memory on a long recording.

Used to answer the only question that matters for multi-hour footage: how long
does a scan take, and does memory stay flat?  Run it against a real plant
recording when you have one; without one it can generate a synthetic recording
of any length.

Examples::

    # generate a 10 minute 720p recording and scan it
    python -m scripts.benchmark_scan --generate 600 --width 1280 --height 720

    # scan a real file
    python -m scripts.benchmark_scan --video /path/to/shift.mp4 --shortest-event 15
"""

from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from app.analysis.coarse_scan import plan_coarse_scan, run_coarse_scan
from app.analysis.motion_detector import MotionActivityScorer
from app.config import CoarseScanConfig
from app.logging_setup import configure_logging
from app.video.metadata import probe_video
from app.video.reader import VideoReader


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in megabytes."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # getrusage reports kilobytes on Linux and bytes on macOS.
    divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    return usage / divisor


def generate_video(path: Path, seconds: float, width: int, height: int, fps: float) -> Path:
    """A long, mostly-idle recording with a short burst of activity every 5 min."""
    rng = np.random.default_rng(7)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("This OpenCV build cannot write MP4 files.")

    background = np.clip(rng.normal(120.0, 2.0, (height, width, 3)), 0, 255).astype(np.uint8)
    total = int(seconds * fps)
    for index in range(total):
        timestamp = index / fps
        frame = background.copy()
        cv2.randn(frame, 0, 3)  # per-frame sensor noise
        frame = cv2.add(background, frame)
        # 20 s of machine movement every 5 minutes.
        phase = timestamp % 300.0
        if phase < 20.0:
            x = int(0.1 * width + (phase / 20.0) * 0.6 * width)
            cv2.rectangle(
                frame,
                (x, height // 3),
                (x + width // 12, height // 3 + height // 5),
                (235, 235, 235),
                -1,
            )
        writer.write(frame)
        if index % (int(fps) * 60) == 0:
            print(f"  generated {timestamp / 60:.0f} min of {seconds / 60:.0f}", flush=True)
    writer.release()
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, help="Existing video to scan")
    parser.add_argument("--generate", type=float, help="Generate a video of N seconds instead")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--shortest-event", type=float, default=15.0)
    parser.add_argument("--out", type=Path, default=Path("benchmark.mp4"))
    args = parser.parse_args()

    configure_logging("INFO")

    if args.video:
        path = args.video
    elif args.generate:
        print(f"Generating {args.generate / 60:.1f} min of {args.width}x{args.height} video...")
        started = time.monotonic()
        path = generate_video(args.out, args.generate, args.width, args.height, args.fps)
        print(
            f"  written in {time.monotonic() - started:.0f} s ({path.stat().st_size / 1e6:.0f} MB)"
        )
    else:
        parser.error("Pass either --video or --generate")

    info = probe_video(path)
    config = CoarseScanConfig(shortest_event_s=args.shortest_event)
    plan = plan_coarse_scan(info, config)
    rss_before = peak_rss_mb()

    started = time.monotonic()
    with VideoReader(path, info) as reader:
        result = run_coarse_scan(reader, MotionActivityScorer(), config)
    elapsed = time.monotonic() - started

    duration = info.duration_s or 0.0
    total_frames = info.frame_count or int(duration * info.fps)
    print("\n=== Coarse scan benchmark ===")
    print(
        f"video            : {info.width}x{info.height} @ {info.fps:.2f} fps, "
        f"{duration / 60:.1f} min ({total_frames} frames)"
    )
    print(
        f"sampling         : every {plan.effective_interval_s:.2f} s "
        f"({plan.step_frames} frames), scale {plan.scale:.3f}"
    )
    print(
        f"frames decoded   : {reader.stats.frames_decoded} "
        f"({reader.stats.frames_decoded / max(1, total_frames):.2%} of the recording)"
    )
    print(
        f"scan time        : {elapsed:.1f} s "
        f"({duration / max(elapsed, 1e-9):.0f}x faster than real time)"
    )
    print(f"candidates       : {len(result.candidates)}")
    print(f"peak memory      : {peak_rss_mb():.0f} MB (before scan {rss_before:.0f} MB)")
    if duration:
        print(f"extrapolated 12 h: {elapsed * (12 * 3600) / duration / 60:.1f} min of scanning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
