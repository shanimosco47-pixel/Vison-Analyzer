"""Author the synthetic frame-timing fixtures in ``tests/fixtures/``.

**This script is not needed to run the test suite.** The fixtures it produces
are committed, and the tests open them with the OpenCV the project already
depends on. It is kept as the record of how those files were made, and to
regenerate or extend them.

Running it needs PyAV, which is deliberately **not** a project dependency —
nothing in ``app/`` or ``tests/`` imports it. Install it into a throwaway
environment:

    python3 -m venv /tmp/fixgen && /tmp/fixgen/bin/pip install av numpy
    /tmp/fixgen/bin/python scripts/make_vfr_fixtures.py tests/fixtures

Why author files at all: OpenCV's writer emits constant-rate containers only,
so a genuinely variable-rate recording cannot be produced with the project's own
dependencies. Every earlier round of the frame-timing work was therefore
validated against scripted timestamp lists, which is exactly how three
successive versions of the detector shipped with the same defect intact. These
files put real containers, real seeks and real ``CAP_PROP_POS_MSEC`` values in
front of the detector.

The content is synthetic — a bar moving across a gradient. Only the *timing*
mirrors a real recording measured during review: a 600/19 base tick with 34 of
its 638 intervals doubled, which a container reports as a plausible 29.98 fps
average.
"""

from __future__ import annotations

import sys
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

WIDTH, HEIGHT = 1280, 720
FRAMES = 639
DOUBLED = 34
TIME_BASE = Fraction(1, 12000)
STEP_TICKS = 380  # 380/12000 s = 31.6667 ms, i.e. the 600/19 fps base tick


def _ticks(doubled_at: set[int], *, first_tick: int = 0) -> list[int]:
    ticks = [first_tick]
    for index in range(1, FRAMES):
        ticks.append(ticks[-1] + (2 * STEP_TICKS if index in doubled_at else STEP_TICKS))
    return ticks


def distributed_ticks() -> list[int]:
    """Doubled intervals spread evenly, so every sampling window sees a few."""
    spacing = (FRAMES - 1) / (DOUBLED + 1)
    return _ticks({round(i * spacing) for i in range(1, DOUBLED + 1)})


def clustered_ticks() -> list[int]:
    """Doubled intervals in one early burst, entirely between sampling windows.

    The windows start at frames 0, 150, 300, 449 and 599 and cover 40 frames
    each, so frames 60-126 are never sampled. Every window is internally
    perfect and every median identical; only the accumulated offset betrays it.
    """
    return _ticks(set(range(60, 60 + 2 * DOUBLED, 2)))


def late_start_ticks() -> list[int]:
    """Perfectly constant cadence whose container starts five seconds in.

    A recording that simply begins at a non-zero timestamp must not be mistaken
    for drift. Note that OpenCV normalises the container's start time away, so
    what reaches the detector is a baseline of zero plus the small linear ramp
    caused by the declared average fps differing from the true tick - real
    error, correctly charged, and comfortably inside the budget.
    """
    return _ticks(set(), first_tick=5 * 12000)


def _image(index: int) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    image[:, :, 1] = np.linspace(20, 90, WIDTH, dtype=np.uint8)[None, :]
    x = int((index / FRAMES) * (WIDTH - 120))
    image[HEIGHT // 3 : 2 * HEIGHT // 3, x : x + 120] = (240, 240, 240)
    return image


def write(path: Path, ticks: list[int]) -> None:
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=30)
    stream.width, stream.height = WIDTH, HEIGHT
    stream.pix_fmt = "yuv420p"
    stream.time_base = TIME_BASE
    stream.codec_context.time_base = TIME_BASE
    stream.options = {"crf": "28", "preset": "veryfast", "x264-params": "keyint=30"}
    for index, pts in enumerate(ticks):
        frame = av.VideoFrame.from_ndarray(_image(index), format="rgb24")
        frame.pts = pts
        frame.time_base = TIME_BASE
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    intervals = [
        (b - a) * float(TIME_BASE) * 1000.0 for a, b in zip(ticks, ticks[1:], strict=False)
    ]
    doubled = sum(1 for value in intervals if value > 45)
    print(
        f"{path.name}: {len(ticks)} frames, {len(intervals)} intervals, "
        f"{doubled} doubled, first PTS {ticks[0] * float(TIME_BASE):.3f} s"
    )


def main(directory: str) -> None:
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    write(out / "vfr_distributed.mp4", distributed_ticks())
    write(out / "vfr_clustered_burst.mp4", clustered_ticks())
    write(out / "cfr_late_start.mp4", late_start_ticks())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures")
