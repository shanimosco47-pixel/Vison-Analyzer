"""Streaming video reader.

Responsibilities:

*   decode only the frames a stage actually asked for;
*   choose between *grabbing* (cheap sequential skip) and *seeking* (jump)
    depending on how far the next sample is;
*   crop and downscale before anything else touches the pixels;
*   survive dropped or corrupt frames instead of aborting the whole analysis;
*   never hold more than one frame at a time, so memory stays flat regardless
    of whether the recording is 10 seconds or 12 hours.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

import cv2
import numpy as np

from ..config import ROI
from ..errors import EmptyVideoError, VideoOpenError
from ..logging_setup import get_logger
from .metadata import VideoInfo, probe_video
from .sampling import SamplingPlan, frame_index_at, frames_to_seconds

logger = get_logger(__name__)

# Skipping frames with grab() avoids the (expensive) colour conversion of a
# full decode but still walks the stream frame by frame.  Beyond this many
# frames a keyframe seek is cheaper, even accounting for seek overhead.
GRAB_SKIP_LIMIT = 90

# A handful of unreadable frames in the middle of a long recording is normal
# for surveillance exports; a long run of them means the file is truncated.
MAX_CONSECUTIVE_READ_FAILURES = 30


@dataclass(frozen=True)
class FrameSample:
    """One frame handed to a scorer.

    ``image`` may be cropped to an ROI, downscaled and/or converted to
    greyscale - whatever the requesting stage asked for.  ``scale`` records the
    downscale factor so diagnostics can be mapped back to source pixels.
    """

    index: int
    timestamp_s: float
    image: np.ndarray
    scale: float = 1.0


@dataclass
class ReaderStats:
    """Bookkeeping used for logging and for confidence estimation."""

    frames_decoded: int = 0
    frames_skipped: int = 0
    read_failures: int = 0
    seeks: int = 0
    extra: dict[str, float] = field(default_factory=dict)


class VideoReader:
    """A seekable, streaming reader around :class:`cv2.VideoCapture`."""

    def __init__(self, path: Path | str, info: VideoInfo | None = None) -> None:
        self.path = Path(path)
        self.info = info or probe_video(self.path)
        self.stats = ReaderStats()
        self._capture: cv2.VideoCapture | None = None
        self._position: int = 0  # index of the next frame to be read

    # -- lifecycle --------------------------------------------------------- #

    def open(self) -> VideoReader:
        if self._capture is None:
            capture = cv2.VideoCapture(str(self.path))
            if not capture.isOpened():
                raise VideoOpenError(detail=f"reopen failed for {self.path.name}")
            self._capture = capture
            self._position = 0
        return self

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> VideoReader:
        return self.open()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def capture(self) -> cv2.VideoCapture:
        if self._capture is None:
            self.open()
        assert self._capture is not None  # for type checkers; open() guarantees it
        return self._capture

    # -- positioning ------------------------------------------------------- #

    def _seek(self, frame_index: int) -> None:
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
        self._position = frame_index
        self.stats.seeks += 1

    def _advance_to(self, target_index: int) -> bool:
        """Move the read head to ``target_index``. Returns False at end of file."""
        if target_index < self._position:
            self._seek(target_index)
            return True

        gap = target_index - self._position
        if gap > GRAB_SKIP_LIMIT:
            self._seek(target_index)
            return True

        for _ in range(gap):
            if not self.capture.grab():
                return False
            self._position += 1
            self.stats.frames_skipped += 1
        return True

    # -- reading ----------------------------------------------------------- #

    def _read_next(self) -> np.ndarray | None:
        ok, frame = self.capture.read()
        if ok and frame is not None:
            self._position += 1
            self.stats.frames_decoded += 1
            return frame
        self._position += 1
        self.stats.read_failures += 1
        return None

    def frame_at(self, timestamp_s: float) -> np.ndarray:
        """Decode a single full-resolution BGR frame at ``timestamp_s``.

        Used for UI previews and diagnostics, not in analysis loops.
        """
        index = frame_index_at(timestamp_s, self.info.fps, self.info.frame_count)
        self._seek(index)
        for _ in range(MAX_CONSECUTIVE_READ_FAILURES):
            frame = self._read_next()
            if frame is not None:
                return frame
        raise EmptyVideoError(
            "No readable frame was found at that position in the video.",
            detail=f"timestamp={timestamp_s:.3f}s index={index}",
        )

    def iter_samples(
        self,
        plan: SamplingPlan,
        *,
        roi: ROI | None = None,
        grayscale: bool = True,
        blur_kernel: int = 0,
    ) -> Iterator[FrameSample]:
        """Yield frames according to ``plan``.

        Args:
            plan: which range, step and scale to use.
            roi: optional crop in *source* coordinates, applied before scaling.
            grayscale: convert to single channel (most scorers only need luma).
            blur_kernel: optional odd Gaussian kernel applied after scaling;
                cheap noise suppression that keeps thresholds stable.
        """
        self.open()
        if roi is not None:
            roi = roi.clipped_to(self.info.width, self.info.height)

        start_index = frame_index_at(plan.start_s, self.info.fps, self.info.frame_count)
        last_index = self._last_index(plan)

        index = start_index
        consecutive_failures = 0
        while index <= last_index:
            if not self._advance_to(index):
                break
            frame = self._read_next()
            if frame is None:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_READ_FAILURES:
                    logger.warning(
                        "Stopping at frame %d after %d consecutive unreadable frames in %s",
                        index,
                        consecutive_failures,
                        self.path.name,
                    )
                    break
                index += 1  # step over the bad frame rather than the whole interval
                continue
            consecutive_failures = 0

            image = _prepare(
                frame, roi=roi, scale=plan.scale, grayscale=grayscale, blur_kernel=blur_kernel
            )
            yield FrameSample(
                index=index,
                timestamp_s=frames_to_seconds(index, self.info.fps),
                image=image,
                scale=plan.scale,
            )
            index += plan.step_frames

    def _last_index(self, plan: SamplingPlan) -> int:
        """Highest frame index the plan may touch."""
        if self.info.frame_count:
            return min(
                frame_index_at(plan.end_s, self.info.fps, self.info.frame_count),
                self.info.frame_count - 1,
            )
        if self.info.duration_s is not None:
            return frame_index_at(min(plan.end_s, self.info.duration_s), self.info.fps)
        # Unknown length: rely on read failures to terminate the loop.
        return frame_index_at(plan.end_s, self.info.fps) if plan.end_s > 0 else 1 << 30


def _prepare(
    frame: np.ndarray,
    *,
    roi: ROI | None,
    scale: float,
    grayscale: bool,
    blur_kernel: int,
) -> np.ndarray:
    """Crop, downscale, grey and blur - in the order that moves fewest bytes."""
    if roi is not None:
        # Slicing is a view; the later resize/cvtColor produces the only copy.
        frame = frame[roi.y : roi.y2, roi.x : roi.x2]

    if scale < 1.0:
        height, width = frame.shape[:2]
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    if grayscale and frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if blur_kernel and blur_kernel >= 3:
        kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        frame = cv2.GaussianBlur(frame, (kernel, kernel), 0)

    return frame


def encode_jpeg(frame: np.ndarray, quality: int = 85) -> bytes:
    """Encode a BGR frame as JPEG bytes (used for UI previews)."""
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise EmptyVideoError("The video frame could not be converted to an image.")
    return buffer.tobytes()
