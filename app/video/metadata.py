"""Video metadata probing.

Metadata from real-world files is frequently wrong or missing: surveillance
exports with no frame count, variable-frame-rate phone recordings, containers
whose reported FPS is 0.  Everything here therefore treats the container's
claims as a hypothesis to be checked, and records *how* each value was
obtained so the UI can warn the user when a value was estimated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from ..errors import EmptyVideoError, VideoMetadataError, VideoOpenError
from ..logging_setup import get_logger
from .sampling import MAX_PLAUSIBLE_FPS, MIN_PLAUSIBLE_FPS

logger = get_logger(__name__)

# Number of frames decoded when the container's FPS has to be re-derived from
# presentation timestamps.
FPS_ESTIMATION_FRAMES = 60


@dataclass(frozen=True)
class VideoInfo:
    """Everything the analysis layer needs to know about a source file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_s: float | None
    fourcc: str
    size_bytes: int
    fps_source: str  # "container" | "timestamps"
    duration_source: str  # "frame_count" | "timestamps" | "unknown"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_variable_frame_rate_suspect(self) -> bool:
        return self.fps_source == "timestamps"

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.path.name,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 4),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 3) if self.duration_s is not None else None,
            "fourcc": self.fourcc,
            "size_bytes": self.size_bytes,
            "fps_source": self.fps_source,
            "duration_source": self.duration_source,
            "warnings": list(self.warnings),
        }


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "unknown"
    chars = [chr((code >> shift) & 0xFF) for shift in (0, 8, 16, 24)]
    text = "".join(c for c in chars if c.isprintable()).strip()
    return text or "unknown"


def _estimate_fps_from_timestamps(capture: cv2.VideoCapture) -> float | None:
    """Derive FPS from presentation timestamps of the first frames.

    Returns ``None`` when the backend does not expose usable timestamps.
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    first_ms: float | None = None
    frames_read = 0
    last_ms = 0.0
    for _ in range(FPS_ESTIMATION_FRAMES):
        ok = capture.grab()
        if not ok:
            break
        position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        if not math.isfinite(position_ms) or position_ms <= 0:
            # Some backends report 0 for every frame; that is unusable.
            if frames_read > 0 and position_ms == last_ms:
                return None
            continue
        if first_ms is None:
            first_ms = position_ms
        last_ms = position_ms
        frames_read += 1

    if first_ms is None or frames_read < 5 or last_ms <= first_ms:
        return None
    elapsed_s = (last_ms - first_ms) / 1000.0
    fps = (frames_read - 1) / elapsed_s
    if not MIN_PLAUSIBLE_FPS <= fps <= MAX_PLAUSIBLE_FPS:
        return None
    return fps


def probe_video(path: Path) -> VideoInfo:
    """Open ``path`` and return validated metadata.

    Raises:
        VideoOpenError: the file cannot be opened/decoded at all.
        EmptyVideoError: the file opens but yields no frames.
        VideoMetadataError: the frame rate cannot be established.
    """
    path = Path(path)
    if not path.is_file():
        raise VideoOpenError("The video file could not be found.", detail=str(path))

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise EmptyVideoError("The uploaded file is empty.")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoOpenError(detail=f"cv2.VideoCapture could not open {path.name}")

    try:
        warnings: list[str] = []

        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise EmptyVideoError(detail=f"no decodable frames in {path.name}")
        height, width = first_frame.shape[:2]

        reported_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        reported_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if (reported_width, reported_height) != (width, height) and reported_width > 0:
            warnings.append(
                "The container reports a different resolution than the decoded frames; "
                "the decoded size is used."
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_source = "container"
        if not math.isfinite(fps) or not MIN_PLAUSIBLE_FPS <= fps <= MAX_PLAUSIBLE_FPS:
            logger.warning("Implausible container FPS %r for %s; estimating", fps, path.name)
            estimated = _estimate_fps_from_timestamps(capture)
            if estimated is None:
                raise VideoMetadataError(detail=f"container fps={fps!r}, estimation failed")
            fps = estimated
            fps_source = "timestamps"
            warnings.append(
                "The frame rate was estimated from timestamps because the file does not "
                "declare a usable one. Timings may be less accurate."
            )

        raw_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        has_frame_count = math.isfinite(raw_count) and raw_count > 0
        frame_count: int | None = int(raw_count) if has_frame_count else None

        duration_s: float | None
        if frame_count:
            duration_s = frame_count / fps
            duration_source = "frame_count"
        else:
            duration_s = _duration_from_end_timestamp(capture)
            duration_source = "timestamps" if duration_s is not None else "unknown"
            if duration_s is None:
                warnings.append(
                    "The video length is unknown; progress reporting will be approximate."
                )

        info = VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=duration_s,
            fourcc=_fourcc_to_str(capture.get(cv2.CAP_PROP_FOURCC)),
            size_bytes=size_bytes,
            fps_source=fps_source,
            duration_source=duration_source,
            warnings=tuple(warnings),
        )
        logger.info(
            "Probed %s: %dx%d @ %.3f fps (%s), %s frames, duration %s s, codec %s, %.1f MB",
            path.name,
            info.width,
            info.height,
            info.fps,
            info.fps_source,
            info.frame_count,
            f"{info.duration_s:.2f}" if info.duration_s else "unknown",
            info.fourcc,
            size_bytes / 1e6,
        )
        return info
    finally:
        capture.release()


def _duration_from_end_timestamp(capture: cv2.VideoCapture) -> float | None:
    """Best-effort duration when the container has no frame count."""
    try:
        # Seeking by relative position works on containers that know their
        # length even when they do not expose a frame count.
        capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
        position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if math.isfinite(position_ms) and position_ms > 0:
            return position_ms / 1000.0
    except cv2.error as exc:  # pragma: no cover - backend dependent
        logger.debug("Duration probing by seek failed: %s", exc)
    return None
