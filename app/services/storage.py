"""Upload storage.

Uploaded videos are untrusted input.  The rules this module enforces:

*   the name supplied by the browser is **never** used as a path.  Files are
    stored as ``<uuid4>.<extension>`` inside one directory, so path traversal
    ("../../etc/passwd") is structurally impossible;
*   only extensions on the allowlist are accepted, and the browser's declared
    MIME type is ignored - the real check is whether OpenCV can decode the
    file, which is content-based;
*   the size limit is enforced *while streaming to disk*, so an oversized or
    endless upload cannot fill the disk before being rejected;
*   a partially written file is always removed, including when the connection
    drops mid-upload.
"""

from __future__ import annotations

import shutil
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from ..config import AppConfig
from ..errors import (
    AnalyzerError,
    NotFoundError,
    StorageError,
    UnsupportedFormatError,
    UploadTooLargeError,
)
from ..logging_setup import get_logger
from ..video.metadata import VideoInfo, probe_video

logger = get_logger(__name__)

UPLOAD_CHUNK_BYTES = 1024 * 1024

# Leave this much head-room on the volume; below it we refuse the upload with a
# clear message instead of failing half-way through writing.
MIN_FREE_DISK_BYTES = 256 * 1024 * 1024


@dataclass
class VideoRecord:
    """A stored upload and its probed metadata."""

    video_id: str
    path: Path
    original_name: str
    info: VideoInfo
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)

    def touch(self) -> None:
        self.last_used_at = time.time()

    def to_dict(self) -> dict[str, Any]:
        payload = self.info.to_dict()
        payload.update(
            {
                "video_id": self.video_id,
                "original_name": self.original_name,
                "uploaded_at": self.created_at,
            }
        )
        return payload


class VideoStore:
    """Thread-safe store of uploaded videos."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._records: dict[str, VideoRecord] = {}
        self._lock = threading.RLock()
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        try:
            self.config.upload_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise StorageError("The upload folder could not be created.", detail=str(exc)) from exc

    # -- writing ------------------------------------------------------------ #

    def save_upload(self, stream: IO[bytes], original_name: str) -> VideoRecord:
        """Stream an upload to disk, validate it, and register it."""
        extension = _safe_extension(original_name, self.config.allowed_extensions)
        self._check_free_space()

        video_id = uuid.uuid4().hex
        destination = self.config.upload_dir / f"{video_id}{extension}"
        written = 0
        try:
            with destination.open("wb") as handle:
                while chunk := stream.read(UPLOAD_CHUNK_BYTES):
                    written += len(chunk)
                    if written > self.config.max_upload_bytes:
                        raise UploadTooLargeError(
                            f"The video is larger than the {self.config.max_upload_mb} MB limit."
                        )
                    handle.write(chunk)
        except OSError as exc:
            _remove_quietly(destination)
            raise StorageError(
                "The video could not be written to disk. Check the available space.",
                detail=str(exc),
            ) from exc
        except BaseException:
            # Includes a dropped connection and the size-limit error above.
            _remove_quietly(destination)
            raise

        if written == 0:
            _remove_quietly(destination)
            raise UnsupportedFormatError("The uploaded file is empty.")

        try:
            info = probe_video(destination)
        except AnalyzerError:
            _remove_quietly(destination)
            raise

        record = VideoRecord(
            video_id=video_id,
            path=destination,
            original_name=_display_name(original_name),
            info=info,
        )
        with self._lock:
            self._records[video_id] = record
        logger.info(
            "Stored upload %s (%s, %.1f MB) as %s",
            record.original_name,
            info.fourcc,
            written / 1e6,
            destination.name,
        )
        return record

    def _check_free_space(self) -> None:
        try:
            usage = shutil.disk_usage(self.config.upload_dir)
        except OSError as exc:  # pragma: no cover - platform dependent
            logger.warning("Could not determine free disk space: %s", exc)
            return
        if usage.free < MIN_FREE_DISK_BYTES:
            raise StorageError("There is not enough free disk space to accept another video.")

    # -- reading ------------------------------------------------------------ #

    def get(self, video_id: str) -> VideoRecord:
        with self._lock:
            record = self._records.get(video_id)
        if record is None or not record.path.is_file():
            raise NotFoundError("That video is no longer available. Please upload it again.")
        record.touch()
        return record

    def list_records(self) -> list[VideoRecord]:
        with self._lock:
            return list(self._records.values())

    # -- cleanup ------------------------------------------------------------ #

    def delete(self, video_id: str) -> None:
        with self._lock:
            record = self._records.pop(video_id, None)
        if record is not None:
            _remove_quietly(record.path)
            logger.info("Deleted upload %s", record.path.name)

    def purge_expired(self) -> int:
        """Delete uploads untouched for longer than the retention period."""
        cutoff = time.time() - self.config.retention_hours * 3600
        with self._lock:
            expired = [r.video_id for r in self._records.values() if r.last_used_at < cutoff]
        for video_id in expired:
            self.delete(video_id)
        if expired:
            logger.info("Purged %d expired upload(s)", len(expired))
        return len(expired)

    def purge_orphans(self) -> int:
        """Remove files in the upload folder with no in-memory record.

        The application keeps its records in memory, so a restart leaves the
        previous session's files behind.  Clearing them at start-up keeps the
        disk from growing without bound.
        """
        with self._lock:
            known = {record.path for record in self._records.values()}
        removed = 0
        for path in self.config.upload_dir.glob("*"):
            if path.is_file() and path not in known:
                _remove_quietly(path)
                removed += 1
        if removed:
            logger.info("Removed %d orphaned upload file(s) from a previous run", removed)
        return removed


def _safe_extension(original_name: str, allowed: tuple[str, ...]) -> str:
    """Validate and return the lower-cased extension - nothing else is reused."""
    suffix = Path(original_name or "").suffix.lower()
    if suffix not in allowed:
        raise UnsupportedFormatError(
            "Only these video formats are supported: " + ", ".join(sorted(allowed))
        )
    return suffix


def _display_name(original_name: str) -> str:
    """A printable version of the client's filename, for display only."""
    name = Path(original_name or "video").name
    cleaned = "".join(character for character in name if character.isprintable())
    return cleaned[:120] or "video"


def _remove_quietly(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("Could not remove temporary file %s: %s", path, exc)
