"""Event log: presentation, wall-clock mapping and CSV export.

The one rule this module enforces is that **elapsed video time and real
clock time are never the same column**.  A video-relative timestamp always
exists; a wall-clock timestamp exists only when the user told us (or the file
reliably recorded) when the recording started.  When it does not, the clock
columns are empty rather than guessed.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from ..analysis.base_detector import Event, EventStatus
from ..video.metadata import VideoInfo

CSV_COLUMNS: tuple[str, ...] = (
    "event",
    "start_video_time",
    "end_video_time",
    "duration_s",
    "start_wall_clock",
    "end_wall_clock",
    "confidence_pct",
    "status",
    "detector",
    "notes",
)

STATUS_LABELS = {
    EventStatus.CONFIRMED: "detected",
    EventStatus.REVIEW: "review recommended",
    EventStatus.FAILED: "detection failed",
}


def format_video_timestamp(seconds: float, *, milliseconds: bool = True) -> str:
    """Format a video-relative time as ``H:MM:SS.mmm``.

    Negative inputs are clamped to zero: a video-relative time before the start
    of the video is meaningless and must not appear in a report.
    """
    seconds = max(0.0, float(seconds))
    hours, remainder = divmod(int(seconds), 3600)
    minutes, whole_seconds = divmod(remainder, 60)
    if not milliseconds:
        return f"{hours}:{minutes:02d}:{whole_seconds:02d}"
    fraction = int(round((seconds - int(seconds)) * 1000))
    if fraction == 1000:  # rounding carried over
        whole_seconds += 1
        fraction = 0
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{fraction:03d}"


def format_duration(seconds: float) -> str:
    """Human-readable duration, e.g. ``23.4 s`` or ``2 m 05 s``."""
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)} m {remainder:04.1f} s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours} h {minutes:02d} m {remainder:04.1f} s"


def to_wall_clock(recording_start: datetime | None, offset_s: float) -> datetime | None:
    """Map a video-relative offset onto real time, or ``None`` if unknown."""
    if recording_start is None:
        return None
    return recording_start + timedelta(seconds=max(0.0, float(offset_s)))


@dataclass
class EventLog:
    """A list of events plus the context needed to present them."""

    events: list[Event] = field(default_factory=list)
    video: VideoInfo | None = None
    recording_start: datetime | None = None
    mode: str = ""

    @property
    def has_wall_clock(self) -> bool:
        return self.recording_start is not None

    def rows(self) -> list[dict[str, Any]]:
        """One dictionary per event, ready for CSV or the browser table."""
        rows: list[dict[str, Any]] = []
        for event in sorted(self.events, key=lambda e: e.start_s):
            start_clock = to_wall_clock(self.recording_start, event.start_s)
            end_clock = to_wall_clock(self.recording_start, event.end_s)
            rows.append(
                {
                    "event": event.label,
                    "start_video_time": format_video_timestamp(event.start_s),
                    "end_video_time": format_video_timestamp(event.end_s),
                    "duration_s": round(event.duration_s, 3),
                    "start_wall_clock": _format_clock(start_clock),
                    "end_wall_clock": _format_clock(end_clock),
                    "confidence_pct": round(event.confidence * 100),
                    "status": STATUS_LABELS.get(event.status, event.status.value),
                    "detector": event.detector,
                    "notes": " | ".join(event.notes),
                }
            )
        return rows

    def to_csv(self) -> str:
        """Render the log as CSV text (UTF-8, Excel-compatible line endings)."""
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, lineterminator="\r\n")
        writer.writeheader()
        for row in self.rows():
            writer.writerow(row)
        return buffer.getvalue()

    def to_dataframe(self) -> Any:
        """Return the log as a pandas DataFrame (pandas imported on demand)."""
        import pandas as pd

        return pd.DataFrame(self.rows(), columns=list(CSV_COLUMNS))

    def suggested_filename(self) -> str:
        """A safe, descriptive download name (no user-supplied path parts)."""
        stem = "analysis"
        if self.video is not None:
            stem = (
                "".join(
                    character if character.isalnum() or character in "-_" else "_"
                    for character in self.video.path.stem
                )[:60]
                or "analysis"
            )
        mode = self.mode or "events"
        return f"{stem}_{mode}_events.csv"

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows(),
            "has_wall_clock": self.has_wall_clock,
            "recording_start": self.recording_start.isoformat() if self.recording_start else None,
            "columns": list(CSV_COLUMNS),
        }


def _format_clock(value: datetime | None) -> str:
    """Wall-clock rendering; empty when the recording start is unknown."""
    return "" if value is None else value.strftime("%Y-%m-%d %H:%M:%S")


def parse_recording_start(raw: str | None) -> datetime | None:
    """Parse a recording start supplied by the user.

    Accepts ISO-8601 (``2026-03-04T07:43:00``) and the ``datetime-local`` form
    the browser produces.  Anything unparseable yields ``None`` - the log then
    simply has no wall-clock column values, which is the honest outcome.
    """
    if not raw:
        return None
    text = raw.strip().replace("Z", "+00:00")
    for parser in (
        datetime.fromisoformat,
        lambda v: datetime.strptime(v, "%Y-%m-%d %H:%M:%S"),
        lambda v: datetime.strptime(v, "%Y-%m-%d %H:%M"),
    ):
        try:
            return parser(text)
        except (ValueError, TypeError):
            continue
    return None


def build_event_log(
    events: Iterable[Event],
    video: VideoInfo | None,
    *,
    recording_start: datetime | None = None,
    mode: str = "",
) -> EventLog:
    return EventLog(events=list(events), video=video, recording_start=recording_start, mode=mode)


def summarise(events: Sequence[Event]) -> dict[str, Any]:
    """Counts used by the UI header ("3 events, 1 needs review")."""
    total = len(events)
    needing_review = sum(1 for e in events if e.status is EventStatus.REVIEW)
    return {
        "total": total,
        "confirmed": sum(1 for e in events if e.status is EventStatus.CONFIRMED),
        "review": needing_review,
        "failed": sum(1 for e in events if e.status is EventStatus.FAILED),
        "total_active_seconds": round(sum(e.duration_s for e in events), 2),
    }
