"""Detector interface and the vocabulary all detectors share.

Adding a new plant-specific detector means implementing :class:`BaseDetector`
(or subclassing :class:`app.analysis.coarse_to_fine.CoarseToFineDetector`) and
registering it - no change to the web layer, the reader, the event log or the
CSV export.  That is the extension point the whole design exists to protect.

A detector's contract:

``__init__(video, params)``   configure and validate settings up front
``describe()``                human-readable summary of what it will do
``run(reader, progress)``     stream frames, report progress, return a result
``DetectorResult``            events + confidence + diagnostics
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, ClassVar, Protocol

import numpy as np

from ..errors import ConfigurationError
from ..video.metadata import VideoInfo
from ..video.reader import FrameSample, VideoReader


class EventStatus(str, Enum):
    """How much trust the software places in one result."""

    CONFIRMED = "confirmed"  # evidence is strong; present as a result
    REVIEW = "review"  # plausible but ambiguous; the user must look
    FAILED = "failed"  # detection did not succeed; no number is invented


@dataclass(frozen=True)
class Event:
    """One detected occurrence, in video-relative time.

    Wall-clock times are *derived* at presentation time from a recording start
    supplied by the user; they are never stored here, so elapsed video time
    and real clock time cannot be silently confused.
    """

    label: str
    start_s: float
    end_s: float
    confidence: float
    detector: str
    status: EventStatus = EventStatus.CONFIRMED
    notes: tuple[str, ...] = field(default_factory=tuple)
    details: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.end_s < self.start_s:
            raise ConfigurationError(
                "An event cannot end before it starts.",
                detail=f"{self.label}: {self.start_s}..{self.end_s}",
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ConfigurationError(
                "Confidence must be between 0 and 1.", detail=f"{self.confidence!r}"
            )

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "start_s": round(self.start_s, 3),
            "end_s": round(self.end_s, 3),
            "duration_s": round(self.duration_s, 3),
            "confidence": round(self.confidence, 4),
            "status": self.status.value,
            "detector": self.detector,
            "notes": list(self.notes),
            "details": dict(self.details),
        }


@dataclass
class ActivityTrace:
    """The score time series a scan produced, kept for plotting and debugging.

    Bounded by construction: the coarse pass stores one float per *sample*, not
    per frame, so a 12-hour scan at one sample per 5 s holds ~8600 values.
    """

    times: list[float] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    disturbed: list[bool] = field(default_factory=list)

    def add(self, timestamp_s: float, score: float, disturbed: bool = False) -> None:
        self.times.append(float(timestamp_s))
        self.scores.append(float(score))
        self.disturbed.append(bool(disturbed))

    def __len__(self) -> int:
        return len(self.times)

    @property
    def disturbed_ratio(self) -> float:
        if not self.disturbed:
            return 0.0
        return sum(self.disturbed) / len(self.disturbed)

    def downsampled(self, max_points: int = 1500) -> dict[str, list[float]]:
        """A small version suitable for sending to the browser."""
        if len(self.times) <= max_points:
            return {"times": list(self.times), "scores": list(self.scores)}
        step = len(self.times) / max_points
        indices = [int(i * step) for i in range(max_points)]
        return {
            "times": [self.times[i] for i in indices],
            "scores": [self.scores[i] for i in indices],
        }


@dataclass
class DetectorResult:
    """What a detector hands back to the service layer."""

    events: list[Event] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    trace: ActivityTrace | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "events": [event.to_dict() for event in self.events],
            "diagnostics": self.diagnostics,
            "summary": self.summary,
            "warnings": self.warnings,
            "trace": self.trace.downsampled() if self.trace is not None else None,
        }


@dataclass(frozen=True)
class ScoreSample:
    """A scorer's verdict on one frame.

    Attributes:
        value: activity score, conventionally 0 (nothing) to 1 (whole region
            changed).  Detectors compare it against thresholds.
        disturbed: the frame is untrustworthy (camera moved, scene-wide light
            change, hand across the lens).  Detectors must not start or stop
            events on such frames.
        mask: optional binary mask for diagnostics; not retained by default.
        extras: scorer-specific measurements used for confidence.
    """

    value: float
    disturbed: bool = False
    mask: np.ndarray | None = None
    extras: Mapping[str, float] = field(default_factory=dict)


class ActivityScorer(ABC):
    """Turns one frame into one number.

    Scorers hold whatever temporal state they need (a background model, the
    previous frame) but never decide what an *event* is - that is the state
    machines' job.  Keeping the two apart is what allows the scoring to be
    swapped for a neural model later without touching the event logic.
    """

    name: ClassVar[str] = "scorer"

    @abstractmethod
    def score(self, sample: FrameSample) -> ScoreSample:
        """Score one frame."""

    def reset(self) -> None:
        """Forget temporal state (called before a new pass over the video)."""


class ProgressReporter(Protocol):
    """Callback used by long-running stages to report progress to the UI."""

    def __call__(self, *, stage: str, fraction: float, message: str) -> None: ...


def null_progress(*, stage: str, fraction: float, message: str) -> None:
    """A :class:`ProgressReporter` that discards everything (tests, CLI)."""


class BaseDetector(ABC):
    """Base class for every analysis mode."""

    name: ClassVar[str] = "base"
    display_name: ClassVar[str] = "Base detector"
    description: ClassVar[str] = ""

    def __init__(self, video: VideoInfo, params: Mapping[str, Any] | None = None) -> None:
        self.video = video
        self.params: dict[str, Any] = dict(params or {})
        self.configure()

    def configure(self) -> None:
        """Validate parameters and build internal configuration.

        Called from ``__init__`` so an invalid configuration is rejected before
        any frame is decoded.
        """

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """Summarise the plan (sampling, thresholds) for the log and the UI."""

    @abstractmethod
    def run(self, reader: VideoReader, progress: ProgressReporter) -> DetectorResult:
        """Analyse the video and return events."""

    # Convenience used by several detectors.
    def _describe_common(self, **extra: Any) -> dict[str, Any]:
        base = {
            "detector": self.name,
            "video_fps": round(self.video.fps, 3),
            "video_duration_s": self.video.duration_s,
            "resolution": f"{self.video.width}x{self.video.height}",
        }
        base.update(extra)
        return base


def dataclass_to_dict(value: Any) -> dict[str, Any]:
    """asdict() that tolerates plain dictionaries (used for logging configs)."""
    if isinstance(value, dict):
        return dict(value)
    return asdict(value)
