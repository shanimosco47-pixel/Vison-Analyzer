"""Optional object detection (Ultralytics YOLO).

**Nothing in the application imports this module at start-up and no analysis
mode uses it today.**  It exists to keep the seam explicit: when a plant
specific detector genuinely needs semantic information ("is that a person or
the robot arm?"), it can call :func:`load_object_detector` inside a candidate
window rather than across the whole recording.

Deliberate constraints:

*   the import is lazy, so Ultralytics is not a required dependency and the
    application starts without it;
*   the interface is a small protocol (:class:`ObjectDetector`), so a different
    detector - a classical template matcher, ONNX Runtime, a customer's own
    model - can replace it without touching the callers;
*   weights are loaded from a local path only.  No download is attempted, so
    the application cannot silently reach the internet with plant footage on
    the machine.

Licensing note (must be settled before this becomes a mandatory dependency):
Ultralytics is published under AGPL-3.0, with a separate commercial licence
available.  AGPL obligations extend to network-served applications, so for
internal commercial use this needs a licence review or a differently licensed
model.  That is why the package is optional and unused by default.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..errors import ModelUnavailableError
from ..logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Detection:
    """One detected object in image coordinates."""

    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float


class ObjectDetector(Protocol):
    """Minimal interface an object detector must provide."""

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Detect objects in a batch of BGR frames, one result list per frame."""


class UltralyticsDetector:
    """Thin adapter over an Ultralytics model held in memory."""

    def __init__(self, model: Any, confidence: float = 0.25) -> None:
        self._model = model
        self._confidence = confidence

    def detect(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not frames:
            return []
        # Batched inference: one call for the whole list is markedly faster
        # than one call per frame, which matters inside candidate windows.
        raw_results = self._model.predict(list(frames), conf=self._confidence, verbose=False)
        names = getattr(self._model, "names", {})
        output: list[list[Detection]] = []
        for result in raw_results:
            detections: list[Detection] = []
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                    class_id = int(box.cls[0])
                    detections.append(
                        Detection(
                            label=str(names.get(class_id, class_id)),
                            confidence=float(box.conf[0]),
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                        )
                    )
            output.append(detections)
        return output


def is_available() -> bool:
    """True when the optional Ultralytics package can be imported."""
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        return False
    return True


def load_object_detector(weights_path: Path, confidence: float = 0.25) -> ObjectDetector:
    """Load a local YOLO weights file.

    Raises:
        ModelUnavailableError: Ultralytics is not installed, or the weights
            file is missing.  Callers should fall back to deterministic
            analysis rather than failing the whole job.
    """
    weights_path = Path(weights_path)
    if not weights_path.is_file():
        raise ModelUnavailableError(
            "The detection model file was not found on this machine.",
            detail=str(weights_path),
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelUnavailableError(
            "Object detection is not installed. Install the optional 'ultralytics' "
            "package to enable it.",
            detail=str(exc),
        ) from exc

    logger.info("Loading object detection weights from %s", weights_path)
    return UltralyticsDetector(YOLO(str(weights_path)), confidence=confidence)
