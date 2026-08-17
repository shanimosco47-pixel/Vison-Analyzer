"""Detector registry.

The single place that knows which analysis modes exist.  Adding a plant
specific detector is: write the class, add one line here.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ..errors import ConfigurationError
from ..video.metadata import VideoInfo
from .base_detector import BaseDetector
from .coarse_to_fine import GenericMotionDetector
from .robot_activity_detector import RobotActivityDetector
from .zahn_detector import ZahnCupDetector

DETECTOR_CLASSES: tuple[type[BaseDetector], ...] = (
    ZahnCupDetector,
    RobotActivityDetector,
    GenericMotionDetector,
)

_BY_NAME: dict[str, type[BaseDetector]] = {cls.name: cls for cls in DETECTOR_CLASSES}


def available_modes() -> list[dict[str, str]]:
    """Modes offered in the UI, in the order they should be presented."""
    return [
        {"name": cls.name, "display_name": cls.display_name, "description": cls.description}
        for cls in DETECTOR_CLASSES
    ]


def get_detector_class(mode: str) -> type[BaseDetector]:
    try:
        return _BY_NAME[mode]
    except KeyError as exc:
        known = ", ".join(sorted(_BY_NAME))
        raise ConfigurationError(
            "That analysis mode is not available.", detail=f"{mode!r} not in {known}"
        ) from exc


def create_detector(
    mode: str, video: VideoInfo, params: Mapping[str, Any] | None = None
) -> BaseDetector:
    """Instantiate a detector, validating its parameters immediately."""
    return get_detector_class(mode)(video, params or {})
