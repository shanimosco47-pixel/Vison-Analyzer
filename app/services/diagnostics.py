"""Visual diagnostics.

Vision algorithms need to be *seen* to be trusted.  This module renders the
overlays the UI shows on demand (region of interest, detected liquid mask) and,
when debug mode is enabled, writes the frames at the detected event boundaries
to disk so a disputed measurement can be inspected afterwards.

Everything here is optional: the analysis never depends on it, and with
``save_diagnostics`` off nothing is written to disk.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from ..analysis.base_detector import Event
from ..config import ROI
from ..logging_setup import get_logger
from ..video.reader import VideoReader

logger = get_logger(__name__)

ROI_COLOUR = (0, 200, 255)  # BGR: amber, visible on most industrial footage
GUARD_COLOUR = (120, 120, 120)
MASK_COLOUR = (0, 0, 255)
LABEL_COLOUR = (255, 255, 255)


def draw_overlay(
    frame: np.ndarray,
    *,
    roi: ROI | None = None,
    guard_roi: ROI | None = None,
    mask: np.ndarray | None = None,
    label: str | None = None,
) -> np.ndarray:
    """Return a copy of ``frame`` with the analysis geometry drawn on it."""
    canvas = frame.copy()
    if guard_roi is not None:
        cv2.rectangle(
            canvas, (guard_roi.x, guard_roi.y), (guard_roi.x2, guard_roi.y2), GUARD_COLOUR, 1
        )
    if roi is not None:
        cv2.rectangle(canvas, (roi.x, roi.y), (roi.x2, roi.y2), ROI_COLOUR, 2)
    if mask is not None and roi is not None and mask.size:
        canvas = _tint_mask(canvas, mask, roi)
    if label:
        cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, LABEL_COLOUR, 1)
    return canvas


def _tint_mask(canvas: np.ndarray, mask: np.ndarray, roi: ROI) -> np.ndarray:
    """Tint the detected pixels red inside the ROI rectangle."""
    resized = cv2.resize(
        mask.astype(np.uint8), (roi.width, roi.height), interpolation=cv2.INTER_NEAREST
    )
    region = canvas[roi.y : roi.y2, roi.x : roi.x2]
    if region.shape[:2] != resized.shape[:2]:  # pragma: no cover - clipped ROI
        return canvas
    overlay = region.copy()
    overlay[resized > 0] = MASK_COLOUR
    canvas[roi.y : roi.y2, roi.x : roi.x2] = cv2.addWeighted(region, 0.6, overlay, 0.4, 0)
    return canvas


def save_event_boundary_frames(
    reader: VideoReader,
    events: Sequence[Event],
    output_dir: Path,
    *,
    roi: ROI | None = None,
    limit: int = 20,
) -> list[Path]:
    """Write the first and last frame of each event, with the ROI drawn.

    Used when debug mode is on.  Failures are logged and skipped: diagnostics
    must never break an otherwise successful analysis.
    """
    saved: list[Path] = []
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        logger.warning("Could not create the diagnostics folder %s: %s", output_dir, exc)
        return saved

    for index, event in enumerate(events[:limit]):
        for moment, timestamp in (("start", event.start_s), ("end", event.end_s)):
            try:
                frame = reader.frame_at(timestamp)
                annotated = draw_overlay(
                    frame,
                    roi=roi,
                    label=f"{event.label} {moment} @ {timestamp:.3f}s",
                )
                path = output_dir / f"event{index:02d}_{moment}_{timestamp:.3f}s.jpg"
                if cv2.imwrite(str(path), annotated):
                    saved.append(path)
            except Exception as exc:  # noqa: BLE001 - diagnostics are best-effort
                logger.warning(
                    "Could not save the %s frame of event %d at %.3fs: %s",
                    moment,
                    index,
                    timestamp,
                    exc,
                )
    if saved:
        logger.info("Saved %d diagnostic frame(s) to %s", len(saved), output_dir)
    return saved
