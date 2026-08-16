"""Deterministic motion / change scoring.

This is the workhorse of the coarse pass and of the generic and robot modes.
No neural network is involved: a background model plus adaptive thresholding
answers "how much of this region genuinely changed?" far more cheaply, and -
just as importantly - explainably.

The measures that make it survive factory footage:

*   the change threshold is derived from the *measured* noise of each frame,
    so a grainy compressed night camera raises its own bar;
*   a global brightness shift (lights switching, auto-exposure) is subtracted
    before thresholding rather than being counted as motion;
*   speckle is removed by morphological opening and by a minimum blob area, so
    single noisy pixels never count;
*   frames where a huge fraction of the region changes at once are flagged as
    *disturbed* (camera knock, someone walking into the lens) rather than
    reported as an event, and the background model is rebuilt from them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import cv2
import numpy as np

from ..errors import ConfigurationError
from ..logging_setup import get_logger
from ..video.reader import FrameSample
from .base_detector import ActivityScorer, ScoreSample

logger = get_logger(__name__)


@dataclass
class MotionScorerConfig:
    """Tunables for :class:`MotionActivityScorer` (see module docstring)."""

    # Absolute grey-level floor: differences below this are never motion, no
    # matter how quiet the camera is.  8-bit sensor noise is typically 2-6.
    min_abs_diff: int = 10

    # Working threshold is max(min_abs_diff, this * measured noise sigma).
    noise_sigma_multiplier: float = 4.0

    # Blobs smaller than this fraction of the analysed region are discarded.
    # 0.0005 of a 320x180 scan frame is ~29 px, roughly a person at distance.
    min_blob_area_ratio: float = 0.0005

    # Morphological opening kernel; removes single-pixel speckle.
    open_kernel_px: int = 3

    # Background model adaptation per sample.  The idle rate lets slow
    # illumination drift into the model within a couple of minutes of samples.
    # The active rate is far slower, and deliberately so: whatever the model
    # absorbs while an object is moving becomes a "ghost" that keeps producing
    # differences after the object stops, which shows up as an event that ends
    # a second or two late.  A stationary object still fades in eventually, it
    # just takes minutes rather than seconds.
    background_alpha_idle: float = 0.05
    background_alpha_active: float = 0.0005

    # Above this changed fraction the frame is "disturbed" rather than active.
    disturbance_area_ratio: float = 0.55

    # Mean brightness jump (grey levels) that also marks a frame disturbed.
    disturbance_brightness_jump: float = 28.0

    # If the *median* pixel differs from the background by this much, more than
    # half the picture changed at once: the camera moved, or the scene was
    # replaced.  This is checked before thresholding, because a scene-wide
    # change also inflates the measured noise and would otherwise hide itself
    # behind its own threshold.
    disturbance_median_diff: float = 18.0

    # After a disturbance the old background is meaningless; rebuild it.
    reset_background_on_disturbance: bool = True

    def validate(self) -> None:
        if self.min_abs_diff < 1:
            raise ConfigurationError("The minimum difference threshold must be at least 1.")
        if self.noise_sigma_multiplier <= 0:
            raise ConfigurationError("The noise multiplier must be greater than zero.")
        if not 0.0 <= self.min_blob_area_ratio < 1.0:
            raise ConfigurationError("The minimum blob area ratio must be in [0, 1).")
        if not 0.0 < self.background_alpha_idle <= 1.0:
            raise ConfigurationError("The idle background rate must be in (0, 1].")
        if not 0.0 < self.background_alpha_active <= 1.0:
            raise ConfigurationError("The active background rate must be in (0, 1].")
        if not 0.0 < self.disturbance_area_ratio <= 1.0:
            raise ConfigurationError("The disturbance ratio must be in (0, 1].")
        if self.disturbance_median_diff <= 0:
            raise ConfigurationError("The scene-change threshold must be greater than zero.")


class MotionActivityScorer(ActivityScorer):
    """Scores a frame by the fraction of its area that meaningfully changed."""

    name: ClassVar[str] = "motion"

    def __init__(self, config: MotionScorerConfig | None = None, *, keep_mask: bool = False):
        self.config = config or MotionScorerConfig()
        self.config.validate()
        self.keep_mask = keep_mask
        self._background: np.ndarray | None = None
        self._previous_mean: float | None = None
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(1, self.config.open_kernel_px), max(1, self.config.open_kernel_px)),
        )

    def reset(self) -> None:
        self._background = None
        self._previous_mean = None

    def score(self, sample: FrameSample) -> ScoreSample:
        frame = sample.image
        if frame.ndim != 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = frame.astype(np.float32, copy=False)
        mean_brightness = float(current.mean())

        if self._background is None:
            self._background = current.copy()
            self._previous_mean = mean_brightness
            # The first frame has nothing to compare against; it is not motion.
            return ScoreSample(value=0.0, extras={"mean_brightness": mean_brightness})

        diff = cv2.absdiff(current, self._background)
        diff_median = float(np.median(diff))
        brightness_jump = abs(mean_brightness - (self._previous_mean or mean_brightness))

        # Scene-wide change (camera knock, lights, someone in front of the
        # lens) is decided *before* thresholding: such a frame also inflates
        # the noise estimate, and would otherwise hide behind its own
        # threshold and be reported as a perfectly quiet scene.
        if (
            diff_median >= self.config.disturbance_median_diff
            or brightness_jump >= self.config.disturbance_brightness_jump
        ):
            self._previous_mean = mean_brightness
            if self.config.reset_background_on_disturbance:
                self._background = current.copy()
            return ScoreSample(
                value=0.0,
                disturbed=True,
                extras={
                    "changed_ratio": 0.0,
                    "median_diff": diff_median,
                    "mean_brightness": mean_brightness,
                    "brightness_jump": brightness_jump,
                },
            )

        # Remove a uniform brightness shift: for an exposure change the whole
        # difference image rises together, so its median is the shift itself.
        residual = diff - diff_median
        noise_sigma = 1.4826 * float(np.median(np.abs(residual))) or 1.0

        threshold = max(
            float(self.config.min_abs_diff),
            self.config.noise_sigma_multiplier * noise_sigma,
        )
        mask: np.ndarray = (residual > threshold).astype(np.uint8)
        if self.config.open_kernel_px >= 2:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        total_pixels = mask.size
        changed_ratio, largest_blob_ratio, blob_count = self._blob_statistics(mask, total_pixels)

        disturbed = changed_ratio >= self.config.disturbance_area_ratio
        self._update_background(current, changed_ratio, disturbed)
        self._previous_mean = mean_brightness

        return ScoreSample(
            value=0.0 if disturbed else changed_ratio,
            disturbed=disturbed,
            mask=mask if self.keep_mask else None,
            extras={
                "changed_ratio": changed_ratio,
                "largest_blob_ratio": largest_blob_ratio,
                "blob_count": float(blob_count),
                "noise_sigma": noise_sigma,
                "threshold": threshold,
                "median_diff": diff_median,
                "mean_brightness": mean_brightness,
                "brightness_jump": brightness_jump,
            },
        )

    def _blob_statistics(self, mask: np.ndarray, total_pixels: int) -> tuple[float, float, int]:
        """Changed-area ratio counting only blobs above the minimum size."""
        min_area = max(1, int(self.config.min_blob_area_ratio * total_pixels))
        if not mask.any():
            return 0.0, 0.0, 0

        count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        kept_area = 0
        largest = 0
        blobs = 0
        for label in range(1, count):  # label 0 is the background
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area >= min_area:
                kept_area += area
                largest = max(largest, area)
                blobs += 1
        return kept_area / total_pixels, largest / total_pixels, blobs

    def _update_background(
        self, current: np.ndarray, changed_ratio: float, disturbed: bool
    ) -> None:
        assert self._background is not None
        if disturbed and self.config.reset_background_on_disturbance:
            # The scene reference is invalid after a camera move or light
            # change; starting again is safer than slowly converging.
            self._background = current.copy()
            return
        alpha = (
            self.config.background_alpha_active
            if changed_ratio > 0.0
            else self.config.background_alpha_idle
        )
        cv2.accumulateWeighted(current, self._background, alpha)
