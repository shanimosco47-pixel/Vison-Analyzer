"""Outlet tracking under a hand-held camera and a hand-held cup.

Stage 0 (``diagnostics/stage0/STAGE0_REPORT.md``) found the Zahn timing error
under hand-held motion was caused almost entirely by the analysis region
staying fixed while the cup drifted out of it - not by contrast, disturbance
detection or the persistence timer, all of which measured as correct or inert
in isolation.  Re-cutting the region around the *true* outlet position, frame
by frame, took the measured error to zero on every hand-held clip tested.

This module is the tracker that makes that re-cutting possible on real
(unlabelled) footage, where the true position is not known in advance.

Method
------

A single point on the outlet is usually too smooth to track - Stage 0
measured its corner response as roughly a sixth of the cup handle's.  So this
tracks a *feature set over the cup body* with pyramidal Lucas-Kanade,
validated by a forward-backward check, fits a similarity transform to the
surviving correspondences with ``estimateAffinePartial2D`` (RANSAC), and
carries the outlet as a point under that transform.  This build's OpenCV
(``opencv-python-headless``) ships no CSRT, KCF or MOSSE tracker - only
``goodFeaturesToTrack``, ``calcOpticalFlowPyrLK``, ``estimateAffinePartial2D``
and ``matchTemplate``, which is what this is built from.

States
------

``TRACKED``   this frame's transform is trustworthy: enough inliers, a
              forward-backward-consistent point set, and a plausible
              per-frame displacement.  Valid for scoring and persistence.
``PREDICTED`` a short constant-velocity bridge across a tracking gap, up to
              ``track_max_bridge_s``.  Diagnostic only - callers must not use
              a predicted position to confirm flow start or end, because
              nothing has actually confirmed the outlet is where this says it
              is.
``LOST``      the bridge ran out, or the estimate left the search window.
              Excluded from scoring entirely.  Reacquisition after ``LOST``
              is *verified independently*: a stored reference patch of the
              cup, captured once at initialisation, must correlate with the
              candidate position above ``track_reacquire_min_correlation``
              before tracking resumes.  A tracker that came back from a gap
              is not proof the outlet was ever seen during that gap - see
              :class:`app.analysis.zahn_detector.FlowStateMachine` for how
              that uncertainty is carried into the reported timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..config import ZahnConfig

# Lucas-Kanade pyramid search parameters. A 21x21 window with 3 pyramid
# levels comfortably covers the per-frame displacements Stage 0 measured
# (single-digit pixels) without the cost of a wider search. Passed as
# individual keyword arguments (not a **dict) so mypy can match cv2's
# overloads precisely.
_LK_WIN_SIZE = (21, 21)
_LK_MAX_LEVEL = 3
_LK_CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

# A point whose forward-then-backward flow does not return within this many
# pixels of where it started is unreliable and is dropped before the
# similarity fit sees it.
_FORWARD_BACKWARD_MAX_PX = 1.0

# RANSAC reprojection tolerance for the similarity fit, in pixels.
_RANSAC_REPROJ_PX = 2.0
_RANSAC_MAX_ITERS = 2000

# Feature detection over the cup: capped so a busy background inside the
# search window cannot make a frame expensive, generous enough that losing
# some points to the forward-backward check still leaves plenty to fit.
_MAX_FEATURES = 150
_FEATURE_QUALITY = 0.01
_FEATURE_MIN_DISTANCE_PX = 5
_FEATURE_BLOCK_SIZE = 5

# Gap kept between the outlet and the feature search area, so a feature
# right at the orifice (which may sit on the liquid, not the rigid cup) is
# not offered to the tracker.
_FEATURE_EXCLUSION_BELOW_OUTLET_PX = 4

# The smallest number of points that can seed tracking at all. Below this,
# initialisation fails outright rather than starting a tracker with nothing
# to lose.
_MIN_INIT_FEATURES = 4


class TrackerInitError(Exception):
    """Raised when the initial frame has too little texture to track at all.

    Callers should treat this as "tracking is not usable for this recording"
    and fall back to the fixed-ROI path - the same behaviour as
    ``zahn_track_outlet=False`` - rather than run a tracker that can never
    lock on.
    """


class TrackState(str, Enum):
    """See the module docstring for the contract each state carries."""

    TRACKED = "tracked"
    PREDICTED = "predicted"
    LOST = "lost"


@dataclass(frozen=True)
class TrackResult:
    """One frame's tracking verdict, in the tracker's local crop coordinates."""

    state: TrackState
    outlet_x: float
    outlet_y: float
    offset_x: float  # outlet - reference outlet, i.e. drift since frame 0
    offset_y: float
    inliers: int
    feature_count: int
    reacquired: bool = False


def _feature_mask(shape: tuple[int, int], outlet_y: float) -> np.ndarray:
    """Restrict feature detection to the cup: everything above the outlet.

    Excluding the area at and below the orifice keeps the liquid itself -
    which moves independently of the rigid cup body - out of the point set
    the similarity transform is fit to.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    bottom = max(1, int(outlet_y) - _FEATURE_EXCLUSION_BELOW_OUTLET_PX)
    mask[:bottom, :] = 255
    return mask


def _detect_features(gray: np.ndarray, outlet_y: float) -> np.ndarray | None:
    mask = _feature_mask(gray.shape[:2], outlet_y)
    return cv2.goodFeaturesToTrack(
        gray,
        maxCorners=_MAX_FEATURES,
        qualityLevel=_FEATURE_QUALITY,
        minDistance=_FEATURE_MIN_DISTANCE_PX,
        mask=mask,
        blockSize=_FEATURE_BLOCK_SIZE,
    )


class OutletTracker:
    """Tracks one outlet position across frames of a fixed capture crop.

    All coordinates are in the *local* pixel space of the crop the caller
    decodes each frame (not source-video coordinates) - the caller is
    responsible for that mapping, and for sizing the crop to comfortably
    contain the drift it expects (``track_search_margin_px``).
    """

    def __init__(
        self,
        config: ZahnConfig,
        reference_gray: np.ndarray,
        outlet_xy: tuple[float, float],
        *,
        patch_half_px: int,
        reference_timestamp_s: float = 0.0,
    ) -> None:
        self._config = config
        self._bounds = reference_gray.shape[:2]  # (height, width)
        self._reference_outlet = np.array(outlet_xy, dtype=np.float64)
        self._outlet = self._reference_outlet.copy()

        points = _detect_features(reference_gray, outlet_xy[1])
        if points is None or len(points) < _MIN_INIT_FEATURES:
            raise TrackerInitError(
                f"Only {0 if points is None else len(points)} trackable feature(s) "
                f"found above the outlet (need >= {_MIN_INIT_FEATURES})."
            )
        self._points: np.ndarray | None = points
        self._prev_gray = reference_gray
        self._state = TrackState.TRACKED
        self._bridge_start_s: float | None = None
        self._last_confident_outlet = self._outlet.copy()
        # Set from construction, not left None until the first successful
        # update(): the reference frame *is* a confident observation, and a
        # gap on the very next frame must still have a timestamp and a
        # (zero, until real motion is observed) velocity to bridge from.
        self._last_confident_timestamp: float | None = reference_timestamp_s
        self._velocity = np.zeros(2, dtype=np.float64)

        # The reacquisition patch is centred on the *strongest* detected
        # feature, not the outlet itself and not the centroid of every
        # feature found: the outlet is a narrow, often near-featureless point
        # (the same reason a single click there is not enough to track frame
        # to frame - see the module docstring), and averaging every feature's
        # position dilutes towards whatever smooth, low-response area most of
        # them sit in. cv2.goodFeaturesToTrack returns points in decreasing
        # order of corner response, so points[0] is the single most
        # distinctive point on the cup (typically the rim or handle) -
        # exactly what normalised template matching needs to tell "the same
        # cup, moved" apart from "a differently-smooth patch of wall that
        # happens to correlate," which is the false-reacquisition failure
        # mode a low-contrast anchor produces. The outlet's position is still
        # recovered precisely via the stored offset from that point.
        anchor_point = points.reshape(-1, 2)[0]
        self._reference_patch, self._patch_outlet_offset = self._extract_patch(
            reference_gray, anchor_point, self._outlet, patch_half_px
        )

    # -- public ------------------------------------------------------------ #

    @property
    def state(self) -> TrackState:
        return self._state

    def update(self, gray: np.ndarray, timestamp_s: float) -> TrackResult:
        """Feed the next frame (same crop geometry as initialisation)."""
        if self._state is TrackState.LOST:
            result = self._attempt_reacquisition(gray, timestamp_s)
        else:
            result = self._attempt_tracking(gray, timestamp_s)
        self._prev_gray = gray
        return result

    # -- tracked/predicted branch ------------------------------------------ #

    def _attempt_tracking(self, gray: np.ndarray, timestamp_s: float) -> TrackResult:
        cfg = self._config
        inliers = 0
        feature_count = 0 if self._points is None else len(self._points)
        accepted_outlet: np.ndarray | None = None
        surviving_points: np.ndarray | None = None

        if self._points is not None and len(self._points) >= cfg.track_min_inliers:
            # cv2's type stub does not mark nextPts as Optional even though
            # passing None - "estimate it from scratch" - is the standard
            # OpenCV idiom used here.
            forward, fwd_ok, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                self._prev_gray,
                gray,
                self._points,
                None,
                winSize=_LK_WIN_SIZE,
                maxLevel=_LK_MAX_LEVEL,
                criteria=_LK_CRITERIA,
            )
            backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
                gray,
                self._prev_gray,
                forward,
                None,
                winSize=_LK_WIN_SIZE,
                maxLevel=_LK_MAX_LEVEL,
                criteria=_LK_CRITERIA,
            )
            good = (fwd_ok.ravel() == 1) & (back_ok.ravel() == 1)
            fb_error = np.linalg.norm(backward - self._points, axis=2).ravel()
            good &= fb_error < _FORWARD_BACKWARD_MAX_PX

            source, target = self._points[good], forward[good]
            if len(source) >= cfg.track_min_inliers:
                matrix, inlier_mask = cv2.estimateAffinePartial2D(
                    source,
                    target,
                    method=cv2.RANSAC,
                    ransacReprojThreshold=_RANSAC_REPROJ_PX,
                    maxIters=_RANSAC_MAX_ITERS,
                )
                if matrix is not None and inlier_mask is not None:
                    inliers = int(inlier_mask.sum())
                    if inliers >= cfg.track_min_inliers:
                        homogeneous = np.array([self._outlet[0], self._outlet[1], 1.0])
                        candidate = matrix @ homogeneous
                        displacement = float(np.hypot(*(candidate - self._outlet)))
                        in_bounds = self._within_bounds(candidate)
                        if displacement <= cfg.track_max_frame_displacement_px and in_bounds:
                            accepted_outlet = candidate
                            surviving_points = target[inlier_mask.ravel() == 1].reshape(-1, 1, 2)

        if accepted_outlet is not None:
            return self._accept_tracked(
                accepted_outlet, surviving_points, gray, timestamp_s, inliers, feature_count
            )
        return self._enter_or_continue_bridge(timestamp_s, inliers, feature_count)

    def _accept_tracked(
        self,
        outlet: np.ndarray,
        points: np.ndarray | None,
        gray: np.ndarray,
        timestamp_s: float,
        inliers: int,
        feature_count: int,
    ) -> TrackResult:
        cfg = self._config
        if self._last_confident_timestamp is not None:
            dt = timestamp_s - self._last_confident_timestamp
            if dt > 0:
                self._velocity = (outlet - self._last_confident_outlet) / dt
        self._outlet = outlet
        self._last_confident_outlet = outlet.copy()
        self._last_confident_timestamp = timestamp_s
        self._bridge_start_s = None
        self._state = TrackState.TRACKED

        if points is not None and len(points) < cfg.track_min_features:
            fresh = _detect_features(gray, outlet[1])
            if fresh is not None and len(fresh) >= cfg.track_min_features:
                points = fresh
        self._points = points

        return self._result(TrackState.TRACKED, inliers=inliers, feature_count=feature_count)

    def _enter_or_continue_bridge(
        self, timestamp_s: float, inliers: int, feature_count: int
    ) -> TrackResult:
        cfg = self._config
        if self._bridge_start_s is None:
            self._bridge_start_s = timestamp_s

        bridge_elapsed = timestamp_s - self._bridge_start_s
        if self._last_confident_timestamp is None or bridge_elapsed > cfg.track_max_bridge_s:
            return self._declare_lost(inliers, feature_count)

        predicted = self._last_confident_outlet + self._velocity * (
            timestamp_s - self._last_confident_timestamp
        )
        if not self._within_bounds(predicted):
            return self._declare_lost(inliers, feature_count)

        self._outlet = predicted
        self._state = TrackState.PREDICTED
        return self._result(TrackState.PREDICTED, inliers=inliers, feature_count=feature_count)

    def _declare_lost(self, inliers: int, feature_count: int) -> TrackResult:
        self._state = TrackState.LOST
        self._points = None
        return self._result(TrackState.LOST, inliers=inliers, feature_count=feature_count)

    # -- lost/reacquisition branch ------------------------------------------ #

    def _attempt_reacquisition(self, gray: np.ndarray, timestamp_s: float) -> TrackResult:
        """Locate the reference patch again, and *verify* the match before trusting it.

        Template matching (translation only) is deliberately simpler than the
        similarity-transform tracking used once locked on: this step only has
        to answer "is this the same thing we lost," not track it precisely.
        The correlation score is exactly that verification, so a reacquired
        position is never accepted on the strength of "a tracker found
        something" alone - it must resemble the cup that was actually there.
        """
        candidate = self._match_reference_patch(gray)
        if candidate is None:
            return self._result(TrackState.LOST, inliers=0, feature_count=0)

        self._outlet = candidate
        self._last_confident_outlet = candidate.copy()
        self._last_confident_timestamp = timestamp_s
        self._velocity = np.zeros(2, dtype=np.float64)
        self._bridge_start_s = None
        self._state = TrackState.TRACKED
        self._points = _detect_features(gray, candidate[1])

        feature_count = 0 if self._points is None else len(self._points)
        return self._result(
            TrackState.TRACKED, inliers=0, feature_count=feature_count, reacquired=True
        )

    def _match_reference_patch(self, gray: np.ndarray) -> np.ndarray | None:
        patch = self._reference_patch
        patch_h, patch_w = patch.shape[:2]
        if gray.shape[0] < patch_h or gray.shape[1] < patch_w:
            return None
        response = cv2.matchTemplate(gray, patch, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(response)
        if max_val < self._config.track_reacquire_min_correlation:
            return None
        outlet_x = max_loc[0] + self._patch_outlet_offset[0]
        outlet_y = max_loc[1] + self._patch_outlet_offset[1]
        candidate = np.array([outlet_x, outlet_y], dtype=np.float64)
        if not self._within_bounds(candidate):
            return None
        return candidate

    # -- helpers ------------------------------------------------------------ #

    def _within_bounds(self, point: np.ndarray) -> bool:
        height, width = self._bounds
        return 0.0 <= point[0] < width and 0.0 <= point[1] < height

    def _extract_patch(
        self,
        gray: np.ndarray,
        center: np.ndarray,
        anchor: np.ndarray,
        half_px: int,
    ) -> tuple[np.ndarray, tuple[float, float]]:
        """Crop a patch around ``center``; return it with ``anchor``'s offset within it."""
        height, width = gray.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        x0, x1 = max(0, cx - half_px), min(width, cx + half_px + 1)
        y0, y1 = max(0, cy - half_px), min(height, cy + half_px + 1)
        patch = gray[y0:y1, x0:x1].copy()
        return patch, (float(anchor[0]) - x0, float(anchor[1]) - y0)

    def _result(
        self,
        state: TrackState,
        *,
        inliers: int,
        feature_count: int,
        reacquired: bool = False,
    ) -> TrackResult:
        offset = self._outlet - self._reference_outlet
        return TrackResult(
            state=state,
            outlet_x=float(self._outlet[0]),
            outlet_y=float(self._outlet[1]),
            offset_x=float(offset[0]),
            offset_y=float(offset[1]),
            inliers=inliers,
            feature_count=feature_count,
            reacquired=reacquired,
        )
