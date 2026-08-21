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

    While ``zahn_track_outlet=True`` (the default), a caller must treat this
    as a hard failure - report it with an explicit reason and no confirmed
    measurement - not fall back to the fixed-ROI path. That path is the
    known-bad geometry this stage exists to move away from, and using it
    silently on the exact condition tracking could not even start under
    would guess under precisely the circumstance guessing is meant to be
    refused. Falling back to fixed geometry is available only via an
    explicit ``zahn_track_outlet=False`` - a caller choice made before
    tracking was ever attempted, not a reaction to this exception. See
    ``ZahnCupDetector._tracking_unavailable_result``.
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


def _feature_mask(
    shape: tuple[int, int],
    outlet_xy: tuple[float, float],
    *,
    half_width_px: int,
    height_above_px: int,
) -> np.ndarray:
    """Restrict feature detection to a cup-sized box above the outlet.

    Bounded on *both* axes, not just vertically: with an independently
    moving camera and cup, an unbounded "everything above the outlet, full
    crop width" mask lets strong background corners into the point set the
    similarity transform is fit to, and the fitted transform then follows
    whichever motion (camera or cup) has the stronger texture rather than
    the cup specifically. The box is centred horizontally on the outlet and
    reaches upward from it, sized by the caller from the cup's own scale
    (the guard region), not the wider search window - the search window
    bounds where the tracker may end up, not what it is allowed to treat as
    "the cup" when picking features.

    Excluding the area at and below the orifice keeps the liquid itself -
    which moves independently of the rigid cup body - out of the point set.
    """
    height, width = shape
    mask = np.zeros(shape, dtype=np.uint8)
    bottom = max(1, int(outlet_xy[1]) - _FEATURE_EXCLUSION_BELOW_OUTLET_PX)
    top = max(0, bottom - height_above_px)
    left = max(0, int(outlet_xy[0]) - half_width_px)
    right = min(width, int(outlet_xy[0]) + half_width_px)
    if bottom > top and right > left:
        mask[top:bottom, left:right] = 255
    return mask


def _detect_features(
    gray: np.ndarray,
    outlet_xy: tuple[float, float],
    *,
    half_width_px: int,
    height_above_px: int,
) -> np.ndarray | None:
    mask = _feature_mask(
        gray.shape[:2], outlet_xy, half_width_px=half_width_px, height_above_px=height_above_px
    )
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
        cup_half_width_px: int,
        cup_height_above_px: int,
        reference_timestamp_s: float = 0.0,
    ) -> None:
        """Build the reference patch/features from ``reference_gray``.

        ``reference_gray`` is always treated as the tracker's first
        confidently-observed frame: every caller either feeds frames
        starting from this exact one, or (for a reference frame that is not
        chronologically first - see ``ZahnCupDetector``'s
        ``outlet_reference_s`` and ``_process_pre_reference_span``) tracks
        *backward* from it through already-decoded frames via ordinary
        optical flow, which needs no separate bootstrap state: LK does not
        care which way time runs, only that consecutive frames are close in
        content.
        """
        self._config = config
        self._bounds = reference_gray.shape[:2]  # (height, width)
        self._reference_outlet = np.array(outlet_xy, dtype=np.float64)
        self._outlet = self._reference_outlet.copy()
        # The cup-relative box feature detection is restricted to, sized by
        # the caller from the cup's own scale (see _feature_mask). Stored so
        # every later redetect - after a thin point set, after reacquisition
        # - uses the same bound, not the full crop.
        self._cup_half_width_px = cup_half_width_px
        self._cup_height_above_px = cup_height_above_px

        points = self._detect_cup_features(reference_gray, outlet_xy)
        if points is None or len(points) < _MIN_INIT_FEATURES:
            raise TrackerInitError(
                f"Only {0 if points is None else len(points)} trackable feature(s) "
                f"found in the cup region above the outlet (need >= {_MIN_INIT_FEATURES})."
            )
        self._prev_gray = reference_gray
        self._bridge_start_s: float | None = None
        self._velocity = np.zeros(2, dtype=np.float64)
        # A reacquisition candidate awaiting a second, consistent frame
        # before it is trusted - see _attempt_reacquisition.
        self._pending_reacquisition: np.ndarray | None = None
        self._points: np.ndarray | None = points
        self._state = TrackState.TRACKED
        self._last_confident_outlet = self._outlet.copy()
        # Set from construction, not left None until the first successful
        # update(): the reference frame *is* a confident observation, and a
        # gap on the very next frame must still have a timestamp and a
        # (zero, until real motion is observed) velocity to bridge from.
        self._last_confident_timestamp: float | None = reference_timestamp_s

        # The reacquisition patch is centred on a strong detected feature
        # near the *top* of the search box, not the single strongest corner
        # wherever it happens to be, not the outlet itself, and not the
        # centroid of every feature found: the outlet is a narrow, often
        # near-featureless point (the same reason a single click there is
        # not enough to track frame to frame - see the module docstring),
        # and averaging every feature's position dilutes towards whatever
        # smooth, low-response area most of them sit in. The top-of-box bias
        # (Codex review, fourth round) is a rim prior: on a translucent,
        # low-texture cup the single strongest corner anywhere in the box
        # can just as easily be background structure showing through the
        # body lower down, while the rim - nearest the top, farthest from
        # the outlet - is "the one feature a real translucent cup usually
        # keeps a visible edge on" (see tests/_synthetic_handheld.py's
        # _draw_translucent_cup). cv2.goodFeaturesToTrack returns points in
        # decreasing order of corner response, so restricting the choice to
        # a small top tier and then taking the highest of those balances
        # "distinctive enough for template matching" against "likely to
        # survive translucency" - a shape prior, not a guarantee: it does
        # nothing for a cup whose rim itself is also low-contrast. The tier
        # is deliberately the top 3 by response, not the top third: this
        # patch also anchors the continuous per-frame patch-correlation
        # gate below (every accepted frame, not just reacquisition), so a
        # weak corner promoted purely for being higher in the box costs
        # ordinary opaque-cup tracking precision, not just reacquisition
        # robustness. Measured on the hand-held regression fixture (Stage 1
        # two-anchor round): a top-third tier left 41% of frames untracked
        # and the recovered start 0.8s off; the top-3 tier matches the
        # unbiased points[0] baseline (single-digit percent untracked, exact
        # start) while still preferring the highest of a genuinely strong
        # few over the single strongest wherever it sits. The outlet's
        # position is still recovered precisely via the stored offset from
        # whichever point is chosen.
        flat_points = points.reshape(-1, 2)
        top_tier = flat_points[: max(1, min(3, len(flat_points) // 3))]
        anchor_point = top_tier[np.argmin(top_tier[:, 1])]
        self._reference_patch, self._patch_outlet_offset = self._extract_patch(
            reference_gray, anchor_point, self._outlet, patch_half_px
        )
        # Codex review: a real translucent, low-texture cup can let
        # goodFeaturesToTrack/RANSAC lock onto structure visible through or
        # around the cup rather than the cup itself - inlier count and
        # per-frame displacement alone do not catch this, since the wrong
        # structure can still move smoothly and pass both. This offset
        # (translation-only; per-frame rotation/scale over a hand-held
        # clip's frame interval is small enough for a plausibility check,
        # unlike for precise tracking) lets _attempt_tracking ask, for any
        # candidate outlet position, "does the anchor feature's implied
        # position still look like the cup we started with" - see
        # _patch_correlation_at.
        self._anchor_offset_from_outlet = anchor_point - self._outlet
        # The reference patch's own top-left corner, relative to the anchor
        # it was cropped around - not necessarily (-patch_half_px,
        # -patch_half_px): _extract_patch clips at the frame edge, so an
        # anchor near an edge (a real possibility - it is simply the
        # strongest corner found, wherever that is) produces a shorter,
        # asymmetric patch. _patch_correlation_at must slide this exact
        # rectangle to a new center, not re-derive a symmetric one from
        # patch_w/patch_h alone, or it recomputes a window that was never
        # actually extracted and can end up entirely outside the frame even
        # at zero displacement.
        anchor_cx = int(round(anchor_point[0]))
        anchor_cy = int(round(anchor_point[1]))
        patch_x0 = max(0, anchor_cx - patch_half_px)
        patch_y0 = max(0, anchor_cy - patch_half_px)
        self._patch_offset_from_anchor = (patch_x0 - anchor_cx, patch_y0 - anchor_cy)

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
                            # Inlier count and displacement alone accept any
                            # transform enough of the point set agrees on -
                            # on a translucent, low-texture cup, that
                            # majority can be structure visible through or
                            # around the cup rather than the cup itself,
                            # and it can move smoothly enough to pass both
                            # checks (Codex review). Verify the anchor
                            # feature's implied position under this
                            # candidate still resembles the reference patch
                            # before trusting it - the same "verified, not
                            # assumed" bar reacquisition already has to
                            # clear, applied continuously rather than only
                            # after a loss.
                            implied_anchor = candidate + self._anchor_offset_from_outlet
                            correlation = self._patch_correlation_at(gray, implied_anchor)
                            if correlation >= cfg.track_min_patch_correlation:
                                accepted_outlet = candidate
                                surviving_points = target[inlier_mask.ravel() == 1].reshape(
                                    -1, 1, 2
                                )

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
            fresh = self._detect_cup_features(gray, (outlet[0], outlet[1]))
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
        self._pending_reacquisition = None  # a fresh loss starts its own confirmation count
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

        A single frame's correlation clearing the threshold is still not
        enough on its own to flip back to `tracked` (Codex review, fourth
        round): a real clip showed two brief (0.166s, 0.067s) spurious
        correlations promote straight to a confident, wrong re-lock. This
        now requires *two consecutive* frames to independently clear the
        threshold and mutually agree on where (within
        `track_max_frame_displacement_px`, the same bound a single tracked
        frame's own displacement is held to) before confirming - a one-off
        spurious match essentially never survives a second independent
        frame's worth of scrutiny, while a genuine reacquisition, sitting on
        the actual cup, keeps matching on the very next frame as a matter of
        course.
        """
        candidate = self._match_reference_patch(gray)
        if candidate is None:
            self._pending_reacquisition = None
            return self._result(TrackState.LOST, inliers=0, feature_count=0)

        if self._pending_reacquisition is None:
            self._pending_reacquisition = candidate
            return self._result(TrackState.LOST, inliers=0, feature_count=0)

        displacement = float(np.hypot(*(candidate - self._pending_reacquisition)))
        if displacement > self._config.track_max_frame_displacement_px:
            # Not a consistent reconfirmation of the pending candidate - but
            # this frame's own hit is still entitled to its own two-frame
            # confirmation starting now, the same as any other first hit.
            self._pending_reacquisition = candidate
            return self._result(TrackState.LOST, inliers=0, feature_count=0)

        self._pending_reacquisition = None
        self._outlet = candidate
        self._last_confident_outlet = candidate.copy()
        self._last_confident_timestamp = timestamp_s
        self._velocity = np.zeros(2, dtype=np.float64)
        self._bridge_start_s = None
        self._state = TrackState.TRACKED
        self._points = self._detect_cup_features(gray, (candidate[0], candidate[1]))

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

    def _patch_correlation_at(self, gray: np.ndarray, center: np.ndarray) -> float:
        """Correlation between the reference patch and this frame at ``center``.

        Same-size, position-only correlation, not a search - cheap enough to
        run on every tracked frame, unlike _match_reference_patch's sliding
        window over the whole crop. Returns -1.0 (below any real threshold)
        when the reference-sized patch does not fully fit at this position,
        so an edge case degrades to "cannot verify," never to a spuriously
        high or low score from a shrunk comparison.
        """
        patch_h, patch_w = self._reference_patch.shape[:2]
        cx, cy = int(round(center[0])), int(round(center[1]))
        dx0, dy0 = self._patch_offset_from_anchor
        x0, y0 = cx + dx0, cy + dy0
        x1, y1 = x0 + patch_w, y0 + patch_h
        if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]:
            return -1.0
        candidate_patch = gray[y0:y1, x0:x1]
        response = cv2.matchTemplate(candidate_patch, self._reference_patch, cv2.TM_CCOEFF_NORMED)
        return float(response[0, 0])

    def reference_patch_correlation(
        self, gray: np.ndarray, outlet_xy: tuple[float, float]
    ) -> float:
        """Correlation between this tracker's stored reference patch and
        ``gray`` at ``outlet_xy`` (an *outlet* position, converted to the
        anchor position ``_patch_correlation_at`` expects the same way
        ``_attempt_tracking`` does).

        Public entry point for verifying a *candidate* position before
        trusting it, without first constructing a whole new tracker there -
        used by ``ZahnCupDetector`` to check a recentre candidate is
        genuinely the cup this tracker was already following, not whatever
        happens to sit at an extrapolated position during a real occlusion
        (Codex review: a fresh tracker built there would report `tracked`
        unconditionally, with no correlation check at all - exactly the
        "verified, not assumed" gap reacquisition already closes for a
        cold search within a fixed window, reopened for a moving one).
        """
        implied_anchor = np.array(outlet_xy, dtype=np.float64) + self._anchor_offset_from_outlet
        return self._patch_correlation_at(gray, implied_anchor)

    # -- helpers ------------------------------------------------------------ #

    def _detect_cup_features(
        self, gray: np.ndarray, outlet_xy: tuple[float, float]
    ) -> np.ndarray | None:
        """goodFeaturesToTrack, bounded to the cup-sized box above ``outlet_xy``.

        The bound is re-centred on the *current* outlet estimate every call
        (init, low-point-count redetect, post-reacquisition redetect), not
        fixed to where the outlet started - the cup has moved by the time any
        of those later calls happen.
        """
        return _detect_features(
            gray,
            outlet_xy,
            half_width_px=self._cup_half_width_px,
            height_above_px=self._cup_height_above_px,
        )

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
