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

Camera-motion compensation (Stage 3)
-------------------------------------

Every check above verifies a candidate against this tracker's *own* stored
reference - inlier count, displacement, patch correlation. On a translucent,
low-texture cup, that verification is circular: if the reference patch was
itself extracted from background visible through the cup, "more of that
same background, wherever it later appears" answers every one of those
checks correctly (diagnostics/stage1/STAGE1_REPORT.md §24.2/§27.2). Closing
that gap needs evidence that is not derived from the reference patch at
all - ``_BackgroundMotionEstimator`` supplies it, tracking features
*outside* the cup/outlet/guard/stream box to estimate the frame's dominant
camera/background motion independently, and ``_background_veto`` /
``_background_veto_since_confident`` reject a candidate whose own
displacement is indistinguishable from that background motion alone - real
cup motion (a hand holding the cup independently of the camera) has a
residual on top of it; background seen through the cup does not. This is
purely a veto: an unavailable estimate (too little background texture) or
one with no real camera motion to test against never grants trust a
candidate would not already have earned from the checks above - see
``ZahnConfig.background_motion_min_signal_px``/``_min_residual_px``.

The follow-up round that made feature *selection* itself draw on this
compensated evidence (``_detect_cup_features``/``_foreground_residual_
mask``, diagnostics/stage1/STAGE1_REPORT.md §29) still failed the real-clip
gate: ``used_residual`` fired on 805/811 frames, but only 17 ended up
trusted, because residual-filtered sparse *corner* features simply do not
exist in enough density on a real translucent cup, however well the
compensation math around them works. ``_contour_candidate`` (§30) is the
response - not another tuning of the same corner-feature path, a different
kind of evidence: the cup's own visible *geometry* (its rim/side/bottom
silhouette), matched as a gradient/edge template against a background-
suppressed edge map, with its own forward-backward geometric-agreement and
score-margin checks (mirroring the corner path's own FB check, but for an
edge template, not LK points), never gated on the raw-intensity
correlation the corner path uses. It is consulted only when the corner/LK
path itself finds no accepted candidate this frame - never a replacement
for it, since the corner path still demonstrably works on opaque cups.
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

# Stage 3: how far a detected corner's own pixel may sit from the nearest
# residual-motion pixel and still count as "drawn from compensated
# evidence" - see OutletTracker._foreground_residual_mask.
_RESIDUAL_DILATE_KERNEL = np.ones((5, 5), dtype=np.uint8)

# Stage 3, round two: radius (in response-map cells, i.e. pixels) suppressed
# around a contour forward match's own best peak before looking for the
# next-best one - see OutletTracker._contour_forward_match's own docstring
# for why a runner-up is checked at all (ZahnConfig.contour_score_margin).
_CONTOUR_PEAK_SUPPRESS_PX = 5


def _contour_subpixel_offset(response: np.ndarray, max_loc: tuple[int, int]) -> tuple[float, float]:
    """Parabolic sub-pixel refinement of ``cv2.matchTemplate``'s own
    integer-pixel peak location - see
    ``OutletTracker._contour_forward_match``'s own docstring for why this
    exists. Fits a parabola through the peak and its immediate neighbours
    independently along each axis; falls back to ``(0.0, 0.0)`` - no
    refinement - wherever a neighbour is not available (the response
    map's own border) or the fit is degenerate (a flat/saturated
    neighbourhood, where a parabola is not a meaningful model).
    """
    x, y = max_loc
    height, width = response.shape[:2]
    dx = dy = 0.0
    if 0 < x < width - 1:
        left, center, right = (
            float(response[y, x - 1]),
            float(response[y, x]),
            float(response[y, x + 1]),
        )
        denom = left - 2.0 * center + right
        if abs(denom) > 1e-6:
            dx = max(-0.5, min(0.5, 0.5 * (left - right) / denom))
    if 0 < y < height - 1:
        top, center, bottom = (
            float(response[y - 1, x]),
            float(response[y, x]),
            float(response[y + 1, x]),
        )
        denom = top - 2.0 * center + bottom
        if abs(denom) > 1e-6:
            dy = max(-0.5, min(0.5, 0.5 * (top - bottom) / denom))
    return dx, dy


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
    # Stage 3 camera-motion compensation - see the module docstring and
    # _BackgroundMotionEstimator. background_dx/dy/available describe this
    # frame's independently-estimated camera/background motion regardless
    # of outcome (0.0/False when unavailable); residual_px is the cup
    # candidate's own displacement once that background motion is
    # subtracted out, only ever populated when there was a background
    # signal to compare against; rejection_reason names *why* a frame that
    # would otherwise have been accepted was not - "background_consistent"
    # today, None whenever the veto did not change the outcome (including
    # every frame before Stage 3 existed). used_residual records whether
    # *this* frame's own feature selection actually drew on compensated
    # (warp-stabilized residual-motion) evidence - see
    # OutletTracker._detect_cup_features/_foreground_residual_mask - as
    # opposed to the veto merely rejecting a candidate after the fact.
    background_dx: float = 0.0
    background_dy: float = 0.0
    background_available: bool = False
    residual_px: float | None = None
    rejection_reason: str | None = None
    used_residual: bool = False
    # Stage 3, round two - see the module docstring and
    # OutletTracker._contour_candidate. contour_available records whether a
    # contour/edge-silhouette candidate search was actually attempted this
    # frame (only ever true when the corner/LK path itself found no accepted
    # candidate, and a large enough search window existed); contour_score is
    # the best forward edge-template match score achieved, whenever a search
    # ran (None otherwise - not merely "low", genuinely never computed);
    # used_contour records whether *this* frame's own accepted outlet
    # position was actually carried from the fitted cup silhouette rather
    # than the corner/LK transform. rejection_reason gains contour-specific
    # values ("contour_low_score", "contour_roundtrip_mismatch",
    # "contour_out_of_bounds") when the corner path itself found nothing to
    # reject (no rejection_reason of its own) but a contour search was
    # attempted and did not clear its own bar.
    contour_available: bool = False
    contour_score: float | None = None
    used_contour: bool = False


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


def _background_mask(
    shape: tuple[int, int],
    outlet_xy: tuple[float, float],
    *,
    half_width_px: int,
    above_px: int,
    below_px: int,
) -> np.ndarray:
    """Everywhere outside a cup+outlet+stream-sized box around the outlet.

    Stage 3 (camera-motion compensation): unlike ``_feature_mask``'s
    cup-only box (all above the outlet, by design - see its own
    docstring), this also excludes a region *below* it, where the stream
    falls - background/camera motion evidence must come from neither the
    cup nor the stream. Deliberately *not* symmetric at the same height in
    both directions: ``above_px`` matches ``_feature_mask``'s own bound
    exactly, so this can never clip into pixels the cup's own feature
    detection already treats as "the cup" (a real regression - an earlier,
    symmetric-at-full-height version of this box occasionally left a sliver
    of true cup texture outside the exclusion zone, on tight synthetic
    geometry where the cup's own patch nearly fills its allotted box).
    ``below_px`` is independently sized, smaller - a coarse box, not a
    precise stream boundary; the point is "not obviously cup or stream,"
    not an exact cut.
    """
    height, width = shape
    mask = np.full(shape, 255, dtype=np.uint8)
    top = max(0, int(outlet_xy[1]) - above_px)
    bottom = min(height, int(outlet_xy[1]) + below_px)
    left = max(0, int(outlet_xy[0]) - half_width_px)
    right = min(width, int(outlet_xy[0]) + half_width_px)
    if bottom > top and right > left:
        mask[top:bottom, left:right] = 0
    return mask


@dataclass(frozen=True)
class _BackgroundMotionResult:
    """One frame's background/camera motion estimate, in the tracker's
    local crop pixel space - see ``_BackgroundMotionEstimator``."""

    available: bool
    dx: float
    dy: float
    inliers: int
    feature_count: int


class _BackgroundMotionEstimator:
    """Frame-to-frame dominant camera/background motion, from features
    outside the cup/outlet/guard/stream box - Stage 3 camera-motion
    compensation (see the module docstring, and diagnostics/stage1/
    STAGE1_REPORT.md §27-28).

    Deliberately simpler than ``OutletTracker``'s own cup tracking: this
    only ever needs *this frame's* motion (plus a running total since
    construction, for the reacquisition path's longer-span comparison),
    never a notion of being "lost" - a frame with too few background
    features to fit reports ``available=False`` and tries a fresh
    detection next frame, the same way a thin cup point set redetects
    rather than failing outright.
    """

    def __init__(
        self,
        config: ZahnConfig,
        reference_gray: np.ndarray,
        outlet_xy: tuple[float, float],
        *,
        half_width_px: int,
        above_px: int,
        below_px: int,
    ) -> None:
        self._config = config
        self._half_width_px = half_width_px
        self._above_px = above_px
        self._below_px = below_px
        self._points = self._detect(reference_gray, outlet_xy)
        # Total background displacement since construction, evaluated at
        # the outlet's own (moving) position each frame - see update()'s
        # "predicted at the outlet" derivation. Only ever accumulated on an
        # *available* frame; an unavailable one contributes nothing, a
        # conservative (understates true motion) rather than incorrect
        # approximation - see the module docstring.
        self.cumulative_dx = 0.0
        self.cumulative_dy = 0.0
        # This frame's full affine background transform (prev_gray -> gray),
        # None whenever unavailable - see update(). Stage 3 compensation
        # (OutletTracker._foreground_residual_mask) warps the previous frame
        # by this to cancel camera motion before looking for the cup's own,
        # independent motion; the scalar dx/dy above are not enough for
        # that, since warpAffine needs the transform itself.
        self.last_matrix: np.ndarray | None = None

    def _detect(self, gray: np.ndarray, outlet_xy: tuple[float, float]) -> np.ndarray | None:
        mask = _background_mask(
            gray.shape[:2],
            outlet_xy,
            half_width_px=self._half_width_px,
            above_px=self._above_px,
            below_px=self._below_px,
        )
        return cv2.goodFeaturesToTrack(
            gray,
            maxCorners=_MAX_FEATURES,
            qualityLevel=_FEATURE_QUALITY,
            minDistance=_FEATURE_MIN_DISTANCE_PX,
            mask=mask,
            blockSize=_FEATURE_BLOCK_SIZE,
        )

    def update(
        self, prev_gray: np.ndarray, gray: np.ndarray, outlet_xy: tuple[float, float]
    ) -> _BackgroundMotionResult:
        cfg = self._config
        min_features = cfg.background_motion_min_features
        self.last_matrix = None  # cleared here; set again only on full success below
        if self._points is None or len(self._points) < min_features:
            self._points = self._detect(gray, outlet_xy)
            feature_count = 0 if self._points is None else len(self._points)
            return _BackgroundMotionResult(
                available=False, dx=0.0, dy=0.0, inliers=0, feature_count=feature_count
            )

        forward, fwd_ok, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
            prev_gray,
            gray,
            self._points,
            None,
            winSize=_LK_WIN_SIZE,
            maxLevel=_LK_MAX_LEVEL,
            criteria=_LK_CRITERIA,
        )
        backward, back_ok, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
            gray,
            prev_gray,
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

        feature_count = len(source)
        if feature_count < min_features:
            self._points = self._detect(gray, outlet_xy)
            return _BackgroundMotionResult(
                available=False, dx=0.0, dy=0.0, inliers=0, feature_count=feature_count
            )

        matrix, inlier_mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=_RANSAC_REPROJ_PX,
            maxIters=_RANSAC_MAX_ITERS,
        )
        if matrix is None or inlier_mask is None:
            self._points = self._detect(gray, outlet_xy)
            return _BackgroundMotionResult(
                available=False, dx=0.0, dy=0.0, inliers=0, feature_count=feature_count
            )

        inliers = int(inlier_mask.sum())
        if inliers < min_features:
            self._points = self._detect(gray, outlet_xy)
            return _BackgroundMotionResult(
                available=False, dx=0.0, dy=0.0, inliers=inliers, feature_count=feature_count
            )

        # The displacement this transform implies *at the outlet's own
        # position*, not the raw matrix translation term - background
        # features sit far from the outlet, so any rotation/scale in the
        # fit would otherwise make the two not directly comparable to the
        # cup transform's own candidate (also evaluated at the outlet -
        # see OutletTracker._attempt_tracking).
        homogeneous = np.array([outlet_xy[0], outlet_xy[1], 1.0])
        predicted = matrix @ homogeneous
        dx = float(predicted[0] - outlet_xy[0])
        dy = float(predicted[1] - outlet_xy[1])
        self.cumulative_dx += dx
        self.cumulative_dy += dy
        self.last_matrix = matrix

        survivors = target[inlier_mask.ravel() == 1].reshape(-1, 1, 2)
        if len(survivors) < min_features:
            fresh = self._detect(gray, outlet_xy)
            if fresh is not None and len(fresh) >= min_features:
                survivors = fresh
        self._points = survivors

        return _BackgroundMotionResult(
            available=True, dx=dx, dy=dy, inliers=inliers, feature_count=feature_count
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
        self._prev_gray = reference_gray

        # Stage 3: dominant camera/background motion, from features outside
        # the cup/outlet/guard/stream box - see _BackgroundMotionEstimator
        # and the module docstring. The width and above-outlet height match
        # cup feature selection's own bound exactly (cup_half_width_px/
        # cup_height_above_px - the same values _detect_cup_features uses),
        # so this can never pick up real cup texture as "background." Below
        # the outlet (the stream's own space) uses a smaller, independent
        # bound - reusing cup_height_above_px there too, symmetrically,
        # regularly exceeds the capture crop the caller decodes each frame
        # (cup_height_above_px is already sized to the guard region's own
        # generous scale), leaving no pixels at all for background feature
        # detection (caught by a Stage 3 regression: a portrait clip's
        # background estimate came back unavailable on effectively every
        # frame). cup_half_width_px is comfortably smaller in every
        # geometry this stage builds (a Zahn cup's guard region is narrow
        # and tall, not wide) and keeps the total exclusion within the
        # crop's own margin (track_search_margin_px) on every side.
        #
        # Constructed *before* the first _detect_cup_features call below,
        # not after: that call (Stage 3 actual compensation - see
        # _foreground_residual_mask) needs self._background to exist, even
        # though its own last_matrix is still None at this exact point (no
        # prior frame to have computed a transform from yet) - the very
        # first frame falls back to raw-image detection precisely because
        # there is nothing to compensate against, not because the
        # attribute is missing.
        self._background = _BackgroundMotionEstimator(
            config,
            reference_gray,
            outlet_xy,
            half_width_px=cup_half_width_px,
            above_px=cup_height_above_px,
            below_px=cup_half_width_px,
        )
        self._last_background = _BackgroundMotionResult(
            available=False, dx=0.0, dy=0.0, inliers=0, feature_count=0
        )
        # Whether the most recent _detect_cup_features call actually used
        # residual (motion-compensated) evidence to pick points, or fell
        # back to raw-image detection - streamed for diagnostics, see
        # ZahnCupDetector._write_tracking_frame_log.
        self._last_detection_used_residual = False

        points = self._detect_cup_features(reference_gray, outlet_xy)
        if points is None or len(points) < _MIN_INIT_FEATURES:
            raise TrackerInitError(
                f"Only {0 if points is None else len(points)} trackable feature(s) "
                f"found in the cup region above the outlet (need >= {_MIN_INIT_FEATURES})."
            )
        self._bridge_start_s: float | None = None
        self._velocity = np.zeros(2, dtype=np.float64)
        # A reacquisition candidate awaiting a second, consistent frame
        # before it is trusted - see _attempt_reacquisition. _source records
        # which mechanism produced it ("intensity" or, Stage 3 round two,
        # "contour") - the two-frame persistence check requires the *same*
        # mechanism to agree with itself twice, not one hit from each,
        # since the two use unrelated evidence and a coincidental
        # cross-mechanism agreement says less than either one's own
        # internal consistency does.
        self._pending_reacquisition: np.ndarray | None = None
        self._pending_reacquisition_source: str | None = None
        self._points: np.ndarray | None = points
        self._state = TrackState.TRACKED
        self._last_confident_outlet = self._outlet.copy()
        # Set from construction, not left None until the first successful
        # update(): the reference frame *is* a confident observation, and a
        # gap on the very next frame must still have a timestamp and a
        # (zero, until real motion is observed) velocity to bridge from.
        self._last_confident_timestamp: float | None = reference_timestamp_s
        # Snapshot of self._background's cumulative offset at the moment
        # _last_confident_outlet/_timestamp were last set - see
        # _background_veto_since_confident. The reference frame is itself a
        # confident observation (same reasoning as _last_confident_timestamp
        # above), so this starts at the estimator's own zero.
        self._last_confident_background_cumulative: tuple[float, float] = (0.0, 0.0)
        # A short rolling history of (timestamp, outlet position, background
        # cumulative offset), bounded to background_motion_window_s - see
        # _background_veto. Comparing *cumulative* displacement over a
        # window, not one frame's, is what lets a real hand's own
        # oscillating motion (its instantaneous velocity crosses zero
        # periodically - tremor, natural sway) look unremarkable near a
        # turning point without a single such frame - or several - being
        # mistaken for background-consistency (Codex review: an earlier,
        # purely per-frame version of this veto rejected clearly-correct
        # opaque-cup tracking at effectively random points in the hand's
        # own motion cycle).
        self._motion_history: list[tuple[float, float, float, float, float]] = []

        self._patch_half_px = patch_half_px
        self._reference_refined = False
        self._build_reference_patch(reference_gray, points, self._outlet)

        # Stage 3, round two: the cup's visible geometry (rim/side/bottom
        # silhouette), as a fallback for when corner/LK evidence itself
        # finds no accepted candidate - see the module docstring and
        # _contour_candidate. Anchored on the exact same feature
        # _build_reference_patch just chose (recovered from the offset it
        # stored, not re-selected independently), so both templates agree
        # on "where the cup's own strongest, rim-biased evidence is" - see
        # _build_reference_patch's own "rim prior" docstring.
        anchor_point = self._outlet + self._anchor_offset_from_outlet
        self._contour_refined = False
        self._last_contour_available = False
        self._last_contour_score: float | None = None
        self._last_contour_rejection: str | None = None
        self._last_contour_anchor: np.ndarray | None = None
        self._last_used_contour = False
        self._build_contour_evidence(reference_gray, anchor_point, self._outlet)

    # -- public ------------------------------------------------------------ #

    @property
    def state(self) -> TrackState:
        return self._state

    def update(self, gray: np.ndarray, timestamp_s: float) -> TrackResult:
        """Feed the next frame (same crop geometry as initialisation)."""
        # Computed unconditionally, tracked/predicted/lost alike - Stage 3's
        # veto checks (in _attempt_tracking/_attempt_reacquisition) and the
        # streamed diagnostics both need this frame's background estimate
        # regardless of the cup's own state, and a LOST run's background
        # motion must keep accumulating so a later reacquisition's "since
        # last confident" comparison (_background_veto_since_confident)
        # covers the whole gap, not just the frames the cup happened to be
        # tracked on.
        self._last_background = self._background.update(
            self._prev_gray, gray, (self._outlet[0], self._outlet[1])
        )
        # Recorded before this frame's own candidate is known - "where did
        # we believe the outlet was, going into this frame" - so
        # _background_veto can compare against a point genuinely
        # window_s ago, not one that already includes this frame's result.
        self._motion_history.append(
            (
                timestamp_s,
                float(self._outlet[0]),
                float(self._outlet[1]),
                self._background.cumulative_dx,
                self._background.cumulative_dy,
            )
        )
        window_s = self._config.background_motion_window_s
        while len(self._motion_history) > 1 and timestamp_s - self._motion_history[0][0] > window_s:
            self._motion_history.pop(0)
        # Reset every frame, unconditionally - these describe *this* call's
        # own contour attempt (or lack of one), the same "never invented,
        # only computed" convention background diagnostics already follow.
        # Set again inside _contour_candidate/_attempt_tracking/
        # _attempt_reacquisition below when a search actually runs.
        self._last_contour_available = False
        self._last_contour_score = None
        self._last_contour_rejection = None
        self._last_used_contour = False
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
        residual_px: float | None = None
        rejection_reason: str | None = None

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
                                residual_px, rejection_reason = self._background_veto(
                                    candidate, timestamp_s
                                )
                                if rejection_reason is None:
                                    accepted_outlet = candidate
                                    surviving_points = target[inlier_mask.ravel() == 1].reshape(
                                        -1, 1, 2
                                    )

        if accepted_outlet is not None:
            return self._accept_tracked(
                accepted_outlet,
                surviving_points,
                gray,
                timestamp_s,
                inliers,
                feature_count,
                residual_px=residual_px,
            )

        # Stage 3, round two: the corner/LK path itself found no accepted
        # candidate this frame - either no plausible transform at all, or
        # one the background veto above just rejected. Rather than falling
        # straight to bridge/lost (the real-clip failure mode this round
        # exists to fix - see the module docstring), try recovering the
        # cup from its own visible geometry instead of interior texture.
        # "Never trust raw-intensity matches through the cup": this
        # acceptance path never calls _patch_correlation_at - the contour
        # match's own forward-backward and score-margin checks are the
        # authority, not the raw-intensity reference patch above.
        # Score is not consumed directly here - _contour_candidate already
        # records it on self._last_contour_score for _result()'s diagnostics.
        contour_outlet, _ = self._contour_candidate(
            gray, self._predict_outlet(timestamp_s), self._config.contour_search_margin_px
        )
        if contour_outlet is not None:
            self._last_used_contour = True
            return self._accept_tracked(
                contour_outlet,
                None,
                gray,
                timestamp_s,
                inliers=0,
                feature_count=0,
                residual_px=residual_px,
            )
        return self._enter_or_continue_bridge(
            timestamp_s,
            inliers,
            feature_count,
            residual_px=residual_px,
            rejection_reason=rejection_reason or self._last_contour_rejection,
        )

    def _predict_outlet(self, timestamp_s: float) -> np.ndarray:
        """Where the outlet is expected to be this frame, from the last
        confident observation and this tracker's own constant-velocity
        model - the same prediction ``_enter_or_continue_bridge`` reports
        as ``predicted``, reused here as the contour search's own center
        so a recovery attempt starts from the tracker's best guess, not
        wherever the cup happened to be last confidently seen."""
        if self._last_confident_timestamp is None:
            return self._outlet
        return self._last_confident_outlet + self._velocity * (
            timestamp_s - self._last_confident_timestamp
        )

    def _background_veto(
        self, candidate: np.ndarray, timestamp_s: float
    ) -> tuple[float | None, str | None]:
        """Stage 3, continuous-tracking path: over the last
        ``background_motion_window_s``, has this cup candidate shown
        genuine motion of its own, or is its cumulative displacement
        exactly explained by background/camera motion alone?

        Only ever a veto - an *available* background estimate with a real
        signal to test against can reject a candidate the checks above
        would otherwise accept; it never accepts one those checks reject,
        and an unavailable, signal-free, or not-yet-full-window estimate
        never changes the outcome at all (see
        ``background_motion_min_signal_px``'s own docstring in
        ``ZahnConfig``).

        A *window*, not a single frame's displacement (Codex review: an
        earlier, purely per-frame version of this check rejected clearly-
        correct opaque-cup tracking at effectively random points in the
        hand's own motion cycle - real hand-held motion oscillates, and its
        instantaneous velocity crosses zero periodically, which looks
        exactly like "no independent motion" even for a genuine, correctly
        tracked cup at that instant). ``_motion_history`` (populated in
        ``update()``) supplies the window's starting point.
        """
        background = self._last_background
        if not background.available:
            return None, None
        window_s = self._config.background_motion_window_s
        history = self._motion_history
        if not history or timestamp_s - history[0][0] < window_s:
            return None, None
        origin_ts, origin_x, origin_y, origin_background_dx, origin_background_dy = history[0]
        del origin_ts
        background_window_dx = self._background.cumulative_dx - origin_background_dx
        background_window_dy = self._background.cumulative_dy - origin_background_dy
        background_speed = float(np.hypot(background_window_dx, background_window_dy))
        if background_speed < self._config.background_motion_min_signal_px:
            return None, None
        cup_window_dx = float(candidate[0] - origin_x)
        cup_window_dy = float(candidate[1] - origin_y)
        residual = float(
            np.hypot(cup_window_dx - background_window_dx, cup_window_dy - background_window_dy)
        )
        if residual < self._config.background_motion_min_residual_px:
            return residual, "background_consistent"
        return residual, None

    def _accept_tracked(
        self,
        outlet: np.ndarray,
        points: np.ndarray | None,
        gray: np.ndarray,
        timestamp_s: float,
        inliers: int,
        feature_count: int,
        *,
        residual_px: float | None = None,
    ) -> TrackResult:
        cfg = self._config
        if self._last_confident_timestamp is not None:
            dt = timestamp_s - self._last_confident_timestamp
            if dt > 0:
                self._velocity = (outlet - self._last_confident_outlet) / dt
        self._outlet = outlet
        self._last_confident_outlet = outlet.copy()
        self._last_confident_timestamp = timestamp_s
        self._last_confident_background_cumulative = (
            self._background.cumulative_dx,
            self._background.cumulative_dy,
        )
        self._bridge_start_s = None
        self._state = TrackState.TRACKED

        # Stage 3 actual compensation: re-seed from residual (motion-
        # compensated) evidence on *every* accepted frame a background
        # transform is available, not only once the LK-propagated point
        # set has thinned out - re-detecting only on a low count means
        # this box's compensated evidence is consulted rarely (observed:
        # 3 times across an 811-frame real-clip run), so an initial point
        # set contaminated by background-through-cup keeps getting
        # LK-tracked forward essentially unchallenged for the rest of the
        # run. Continuous re-seeding gives every accepted frame a chance
        # to correct course onto genuine cup evidence, not just the rare
        # frame where the old point set happened to run thin. Falls back
        # to the low-count-only redetect when no compensation is available
        # this frame (unchanged from before Stage 3).
        point_count_thin = points is None or len(points) < cfg.track_min_features
        if self._background.last_matrix is not None or point_count_thin:
            fresh = self._detect_cup_features(gray, (outlet[0], outlet[1]))
            fresh_usable = fresh is not None and len(fresh) >= cfg.track_min_features
            if fresh_usable and (self._last_detection_used_residual or point_count_thin):
                points = fresh
        self._points = points
        self._maybe_refine_reference_patch(gray, outlet)
        self._maybe_refine_contour_evidence(gray, outlet)

        return self._result(
            TrackState.TRACKED,
            inliers=inliers,
            feature_count=feature_count,
            residual_px=residual_px,
        )

    def _enter_or_continue_bridge(
        self,
        timestamp_s: float,
        inliers: int,
        feature_count: int,
        *,
        residual_px: float | None = None,
        rejection_reason: str | None = None,
    ) -> TrackResult:
        cfg = self._config
        if self._bridge_start_s is None:
            self._bridge_start_s = timestamp_s

        bridge_elapsed = timestamp_s - self._bridge_start_s
        if self._last_confident_timestamp is None or bridge_elapsed > cfg.track_max_bridge_s:
            return self._declare_lost(
                inliers, feature_count, residual_px=residual_px, rejection_reason=rejection_reason
            )

        predicted = self._predict_outlet(timestamp_s)
        if not self._within_bounds(predicted):
            return self._declare_lost(
                inliers, feature_count, residual_px=residual_px, rejection_reason=rejection_reason
            )

        self._outlet = predicted
        self._state = TrackState.PREDICTED
        return self._result(
            TrackState.PREDICTED,
            inliers=inliers,
            feature_count=feature_count,
            residual_px=residual_px,
            rejection_reason=rejection_reason,
        )

    def _declare_lost(
        self,
        inliers: int,
        feature_count: int,
        *,
        residual_px: float | None = None,
        rejection_reason: str | None = None,
    ) -> TrackResult:
        self._state = TrackState.LOST
        self._points = None
        self._pending_reacquisition = None  # a fresh loss starts its own confirmation count
        self._pending_reacquisition_source = None
        return self._result(
            TrackState.LOST,
            inliers=inliers,
            feature_count=feature_count,
            residual_px=residual_px,
            rejection_reason=rejection_reason,
        )

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

        Stage 3 adds one more gate after the persistence check passes: the
        candidate's *total* displacement since the last confident
        observation must not be exactly explained by how far the
        background/camera has moved over that same span (see
        ``_background_veto_since_confident``) - the reacquisition-path
        counterpart of ``_attempt_tracking``'s own veto, closing the same
        circular-verification gap for template matching that the continuous
        patch-correlation check already had for the similarity-transform fit.

        Stage 3, round two: when the raw-intensity match itself finds
        nothing, try the same edge/contour silhouette match
        ``_attempt_tracking`` falls back to, searched over the *whole* crop
        (a gap's length since loss is not known in advance, the same reason
        ``_match_reference_patch`` already searches the whole crop rather
        than a window). The two-frame persistence and background-veto
        checks above apply identically regardless of which mechanism
        produced the candidate - only the *source* of two consecutive
        candidates must agree with itself (see ``_pending_reacquisition_
        source``'s own docstring), not merely their positions.
        """
        candidate = self._match_reference_patch(gray)
        source = "intensity"
        if candidate is None:
            height, width = self._bounds
            candidate, _ = self._contour_candidate(
                gray, self._last_confident_outlet, max(width, height)
            )
            source = "contour"

        if candidate is None:
            self._pending_reacquisition = None
            self._pending_reacquisition_source = None
            return self._result(TrackState.LOST, inliers=0, feature_count=0)

        if self._pending_reacquisition is None or self._pending_reacquisition_source != source:
            self._pending_reacquisition = candidate
            self._pending_reacquisition_source = source
            return self._result(TrackState.LOST, inliers=0, feature_count=0)

        displacement = float(np.hypot(*(candidate - self._pending_reacquisition)))
        if displacement > self._config.track_max_frame_displacement_px:
            # Not a consistent reconfirmation of the pending candidate - but
            # this frame's own hit is still entitled to its own two-frame
            # confirmation starting now, the same as any other first hit.
            self._pending_reacquisition = candidate
            self._pending_reacquisition_source = source
            return self._result(TrackState.LOST, inliers=0, feature_count=0)

        residual_px, rejection_reason = self._background_veto_since_confident(candidate)
        if rejection_reason is not None:
            # Background-consistent even after two independent frames
            # agreeing with each other - still not proof it is the cup,
            # only that it is not a one-off spurious match. Restart the
            # persistence count from this candidate, the same as an
            # inconsistent hit above: a later frame might yet show genuine
            # residual motion once the candidate (or the true cup) actually
            # moves independently of the camera.
            self._pending_reacquisition = candidate
            self._pending_reacquisition_source = source
            return self._result(
                TrackState.LOST,
                inliers=0,
                feature_count=0,
                residual_px=residual_px,
                rejection_reason=rejection_reason,
            )

        self._pending_reacquisition = None
        self._pending_reacquisition_source = None
        if source == "contour":
            self._last_used_contour = True
        self._outlet = candidate
        self._last_confident_outlet = candidate.copy()
        self._last_confident_timestamp = timestamp_s
        self._last_confident_background_cumulative = (
            self._background.cumulative_dx,
            self._background.cumulative_dy,
        )
        self._velocity = np.zeros(2, dtype=np.float64)
        self._bridge_start_s = None
        self._state = TrackState.TRACKED
        self._points = self._detect_cup_features(gray, (candidate[0], candidate[1]))
        self._maybe_refine_contour_evidence(gray, candidate)

        feature_count = 0 if self._points is None else len(self._points)
        return self._result(
            TrackState.TRACKED,
            inliers=0,
            feature_count=feature_count,
            reacquired=True,
            residual_px=residual_px,
        )

    def _background_veto_since_confident(
        self, candidate: np.ndarray
    ) -> tuple[float | None, str | None]:
        """Stage 3, reacquisition path: does the candidate's *total*
        displacement since the last confident observation have genuine
        motion of its own, or is it exactly explained by how far the
        background/camera has moved over that same span?

        Mirrors ``_background_veto``'s continuous-tracking check, but
        against the *cumulative* background offset since last confidence -
        a reacquisition can follow an arbitrarily long gap, not just one
        frame - using this tracker's own reference-patch template match,
        not a fresh similarity fit.
        """
        background = self._last_background
        if not background.available:
            return None, None
        cumulative_dx = (
            self._background.cumulative_dx - self._last_confident_background_cumulative[0]
        )
        cumulative_dy = (
            self._background.cumulative_dy - self._last_confident_background_cumulative[1]
        )
        background_speed = float(np.hypot(cumulative_dx, cumulative_dy))
        if background_speed < self._config.background_motion_min_signal_px:
            return None, None
        cup_dx = float(candidate[0] - self._last_confident_outlet[0])
        cup_dy = float(candidate[1] - self._last_confident_outlet[1])
        residual = float(np.hypot(cup_dx - cumulative_dx, cup_dy - cumulative_dy))
        if residual < self._config.background_motion_min_residual_px:
            return residual, "background_consistent"
        return residual, None

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
        """goodFeaturesToTrack, bounded to the cup-sized box above ``outlet_xy``,
        preferring residual (motion-compensated) evidence when there is
        enough of it - see ``_foreground_residual_mask``.

        The bound is re-centred on the *current* outlet estimate every call
        (init, low-point-count redetect, post-reacquisition redetect), not
        fixed to where the outlet started - the cup has moved by the time any
        of those later calls happen.

        Stage 3 actual compensation, not only the veto in ``_background_
        veto``/``_background_veto_since_confident``: those can only ever
        *reject* a candidate the rest of the pipeline already found, which
        does nothing when the raw image never offered a plausible candidate
        to begin with - exactly the real-clip failure mode (Codex review:
        "the implementation only removes trust from a candidate after the
        Stage 1 tracker already finds one; it provides no mechanism to
        recover or follow the cup when Stage 1 loses it"). Restricting
        feature selection itself to pixels that moved independently of the
        background gives the tracker genuine cup/rim evidence to seed
        LK/RANSAC from, on a translucent cup where the raw image's
        strongest corners are just as often background showing through.
        """
        # Detect corners over the *whole* cup box first, exactly as before
        # Stage 3, then filter to the ones that also show residual motion -
        # not the other way around (restricting goodFeaturesToTrack's own
        # search to only the residual-masked pixels). A residual region
        # highlights *where* something moved independently of the camera,
        # but corner detection itself needs real local texture to find
        # anything there at all - a smooth-bodied cup's own residual blob
        # can easily contain no strong corners despite being a large,
        # genuine motion signal, while restricting the search to it starves
        # the search of exactly the well-textured points (a rim's own
        # edge, most often) that both approaches would otherwise agree on.
        # Filtering detected corners by residual, instead, only needs each
        # already-strong corner to individually clear the residual check at
        # its own location - corners sit on intensity edges, and edges are
        # exactly where a real position difference between frames shows up
        # most clearly after the background warp, so this is both cheaper
        # and, empirically, the one that actually finds points (a masked-
        # detection version of this method found usable residual points on
        # 4 of 402 calls in one regression clip; this version - unchanged
        # in every other respect - found them on the clear majority).
        points = _detect_features(
            gray,
            outlet_xy,
            half_width_px=self._cup_half_width_px,
            height_above_px=self._cup_height_above_px,
        )
        residual_mask = self._foreground_residual_mask(gray)
        if points is not None and residual_mask is not None:
            height, width = residual_mask.shape[:2]
            residual_points = []
            for point in points.reshape(-1, 2):
                x, y = int(round(point[0])), int(round(point[1]))
                if 0 <= x < width and 0 <= y < height and residual_mask[y, x] != 0:
                    residual_points.append(point)
            if len(residual_points) >= _MIN_INIT_FEATURES:
                self._last_detection_used_residual = True
                return np.array(residual_points, dtype=points.dtype).reshape(-1, 1, 2)

        self._last_detection_used_residual = False
        return points

    def _foreground_residual_mask(self, gray: np.ndarray) -> np.ndarray | None:
        """Pixels that moved *more* than the background transform alone
        explains - genuine independent (cup) motion, not background moving
        with the camera.

        Warps the previous frame by this frame's background transform
        (``self._background.last_matrix``) and diffs it against the
        current frame: a pixel that is purely background, however
        strongly textured, lines back up after that warp and shows near-
        zero difference; a pixel on the cup - moving with the operator's
        hand, independently of the camera - does not. ``None`` whenever no
        transform is available this frame (construction, before any
        ``update()``; or the background estimator itself lacked enough
        texture/inliers) - callers fall back to plain cup-box detection,
        exactly Stage 1's original behaviour, never an invented signal.
        """
        matrix = self._background.last_matrix
        if matrix is None:
            return None
        height, width = self._bounds
        warped_prev = cv2.warpAffine(self._prev_gray, matrix, (width, height))
        diff = cv2.absdiff(gray, warped_prev)
        _, mask = cv2.threshold(
            diff,
            self._config.background_motion_residual_intensity_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        # Dilated by a few pixels: a detected corner's own coordinate and
        # the residual signal it should coincide with are not pixel-exact -
        # sub-pixel motion, warp interpolation and ordinary sensor noise all
        # shift or soften a real edge's residual by a pixel or two, and
        # goodFeaturesToTrack's own corner localisation has similar slack.
        # Without this, a genuinely-moving corner's own pixel can land just
        # outside its true residual region and be filtered out as if it
        # were background.
        return cv2.dilate(mask, _RESIDUAL_DILATE_KERNEL)

    # -- Stage 3, round two: cup silhouette (edge/contour) evidence -------- #

    def _contour_raw_edges(self, gray: np.ndarray) -> np.ndarray:
        """Gradient-magnitude edge map, no background suppression at all -
        see ``_contour_edge_map``'s own docstring for why this is clipped,
        not per-frame min-max normalised.
        """
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)
        return np.clip(magnitude, 0, 255).astype(np.uint8)

    def _contour_edge_map(self, gray: np.ndarray) -> np.ndarray:
        """Gradient-magnitude edge map, background-suppressed when a Stage
        3 residual signal is available this frame, raw otherwise
        (construction - no prior frame to compensate against yet; or a
        frame with too little background texture to estimate motion at
        all) - the same fallback convention ``_foreground_residual_mask``
        itself follows, one level up.

        A translucent, low-texture cup rarely offers enough *interior*
        corner texture for the corner/LK path to find - the real-clip
        numbers in the module docstring - but its rim/side/bottom boundary
        is a strong intensity edge almost by definition, the same physical
        feature ``_build_reference_patch``'s own "rim prior" already leans
        on for the raw-intensity anchor choice. Suppressing pixels the
        background transform already explains keeps a strongly-textured
        background from contributing its own, unrelated edges to the
        template match.

        Clipped to ``[0, 255]``, not per-frame min-max *normalised*: a
        translucent rim's own edge is genuinely faint (single-digit-to-
        low-tens raw Sobel magnitude, for the low grey-level contrast this
        stage's own cup rendering uses), and a full-frame adaptive
        normalisation rescales it relative to whatever the single
        strongest edge *anywhere* in the frame happens to be that frame -
        a stray strong edge elsewhere (the guard-region boundary, a
        distractor at the crop's far side) then dominates the stretch and
        squashes the rim down to a small, inconsistent fraction of the
        template's own dynamic range, frame to frame (a real bug found
        while calibrating this round's own synthetic fixture: the
        template's peak value dropped from ~240/255 unclipped to ~70/255
        normalised, purely from an unrelated bright region elsewhere in
        the same frame). ``cv2.matchTemplate``'s ``TM_CCOEFF_NORMED`` is
        already invariant to a uniform linear rescale (it normalises by
        each patch's own local mean/variance) - the fix is not more
        stretching, it is removing the per-frame stretch that made "the
        same physical edge" mean a different template value on different
        frames.
        """
        edges = self._contour_raw_edges(gray)
        residual_mask = self._foreground_residual_mask(gray)
        if residual_mask is None:
            return edges
        return cv2.bitwise_and(edges, edges, mask=residual_mask)

    def _build_contour_evidence(
        self, gray: np.ndarray, anchor_point: np.ndarray, outlet: np.ndarray
    ) -> None:
        """(Re)build the cup-silhouette edge template and its round-trip
        context crop, both centred on ``anchor_point`` - see
        ``_contour_candidate`` and ``ZahnConfig.contour_max_roundtrip_px``'s
        own docstring for what the context crop is for.

        Called once at construction (from the same anchor
        ``_build_reference_patch`` just chose, recovered via
        ``_anchor_offset_from_outlet`` rather than re-selected
        independently, so both templates agree on where the cup's
        strongest evidence is) and, Stage 3 round two, at most once more
        (see ``_maybe_refine_contour_evidence``) - the same one-time-
        refinement discipline ``_build_reference_patch`` already
        established, for the same reason: the initial template can only
        ever be built from whatever edge signal exists at construction,
        which a translucent cup's rim may or may not offer strongly, and a
        later frame that has already been independently verified (a
        confirmed contour or corner accept) gives the template a second,
        possibly better chance - bounded to once, not continuous, so this
        cannot become the unverified self-drift the round-trip check
        exists to catch.
        """
        edge_map = self._contour_edge_map(gray)
        self._contour_template, self._contour_outlet_offset = self._extract_patch(
            edge_map, anchor_point, outlet, self._patch_half_px
        )
        context_half = self._patch_half_px + self._config.contour_search_margin_px
        self._contour_context, self._contour_context_anchor_local = self._extract_patch(
            edge_map, anchor_point, anchor_point, context_half
        )
        # The anchor's own offset from the outlet - a Zahn cup's rim (what
        # the template is centred on) sits well above the outlet itself
        # (cup_height_above_px), not beside it, so a contour search
        # centred on the *outlet* prediction with only
        # contour_search_margin_px of margin would not even reach the
        # rim's true location. _contour_candidate uses this to translate
        # a predicted *outlet* position into the anchor position it
        # should actually search around.
        self._contour_anchor_offset_from_outlet = anchor_point - outlet

    def _contour_forward_match(
        self, edge_region: np.ndarray, region_offset: tuple[int, int]
    ) -> tuple[tuple[float, float] | None, float | None]:
        """Best-scoring location of the stored contour template within
        ``edge_region`` (a crop of the current frame's edge map, top-left
        at ``region_offset`` in the tracker's own local coordinates), or
        ``(None, None)`` unless a location clears both
        ``contour_min_match_score`` and ``contour_score_margin`` over the
        next-best, non-overlapping local peak - see
        ``ZahnConfig.contour_score_margin``'s own docstring for why a
        runner-up is checked at all, not just an absolute floor.

        The returned location is parabolically sub-pixel refined
        (``_contour_subpixel_offset``), not the raw integer-pixel
        ``matchTemplate`` peak: a real regression found while calibrating
        this round (a slow, steady pan under 1px/frame) showed integer-
        only matching freezing at the same whole-pixel peak for several
        consecutive frames, which then froze this tracker's own velocity
        estimate - and, with it, where the *next* frame's search window
        was even centred - since consecutive identical positions imply
        zero velocity. Sub-pixel refinement is what lets a true, real
        motion smaller than one pixel still register as motion.
        """
        template = self._contour_template
        th, tw = template.shape[:2]
        if edge_region.shape[0] < th or edge_region.shape[1] < tw:
            return None, None
        response = cv2.matchTemplate(edge_region, template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(response)
        if max_val < self._config.contour_min_match_score:
            return None, None
        suppressed = response.copy()
        y0 = max(0, max_loc[1] - _CONTOUR_PEAK_SUPPRESS_PX)
        y1 = min(suppressed.shape[0], max_loc[1] + _CONTOUR_PEAK_SUPPRESS_PX + 1)
        x0 = max(0, max_loc[0] - _CONTOUR_PEAK_SUPPRESS_PX)
        x1 = min(suppressed.shape[1], max_loc[0] + _CONTOUR_PEAK_SUPPRESS_PX + 1)
        suppressed[y0:y1, x0:x1] = -1.0
        second_val = float(suppressed.max()) if suppressed.size else -1.0
        if max_val - second_val < self._config.contour_score_margin:
            return None, None
        max_loc_xy = (int(max_loc[0]), int(max_loc[1]))
        sub_dx, sub_dy = _contour_subpixel_offset(response, max_loc_xy)
        topleft = (
            region_offset[0] + max_loc_xy[0] + sub_dx,
            region_offset[1] + max_loc_xy[1] + sub_dy,
        )
        return topleft, float(max_val)

    def _contour_roundtrip_ok(self, edge_map: np.ndarray, topleft: tuple[int, int]) -> bool:
        """Bidirectional/geometric-agreement check: the same-sized patch
        the forward match just found *in the current frame* is matched
        back against the round-trip context crop around the true anchor in
        the *reference* frame (cached at construction, refreshed at most
        once - see ``_build_contour_evidence``) - the edge-template
        analogue of the corner path's own forward-backward LK check. A
        genuine match's round trip lands back within
        ``contour_max_roundtrip_px`` of the true anchor; a coincidental
        one, resembling the template's shape without truly being the same
        cup structure, generally does not.
        """
        template = self._contour_template
        context = self._contour_context
        th, tw = template.shape[:2]
        x0, y0 = topleft
        candidate_patch = edge_map[y0 : y0 + th, x0 : x0 + tw]
        if candidate_patch.shape[:2] != (th, tw):
            return False
        if context.shape[0] < th or context.shape[1] < tw:
            return False
        response = cv2.matchTemplate(context, candidate_patch, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(response)
        found_center = (max_loc[0] + tw / 2.0, max_loc[1] + th / 2.0)
        expected = self._contour_context_anchor_local
        distance = float(np.hypot(found_center[0] - expected[0], found_center[1] - expected[1]))
        return distance <= self._config.contour_max_roundtrip_px

    def _contour_candidate(
        self, gray: np.ndarray, search_center: np.ndarray, margin: int
    ) -> tuple[np.ndarray | None, float | None]:
        """Locate the cup from its own edge/silhouette geometry, independent
        of the corner/LK path - see the module docstring and
        ``ZahnConfig.contour_min_match_score``'s own docstring for the
        thresholds this composes. ``search_center`` is an *outlet* position
        (the same space every caller already predicts in), translated
        below to the anchor/rim position the template is actually built
        around (``_contour_anchor_offset_from_outlet``) - a Zahn cup's rim
        sits well above the outlet itself, not beside it, so searching
        around the outlet position directly would not even reach it.

        Only ever called when the corner/LK path itself found no accepted
        candidate this frame (``_attempt_tracking``) or the raw-intensity
        reacquisition match failed (``_attempt_reacquisition``) - never
        overrides a candidate either of those already accepted. Records
        ``self._last_contour_available``/``_score``/``_rejection`` for the
        caller's own diagnostics regardless of outcome (reset once per
        frame in ``update()``); ``contour_available`` stays ``False`` when
        the search window itself was too small to hold the template
        (construction geometry - a search this close to the crop's own
        edge, not a rejection of any evidence), the same "never invented,
        only computed" convention ``_foreground_residual_mask`` follows.
        """
        edge_map = self._contour_edge_map(gray)
        template = self._contour_template
        th, tw = template.shape[:2]
        half_w = tw // 2 + margin
        half_h = th // 2 + margin
        height, width = self._bounds
        anchor_center = search_center + self._contour_anchor_offset_from_outlet
        cx, cy = int(round(anchor_center[0])), int(round(anchor_center[1]))
        x0, x1 = max(0, cx - half_w), min(width, cx + half_w)
        y0, y1 = max(0, cy - half_h), min(height, cy + half_h)
        if x1 - x0 < tw or y1 - y0 < th:
            return None, None
        self._last_contour_available = True
        region = edge_map[y0:y1, x0:x1]
        if not region.any():
            # Nothing survived background suppression *in this specific
            # search window* - not proof there is no cup here, only that
            # nothing moved enough this exact frame to register against
            # background_motion_residual_intensity_threshold (see
            # _contour_edge_map's own docstring: real independent cup
            # motion is not constant, and a single slow frame can leave a
            # genuinely-moving rim's own diff below that deliberately
            # tight, pixel-precision threshold). Falling back to the raw,
            # unsuppressed edge map for *this* window - not silently
            # matching against an empty region - is the same "an absence
            # of evidence is not evidence of absence" fallback
            # _foreground_residual_mask itself already follows one level
            # up for an unavailable transform.
            edge_map = self._contour_raw_edges(gray)
            region = edge_map[y0:y1, x0:x1]
        topleft, score = self._contour_forward_match(region, (x0, y0))
        self._last_contour_score = score
        if topleft is None:
            self._last_contour_rejection = "contour_low_score"
            return None, None
        # The round-trip check works on whole-pixel crops (it only needs
        # ZahnConfig.contour_max_roundtrip_px of tolerance, well above one
        # pixel) - round the sub-pixel-refined location back to int only
        # for that slice; the outlet/anchor positions below keep the full
        # sub-pixel precision (see _contour_forward_match's own docstring
        # for why that precision matters).
        int_topleft = (int(round(topleft[0])), int(round(topleft[1])))
        if not self._contour_roundtrip_ok(edge_map, int_topleft):
            self._last_contour_rejection = "contour_roundtrip_mismatch"
            return None, None
        outlet_xy = (
            topleft[0] + self._contour_outlet_offset[0],
            topleft[1] + self._contour_outlet_offset[1],
        )
        candidate = np.array(outlet_xy, dtype=np.float64)
        if not self._within_bounds(candidate):
            self._last_contour_rejection = "contour_out_of_bounds"
            return None, None
        self._last_contour_rejection = None
        self._last_contour_anchor = np.array(
            [topleft[0] + tw / 2.0, topleft[1] + th / 2.0], dtype=np.float64
        )
        return candidate, score

    def _maybe_refine_contour_evidence(self, gray: np.ndarray, outlet: np.ndarray) -> None:
        """Stage 3, round two: the one-time contour-template refinement
        ``_build_contour_evidence`` describes - called only from
        ``_accept_tracked``/``_attempt_reacquisition``, i.e. only on a
        frame that has already cleared every contour check (forward score
        and margin, round-trip agreement) or the corner path's own checks,
        and only once per tracker (see ``self._contour_refined``), and
        only when *this* frame's own accepted position actually came from
        the contour path (``self._last_used_contour``) - not merely that a
        contour search happened to run.
        """
        if self._contour_refined or not self._last_used_contour:
            return
        if self._last_contour_anchor is None:
            return
        self._contour_refined = True
        self._build_contour_evidence(gray, self._last_contour_anchor, outlet)

    def _build_reference_patch(
        self, gray: np.ndarray, points: np.ndarray, outlet: np.ndarray
    ) -> None:
        """(Re)build the reacquisition/correlation reference patch from
        ``points`` - centred on a strong detected feature near the *top*
        of the search box, not the single strongest corner wherever it
        happens to be, not the outlet itself, and not the centroid of
        every feature found: the outlet is a narrow, often near-
        featureless point (the same reason a single click there is not
        enough to track frame to frame - see the module docstring), and
        averaging every feature's position dilutes towards whatever
        smooth, low-response area most of them sit in. The top-of-box bias
        (Codex review, fourth round) is a rim prior: on a translucent,
        low-texture cup the single strongest corner anywhere in the box
        can just as easily be background structure showing through the
        body lower down, while the rim - nearest the top, farthest from
        the outlet - is "the one feature a real translucent cup usually
        keeps a visible edge on" (see tests/_synthetic_handheld.py's
        _draw_translucent_cup). cv2.goodFeaturesToTrack returns points in
        decreasing order of corner response, so restricting the choice to
        a small top tier and then taking the highest of those balances
        "distinctive enough for template matching" against "likely to
        survive translucency" - a shape prior, not a guarantee: it does
        nothing for a cup whose rim itself is also low-contrast. The tier
        is deliberately the top 3 by response, not the top third: this
        patch also anchors the continuous per-frame patch-correlation
        gate below (every accepted frame, not just reacquisition), so a
        weak corner promoted purely for being higher in the box costs
        ordinary opaque-cup tracking precision, not just reacquisition
        robustness. Measured on the hand-held regression fixture (Stage 1
        two-anchor round): a top-third tier left 41% of frames untracked
        and the recovered start 0.8s off; the top-3 tier matches the
        unbiased points[0] baseline (single-digit percent untracked, exact
        start) while still preferring the highest of a genuinely strong
        few over the single strongest wherever it sits. The outlet's
        position is still recovered precisely via the stored offset from
        whichever point is chosen.

        Called once at construction, and - Stage 3 - at most once more
        after that (see ``_maybe_refine_reference_patch``): the initial
        call has no compensated evidence to draw on (no prior frame
        exists yet to have computed a background transform from), so a
        translucent cup's very first reference patch can only ever be
        built from raw, uncompensated pixels - exactly the reference this
        stage's correlation gate then holds every later frame to,
        regardless of how much better later feature selection becomes.
        Rebuilding once, on the first already-independently-verified frame
        where compensated evidence is available, gives the correlation
        gate itself a chance at a genuinely-cup patch instead - bounded to
        once, not continuous, so this cannot become the same unverified
        self-drift the gate exists to prevent.
        """
        flat_points = points.reshape(-1, 2)
        top_tier = flat_points[: max(1, min(3, len(flat_points) // 3))]
        anchor_point = top_tier[np.argmin(top_tier[:, 1])]
        self._reference_patch, self._patch_outlet_offset = self._extract_patch(
            gray, anchor_point, outlet, self._patch_half_px
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
        self._anchor_offset_from_outlet = anchor_point - outlet
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
        patch_x0 = max(0, anchor_cx - self._patch_half_px)
        patch_y0 = max(0, anchor_cy - self._patch_half_px)
        self._patch_offset_from_anchor = (patch_x0 - anchor_cx, patch_y0 - anchor_cy)

    def _maybe_refine_reference_patch(self, gray: np.ndarray, outlet: np.ndarray) -> None:
        """Stage 3: the one-time reference-patch refinement
        ``_build_reference_patch`` describes - called only from
        ``_accept_tracked``, i.e. only on a frame that has *already*
        cleared every existing check (inliers, displacement, correlation
        against the original patch), and only once per tracker (see
        ``self._reference_refined``), and only when that frame's own
        feature selection actually drew on compensated evidence
        (``self._last_detection_used_residual``) - not merely that a
        background transform happened to be available.
        """
        if self._reference_refined or not self._last_detection_used_residual:
            return
        if self._points is None or len(self._points) < _MIN_INIT_FEATURES:
            return
        self._reference_refined = True
        self._build_reference_patch(gray, self._points, outlet)

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
        residual_px: float | None = None,
        rejection_reason: str | None = None,
    ) -> TrackResult:
        offset = self._outlet - self._reference_outlet
        background = self._last_background
        return TrackResult(
            state=state,
            outlet_x=float(self._outlet[0]),
            outlet_y=float(self._outlet[1]),
            offset_x=float(offset[0]),
            offset_y=float(offset[1]),
            inliers=inliers,
            feature_count=feature_count,
            reacquired=reacquired,
            background_dx=background.dx,
            background_dy=background.dy,
            background_available=background.available,
            residual_px=residual_px,
            rejection_reason=rejection_reason,
            used_residual=self._last_detection_used_residual,
            contour_available=self._last_contour_available,
            contour_score=self._last_contour_score,
            used_contour=self._last_used_contour,
        )
