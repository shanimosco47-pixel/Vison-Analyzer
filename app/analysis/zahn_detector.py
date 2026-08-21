"""Zahn cup efflux timing.

A Zahn cup is a viscosity cup with a calibrated orifice: the operator lifts the
full cup, the liquid drains through the hole and the *efflux time* - from the
first liquid leaving the orifice to the moment the stream breaks for good - is
converted to viscosity from the cup's table.  The only thing this module has to
do is measure that interval honestly.

Method
------

1.  The user marks the outlet.  An analysis region is built extending
    **downward** from it, and only those pixels are decoded and analysed.  A
    surrounding *guard* band is watched too, but only to notice disturbances.
2.  Each frame is compared with a background model of the empty region.  The
    threshold is derived from the frame's own measured noise, so compression
    grain does not become "liquid".
3.  Connected components are classified by shape: the stream is *tall and
    thin*, a falling drop is *small and compact*.  Wide blobs (a hand, a cloth,
    a shadow sweeping through) do not qualify.
4.  Flow start requires the stream to be present **at the outlet** for
    ``flow_start_persistence_s`` continuously.  The reported start is the first
    frame of that run, not the moment persistence was satisfied.
5.  Flow end requires a sustained *absence* of any liquid activity - stream or
    drops - for ``flow_end_persistence_s``.  The reported end is the last frame
    that showed activity.  The tail of a Zahn run is stream -> weak stream ->
    intermittent drops, so a single broken frame must never stop the clock.
6.  A confidence is computed from measurable evidence.  If it is poor the
    result is reported as "review recommended" or "failed"; no number is
    invented.

Every threshold lives in :class:`app.config.ZahnConfig`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TextIO

import cv2
import numpy as np

from ..config import ROI, ZahnConfig, apply_overrides
from ..errors import ConfigurationError, InvalidROIError
from ..logging_setup import get_logger
from ..video.metadata import VideoInfo
from ..video.reader import FrameSample, VideoReader
from ..video.sampling import (
    build_sampling_plan,
    frame_index_at,
    frames_to_seconds,
    scale_factor_for_width,
)
from .base_detector import (
    ActivityScorer,
    ActivityTrace,
    BaseDetector,
    DetectorResult,
    Event,
    EventStatus,
    ProgressReporter,
    ScoreSample,
    null_progress,
)
from .outlet_tracker import OutletTracker, TrackerInitError, TrackResult, TrackState
from .temporal import PersistenceTimer, clamp, safe_ratio

logger = get_logger(__name__)

# The ROI is analysed at this width; a Zahn stream is a few pixels wide, and
# 240 px across the region keeps it several pixels wide while bounding cost.
ANALYSIS_WIDTH_PX = 240

# Progress updates are throttled to keep the job state cheap to poll.
PROGRESS_UPDATE_INTERVAL_S = 0.4

# Diagnostic tracking-state transitions are recorded (timestamp + geometry,
# not pixels) as they happen, bounded so a long run of flicker between states
# cannot grow this without limit.
MAX_DIAGNOSTIC_TRANSITIONS = 60

# How many times one run may recentre its capture/search window on the
# tracker's last credible estimate (see ZahnCupDetector._run_segment). Each
# recentre costs one extra out-of-band frame read; bounding the count caps
# that cost and stops a pathological run (repeatedly drifting back to an
# edge) from recentring without limit, while still comfortably covering the
# handful of large reframings a real hand-held clip actually needs.
MAX_TRACKING_RECENTRES = 8


# --------------------------------------------------------------------------- #
# Region of interest construction
# --------------------------------------------------------------------------- #


def build_roi_from_click(
    click_x: int,
    click_y: int,
    video: VideoInfo,
    config: ZahnConfig,
) -> ROI:
    """Build the analysis region from a single click on the outlet hole.

    The region is centred horizontally on the click and extends downward,
    because that is where the liquid goes.  A few pixels above the click are
    included so the very first liquid leaving the orifice is inside the region.
    """
    if not (0 <= click_x < video.width and 0 <= click_y < video.height):
        raise InvalidROIError("The marked point is outside the video frame.")

    width = min(config.default_roi_width_px, video.width)
    height = max(16, int(video.height * config.roi_height_fraction))

    above = max(2, int(0.02 * video.height))  # small margin above the orifice
    x = int(round(click_x - width / 2))
    y = click_y - above

    x = max(0, min(x, video.width - width))
    y = max(0, min(y, video.height - 1))
    height = min(height, video.height - y)
    # Codex review: on a tall portrait frame, roi_height_fraction can eat
    # nearly all the space below the click - build_guard_roi's downward
    # margin (guard_margin_px // 2) plus the tracker's own search budget
    # (track_search_margin_px, see _build_capture_roi) then has nowhere
    # left to extend, so the *geometry* clips against the frame edge as
    # soon as the tracked outlet drifts down at all, marking otherwise-
    # correctly-tracked frames untrusted for a reason that has nothing to
    # do with tracking quality. Reserve that downward headroom up front so
    # the guard/capture geometry stays valid across the tracker's full
    # configured displacement budget, the same way it already does
    # horizontally via x's clamp above. A pathologically low click still
    # gets the best available height (floor of 16px, matching the
    # unconstrained case above) rather than an error - a shorter analysis
    # region that works beats a taller one that clips.
    downward_headroom = config.guard_margin_px // 2 + config.track_search_margin_px
    height = max(16, min(height, video.height - y - downward_headroom))
    roi = ROI(x=x, y=y, width=width, height=height)
    roi.validate(video.width, video.height)
    return roi


def build_guard_roi(roi: ROI, video: VideoInfo, margin_px: int) -> ROI:
    """The analysis region grown sideways/upward, used for disturbance checks."""
    x = max(0, roi.x - margin_px)
    y = max(0, roi.y - margin_px)
    x2 = min(video.width, roi.x2 + margin_px)
    y2 = min(video.height, roi.y2 + margin_px // 2)
    return ROI(x=x, y=y, width=x2 - x, height=y2 - y)


def _safe_slice(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray | None:
    """A ``width``x``height`` window at ``(x, y)``, or None if it does not fully fit.

    Used only for outlet tracking's guard sub-crop: a partially-clipped slice
    would hand the scorer a different shape than the guard region it was
    built for, so a window that does not fully fit is treated as untrusted
    for that frame rather than silently shrunk.
    """
    source_height, source_width = image.shape[:2]
    x2, y2 = x + width, y + height
    if x < 0 or y < 0 or x2 > source_width or y2 > source_height:
        return None
    return image[y:y2, x:x2]


def _prepare_for_scoring(gray: np.ndarray, scale: float, blur_kernel: int = 3) -> np.ndarray:
    """Downscale and blur a guard sub-crop the way the reader used to.

    Outlet tracking decodes one wide, unscaled, unblurred capture window per
    frame so the tracker sees full-resolution detail; this replicates the
    resize-then-blur step :func:`app.video.reader._prepare` applied to the
    whole guard crop before tracking existed, so the scorer's thresholds stay
    calibrated the same way regardless of which path produced its input.
    """
    if scale < 1.0:
        height, width = gray.shape[:2]
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        gray = cv2.resize(gray, new_size, interpolation=cv2.INTER_AREA)
    if blur_kernel >= 3:
        kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
        gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    return gray


# --------------------------------------------------------------------------- #
# Frame scoring
# --------------------------------------------------------------------------- #


@dataclass
class _Rect:
    """Inner region position inside the guard crop, in analysis pixels."""

    x: int
    y: int
    width: int
    height: int


class StreamActivityScorer(ActivityScorer):
    """Detects liquid (stream or drops) inside the analysis region.

    The scorer receives the *guard* crop; the analysis region is a rectangle
    inside it.  Working from one crop means one memory copy per frame and lets
    the surrounding band be checked for disturbance at no extra decode cost.
    """

    name: ClassVar[str] = "zahn_stream"

    def __init__(self, config: ZahnConfig, inner: _Rect, *, keep_mask: bool = False) -> None:
        self.config = config
        self.inner = inner
        self.keep_mask = keep_mask
        self._background: np.ndarray | None = None
        self._outside_mask: np.ndarray | None = None
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (max(1, config.open_kernel_px), max(1, config.open_kernel_px)),
        )

    def reset(self) -> None:
        self._background = None
        self._outside_mask = None

    # -- helpers ----------------------------------------------------------- #

    def _ensure_outside_mask(self, shape: tuple[int, int]) -> np.ndarray:
        if self._outside_mask is None or self._outside_mask.shape != shape:
            mask = np.ones(shape, dtype=bool)
            inner = self.inner
            mask[inner.y : inner.y + inner.height, inner.x : inner.x + inner.width] = False
            self._outside_mask = mask
        return self._outside_mask

    def _inner_view(self, image: np.ndarray) -> np.ndarray:
        inner = self.inner
        return image[inner.y : inner.y + inner.height, inner.x : inner.x + inner.width]

    # -- scoring ----------------------------------------------------------- #

    def score(self, sample: FrameSample) -> ScoreSample:
        frame = sample.image
        if frame.ndim != 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = frame.astype(np.float32, copy=False)

        if self._background is None:
            self._background = current.copy()
            return ScoreSample(value=0.0, extras={"outlet_score": 0.0, "noise_sigma": 0.0})

        diff = cv2.absdiff(current, self._background)
        diff_median = float(np.median(diff))

        # More than half the watched area changed: the camera was knocked, the
        # light changed, or something crossed the cup.  Such a frame proves
        # nothing about the liquid, and it must be recognised before
        # thresholding because it also inflates the measured noise.
        if diff_median >= self.config.disturbance_median_diff:
            self._background = current.copy()
            return ScoreSample(
                value=0.0,
                disturbed=True,
                extras={
                    "outlet_score": 0.0,
                    "median_diff": diff_median,
                    "noise_sigma": 0.0,
                },
            )

        residual = diff - diff_median
        noise_sigma = 1.4826 * float(np.median(np.abs(residual))) or 1.0
        threshold = max(
            float(self.config.min_abs_diff), self.config.noise_sigma_multiplier * noise_sigma
        )

        mask: np.ndarray = (residual > threshold).astype(np.uint8)
        if self.config.open_kernel_px >= 2:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        # --- disturbance: did the world outside the stream region move? ---- #
        outside = self._ensure_outside_mask(mask.shape)
        outside_pixels = int(outside.sum())
        outside_changed = float(mask[outside].sum()) / outside_pixels if outside_pixels else 0.0
        disturbed = outside_changed >= self.config.disturbance_area_ratio

        inner_mask = self._inner_view(mask)
        activity, outlet, has_liquid, evidence = self._classify(
            inner_mask, self._inner_view(residual)
        )

        # The background is only refreshed while nothing is happening, so the
        # stream can never dissolve into the reference image.
        if not disturbed and activity == 0.0:
            cv2.accumulateWeighted(current, self._background, 0.02)
        elif disturbed:
            self._background = current.copy()

        return ScoreSample(
            value=activity,
            disturbed=disturbed,
            mask=inner_mask if self.keep_mask else None,
            extras={
                "outlet_score": outlet,
                # Any liquid at all - stream *or* a single falling drop.  The
                # end-of-flow rule needs this rather than the coverage score,
                # because one drop covers only a few percent of the region's
                # height and would otherwise read as "no liquid".
                "liquid_present": 1.0 if has_liquid else 0.0,
                "outside_changed_ratio": outside_changed,
                "noise_sigma": noise_sigma,
                "threshold": threshold,
                **evidence,
            },
        )

    def _classify(
        self, mask: np.ndarray, residual: np.ndarray
    ) -> tuple[float, float, bool, dict[str, float]]:
        """Score the analysis region by liquid-shaped connected components.

        Returns:
            ``(activity, outlet, has_liquid, evidence)`` where ``activity`` is
            the fraction of region rows containing liquid, ``outlet`` is the
            same measure restricted to the band under the orifice, and
            ``has_liquid`` is True when any liquid-shaped component survived
            the shape and size filters (a lone drop included).
        """
        height, width = mask.shape
        band_height = max(1, int(round(height * self.config.outlet_band_fraction)))
        empty_evidence = {"stream_contrast": 0.0, "liquid_area_ratio": 0.0, "blob_count": 0.0}
        if not mask.any():
            return 0.0, 0.0, False, empty_evidence

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(mask.shape, dtype=bool)
        kept_area = 0
        blobs = 0

        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            blob_w = int(stats[label, cv2.CC_STAT_WIDTH])
            blob_h = int(stats[label, cv2.CC_STAT_HEIGHT])

            tall_and_thin = (
                blob_h >= self.config.min_stream_height_fraction * height
                and blob_w <= self.config.max_stream_width_fraction * width
            )
            compact_drop = (
                area >= self.config.min_drop_area_px
                and blob_w <= self.config.max_stream_width_fraction * width
                and blob_h >= 2
            )
            if tall_and_thin or compact_drop:
                keep |= labels == label
                kept_area += area
                blobs += 1

        if kept_area == 0:
            return 0.0, 0.0, False, empty_evidence

        rows_with_liquid = np.asarray(keep.any(axis=1))
        activity = float(rows_with_liquid.sum()) / height
        outlet = float(rows_with_liquid[:band_height].sum()) / band_height
        contrast = float(residual[keep].mean()) if keep.any() else 0.0

        return (
            activity,
            outlet,
            True,
            {
                "stream_contrast": contrast,
                "liquid_area_ratio": kept_area / mask.size,
                "blob_count": float(blobs),
            },
        )


# --------------------------------------------------------------------------- #
# Flow state machine
# --------------------------------------------------------------------------- #


@dataclass
class FlowMeasurement:
    """Result of following one Zahn run from start to finish.

    The ``timed_*`` counters cover only the measured interval (flow start to
    flow end); the plain counters cover everything that was analysed, including
    the wait before the cup was lifted.
    """

    start_s: float | None = None
    end_s: float | None = None
    end_confirmed: bool = False
    frames_analysed: int = 0
    frames_with_liquid: int = 0
    frames_disturbed: int = 0
    frames_untracked: int = 0
    timed_frames: int = 0
    timed_frames_with_liquid: int = 0
    breaks: list[tuple[float, float]] = field(default_factory=list)
    start_margin: float = 0.0
    mean_contrast: float = 0.0
    mean_noise_sigma: float = 0.0
    stopped_early: bool = False
    # Set when the reported end sits right after an untrusted (lost/predicted
    # tracking) span wider than zahn_max_endpoint_uncertainty_s: the true
    # break could have happened anywhere in end_uncertainty_bounds, so
    # end_confirmed is also forced False rather than reporting a falsely
    # precise duration. Only set from a gap with no trusted liquid observed
    # between it closing and the end being confirmed - see
    # end_gap_unresolved below for the case where liquid *did* return.
    end_uncertain: bool = False
    end_uncertainty_bounds: tuple[float, float] | None = None
    # Set when an untrusted span wider than zahn_max_endpoint_uncertainty_s
    # occurred anywhere earlier while flow was ongoing, and trusted liquid
    # was seen again afterward - flow visibly continued past it. Codex
    # review (second round): unlike end_uncertainty_bounds above, this gap's
    # own span must NOT be reported as bounding the end - trusted liquid
    # after it proves the true end is not "somewhere in that gap", so using
    # the gap's timestamps there would be a fabricated bound that could
    # exclude the actual later end entirely. What the gap genuinely
    # establishes is narrower: continuity through that span could not be
    # verified, so the measurement as a whole is not confirmed - end_s is
    # still the best available candidate, but end_confirmed is forced False
    # and no precise duration/Event is produced (same suppression as
    # end_uncertain), without claiming to know where the true end is.
    end_gap_unresolved: bool = False
    # Symmetric case: the persistence run that confirmed the start began
    # immediately after an untrusted span wider than
    # zahn_max_endpoint_uncertainty_s. Flow could have started anywhere in
    # start_uncertainty_bounds - a late-but-precise start still reports a
    # falsely precise (shortened) duration, which is exactly as dishonest as
    # a falsely precise end.
    start_uncertain: bool = False
    start_uncertainty_bounds: tuple[float, float] | None = None

    @property
    def efflux_s(self) -> float | None:
        if self.start_s is None or self.end_s is None:
            return None
        return max(0.0, self.end_s - self.start_s)

    @property
    def continuity(self) -> float:
        """Fraction of the timed interval in which liquid was actually visible."""
        if self.timed_frames <= 0:
            return 0.0
        return self.timed_frames_with_liquid / self.timed_frames


class FlowStateMachine:
    """Turns a stream of per-frame observations into start/end timestamps.

    Deliberately independent of OpenCV so it can be tested with synthetic
    sequences (continuous stream, brief flicker, drops at the end, ...).
    """

    def __init__(self, config: ZahnConfig) -> None:
        config.validate()
        self.config = config
        self.start_timer = PersistenceTimer(config.flow_start_persistence_s)
        self.absence_timer = PersistenceTimer(config.flow_end_persistence_s)
        self.measurement = FlowMeasurement()
        self._flowing = False
        self._finished = False
        self._last_activity_s: float | None = None
        self._gap_started_s: float | None = None
        self._frames_since_activity = 0
        self._start_margins: list[float] = []
        self._contrasts: list[float] = []
        self._noise: list[float] = []
        # An "untrusted" span (disturbed, or geometry not confidently known)
        # currently open, and the most recent one that closed after the last
        # confirmed liquid sighting - the window a reported end's true
        # timing could actually fall within. See update()'s `trusted` param.
        self._gap_open = False
        self._gap_after_activity: tuple[float, float] | None = None
        # Codex review (first round): _gap_after_activity alone is forgotten
        # the moment fresh trusted liquid arrives (see _update_flowing), so
        # a long gap in the *middle* of an otherwise continuous-looking run
        # - liquid before, liquid after - left no trace by the time a
        # later, clean end was confirmed. Set once (first qualifying gap
        # only) and never cleared, so it still poisons the eventual end
        # confirmation however long afterward flow actually stops.
        #
        # Deliberately a bool, not the gap's own (start, end) span (Codex
        # review, second round): trusted liquid seen again after this gap
        # closed proves the true end is not "somewhere in that gap" -
        # reporting the gap's own timestamps as end_uncertainty_bounds would
        # be a fabricated bound that could exclude the actual later end
        # entirely. All this flag may honestly claim is that continuity
        # through that span could not be verified - see
        # FlowMeasurement.end_gap_unresolved.
        #
        # Set only once liquid *actually resumes* after a qualifying gap -
        # in _update_flowing's liquid-present branch, not when the gap
        # itself closes (Codex review, third round). A gap closing proves
        # nothing yet about what follows: setting this at close time also
        # mislabelled the ordinary case where trusted absence follows
        # straight through to a confirmed end (liquid never resumed) as
        # "mid-flow", when that case is genuinely just the adjacent-gap
        # one _gap_after_activity already exists for.
        self._unresolved_flow_gap = False
        # Symmetric bookkeeping for the start side: the timestamp of the
        # last trusted frame seen at all (regardless of content - "flow
        # hadn't started as of here" is valid negative evidence even from a
        # trusted, quiet frame), and the most recent gap that closed while
        # still waiting for flow to start. A late-but-precise start is just
        # as falsely precise as a late end: the *duration* comes out short,
        # reported with full confidence, on an interval that could not
        # actually have been observed from the beginning.
        self._last_trusted_timestamp: float | None = None
        self._gap_start_ts: float | None = None
        self._pending_start_gap: tuple[float, float] | None = None

    @property
    def finished(self) -> bool:
        """True once the end of flow has been confirmed (analysis may stop)."""
        return self._finished

    @property
    def flowing(self) -> bool:
        return self._flowing

    def update(self, timestamp_s: float, sample: ScoreSample, *, trusted: bool = True) -> None:
        """Feed one scored frame.

        ``trusted`` is False whenever the frame's geometry is not confidently
        known this instant - a `lost` or `predicted` outlet-tracking state,
        layered on top of (not instead of) the scorer's own
        ``sample.disturbed``. An untrusted frame can neither start nor end the
        flow, the same rule already applied to disturbed frames below. The
        one new behaviour: a gap adjacent to a reported end is remembered so
        the endpoint can be marked uncertain, with explicit bounds, instead of
        falsely precise - see ``zahn_max_endpoint_uncertainty_s``.
        """
        if self._finished:
            return

        measurement = self.measurement
        measurement.frames_analysed += 1
        self._noise.append(float(sample.extras.get("noise_sigma", 0.0)))

        if sample.disturbed or not trusted:
            if sample.disturbed:
                measurement.frames_disturbed += 1
            if not trusted:
                measurement.frames_untracked += 1
                # Scene-wide disturbance (camera knock, a hand crossing) is
                # deliberately NOT gap-tracked here: fresh trusted evidence
                # afterward has always been enough to confirm an end, however
                # long the disturbance ran (see
                # test_disturbance_does_not_count_toward_the_end_persistence).
                # Not knowing *where the outlet is at all* is a qualitatively
                # different, wider uncertainty - that is what
                # zahn_max_endpoint_uncertainty_s bounds, and it is why only
                # `not trusted` (lost/predicted tracking) opens a gap here.
                if not self._gap_open:
                    self._gap_start_ts = (
                        self._last_trusted_timestamp
                        if self._last_trusted_timestamp is not None
                        else timestamp_s
                    )
                self._gap_open = True
            # An untrusted frame proves nothing: it may neither start the
            # clock nor contribute to the "liquid has stopped" evidence. Both
            # persistence runs therefore restart, because each one must be a
            # contiguous run of frames we actually trust. Resetting (rather
            # than pausing) also errs toward continuing to measure: the failure
            # mode becomes "end not confirmed", which is reported for review,
            # instead of an end confirmed on evidence we never had.
            self.start_timer.reset()
            self.absence_timer.reset()
            self._gap_started_s = None
            return

        if self._gap_open:
            self._gap_open = False
            if self._flowing:
                if self._last_activity_s is not None:
                    # The true transition could have happened any time from
                    # the last confirmed liquid sighting through to this
                    # first trusted frame - not just during the untrusted
                    # span itself, since ordinary quiet-but-trusted frames
                    # right before it are equally unable to prove flow had
                    # already ended. Only a candidate here: whether this
                    # becomes an honest end-adjacent bound or gets promoted
                    # to end_gap_unresolved is decided by what happens next,
                    # not by the gap closing alone (Codex review, third
                    # round - see _update_flowing's liquid-present branch).
                    self._gap_after_activity = (self._last_activity_s, timestamp_s)
            elif self._gap_start_ts is not None:
                # Symmetric case, still waiting for flow to start: the last
                # trusted frame before the gap - whatever it showed - is
                # valid negative evidence flow had not started yet. Only
                # relevant if a persistence run beginning exactly here goes
                # on to confirm a start; overwritten by any later gap that
                # closes before that happens.
                self._pending_start_gap = (self._gap_start_ts, timestamp_s)
        self._last_trusted_timestamp = timestamp_s

        activity_score = sample.value
        outlet_score = float(sample.extras.get("outlet_score", 0.0))
        # The scorer reports "any liquid at all" explicitly, because a single
        # falling drop covers only a few percent of the region's height and
        # would never reach the coverage threshold that a stream does.  When a
        # scorer does not supply the flag, fall back to the coverage score.
        explicit_presence = sample.extras.get("liquid_present")
        liquid_present = (
            explicit_presence >= 0.5
            if explicit_presence is not None
            else activity_score >= self.config.activity_threshold
        )

        if liquid_present:
            measurement.frames_with_liquid += 1
            contrast = float(sample.extras.get("stream_contrast", 0.0))
            if contrast:
                self._contrasts.append(contrast)

        if not self._flowing:
            self._update_waiting(timestamp_s, outlet_score)
        else:
            self._update_flowing(timestamp_s, liquid_present)

    def _update_waiting(self, timestamp_s: float, outlet_score: float) -> None:
        """Look for liquid *at the orifice* persisting long enough to be real."""
        at_outlet = outlet_score >= self.config.activity_threshold
        if at_outlet:
            self._start_margins.append(
                safe_ratio(
                    outlet_score - self.config.activity_threshold, self.config.activity_threshold
                )
            )
        else:
            self._start_margins.clear()

        if self.start_timer.update(timestamp_s, at_outlet):
            self._flowing = True
            self.measurement.start_s = self.start_timer.run_start_s
            self.measurement.start_margin = (
                sum(self._start_margins) / len(self._start_margins) if self._start_margins else 0.0
            )
            self._last_activity_s = timestamp_s
            if (
                self._pending_start_gap is not None
                and self._pending_start_gap[1] == self.measurement.start_s
            ):
                gap_start, gap_end = self._pending_start_gap
                if (gap_end - gap_start) > self.config.zahn_max_endpoint_uncertainty_s:
                    # This persistence run began immediately after an
                    # untrusted span: flow could genuinely have started
                    # anywhere in it, not at the first frame we happened to
                    # trust again. Reporting a confirmed, precise start (and
                    # therefore a precise duration) here would invent
                    # certainty the evidence does not support.
                    self.measurement.start_uncertain = True
                    self.measurement.start_uncertainty_bounds = (gap_start, gap_end)
            self._pending_start_gap = None
            logger.info(
                "Zahn flow start detected at %.3fs (persistence %.2fs satisfied at %.3fs)",
                self.measurement.start_s,
                self.config.flow_start_persistence_s,
                timestamp_s,
            )

    def _update_flowing(self, timestamp_s: float, liquid_present: bool) -> None:
        """Wait for a *sustained* absence of stream and drops."""
        self.measurement.timed_frames += 1
        if liquid_present:
            self.measurement.timed_frames_with_liquid += 1
            self._frames_since_activity = 0
            if self._gap_started_s is not None:
                gap_length = timestamp_s - self._gap_started_s
                if gap_length >= self.config.break_report_threshold_s:
                    self.measurement.breaks.append((self._gap_started_s, timestamp_s))
                self._gap_started_s = None
            self._last_activity_s = timestamp_s
            self.absence_timer.reset()
            # Fresh, trusted liquid supersedes the *adjacency* signal: flow
            # demonstrably continued past it, so this particular gap is no
            # longer immediately next to whatever end eventually gets
            # reported - it is promoted here, at the moment liquid actually
            # proves resumption, to _unresolved_flow_gap instead (Codex
            # review, third round: promoting this at gap-close time instead
            # - before knowing whether liquid or absence follows - wrongly
            # relabelled the ordinary adjacent-to-end case too, since that
            # case never reaches this liquid-present branch at all).
            if self._gap_after_activity is not None:
                gap_start, gap_end = self._gap_after_activity
                if (gap_end - gap_start) > self.config.zahn_max_endpoint_uncertainty_s:
                    self._unresolved_flow_gap = True
            self._gap_after_activity = None
            return

        self._frames_since_activity += 1
        if self._gap_started_s is None:
            self._gap_started_s = timestamp_s

        if self.absence_timer.update(timestamp_s, True):
            self.measurement.end_s = self._last_activity_s
            self.measurement.end_confirmed = True
            self._finished = True
            # The frames spent proving the absence are not part of the timed
            # interval; removing them keeps the continuity ratio honest.
            self.measurement.timed_frames -= self._frames_since_activity

            if self._gap_after_activity is not None:
                gap_start, gap_end = self._gap_after_activity
                if (gap_end - gap_start) > self.config.zahn_max_endpoint_uncertainty_s:
                    # No trusted liquid was seen between this gap closing and
                    # the end being confirmed, so the true break could
                    # genuinely have happened anywhere in that span:
                    # reporting a precise, confirmed number would be
                    # inventing certainty the evidence does not support.
                    self.measurement.end_confirmed = False
                    self.measurement.end_uncertain = True
                    self.measurement.end_uncertainty_bounds = (gap_start, gap_end)
            if self._unresolved_flow_gap:
                # An earlier gap whose continuity could not be verified -
                # but trusted liquid *was* seen again afterward, so (Codex
                # review, second round) that gap's own span must NOT be
                # folded into end_uncertainty_bounds: doing so would claim
                # the true end sits somewhere back in that gap, when the
                # evidence actually shows the opposite (flow continued past
                # it). This only forces the measurement unconfirmed; it does
                # not - and must not - narrow where the true end is. Any
                # end_uncertainty_bounds set above (a separate, genuinely
                # adjacent gap) is left untouched, since that one still
                # honestly bounds the end regardless of this concern.
                self.measurement.end_confirmed = False
                self.measurement.end_uncertain = True
                self.measurement.end_gap_unresolved = True
            logger.info(
                "Zahn flow end detected at %.3fs (no liquid for %.2fs, confirmed at %.3fs)",
                self.measurement.end_s if self.measurement.end_s is not None else -1.0,
                self.config.flow_end_persistence_s,
                timestamp_s,
            )

    def finalize(self, last_timestamp_s: float) -> FlowMeasurement:
        """Close the measurement at the end of the analysed range."""
        measurement = self.measurement
        if self._flowing and not measurement.end_confirmed:
            # The recording ran out while liquid was still being seen.  The
            # last activity is a *lower bound* for the end, not the end.
            measurement.end_s = (
                self._last_activity_s if self._last_activity_s is not None else last_timestamp_s
            )
            measurement.end_confirmed = False
        measurement.mean_contrast = (
            sum(self._contrasts) / len(self._contrasts) if self._contrasts else 0.0
        )
        measurement.mean_noise_sigma = sum(self._noise) / len(self._noise) if self._noise else 0.0
        return measurement


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #


@dataclass
class _SegmentResult:
    """How one decode segment (``ZahnCupDetector._run_segment``) ended."""

    status: str  # "finished" | "exhausted" | "recentre" | "init_failed"
    last_timestamp: float
    last_state: TrackState | None
    last_report: float
    # The tracker as of this segment's end - may be a *new* instance the
    # segment constructed itself (a cold start on its own first frame), so
    # the caller must always take this over whatever it passed in, not
    # assume its own reference is still current (Codex review: a stale
    # caller-side None here, when the segment had actually cold-started one
    # internally, broke the recentre-verification path's assumption that a
    # tracker is always present at that point).
    tracker: OutletTracker | None = None
    recentre_source_xy: tuple[float, float] | None = None
    resume_timestamp_s: float | None = None
    reason: str | None = None


@dataclass
class _PreReferenceResult:
    """How ``ZahnCupDetector._process_pre_reference_span`` ended."""

    status: str  # "ready" | "finished" | "failed"
    tracker: OutletTracker | None = None
    last_state: TrackState | None = None
    last_timestamp: float = 0.0
    reason: str | None = None


class ZahnCupDetector(BaseDetector):
    """Measures Zahn cup efflux time from a side-on recording."""

    name: ClassVar[str] = "zahn_cup"
    display_name: ClassVar[str] = "Zahn cup viscosity"
    description: ClassVar[str] = (
        "Measures the efflux time of a Zahn cup by watching the liquid stream "
        "below the outlet hole."
    )

    def configure(self) -> None:
        self.config = _zahn_config_from_params(self.params)
        self.config.validate()

        roi_param = self.params.get("roi")
        outlet = self.params.get("outlet")
        outlet_end = self.params.get("outlet_end")
        if roi_param:
            roi = ROI.from_dict(roi_param)
            # A manually drawn region has no click to track; the horizontal
            # centre of its top edge is the closest thing to "the outlet" and
            # matches where build_roi_from_click positions a clicked one. A
            # drawn region also has no *second* anchor - the two-anchor
            # contract below only applies to the outlet-click flow, where
            # there is an actual point to track drift from.
            self.outlet_xy: tuple[float, float] = (roi.x + roi.width / 2.0, float(roi.y))
            self.outlet_end_xy: tuple[float, float] | None = None
        elif outlet or outlet_end:
            # Product decision (Codex review, fourth round): a single click
            # is not enough to bound how far unsupervised tracking has to
            # run without a human-verified checkpoint over a whole clip -
            # both an early and a late outlet mark are required together
            # *while tracking is actually on*. zahn_track_outlet=False is
            # still the full, sanctioned rollback to the original
            # single-click, fixed-ROI path (self.outlet_end_xy simply stays
            # unused there - see run()) - it must not also gain a new UI
            # requirement an operator explicitly opting out of tracking has
            # no reason to satisfy.
            if self.config.zahn_track_outlet and (not outlet or not outlet_end):
                raise InvalidROIError(
                    "Mark the outlet on two frames before analysing: an early "
                    "one where it is clearly visible, and a late one near the "
                    "expected end of the stream. Only one of the two marks "
                    "was given."
                )
            if not outlet:
                raise InvalidROIError(
                    "Mark the outlet hole of the cup (or draw a region below it) before analysing."
                )
            try:
                early_x, early_y = int(outlet["x"]), int(outlet["y"])
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidROIError(
                    "The marked outlet point is not valid.", detail=str(exc)
                ) from exc
            try:
                roi = build_roi_from_click(early_x, early_y, self.video, self.config)
            except (KeyError, TypeError, ValueError) as exc:
                raise InvalidROIError(
                    "The marked outlet point is not valid.", detail=str(exc)
                ) from exc
            self.outlet_xy = (float(early_x), float(early_y))
            if outlet_end:
                try:
                    late_x, late_y = int(outlet_end["x"]), int(outlet_end["y"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise InvalidROIError(
                        "The late outlet mark is not valid.", detail=str(exc)
                    ) from exc
                if not (0 <= late_x < self.video.width and 0 <= late_y < self.video.height):
                    raise InvalidROIError("The late outlet mark is outside the video frame.")
                self.outlet_end_xy = (float(late_x), float(late_y))
            else:
                self.outlet_end_xy = None
        else:
            raise InvalidROIError(
                "Mark the outlet hole of the cup on two frames (or draw a region "
                "below it) before analysing."
            )

        roi.validate(self.video.width, self.video.height)
        self.roi = roi
        self.guard_roi = build_guard_roi(roi, self.video, self.config.guard_margin_px)

        duration = self.video.duration_s or 0.0
        self.start_s = float(self.params.get("analysis_start_s", 0.0) or 0.0)
        end_param = self.params.get("analysis_end_s")
        self.end_s = float(end_param) if end_param else duration
        if duration and self.end_s > duration:
            self.end_s = duration
        if self.end_s and self.end_s <= self.start_s:
            raise ConfigurationError("The analysis end time must be after the start time.")

        if self.outlet_end_xy is not None:
            # Two-anchor contract: both timestamps are required, not
            # defaulted - unlike the old single-reference design, there is
            # no sane default for "when was the late anchor marked."
            early_ref_param = self.params.get("outlet_reference_s")
            late_ref_param = self.params.get("outlet_end_reference_s")
            if early_ref_param is None or late_ref_param is None:
                raise InvalidROIError(
                    "Both outlet marks need the timestamp of the frame they "
                    "were made on (outlet_reference_s and "
                    "outlet_end_reference_s)."
                )
            try:
                early_ref = float(early_ref_param)
                late_ref = float(late_ref_param)
            except (TypeError, ValueError) as exc:
                raise InvalidROIError(
                    "The outlet marks' timestamps are not valid.", detail=str(exc)
                ) from exc
            bounds_hi = self.end_s or self.start_s
            if not self.start_s <= early_ref <= bounds_hi:
                raise InvalidROIError(
                    "The early outlet mark's timestamp falls outside the analysed range."
                )
            if not self.start_s <= late_ref <= bounds_hi:
                raise InvalidROIError(
                    "The late outlet mark's timestamp falls outside the analysed range."
                )
            if late_ref <= early_ref:
                raise InvalidROIError(
                    "The late outlet mark must be on a later frame than the "
                    "early one (they cannot share, or invert, a timestamp)."
                )
            self.outlet_reference_s = early_ref
            self.outlet_end_reference_s: float | None = late_ref
        else:
            # The timestamp of the frame the outlet/ROI coordinates above
            # were actually marked on - distinct from analysis_start_s,
            # where scoring/decoding begins (Codex review: a real
            # hand-held clip's outlet can become markable only after the
            # camera has already panned well past frame 0, e.g. a large
            # early reframing). Only a click carries a meaningful
            # reference frame; a manually drawn ROI has none, so it
            # defaults to analysis_start_s like an unsupplied value.
            # Clamped into [start_s, end_s] defensively - a stale or
            # out-of-range value from a caller should degrade to "no
            # reference frame given" rather than fail the whole analysis.
            reference_param = self.params.get("outlet_reference_s") if outlet else None
            self.outlet_reference_s = (
                clamp(float(reference_param), self.start_s, self.end_s or self.start_s)
                if reference_param is not None
                else self.start_s
            )
            self.outlet_end_reference_s = None

        self.keep_diagnostics = bool(self.params.get("save_diagnostics", False))
        # Set by the service layer (job.job_id under config.diagnostics_dir),
        # not by a user-facing parameter. When present and save_diagnostics is
        # on, per-frame tracking evidence is *streamed* to a file here rather
        # than accumulated in memory - the transitions-only summary in
        # DetectorResult.diagnostics is enough to see the shape of a run, but
        # not enough to inspect a specific frame's tracked outlet, search
        # window, scoring ROI and state, which is what this file is for.
        diagnostics_dir_param = self.params.get("diagnostics_dir")
        self._diagnostics_dir = Path(diagnostics_dir_param) if diagnostics_dir_param else None
        self._tracking_frames_log_path: Path | None = None

    # -- planning ---------------------------------------------------------- #

    def describe(self) -> dict[str, Any]:
        return self._describe_common(
            roi=self.roi.to_dict(),
            guard_roi=self.guard_roi.to_dict(),
            analysis_range_s=[round(self.start_s, 3), round(self.end_s, 3)],
            frame_step="every frame",
            flow_start_persistence_s=self.config.flow_start_persistence_s,
            flow_end_persistence_s=self.config.flow_end_persistence_s,
            activity_threshold=self.config.activity_threshold,
            track_outlet=self.config.zahn_track_outlet,
        )

    def _build_capture_roi(self, center_xy: tuple[float, float] | None = None) -> ROI:
        """A guard-sized window, padded by ``track_search_margin_px``, that
        the tracker searches within for one contiguous run.

        Centred on ``center_xy`` (source coordinates) when given - used to
        *recentre* the decode/search window on the tracker's last credible
        estimate when the outlet has drifted toward the edge of the current
        window (see ``_run_segment``'s recentre trigger). A window pinned
        forever to the original click cannot see the outlet at all once real
        camera motion carries it further than this margin away - the pixels
        are simply never decoded, however good the tracker or its
        reacquisition search is (Codex review, against real footage). With
        no ``center_xy``, this reproduces the original click-anchored window
        exactly - the guard region's own centre.
        """
        margin = self.config.track_search_margin_px
        if center_xy is None:
            center_xy = (
                self.guard_roi.x + self.guard_roi.width / 2.0,
                self.guard_roi.y + self.guard_roi.height / 2.0,
            )
        half_w = self.guard_roi.width / 2.0 + margin
        half_h = self.guard_roi.height / 2.0 + margin
        x = int(round(center_xy[0] - half_w))
        y = int(round(center_xy[1] - half_h))
        x2 = int(round(center_xy[0] + half_w))
        y2 = int(round(center_xy[1] + half_h))
        return ROI(x=x, y=y, width=max(1, x2 - x), height=max(1, y2 - y)).clipped_to(
            self.video.width, self.video.height
        )

    def _read_capture_frame_gray(
        self, reader: VideoReader, timestamp_s: float, capture_roi: ROI
    ) -> np.ndarray:
        """The single frame at ``timestamp_s``, cropped to ``capture_roi`` and
        greyed - the wide-capture contract (scale=1.0, no blur - see
        ``run()``). One out-of-band read via ``VideoReader.frame_at``, used
        only *between* streaming passes, never while an ``iter_samples()``
        generator is still active: both share the reader's read position,
        and seeking mid-generator would corrupt it. Used to seed a fresh
        reference frame at the reference-frame span's boundary and at each
        recentre of the search window.
        """
        frame = reader.frame_at(timestamp_s)
        frame = frame[capture_roi.y : capture_roi.y2, capture_roi.x : capture_roi.x2]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _process_tracked_frame(
        self,
        *,
        capture_gray: np.ndarray,
        sample_index: int,
        timestamp_s: float,
        state: TrackState,
        local_outlet_x: float,
        local_outlet_y: float,
        reacquired: bool,
        capture_roi: ROI,
        scale: float,
        scorer: StreamActivityScorer,
        machine: FlowStateMachine,
        trace: ActivityTrace,
        frames_log: TextIO | None,
    ) -> tuple[bool, float, float]:
        """Score one tracked frame and feed it into the shared scorer/state
        machine/trace/diagnostics log - the wide-capture (tracking-on) path.

        ``local_outlet_x/y`` is the tracker's outlet position local to
        whichever ``capture_roi`` is currently in effect - which can change
        mid-run (a recentre re-anchors the search window - see
        ``_run_segment``). The returned ``offset_x/offset_y`` is therefore
        always recomputed from ``capture_roi`` and ``self.outlet_xy`` (the
        *original* click) fresh here, never carried over from an earlier
        window's own coordinate space: it is drift in source pixels since
        the original click, not since whichever window last produced this
        frame, which is what lets the guard/ROI cut below - and every
        diagnostic derived from ``offset_x/offset_y`` - use one formula
        regardless of how many recentres have happened.
        """
        offset_x = capture_roi.x + local_outlet_x - self.outlet_xy[0]
        offset_y = capture_roi.y + local_outlet_y - self.outlet_xy[1]
        trusted = state == TrackState.TRACKED
        scored: ScoreSample | None = None
        if state != TrackState.LOST:
            local_x = int(round((self.guard_roi.x - capture_roi.x) + offset_x))
            local_y = int(round((self.guard_roi.y - capture_roi.y) + offset_y))
            sub = _safe_slice(
                capture_gray, local_x, local_y, self.guard_roi.width, self.guard_roi.height
            )
            if sub is not None:
                prepared = _prepare_for_scoring(sub, scale)
                scored = scorer.score(
                    FrameSample(
                        index=sample_index, timestamp_s=timestamp_s, image=prepared, scale=scale
                    )
                )
            else:
                trusted = False

        if scored is None:
            scored = ScoreSample(value=0.0, extras={"outlet_score": 0.0, "noise_sigma": 0.0})

        trace.add(timestamp_s, scored.value, scored.disturbed or not trusted)
        machine.update(timestamp_s, scored, trusted=trusted)

        if frames_log is not None:
            self._write_tracking_frame_log(
                frames_log, timestamp_s, state, offset_x, offset_y, reacquired, trusted, capture_roi
            )
        return trusted, offset_x, offset_y

    def _process_pre_reference_span(
        self,
        *,
        reader: VideoReader,
        capture_roi: ROI,
        reference_outlet_local: tuple[float, float],
        patch_half_px: int,
        cup_half_width_px: int,
        cup_height_above_px: int,
        scale: float,
        scorer: StreamActivityScorer,
        machine: FlowStateMachine,
        trace: ActivityTrace,
        transitions: list[dict[str, Any]],
        frames_log: TextIO | None,
    ) -> _PreReferenceResult:
        """Score every frame from ``self.start_s`` to ``self.outlet_reference_s``
        and hand back a tracker anchored at the reference frame, ready to
        continue forward.

        The user's outlet mark is only trustworthy on the frame it was made
        on (``self.outlet_reference_s``), which can sit well after
        ``self.start_s`` on a real hand-held clip - a large early reframing
        before the outlet becomes markable at all (Codex review). The
        *previous* round pre-armed a tracker at the reference frame but then
        searched for it *chronologically forward from ``self.start_s``* by
        template match alone: a blind, cold search with no motion continuity
        between consecutive frames, which a supervisor review against real
        footage found could falsely "reacquire" on an unrelated patch of the
        scene right at frame zero (diagnostics/stage1/STAGE1_REPORT.md).
        This replaces that with genuine optical-flow continuity in *both*
        directions from a single trusted anchor: the reference frame is the
        only frame this run ever treats as trustworthy without
        verification; every other frame's position is reached from it by
        tracking frame-to-frame, never by guessing where in the frame a
        stored patch might match.

        Method: decode every frame from ``self.start_s`` to
        ``self.outlet_reference_s`` once, forward - an ordinary, cheap
        sequential read, no seeking - then track *backward* through that
        buffer, from the reference frame to the first one, with the same
        Lucas-Kanade tracker used going forward (optical flow does not care
        which way time runs, only that consecutive frames are close in
        content - see ``OutletTracker``). Every one of those frames is then
        scored in the true, forward chronological order
        ``StreamActivityScorer``'s background model and ``FlowStateMachine``
        require, using the backward pass's per-frame positions. Bounded in
        memory by ``self.outlet_reference_s - self.start_s`` - the gap
        between where a real clip's outlet first becomes markable and where
        analysis should start, which is inherently small in the workflow
        this exists for (the user marks the outlet as soon as they can after
        the recording begins), not the length of the whole recording.

        The capture/search window does not recentre during this span (see
        ``_run_segment`` for that, forward of the reference frame only) - a
        known, deliberately scoped limit: this span is inherently short, and
        the bidirectional redesign above already resolves the specific
        false-reacquisition failure a supervisor review found on real
        footage. See diagnostics/stage1/STAGE1_REPORT.md.
        """
        plan = build_sampling_plan(
            fps=self.video.fps,
            duration_s=self.end_s or (self.video.duration_s or 0.0),
            interval_s=1.0 / self.video.fps,
            scale=1.0,
            start_s=self.start_s,
            end_s=self.outlet_reference_s,
        )
        samples = list(reader.iter_samples(plan, roi=capture_roi, grayscale=True, blur_kernel=0))
        if not samples:
            return _PreReferenceResult(
                status="failed", reason="No frames could be decoded up to the reference frame."
            )

        reference_sample = samples[-1]
        try:
            backward_tracker = OutletTracker(
                self.config,
                reference_sample.image,
                reference_outlet_local,
                patch_half_px=patch_half_px,
                cup_half_width_px=cup_half_width_px,
                cup_height_above_px=cup_height_above_px,
                reference_timestamp_s=reference_sample.timestamp_s,
            )
        except TrackerInitError as exc:
            return _PreReferenceResult(status="failed", reason=str(exc))

        backward_results: dict[int, TrackResult] = {
            sample.index: backward_tracker.update(sample.image, sample.timestamp_s)
            for sample in reversed(samples[:-1])
        }

        last_state: TrackState | None = None
        last_timestamp = self.start_s
        for sample in samples:
            if sample is reference_sample:
                state, local_x, local_y, reacquired = (
                    TrackState.TRACKED,
                    reference_outlet_local[0],
                    reference_outlet_local[1],
                    False,
                )
            else:
                result = backward_results[sample.index]
                state, local_x, local_y, reacquired = (
                    result.state,
                    result.outlet_x,
                    result.outlet_y,
                    result.reacquired,
                )

            _trusted, offset_x, offset_y = self._process_tracked_frame(
                capture_gray=sample.image,
                sample_index=sample.index,
                timestamp_s=sample.timestamp_s,
                state=state,
                local_outlet_x=local_x,
                local_outlet_y=local_y,
                reacquired=reacquired,
                capture_roi=capture_roi,
                scale=scale,
                scorer=scorer,
                machine=machine,
                trace=trace,
                frames_log=frames_log,
            )
            last_timestamp = sample.timestamp_s

            if (
                self.keep_diagnostics
                and (state != last_state or reacquired)
                and len(transitions) < MAX_DIAGNOSTIC_TRANSITIONS
            ):
                last_state = state
                entry = self._diagnostic_transition(
                    sample.timestamp_s, state, offset_x, offset_y, reacquired, capture_roi
                )
                if entry is not None:
                    transitions.append(entry)

            if machine.finished:
                machine.measurement.stopped_early = True
                return _PreReferenceResult(
                    status="finished", last_state=state, last_timestamp=last_timestamp
                )

        try:
            forward_tracker = OutletTracker(
                self.config,
                reference_sample.image,
                reference_outlet_local,
                patch_half_px=patch_half_px,
                cup_half_width_px=cup_half_width_px,
                cup_height_above_px=cup_height_above_px,
                reference_timestamp_s=reference_sample.timestamp_s,
            )
        except TrackerInitError as exc:
            return _PreReferenceResult(status="failed", reason=str(exc))

        return _PreReferenceResult(
            status="ready",
            tracker=forward_tracker,
            last_state=last_state,
            last_timestamp=last_timestamp,
        )

    def _build_capture_roi_for_anchors(
        self, early_xy: tuple[float, float], late_xy: tuple[float, float]
    ) -> ROI:
        """A window sized and positioned to contain the guard region at
        *either* anchor's position, for the span between them.

        Same margin as ``_build_capture_roi``, but that method centres on a
        single point; here the window must comfortably hold both anchors'
        guard regions at once, however far apart they are, since either one
        may need to be decoded before the tracker has established which
        direction (forward or backward) is providing which frame. Shifting
        each anchor by the guard region's own fixed offset from the
        original click - not just each anchor's raw click point - before
        taking the bounding box is what keeps this correct for the same
        reason ``_run_loop``'s recentring does (Codex review): the guard
        region is not symmetric around the outlet.
        """
        margin = self.config.track_search_margin_px
        half_w = self.guard_roi.width / 2.0 + margin
        half_h = self.guard_roi.height / 2.0 + margin
        guard_dx = self.guard_roi.x + self.guard_roi.width / 2.0 - self.outlet_xy[0]
        guard_dy = self.guard_roi.y + self.guard_roi.height / 2.0 - self.outlet_xy[1]
        centers_x = [early_xy[0] + guard_dx, late_xy[0] + guard_dx]
        centers_y = [early_xy[1] + guard_dy, late_xy[1] + guard_dy]
        x0 = min(centers_x) - half_w
        x1 = max(centers_x) + half_w
        y0 = min(centers_y) - half_h
        y1 = max(centers_y) + half_h
        return ROI(
            x=int(round(x0)),
            y=int(round(y0)),
            width=max(1, int(round(x1 - x0))),
            height=max(1, int(round(y1 - y0))),
        ).clipped_to(self.video.width, self.video.height)

    def _reconcile_between_anchors(
        self, forward: TrackResult, backward: TrackResult
    ) -> tuple[TrackState, float, float, bool]:
        """Trust an interior frame between two anchors only when *both*
        directions independently agree it is tracked, and on where.

        The two-anchor contract (Codex review, fourth round): a candidate
        position is never trusted merely for lying between two human-
        verified anchors. Either direction alone reporting anything but
        `tracked` (a bridge, a loss, an unconfirmed reacquisition candidate)
        or the two directions disagreeing on the position by more than
        ``track_reconciliation_max_disagreement_px`` both fall back to
        `lost` here, feeding the same untracked/uncertainty machinery a
        genuine loss already does - not a special "probably fine, it's
        between two anchors" exemption.
        """
        if forward.state is not TrackState.TRACKED or backward.state is not TrackState.TRACKED:
            return TrackState.LOST, 0.0, 0.0, False
        displacement = float(
            np.hypot(forward.outlet_x - backward.outlet_x, forward.outlet_y - backward.outlet_y)
        )
        if displacement > self.config.track_reconciliation_max_disagreement_px:
            return TrackState.LOST, 0.0, 0.0, False
        x = (forward.outlet_x + backward.outlet_x) / 2.0
        y = (forward.outlet_y + backward.outlet_y) / 2.0
        return TrackState.TRACKED, x, y, forward.reacquired or backward.reacquired

    def _process_between_anchors_span(
        self,
        *,
        reader: VideoReader,
        capture_roi: ROI,
        early_local: tuple[float, float],
        late_local: tuple[float, float],
        include_early_frame: bool,
        patch_half_px: int,
        cup_half_width_px: int,
        cup_height_above_px: int,
        scale: float,
        scorer: StreamActivityScorer,
        machine: FlowStateMachine,
        trace: ActivityTrace,
        transitions: list[dict[str, Any]],
        frames_log: TextIO | None,
    ) -> _PreReferenceResult:
        """Score every frame from the early anchor to the late anchor,
        trusting an interior frame only when independent forward and
        backward tracks agree (see ``_reconcile_between_anchors``), and
        hand back a tracker anchored at the late anchor, ready to continue
        forward.

        ``include_early_frame`` is False when ``_process_pre_reference_span``
        already scored the early anchor's own frame as the last frame of its
        own span - this method must not score it a second time (that
        produced an extra, duplicate analysed row - the frame-boundary bug
        a supervisor review's real-clip diagnostics caught: an 811-frame
        clip should never produce 812 analysed rows). It is True only when
        the early anchor coincides with ``self.start_s`` itself, so nothing
        upstream has scored it yet.

        Method: decode every frame from the early to the late anchor once,
        forward; track forward from the early anchor and, separately,
        backward from the late anchor through the same buffer (optical flow
        does not care which way time runs - see ``OutletTracker``); score
        every frame in true chronological order using whichever of the two
        tracks agree, or `lost` when they do not. Bounded in memory by the
        gap between the two anchors, which for a typical Zahn run is most
        of the clip's own (short) duration, not an unboundedly long one.
        """
        plan = build_sampling_plan(
            fps=self.video.fps,
            duration_s=self.end_s or (self.video.duration_s or 0.0),
            interval_s=1.0 / self.video.fps,
            scale=1.0,
            start_s=self.outlet_reference_s,
            end_s=self.outlet_end_reference_s,
        )
        samples = list(reader.iter_samples(plan, roi=capture_roi, grayscale=True, blur_kernel=0))
        if not samples:
            return _PreReferenceResult(
                status="failed", reason="No frames could be decoded between the two anchors."
            )
        early_sample = samples[0]
        late_sample = samples[-1]
        if early_sample is late_sample:
            return _PreReferenceResult(
                status="failed",
                reason=(
                    "The two outlet marks resolve to the same frame; they must be further apart."
                ),
            )

        try:
            forward_tracker = OutletTracker(
                self.config,
                early_sample.image,
                early_local,
                patch_half_px=patch_half_px,
                cup_half_width_px=cup_half_width_px,
                cup_height_above_px=cup_height_above_px,
                reference_timestamp_s=early_sample.timestamp_s,
            )
        except TrackerInitError as exc:
            return _PreReferenceResult(status="failed", reason=str(exc))
        forward_results: dict[int, TrackResult] = {
            sample.index: forward_tracker.update(sample.image, sample.timestamp_s)
            for sample in samples[1:]
        }

        try:
            backward_tracker = OutletTracker(
                self.config,
                late_sample.image,
                late_local,
                patch_half_px=patch_half_px,
                cup_half_width_px=cup_half_width_px,
                cup_height_above_px=cup_height_above_px,
                reference_timestamp_s=late_sample.timestamp_s,
            )
        except TrackerInitError as exc:
            return _PreReferenceResult(status="failed", reason=str(exc))
        backward_results: dict[int, TrackResult] = {
            sample.index: backward_tracker.update(sample.image, sample.timestamp_s)
            for sample in reversed(samples[:-1])
        }

        last_state: TrackState | None = None
        last_timestamp = self.outlet_reference_s
        frames_to_score = samples if include_early_frame else samples[1:]
        for sample in frames_to_score:
            if sample is early_sample:
                state, local_x, local_y, reacquired = (
                    TrackState.TRACKED,
                    early_local[0],
                    early_local[1],
                    False,
                )
            elif sample is late_sample:
                state, local_x, local_y, reacquired = (
                    TrackState.TRACKED,
                    late_local[0],
                    late_local[1],
                    False,
                )
            else:
                state, local_x, local_y, reacquired = self._reconcile_between_anchors(
                    forward_results[sample.index], backward_results[sample.index]
                )

            _trusted, offset_x, offset_y = self._process_tracked_frame(
                capture_gray=sample.image,
                sample_index=sample.index,
                timestamp_s=sample.timestamp_s,
                state=state,
                local_outlet_x=local_x,
                local_outlet_y=local_y,
                reacquired=reacquired,
                capture_roi=capture_roi,
                scale=scale,
                scorer=scorer,
                machine=machine,
                trace=trace,
                frames_log=frames_log,
            )
            last_timestamp = sample.timestamp_s

            if (
                self.keep_diagnostics
                and (state != last_state or reacquired)
                and len(transitions) < MAX_DIAGNOSTIC_TRANSITIONS
            ):
                last_state = state
                entry = self._diagnostic_transition(
                    sample.timestamp_s, state, offset_x, offset_y, reacquired, capture_roi
                )
                if entry is not None:
                    transitions.append(entry)

            if machine.finished:
                machine.measurement.stopped_early = True
                return _PreReferenceResult(
                    status="finished", last_state=state, last_timestamp=last_timestamp
                )

        try:
            continuation_tracker = OutletTracker(
                self.config,
                late_sample.image,
                late_local,
                patch_half_px=patch_half_px,
                cup_half_width_px=cup_half_width_px,
                cup_height_above_px=cup_height_above_px,
                reference_timestamp_s=late_sample.timestamp_s,
            )
        except TrackerInitError as exc:
            return _PreReferenceResult(status="failed", reason=str(exc))

        return _PreReferenceResult(
            status="ready",
            tracker=continuation_tracker,
            last_state=last_state,
            last_timestamp=last_timestamp,
        )

    # -- execution --------------------------------------------------------- #

    def run(
        self, reader: VideoReader, progress: ProgressReporter = null_progress
    ) -> DetectorResult:
        """Analyse every frame and time the efflux, tracking the outlet if enabled.

        With tracking on, one crop - the *capture window* around the guard
        region - is decoded per frame at source resolution. The tracker
        estimates the outlet's drift within it; the guard/ROI sub-window
        used for scoring is then re-cut from that same crop at the tracked
        position, resized and blurred to match the scale scoring has always
        run at. The capture window is not fixed for the whole run: it
        recentres on the tracker's last credible estimate when the outlet
        drifts toward its edge (see ``_run_segment``), so real camera motion
        carrying the outlet well away from where it was clicked does not
        simply run the tracker out of decoded pixels.

        When both outlet anchors are given (``self.outlet_end_xy`` is not
        None - the two-anchor contract, required for every outlet-click
        configuration), tracking runs in three segments: backward from the
        early anchor to analysis start if needed
        (``_process_pre_reference_span``), forward-from-early reconciled
        against backward-from-late between the two anchors
        (``_process_between_anchors_span`` - an interior frame is trusted
        only when both directions agree), then forward from the late anchor
        to analysis end via the ordinary recentring path
        (``_run_two_anchor``). A manually drawn region has no second anchor
        and keeps the single-reference behaviour below.

        Tracking off (or unable to start - see TrackerInitError) behaves
        exactly as before: the guard region is decoded directly at analysis
        scale, fixed for the whole run.
        """
        scale = scale_factor_for_width(self.guard_roi.width, ANALYSIS_WIDTH_PX)
        inner = self._inner_rect(scale)
        scorer = StreamActivityScorer(self.config, inner, keep_mask=False)
        machine = FlowStateMachine(self.config)
        trace = ActivityTrace()

        wide_capture = self.config.zahn_track_outlet
        capture_roi = self._build_capture_roi() if wide_capture else self.guard_roi
        capture_scale = 1.0 if wide_capture else scale
        plan = build_sampling_plan(
            fps=self.video.fps,
            duration_s=self.end_s or (self.video.duration_s or 0.0),
            interval_s=1.0 / self.video.fps,  # Zahn events are short: no skipping
            scale=capture_scale,
            start_s=self.start_s,
            end_s=self.end_s or None,
        )
        blur_kernel = 0 if wide_capture else 3  # tracking blurs only the scored sub-crop

        patch_half_px = max(12, int(round(0.4 * self.roi.width)))
        # Feature detection is bounded to a cup-sized box near/above the
        # outlet - the guard region's own scale, not the wider capture
        # window - so an independently moving, strongly textured background
        # cannot out-compete the cup for the similarity fit or the
        # reacquisition anchor. See OutletTracker._detect_cup_features.
        cup_half_width_px = max(1, self.guard_roi.width // 2)
        cup_height_above_px = max(1, self.guard_roi.height)

        logger.info(
            "Zahn analysis of %s: roi=%s guard=%s capture=%s track_outlet_requested=%s %s",
            self.video.path.name,
            self.roi.to_dict(),
            self.guard_roi.to_dict(),
            capture_roi.to_dict(),
            wide_capture,
            plan.describe(),
        )

        transitions: list[dict[str, Any]] = []
        last_state: TrackState | None = None
        tracker: OutletTracker | None = None
        first_frame = True

        started = time.monotonic()
        last_report = 0.0
        last_timestamp = self.start_s
        total_span = max(plan.end_s - plan.start_s, 1e-6)
        segment_start_s = self.start_s

        frames_log = self._open_tracking_frames_log() if wide_capture else None
        try:
            if wide_capture and self.outlet_end_xy is not None:
                return self._run_two_anchor(
                    reader=reader,
                    progress=progress,
                    capture_roi=capture_roi,
                    scale=scale,
                    capture_scale=capture_scale,
                    blur_kernel=blur_kernel,
                    patch_half_px=patch_half_px,
                    cup_half_width_px=cup_half_width_px,
                    cup_height_above_px=cup_height_above_px,
                    scorer=scorer,
                    machine=machine,
                    trace=trace,
                    transitions=transitions,
                    started=started,
                    plan=plan,
                    total_span=total_span,
                    frames_log=frames_log,
                )

            if wide_capture and self.outlet_reference_s > self.start_s:
                reference_outlet_local = (
                    self.outlet_xy[0] - capture_roi.x,
                    self.outlet_xy[1] - capture_roi.y,
                )
                pre = self._process_pre_reference_span(
                    reader=reader,
                    capture_roi=capture_roi,
                    reference_outlet_local=reference_outlet_local,
                    patch_half_px=patch_half_px,
                    cup_half_width_px=cup_half_width_px,
                    cup_height_above_px=cup_height_above_px,
                    scale=scale,
                    scorer=scorer,
                    machine=machine,
                    trace=trace,
                    transitions=transitions,
                    frames_log=frames_log,
                )
                if pre.status == "failed":
                    logger.warning(
                        "Zahn outlet tracking could not start for %s: %s",
                        self.video.path.name,
                        pre.reason,
                    )
                    return self._tracking_unavailable_result(
                        pre.reason or "Tracking could not be started at the reference frame."
                    )
                last_state = pre.last_state
                last_timestamp = pre.last_timestamp
                if pre.status == "finished":
                    measurement = machine.finalize(last_timestamp)
                    elapsed = time.monotonic() - started
                    return self._build_result(
                        measurement, trace, elapsed, plan.effective_interval_s, transitions, True
                    )
                tracker = pre.tracker
                first_frame = False
                segment_start_s = last_timestamp + plan.effective_interval_s

            if wide_capture and segment_start_s > (self.end_s or last_timestamp):
                # The reference frame was the last frame in the analysed
                # range - nothing forward to track.
                measurement = machine.finalize(last_timestamp)
                elapsed = time.monotonic() - started
                return self._build_result(
                    measurement, trace, elapsed, plan.effective_interval_s, transitions, True
                )

            return self._run_loop(
                reader=reader,
                progress=progress,
                capture_roi=capture_roi,
                wide_capture=wide_capture,
                scale=scale,
                capture_scale=capture_scale,
                blur_kernel=blur_kernel,
                patch_half_px=patch_half_px,
                cup_half_width_px=cup_half_width_px,
                cup_height_above_px=cup_height_above_px,
                scorer=scorer,
                machine=machine,
                trace=trace,
                tracker=tracker,
                transitions=transitions,
                last_state=last_state,
                first_frame=first_frame,
                started=started,
                last_report=last_report,
                segment_start_s=segment_start_s,
                progress_start_s=plan.start_s,
                total_span=total_span,
                sample_interval_s=plan.effective_interval_s,
                frames_log=frames_log,
            )
        finally:
            if frames_log is not None:
                frames_log.close()

    def _run_two_anchor(  # noqa: PLR0913 - internal, keeps run() readable
        self,
        *,
        reader: VideoReader,
        progress: ProgressReporter,
        capture_roi: ROI,
        scale: float,
        capture_scale: float,
        blur_kernel: int,
        patch_half_px: int,
        cup_half_width_px: int,
        cup_height_above_px: int,
        scorer: StreamActivityScorer,
        machine: FlowStateMachine,
        trace: ActivityTrace,
        transitions: list[dict[str, Any]],
        started: float,
        plan: Any,
        total_span: float,
        frames_log: TextIO | None,
    ) -> DetectorResult:
        """Drive the two-anchor tracking contract end to end.

        Three segments, run in chronological order: backward from the early
        anchor to analysis start if there is a span to cover
        (``_process_pre_reference_span``, unchanged from the single-anchor
        design); forward-from-early reconciled against backward-from-late
        between the two anchors (``_process_between_anchors_span`` - an
        interior frame is trusted only when both directions agree); then
        forward from the late anchor to analysis end via the ordinary
        recentring-capable path (``_run_loop``, unchanged). The transition
        into segment C computes its start frame by *integer* index
        (``frame_index_at``/``frames_to_seconds``), not by adding a
        floating-point interval to a timestamp and re-deriving an index from
        the sum - the previous single-anchor design did exactly that and a
        real clip's diagnostics caught the fencepost it occasionally lands
        on: 812 analysed rows for an 811-frame clip, the late anchor's own
        frame re-scored a second time.
        """
        assert self.outlet_end_xy is not None
        assert self.outlet_end_reference_s is not None
        sample_interval_s = plan.effective_interval_s
        last_state: TrackState | None = None
        last_timestamp = self.start_s
        include_early_frame = True

        if self.outlet_reference_s > self.start_s:
            early_local = (
                self.outlet_xy[0] - capture_roi.x,
                self.outlet_xy[1] - capture_roi.y,
            )
            pre = self._process_pre_reference_span(
                reader=reader,
                capture_roi=capture_roi,
                reference_outlet_local=early_local,
                patch_half_px=patch_half_px,
                cup_half_width_px=cup_half_width_px,
                cup_height_above_px=cup_height_above_px,
                scale=scale,
                scorer=scorer,
                machine=machine,
                trace=trace,
                transitions=transitions,
                frames_log=frames_log,
            )
            if pre.status == "failed":
                logger.warning(
                    "Zahn outlet tracking could not start for %s: %s",
                    self.video.path.name,
                    pre.reason,
                )
                return self._tracking_unavailable_result(
                    pre.reason or "Tracking could not be started at the early anchor."
                )
            last_state, last_timestamp = pre.last_state, pre.last_timestamp
            if pre.status == "finished":
                return self._finalize_two_anchor(
                    machine, trace, started, sample_interval_s, transitions, last_timestamp
                )
            # _process_pre_reference_span already scored the early anchor's
            # own frame as the last frame of its span - the between-anchors
            # segment below must not score it a second time.
            include_early_frame = False

        capture_roi_between = self._build_capture_roi_for_anchors(
            self.outlet_xy, self.outlet_end_xy
        )
        early_local = (
            self.outlet_xy[0] - capture_roi_between.x,
            self.outlet_xy[1] - capture_roi_between.y,
        )
        late_local = (
            self.outlet_end_xy[0] - capture_roi_between.x,
            self.outlet_end_xy[1] - capture_roi_between.y,
        )
        between = self._process_between_anchors_span(
            reader=reader,
            capture_roi=capture_roi_between,
            early_local=early_local,
            late_local=late_local,
            include_early_frame=include_early_frame,
            patch_half_px=patch_half_px,
            cup_half_width_px=cup_half_width_px,
            cup_height_above_px=cup_height_above_px,
            scale=scale,
            scorer=scorer,
            machine=machine,
            trace=trace,
            transitions=transitions,
            frames_log=frames_log,
        )
        if between.status == "failed":
            logger.warning(
                "Zahn outlet tracking could not start for %s: %s",
                self.video.path.name,
                between.reason,
            )
            return self._tracking_unavailable_result(
                between.reason or "Tracking could not be started between the two anchors."
            )
        last_state, last_timestamp = between.last_state, between.last_timestamp
        if between.status == "finished":
            return self._finalize_two_anchor(
                machine, trace, started, sample_interval_s, transitions, last_timestamp
            )
        tracker = between.tracker

        late_index = frame_index_at(
            self.outlet_end_reference_s, self.video.fps, self.video.frame_count
        )
        forward_start_s = frames_to_seconds(late_index + 1, self.video.fps)
        if forward_start_s > (self.end_s or last_timestamp):
            # The late anchor was the last frame in the analysed range -
            # nothing forward to track.
            return self._finalize_two_anchor(
                machine, trace, started, sample_interval_s, transitions, last_timestamp
            )

        return self._run_loop(
            reader=reader,
            progress=progress,
            capture_roi=capture_roi_between,
            wide_capture=True,
            scale=scale,
            capture_scale=capture_scale,
            blur_kernel=blur_kernel,
            patch_half_px=patch_half_px,
            cup_half_width_px=cup_half_width_px,
            cup_height_above_px=cup_height_above_px,
            scorer=scorer,
            machine=machine,
            trace=trace,
            tracker=tracker,
            transitions=transitions,
            last_state=last_state,
            first_frame=False,
            started=started,
            last_report=0.0,
            segment_start_s=forward_start_s,
            progress_start_s=plan.start_s,
            total_span=total_span,
            sample_interval_s=sample_interval_s,
            frames_log=frames_log,
        )

    def _finalize_two_anchor(
        self,
        machine: FlowStateMachine,
        trace: ActivityTrace,
        started: float,
        sample_interval_s: float,
        transitions: list[dict[str, Any]],
        last_timestamp: float,
    ) -> DetectorResult:
        measurement = machine.finalize(last_timestamp)
        elapsed = time.monotonic() - started
        return self._build_result(measurement, trace, elapsed, sample_interval_s, transitions, True)

    def _run_loop(  # noqa: PLR0913 - internal, keeps run() readable and the log guaranteed to close
        self,
        *,
        reader: VideoReader,
        progress: ProgressReporter,
        capture_roi: ROI,
        wide_capture: bool,
        scale: float,
        capture_scale: float,
        blur_kernel: int,
        patch_half_px: int,
        cup_half_width_px: int,
        cup_height_above_px: int,
        scorer: StreamActivityScorer,
        machine: FlowStateMachine,
        trace: ActivityTrace,
        tracker: OutletTracker | None,
        transitions: list[dict[str, Any]],
        last_state: TrackState | None,
        first_frame: bool,
        started: float,
        last_report: float,
        segment_start_s: float,
        progress_start_s: float,
        total_span: float,
        sample_interval_s: float,
        frames_log: TextIO | None,
    ) -> DetectorResult:
        """Drive one or more decode segments to the end of the analysed range.

        Almost always one segment. A second (or third, ...) begins only when
        a segment ends by requesting a *recentre*: the tracked outlet
        drifted near the edge of the current capture window and was then
        lost (see ``_run_segment``), so the window is rebuilt around the
        last credible estimate rather than staying pinned to the original
        click, and decoding resumes from the next frame. Bounded by
        ``MAX_TRACKING_RECENTRES`` so a pathological run cannot recentre
        without limit; once exhausted, remaining frames fall back to the
        tracker's own in-window reacquisition, exactly as before this
        capability existed.
        """
        last_timestamp = segment_start_s
        recentre_budget = MAX_TRACKING_RECENTRES if wide_capture else 0
        recentre_events: list[dict[str, Any]] = []

        while True:
            segment_plan = build_sampling_plan(
                fps=self.video.fps,
                duration_s=self.end_s or (self.video.duration_s or 0.0),
                interval_s=1.0 / self.video.fps,
                scale=capture_scale,
                start_s=segment_start_s,
                end_s=self.end_s or None,
            )
            segment = self._run_segment(
                reader=reader,
                progress=progress,
                plan=segment_plan,
                capture_roi=capture_roi,
                wide_capture=wide_capture,
                allow_recentre=recentre_budget > 0,
                scale=scale,
                blur_kernel=blur_kernel,
                patch_half_px=patch_half_px,
                cup_half_width_px=cup_half_width_px,
                cup_height_above_px=cup_height_above_px,
                scorer=scorer,
                machine=machine,
                trace=trace,
                tracker=tracker,
                transitions=transitions,
                last_state=last_state,
                first_frame=first_frame,
                started=started,
                last_report=last_report,
                progress_start_s=progress_start_s,
                total_span=total_span,
                frames_log=frames_log,
            )
            last_timestamp = segment.last_timestamp
            last_state = segment.last_state
            last_report = segment.last_report
            tracker = segment.tracker

            if segment.status == "init_failed":
                return self._tracking_unavailable_result(
                    segment.reason or "Outlet tracking could not be started."
                )
            if segment.status in ("finished", "exhausted"):
                break

            # status == "recentre": rebuild the search window on the last
            # credible estimate and keep going from the next frame. Failing
            # to re-anchor there (too little texture at the new position) is
            # not fatal - fall back to a cold start at the *original*
            # window, the same recovery a fresh loss has always had.
            recentre_budget -= 1
            assert segment.recentre_source_xy is not None
            assert segment.resume_timestamp_s is not None
            assert tracker is not None  # recentre only ever follows a fresh loss
            # _build_capture_roi centres its window on the *guard region's*
            # own centre, not on the outlet - the guard region is not
            # symmetric around the outlet (it reaches much further down,
            # for the stream, than up, for the cup body), so recentring on
            # the raw outlet position directly would size/place the window
            # off-centre from where the (fixed-size, fixed-shape-relative-
            # to-outlet) guard region actually needs to sit, and a video-
            # edge clip could then leave it too short to contain the guard
            # region at all - every frame's guard-cut silently failing
            # (Codex review; caught by TestConfidenceReflectsMotion's
            # steady-video regression test, which this recentring feature
            # was breaking even on a *static* cup with no drift at all).
            # Shift the candidate outlet position by the same fixed offset
            # the guard region already sits at relative to the original
            # click, to get where the guard region's centre would be if it
            # had moved rigidly with the outlet.
            guard_center_offset = (
                self.guard_roi.x + self.guard_roi.width / 2.0 - self.outlet_xy[0],
                self.guard_roi.y + self.guard_roi.height / 2.0 - self.outlet_xy[1],
            )
            new_capture_roi = self._build_capture_roi(
                (
                    segment.recentre_source_xy[0] + guard_center_offset[0],
                    segment.recentre_source_xy[1] + guard_center_offset[1],
                )
            )
            local_outlet = (
                segment.recentre_source_xy[0] - new_capture_roi.x,
                segment.recentre_source_xy[1] - new_capture_roi.y,
            )
            verified = False
            try:
                reference_gray = self._read_capture_frame_gray(
                    reader, segment.resume_timestamp_s, new_capture_roi
                )
                # A recentre candidate must be *verified* against this
                # tracker's own stored reference patch before it is
                # trusted - the same bar a cold reacquisition search
                # already holds a candidate to. Skipping this would let
                # the fresh tracker construction below "confirm" whatever
                # happens to sit at an extrapolated position - a real,
                # prolonged occlusion included - as `tracked`, with no
                # evidence it is the cup at all (Codex review; caught by
                # tests/test_zahn_tracking_integration.py's occlusion-gap
                # regression tests, which this recentring feature was
                # briefly breaking before this check was added).
                correlation = tracker.reference_patch_correlation(reference_gray, local_outlet)
                if correlation >= self.config.track_reacquire_min_correlation:
                    # Two-frame persistence gate (Codex review, fourth
                    # round), the same principle as
                    # OutletTracker._attempt_reacquisition's own: one
                    # frame's correlation clearing the threshold is not
                    # enough to commit the whole run to a new search
                    # window - a disposable probe tracker must also track
                    # the *next* frame as `tracked` before this candidate
                    # is trusted. Discarded either way; the run's actual
                    # continuation tracker (below) is anchored fresh at
                    # ``resume_timestamp_s`` itself so every frame still
                    # gets scored, none skipped as "already used for
                    # verification."
                    probe_timestamp_s = (
                        segment.resume_timestamp_s + segment_plan.effective_interval_s
                    )
                    probe_tracker = OutletTracker(
                        self.config,
                        reference_gray,
                        local_outlet,
                        patch_half_px=patch_half_px,
                        cup_half_width_px=cup_half_width_px,
                        cup_height_above_px=cup_height_above_px,
                        reference_timestamp_s=segment.resume_timestamp_s,
                    )
                    probe_gray = self._read_capture_frame_gray(
                        reader, probe_timestamp_s, new_capture_roi
                    )
                    probe_result = probe_tracker.update(probe_gray, probe_timestamp_s)
                    if probe_result.state is TrackState.TRACKED:
                        tracker = OutletTracker(
                            self.config,
                            reference_gray,
                            local_outlet,
                            patch_half_px=patch_half_px,
                            cup_half_width_px=cup_half_width_px,
                            cup_height_above_px=cup_height_above_px,
                            reference_timestamp_s=segment.resume_timestamp_s,
                        )
                        verified = True
            except TrackerInitError:
                verified = False

            if verified:
                recentre_events.append(
                    {
                        "timestamp_s": round(segment.resume_timestamp_s, 3),
                        "source_xy": [
                            round(segment.recentre_source_xy[0], 1),
                            round(segment.recentre_source_xy[1], 1),
                        ],
                        "from_capture_roi": capture_roi.to_dict(),
                        "to_capture_roi": new_capture_roi.to_dict(),
                        "reanchored": True,
                    }
                )
                capture_roi = new_capture_roi
                first_frame = False
            else:
                # Verification refused the candidate - do not touch
                # `tracker` or `capture_roi` at all. The still-LOST tracker
                # keeps trying its own verified in-window reacquisition
                # every frame from here, exactly as it did before this
                # capability existed (Codex review: an earlier version of
                # this fallback cold-started a *fresh* tracker instead,
                # which has no correlation check of its own - reopening,
                # one level deeper, the exact false-positive hole
                # verification here exists to close. A real, prolonged
                # occlusion must stay honestly `lost`, not get "confirmed"
                # by a fresh init that happened to find >=4 corners in
                # plain background noise).
                recentre_events.append(
                    {
                        "timestamp_s": round(segment.resume_timestamp_s, 3),
                        "source_xy": [
                            round(segment.recentre_source_xy[0], 1),
                            round(segment.recentre_source_xy[1], 1),
                        ],
                        "from_capture_roi": capture_roi.to_dict(),
                        "to_capture_roi": None,
                        "reanchored": False,
                    }
                )
                first_frame = False
            segment_start_s = segment.resume_timestamp_s

        measurement = machine.finalize(last_timestamp)
        elapsed = time.monotonic() - started
        return self._build_result(
            measurement,
            trace,
            elapsed,
            sample_interval_s,
            transitions,
            wide_capture,
            recentre_events=recentre_events,
        )

    def _run_segment(  # noqa: PLR0913 - internal; keeps _run_loop's own signature manageable
        self,
        *,
        reader: VideoReader,
        progress: ProgressReporter,
        plan: Any,
        capture_roi: ROI,
        wide_capture: bool,
        allow_recentre: bool,
        scale: float,
        blur_kernel: int,
        patch_half_px: int,
        cup_half_width_px: int,
        cup_height_above_px: int,
        scorer: StreamActivityScorer,
        machine: FlowStateMachine,
        trace: ActivityTrace,
        tracker: OutletTracker | None,
        transitions: list[dict[str, Any]],
        last_state: TrackState | None,
        first_frame: bool,
        started: float,
        last_report: float,
        progress_start_s: float,
        total_span: float,
        frames_log: TextIO | None,
    ) -> _SegmentResult:
        """Decode and process one contiguous run of frames from ``plan``.

        Ends normally when the plan's frames are exhausted (``exhausted``)
        or the flow's end is confirmed (``finished``); ends early
        (``recentre``) on a *fresh* loss - not a bad-frame blip, but the
        first frame of a genuinely new `lost` run - so the search window can
        be rebuilt around the last credible estimate rather than staying
        pinned to the original click.

        Deliberately not gated on how close to the edge of ``capture_roi``
        that loss happened: a first attempt at this gated only on
        edge-proximity missed the dominant real failure mode on a fast,
        sustained swing - a `predicted` bridge timing out
        (``track_max_bridge_s``) while still comfortably in-bounds, well
        before the extrapolated position ever nears the window edge (see
        diagnostics/stage1/STAGE1_REPORT.md and
        tests/test_zahn_tracking_integration.py's
        ``TestSearchWindowRecentres``). Every fresh loss is a signal the
        tracker no longer knows where the outlet is *right now*, which is
        reason enough to re-anchor - ``previous_state is not TrackState.LOST``
        below already stops this from repeating every frame while stuck
        lost, and ``MAX_TRACKING_RECENTRES`` bounds the total cost.
        """
        last_timestamp = plan.start_s
        previous_state: TrackState | None = None

        for sample in reader.iter_samples(
            plan, roi=capture_roi, grayscale=True, blur_kernel=blur_kernel
        ):
            capture_gray = sample.image

            if first_frame:
                first_frame = False
                if wide_capture and tracker is None:
                    try:
                        tracker = OutletTracker(
                            self.config,
                            capture_gray,
                            (self.outlet_xy[0] - capture_roi.x, self.outlet_xy[1] - capture_roi.y),
                            patch_half_px=patch_half_px,
                            cup_half_width_px=cup_half_width_px,
                            cup_height_above_px=cup_height_above_px,
                            reference_timestamp_s=sample.timestamp_s,
                        )
                    except TrackerInitError as exc:
                        # zahn_track_outlet=True is the product decision:
                        # tracking is not optional here, and the config flag
                        # is the *only* sanctioned rollback. Silently falling
                        # back to the fixed-ROI path would both violate that
                        # and let a frame that could not even be initialised
                        # for tracking come back CONFIRMED on stale, fixed
                        # geometry - guessing under exactly the condition
                        # tracking exists to refuse to guess under. Report
                        # failure instead; only an explicit
                        # zahn_track_outlet=False may use the fixed path.
                        logger.warning(
                            "Zahn outlet tracking could not start for %s: %s",
                            self.video.path.name,
                            exc,
                        )
                        return _SegmentResult(
                            status="init_failed",
                            last_timestamp=last_timestamp,
                            last_state=last_state,
                            last_report=last_report,
                            tracker=None,
                            reason=str(exc),
                        )

            state = TrackState.TRACKED
            local_outlet_x = local_outlet_y = 0.0
            offset_x = offset_y = 0.0
            reacquired = False
            if tracker is not None:
                result = tracker.update(capture_gray, sample.timestamp_s)
                state = result.state
                local_outlet_x, local_outlet_y = result.outlet_x, result.outlet_y
                reacquired = result.reacquired

            if wide_capture:
                _trusted, offset_x, offset_y = self._process_tracked_frame(
                    capture_gray=capture_gray,
                    sample_index=sample.index,
                    timestamp_s=sample.timestamp_s,
                    state=state,
                    local_outlet_x=local_outlet_x,
                    local_outlet_y=local_outlet_y,
                    reacquired=reacquired,
                    capture_roi=capture_roi,
                    scale=scale,
                    scorer=scorer,
                    machine=machine,
                    trace=trace,
                    frames_log=frames_log,
                )
            else:
                # capture_gray is already the guard region, decoded directly
                # at analysis scale - the pre-tracking path.
                scored = scorer.score(sample)
                trace.add(sample.timestamp_s, scored.value, scored.disturbed)
                machine.update(sample.timestamp_s, scored, trusted=True)

            last_timestamp = sample.timestamp_s

            if (
                self.keep_diagnostics
                and tracker is not None
                and (state != last_state or reacquired)
                and len(transitions) < MAX_DIAGNOSTIC_TRANSITIONS
            ):
                last_state = state
                entry = self._diagnostic_transition(
                    sample.timestamp_s, state, offset_x, offset_y, reacquired, capture_roi
                )
                if entry is not None:
                    transitions.append(entry)

            now = time.monotonic()
            if now - last_report >= PROGRESS_UPDATE_INTERVAL_S:
                last_report = now
                fraction = clamp((sample.timestamp_s - progress_start_s) / total_span, 0.0, 1.0)
                progress(
                    stage="analysing",
                    fraction=fraction,
                    message=(
                        "Measuring liquid flow"
                        if machine.flowing
                        else "Watching for the start of flow"
                    ),
                )

            if machine.finished:
                # The end of flow is confirmed; decoding the rest of the file
                # would tell us nothing.
                machine.measurement.stopped_early = True
                return _SegmentResult(
                    status="finished",
                    last_timestamp=last_timestamp,
                    last_state=state,
                    last_report=last_report,
                    tracker=tracker,
                )

            if (
                allow_recentre
                and wide_capture
                and tracker is not None
                and state is TrackState.LOST
                and previous_state is not None
                and previous_state is not TrackState.LOST
            ):
                return _SegmentResult(
                    status="recentre",
                    last_timestamp=last_timestamp,
                    last_state=state,
                    last_report=last_report,
                    tracker=tracker,
                    recentre_source_xy=(
                        capture_roi.x + local_outlet_x,
                        capture_roi.y + local_outlet_y,
                    ),
                    resume_timestamp_s=sample.timestamp_s + plan.effective_interval_s,
                )

            previous_state = state

        return _SegmentResult(
            status="exhausted",
            last_timestamp=last_timestamp,
            last_state=last_state,
            last_report=last_report,
            tracker=tracker,
        )

    def _tracked_geometry(self, offset_x: float, offset_y: float) -> tuple[ROI, ROI] | None:
        """The analysis ROI and guard ROI at the tracker's current offset.

        ``None`` when the offset carries the geometry off-frame - best-effort,
        like the rest of diagnostics: this must never raise into the analysis
        loop.
        """
        try:
            roi = ROI(
                x=int(round(self.roi.x + offset_x)),
                y=int(round(self.roi.y + offset_y)),
                width=self.roi.width,
                height=self.roi.height,
            ).clipped_to(self.video.width, self.video.height)
            guard_roi = ROI(
                x=int(round(self.guard_roi.x + offset_x)),
                y=int(round(self.guard_roi.y + offset_y)),
                width=self.guard_roi.width,
                height=self.guard_roi.height,
            ).clipped_to(self.video.width, self.video.height)
        except InvalidROIError:
            return None
        return roi, guard_roi

    def _diagnostic_transition(
        self,
        timestamp_s: float,
        state: TrackState,
        offset_x: float,
        offset_y: float,
        reacquired: bool,
        capture_roi: ROI,
    ) -> dict[str, Any] | None:
        """A tracking-state snapshot for --save-diagnostics: geometry, not pixels."""
        geometry = self._tracked_geometry(offset_x, offset_y)
        if geometry is None:
            return None
        roi, guard_roi = geometry
        return {
            "timestamp_s": round(timestamp_s, 3),
            "state": state.value,
            "reacquired": bool(reacquired),
            "roi": roi.to_dict(),
            "guard_roi": guard_roi.to_dict(),
            "search_roi": capture_roi.to_dict(),
        }

    def _open_tracking_frames_log(self) -> TextIO | None:
        """Open a streamed, bounded-memory per-frame tracking evidence file.

        Codex review: state-transition JSON alone does not show the tracked
        outlet, search window, scoring ROI and state for the frames actually
        analysed. One JSON line per frame, written and flushed as the run
        progresses, gives that without loading the whole video (or a
        per-frame list) into memory. Gated on ``diagnostics_dir`` - set by the
        service layer only when the operator has diagnostics enabled - not on
        ``keep_diagnostics``, which separately controls the lightweight
        transitions-only summary above. Best-effort like the rest of
        diagnostics: any failure to create or open the file simply disables
        this stream rather than failing the analysis.
        """
        if self._diagnostics_dir is None:
            return None
        try:
            self._diagnostics_dir.mkdir(parents=True, exist_ok=True)
            path = self._diagnostics_dir / "zahn_tracking_frames.jsonl"
            handle = path.open("w", encoding="utf-8")
        except OSError:
            logger.warning("Could not open tracking frames log under %s", self._diagnostics_dir)
            return None
        self._tracking_frames_log_path = path
        return handle

    def _write_tracking_frame_log(
        self,
        frames_log: TextIO,
        timestamp_s: float,
        state: TrackState,
        offset_x: float,
        offset_y: float,
        reacquired: bool,
        trusted: bool,
        capture_roi: ROI,
    ) -> None:
        geometry = self._tracked_geometry(offset_x, offset_y)
        roi, guard_roi = geometry if geometry is not None else (None, None)
        record = {
            "timestamp_s": round(timestamp_s, 3),
            "state": state.value,
            "trusted": trusted,
            "reacquired": bool(reacquired),
            "search_roi": capture_roi.to_dict(),
            "roi": roi.to_dict() if roi is not None else None,
            "guard_roi": guard_roi.to_dict() if guard_roi is not None else None,
        }
        try:
            frames_log.write(json.dumps(record) + "\n")
            # Deliberate, not incidental: the point of streaming instead of
            # accumulating is that an operator can inspect the file while
            # the run is still in progress (e.g. `tail -f`), not only after
            # it closes. Python's own buffering would otherwise hold lines
            # back for an arbitrary, unbounded time (Codex review).
            frames_log.flush()
        except OSError:
            logger.warning("Could not write to tracking frames log", exc_info=True)

    def _inner_rect(self, scale: float) -> _Rect:
        """Where the analysis ROI sits inside the (scaled) guard crop."""
        offset_x = int(round((self.roi.x - self.guard_roi.x) * scale))
        offset_y = int(round((self.roi.y - self.guard_roi.y) * scale))
        guard_w = max(1, int(round(self.guard_roi.width * scale)))
        guard_h = max(1, int(round(self.guard_roi.height * scale)))
        width = max(1, min(int(round(self.roi.width * scale)), guard_w - offset_x))
        height = max(1, min(int(round(self.roi.height * scale)), guard_h - offset_y))
        return _Rect(x=max(0, offset_x), y=max(0, offset_y), width=width, height=height)

    # -- reporting --------------------------------------------------------- #

    def _tracking_unavailable_result(self, reason: str) -> DetectorResult:
        """Outlet tracking was requested but could not be started.

        No frame has been scored, so there is no honest number to report -
        not even a lower bound. This is a FAILED result, distinct from a
        REVIEW one: the operator needs to re-mark the outlet or explicitly
        opt out of tracking (``zahn_track_outlet=False``), not merely double
        check an uncertain measurement.
        """
        message = (
            "Outlet tracking is enabled but could not be started: "
            f"{reason} Mark the outlet on a more textured part of the cup, "
            "or set zahn_track_outlet=False to use a fixed region instead."
        )
        return DetectorResult(
            events=[],
            diagnostics={
                "roi": self.roi.to_dict(),
                "guard_roi": self.guard_roi.to_dict(),
                "config": _zahn_config_to_dict(self.config),
                "track_outlet": False,
                "tracking_transitions": [],
                "recentre_events": [],
                "tracking_init_failed": True,
                "tracking_frames_log": (
                    str(self._tracking_frames_log_path)
                    if self._tracking_frames_log_path is not None
                    else None
                ),
            },
            trace=ActivityTrace(),
            summary={
                "mode": self.name,
                "flow_start_s": None,
                "flow_end_s": None,
                "efflux_seconds": None,
                "efflux_seconds_bounds": None,
                "end_confirmed": False,
                "end_uncertain": False,
                "end_gap_unresolved": False,
                "start_uncertain": False,
                "fps": round(self.video.fps, 4),
                "frames_analysed": 0,
                "frames_with_liquid": 0,
                "frames_disturbed": 0,
                "frames_untracked": 0,
                "timed_frames": 0,
                "continuity": 0.0,
                "stream_breaks": 0,
                "confidence": 0.0,
                "status": EventStatus.FAILED.value,
                "reasons": [message],
                "processing_seconds": 0.0,
                "roi": self.roi.to_dict(),
                "sample_interval_s": round(1.0 / self.video.fps, 5),
                "track_outlet": False,
            },
            warnings=[message],
        )

    def _build_result(
        self,
        measurement: FlowMeasurement,
        trace: ActivityTrace,
        elapsed_s: float,
        sample_interval_s: float,
        transitions: list[dict[str, Any]] | None = None,
        tracking_used: bool = False,
        recentre_events: list[dict[str, Any]] | None = None,
    ) -> DetectorResult:
        confidence, reasons = score_confidence(measurement, self.config)
        status = self._status_for(measurement, confidence)
        uncertain = measurement.start_uncertain or measurement.end_uncertain
        # An uncertain endpoint must not be presented as a precise, confirmed
        # duration - the point of end_uncertain/start_uncertain existing at
        # all. efflux_seconds is suppressed exactly like the FAILED case;
        # efflux_seconds_bounds carries what the evidence actually supports.
        efflux = (
            measurement.efflux_s if status is not EventStatus.FAILED and not uncertain else None
        )
        efflux_bounds = self._efflux_bounds(measurement) if uncertain else None

        summary: dict[str, Any] = {
            "mode": self.name,
            "flow_start_s": measurement.start_s,
            "flow_end_s": measurement.end_s,
            "efflux_seconds": round(efflux, 3) if efflux is not None else None,
            "efflux_seconds_bounds": (
                [round(efflux_bounds[0], 3), round(efflux_bounds[1], 3)]
                if efflux_bounds is not None
                else None
            ),
            "end_confirmed": measurement.end_confirmed,
            "end_uncertain": measurement.end_uncertain,
            "end_uncertainty_bounds": (
                [round(bound, 3) for bound in measurement.end_uncertainty_bounds]
                if measurement.end_uncertainty_bounds is not None
                else None
            ),
            "end_gap_unresolved": measurement.end_gap_unresolved,
            "start_uncertain": measurement.start_uncertain,
            "start_uncertainty_bounds": (
                [round(bound, 3) for bound in measurement.start_uncertainty_bounds]
                if measurement.start_uncertainty_bounds is not None
                else None
            ),
            "fps": round(self.video.fps, 4),
            "frames_analysed": measurement.frames_analysed,
            "frames_with_liquid": measurement.frames_with_liquid,
            "frames_disturbed": measurement.frames_disturbed,
            "frames_untracked": measurement.frames_untracked,
            "timed_frames": measurement.timed_frames,
            "continuity": round(measurement.continuity, 4),
            "stream_breaks": len(measurement.breaks),
            "confidence": round(confidence, 4),
            "status": status.value,
            "reasons": reasons,
            "processing_seconds": round(elapsed_s, 2),
            "roi": self.roi.to_dict(),
            "sample_interval_s": round(sample_interval_s, 5),
            "track_outlet": tracking_used,
        }

        events: list[Event] = []
        # Codex review: Event.duration_s is always end_s - start_s, computed
        # fresh by the shared Event/EventLog contract regardless of what
        # `details` says - build_event_log's CSV rows, DetectorResult.to_dict
        # and summarise()'s total_active_seconds all read it directly. Only
        # suppressing "efflux_seconds" in `details` therefore did not stop an
        # uncertain measurement from presenting a precise duration elsewhere.
        # Changing that shared contract to carry a genuinely unknown/bounded
        # duration is out of scope here, so the focused fix is this: no Event
        # at all for an uncertain measurement. The bounded candidate stays
        # visible in `summary` (flow_start_s/flow_end_s/efflux_seconds_bounds
        # /start_uncertain/end_uncertain), just never as an Event a
        # downstream consumer could mistake for a confirmed occurrence.
        if measurement.start_s is not None and measurement.end_s is not None and not uncertain:
            events.append(
                Event(
                    label="Zahn cup efflux",
                    start_s=measurement.start_s,
                    end_s=measurement.end_s,
                    confidence=confidence,
                    detector=self.name,
                    status=status,
                    notes=tuple(reasons),
                    details={
                        "efflux_seconds": round(efflux, 3) if efflux is not None else None,
                        "efflux_seconds_bounds": summary["efflux_seconds_bounds"],
                        "end_confirmed": measurement.end_confirmed,
                        "end_uncertain": measurement.end_uncertain,
                        "end_uncertainty_bounds": summary["end_uncertainty_bounds"],
                        "end_gap_unresolved": measurement.end_gap_unresolved,
                        "start_uncertain": measurement.start_uncertain,
                        "start_uncertainty_bounds": summary["start_uncertainty_bounds"],
                        "stream_breaks": [
                            [round(a, 3), round(b, 3)] for a, b in measurement.breaks
                        ],
                        "frames_analysed": measurement.frames_analysed,
                        "frames_untracked": measurement.frames_untracked,
                        "mean_stream_contrast": round(measurement.mean_contrast, 2),
                        "mean_noise_sigma": round(measurement.mean_noise_sigma, 2),
                    },
                )
            )

        warnings: list[str] = []
        if status is EventStatus.FAILED:
            warnings.append(
                "No reliable efflux time could be measured. Check that the outlet "
                "marker is on the hole and that the stream is visible against the "
                "background."
            )
        elif status is EventStatus.REVIEW:
            # Independent ifs, not if/elif (Codex review, third round): a
            # mid-flow gap (end_gap_unresolved), a separate genuinely
            # adjacent end gap, and start uncertainty can each independently
            # apply to the same measurement (e.g. two different gaps in one
            # run) - every concern that actually applies gets its own
            # warning, rather than the first check found masking the rest.
            warned = False
            if measurement.end_gap_unresolved:
                # Deliberately does not say the true end "could have
                # occurred earlier" - trusted liquid seen again after the
                # gap proves the opposite (Codex review, second round).
                warnings.append(
                    "The outlet tracker lost the outlet for long enough, during the "
                    "flow, that the stream's continuity through that span could not "
                    "be confirmed, even though liquid was seen again afterward. The "
                    "reported end may not reflect a single continuous stream - please "
                    "review the footage directly before using this measurement."
                )
                warned = True

            has_adjacent_end_bound = (
                measurement.end_uncertain and measurement.end_uncertainty_bounds is not None
            )
            if has_adjacent_end_bound and measurement.start_uncertain:
                warnings.append(
                    "The outlet was not confidently tracked around the reported start "
                    "or the reported end - the true efflux time could be shorter or "
                    "longer than shown. Please review before using this number."
                )
                warned = True
            elif has_adjacent_end_bound:
                warnings.append(
                    "The outlet was not confidently tracked around the reported end - "
                    "the true end could have occurred earlier. Please review before "
                    "using this number."
                )
                warned = True
            elif measurement.start_uncertain:
                warnings.append(
                    "The outlet was not confidently tracked around the reported start "
                    "- the true start could have occurred earlier, making the efflux "
                    "time reported here too short. Please review before using this "
                    "number."
                )
                warned = True

            if not warned:
                warnings.append(
                    "The measurement is uncertain - please review the marked start and "
                    "end before using this number."
                )

        return DetectorResult(
            events=events,
            diagnostics={
                "roi": self.roi.to_dict(),
                "guard_roi": self.guard_roi.to_dict(),
                "config": _zahn_config_to_dict(self.config),
                "breaks": [[round(a, 3), round(b, 3)] for a, b in measurement.breaks],
                "stopped_early": measurement.stopped_early,
                "track_outlet": tracking_used,
                "tracking_transitions": transitions or [],
                "recentre_events": recentre_events or [],
                "tracking_frames_log": (
                    str(self._tracking_frames_log_path)
                    if self._tracking_frames_log_path is not None
                    else None
                ),
            },
            trace=trace,
            summary=summary,
            warnings=warnings,
        )

    def _efflux_bounds(self, measurement: FlowMeasurement) -> tuple[float, float] | None:
        """The range the true efflux time could fall in, given uncertain endpoints.

        Combines whichever of start/end is uncertain with the other's exact
        value (or both, worst case, if both are uncertain) - the widest
        duration comes from the latest-plausible start and the
        earliest-plausible... no: from the *earliest* start and *latest* end
        for the upper bound, and the reverse for the lower bound.

        Returns ``None`` - no range at all, not a degenerate one collapsed
        onto the raw measurement - when an unresolved mid-flow gap
        (``end_gap_unresolved``) leaves no genuine bound on the end (Codex
        review, second round): ``end_s`` is not trustworthy even as a point
        estimate there, so a "range" built from it would still be inventing
        precision the evidence does not support. When a separate, genuinely
        adjacent gap *did* also produce real ``end_uncertainty_bounds``,
        that case does not apply and the normal computation below runs.
        """
        if measurement.start_s is None or measurement.end_s is None:
            return None
        if measurement.end_gap_unresolved and measurement.end_uncertainty_bounds is None:
            return None
        start_lo, start_hi = (
            measurement.start_uncertainty_bounds
            if measurement.start_uncertain and measurement.start_uncertainty_bounds is not None
            else (measurement.start_s, measurement.start_s)
        )
        end_lo, end_hi = (
            measurement.end_uncertainty_bounds
            if measurement.end_uncertain and measurement.end_uncertainty_bounds is not None
            else (measurement.end_s, measurement.end_s)
        )
        return max(0.0, end_lo - start_hi), max(0.0, end_hi - start_lo)

    def _status_for(self, measurement: FlowMeasurement, confidence: float) -> EventStatus:
        if measurement.start_s is None or measurement.end_s is None:
            return EventStatus.FAILED
        if confidence < self.config.fail_confidence:
            return EventStatus.FAILED
        if (
            confidence < self.config.review_confidence
            or not measurement.end_confirmed
            or measurement.start_uncertain
        ):
            return EventStatus.REVIEW
        return EventStatus.CONFIRMED


def score_confidence(measurement: FlowMeasurement, config: ZahnConfig) -> tuple[float, list[str]]:
    """Confidence in a Zahn measurement, from evidence that was actually measured.

    Components (each documented in the returned reason list):

    *   base 0.35 once both a start and an end exist;
    *   +0.20 x how far the stream at the outlet exceeded the detection
        threshold when flow started (a marginal start is a weak start);
    *   +0.20 x continuity - the fraction of the timed interval in which
        liquid was actually visible;
    *   +0.15 x signal-to-noise of the stream against the region's own noise;
    *   +0.10 when the ending is clean (no long breaks in the final stream);
    *   -0.35 x the fraction of frames that were disturbed;
    *   -0.25 x the fraction of frames the outlet tracker could not trust
        (`lost` or `predicted`) - hand-held motion earns a visibly lower
        confidence than a steady run, distinct from scene-wide disturbance;
    *   the whole result is capped at 0.5 when the end was never confirmed,
        whether because the video ended mid-flow or because a tracking/
        visibility gap left the true end uncertain, and likewise capped at
        0.5 when the *start* was confirmed right after such a gap - a
        late-but-precise start reports a falsely short duration just as
        surely as a falsely late end reports a falsely long one (see below).
    """
    reasons: list[str] = []
    if measurement.start_s is None:
        return 0.0, ["No liquid stream was detected at the outlet."]
    if measurement.end_s is None:
        return 0.0, ["Flow started but no end of flow could be determined."]

    efflux = measurement.efflux_s or 0.0
    start_margin = clamp(measurement.start_margin, 0.0, 1.0)
    continuity = clamp(measurement.continuity, 0.0, 1.0)
    snr = clamp(
        safe_ratio(measurement.mean_contrast, max(measurement.mean_noise_sigma, 1e-6)) / 8.0,
        0.0,
        1.0,
    )
    disturbed_ratio = clamp(
        safe_ratio(measurement.frames_disturbed, max(1, measurement.frames_analysed)), 0.0, 1.0
    )
    untracked_ratio = clamp(
        safe_ratio(measurement.frames_untracked, max(1, measurement.frames_analysed)), 0.0, 1.0
    )
    long_breaks = [
        b for b in measurement.breaks if (b[1] - b[0]) >= config.break_report_threshold_s
    ]
    clean_ending = 0.10 if len(long_breaks) <= 1 else 0.0

    confidence = (
        0.35
        + 0.20 * start_margin
        + 0.20 * continuity
        + 0.15 * snr
        + clean_ending
        - 0.35 * disturbed_ratio
        - 0.25 * untracked_ratio
    )

    if not measurement.end_confirmed:
        confidence = min(confidence, 0.50)
        # Independent ifs, not if/elif (Codex review, third round): a
        # mid-flow gap and a separate, genuinely adjacent end gap can
        # coexist in the same run (two different gaps) - each is a real,
        # distinct concern and both reasons must be reported, not just
        # whichever is checked first.
        has_adjacent_end_bound = (
            measurement.end_uncertain and measurement.end_uncertainty_bounds is not None
        )
        if measurement.end_gap_unresolved:
            # Deliberately does NOT say the true end "could have occurred
            # earlier" (Codex review, second round): trusted liquid was
            # seen again after the gap, which proves the opposite - flow
            # continued past it. What is actually unverified is continuity
            # through that span, not the end's location.
            reasons.append(
                "The outlet tracker lost the outlet for long enough, during the "
                "flow, that the stream's continuity through that span could not be "
                "confirmed - even though liquid was seen again afterward. The "
                "reported end may not reflect a single continuous stream, so the "
                "efflux time is not reported as precise."
            )
        if has_adjacent_end_bound and measurement.end_uncertainty_bounds is not None:
            gap_start, gap_end = measurement.end_uncertainty_bounds
            reasons.append(
                f"The outlet was not confidently tracked for {gap_end - gap_start:.2f}s "
                f"around the reported end ({gap_start:.2f}-{gap_end:.2f}s); the true end "
                "could have occurred anywhere in that span, so the efflux time is not "
                "reported as precise."
            )
        if not measurement.end_gap_unresolved and not has_adjacent_end_bound:
            reasons.append(
                "The video ended while liquid was still visible, so the efflux time is a "
                "lower bound rather than a measurement."
            )
    if measurement.start_uncertain and measurement.start_uncertainty_bounds is not None:
        confidence = min(confidence, 0.50)
        gap_start, gap_end = measurement.start_uncertainty_bounds
        reasons.append(
            f"The outlet was not confidently tracked for {gap_end - gap_start:.2f}s "
            f"around the reported start ({gap_start:.2f}-{gap_end:.2f}s); the true start "
            "could have occurred anywhere in that span, so the efflux time is not "
            "reported as precise."
        )
    if untracked_ratio > 0.02:
        reasons.append(
            f"The outlet tracker could not confidently place the outlet in "
            f"{untracked_ratio:.0%} of frames (camera or cup motion outran tracking)."
        )
    if continuity < 0.6:
        reasons.append(
            f"The stream was only visible in {continuity:.0%} of the timed interval, "
            "which makes the end of flow ambiguous."
        )
    if len(long_breaks) > 1:
        reasons.append(f"{len(long_breaks)} interruptions of the stream were seen before the end.")
    if disturbed_ratio > 0.02:
        reasons.append(
            f"{disturbed_ratio:.0%} of frames showed movement around the cup "
            "(camera or hand movement)."
        )
    if snr < 0.3:
        reasons.append(
            "The stream contrast against the background is low; consider improving "
            "lighting or the background behind the stream."
        )
    if efflux and efflux < config.min_plausible_efflux_s:
        confidence = min(confidence, 0.45)
        reasons.append(f"The measured interval ({efflux:.2f} s) is very short for a Zahn cup test.")
    if not reasons:
        reasons.append("Stream detected continuously with a clear start and a sustained end.")

    return clamp(confidence, 0.0, 0.98), reasons


def _zahn_config_from_params(params: Mapping[str, Any]) -> ZahnConfig:
    """Build a :class:`ZahnConfig`, overriding only the keys supplied.

    Delegates to the shared :func:`app.config.apply_overrides` rather than
    reimplementing it, so a boolean field such as ``zahn_track_outlet`` is
    coerced correctly (``"false"`` from a web form must not become
    ``bool("false") == True``) the same way every other detector's params do.
    """
    return apply_overrides(ZahnConfig(), params, ignore={"roi", "outlet"})


def _zahn_config_to_dict(config: ZahnConfig) -> dict[str, Any]:
    return {
        key: getattr(config, key)
        for key in (
            "activity_threshold",
            "flow_start_persistence_s",
            "flow_end_persistence_s",
            "min_abs_diff",
            "noise_sigma_multiplier",
            "min_stream_height_fraction",
            "max_stream_width_fraction",
            "min_drop_area_px",
            "outlet_band_fraction",
            "disturbance_area_ratio",
            "review_confidence",
            "fail_confidence",
            "zahn_track_outlet",
            "track_search_margin_px",
            "track_max_frame_displacement_px",
            "track_min_inliers",
            "track_min_features",
            "track_max_bridge_s",
            "track_reacquire_min_correlation",
            "track_min_patch_correlation",
            "track_reconciliation_max_disagreement_px",
            "zahn_max_endpoint_uncertainty_s",
        )
    }
