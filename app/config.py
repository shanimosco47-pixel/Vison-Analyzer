"""Central configuration.

Design rules for this module:

*   Every parameter that represents *time* is stored in **seconds**, never in
    frames.  Frame indices are derived at runtime from the video's real FPS
    (see :mod:`app.video.sampling`).  A rule such as "15 frames" silently
    changes meaning between a 15 FPS and a 60 FPS recording.
*   Every value has a comment explaining what it controls and how to tune it.
*   Nothing here reads the network, and nothing here is machine specific.

The dataclasses are plain and JSON-friendly so the web layer can override
individual fields from the UI without the analysis code knowing about HTTP.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .errors import ConfigurationError, InvalidROIError

# --------------------------------------------------------------------------- #
# Region of interest
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ROI:
    """An axis-aligned region in *source* pixel coordinates.

    ROIs are always expressed against the original video resolution.  Scaling
    to whatever resolution a stage happens to use is done at the point of use,
    so a stored ROI stays valid when the scan resolution changes.
    """

    x: int
    y: int
    width: int
    height: int

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    def validate(self, frame_width: int, frame_height: int, *, minimum_side: int = 4) -> None:
        """Raise :class:`InvalidROIError` unless the ROI fits in the frame."""
        if self.width < minimum_side or self.height < minimum_side:
            raise InvalidROIError(
                f"The selected region is too small (minimum {minimum_side}x{minimum_side} pixels)."
            )
        if self.x < 0 or self.y < 0 or self.x2 > frame_width or self.y2 > frame_height:
            raise InvalidROIError(
                "The selected region extends outside the video frame "
                f"({frame_width}x{frame_height} pixels)."
            )

    def clipped_to(self, frame_width: int, frame_height: int) -> ROI:
        """Intersect with the frame; raises if nothing of the ROI is visible."""
        x = max(0, self.x)
        y = max(0, self.y)
        x2 = min(frame_width, self.x2)
        y2 = min(frame_height, self.y2)
        if x2 <= x or y2 <= y:
            raise InvalidROIError()
        return ROI(x, y, x2 - x, y2 - y)

    def scaled(self, factor: float) -> ROI:
        """Return the ROI mapped into an image scaled by ``factor``."""
        if factor <= 0:
            raise ConfigurationError("Scale factor must be positive.")
        return ROI(
            int(round(self.x * factor)),
            int(round(self.y * factor)),
            max(1, int(round(self.width * factor))),
            max(1, int(round(self.height * factor))),
        )

    def to_dict(self) -> dict[str, int]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ROI:
        try:
            return cls(
                x=int(data["x"]),
                y=int(data["y"]),
                width=int(data["width"]),
                height=int(data["height"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise InvalidROIError("The selected region is not valid.", detail=str(exc)) from exc


# --------------------------------------------------------------------------- #
# Stage A - coarse scan
# --------------------------------------------------------------------------- #


@dataclass
class CoarseScanConfig:
    """Stage A: cheap pass over the whole recording to find *candidate* windows."""

    # The shortest event that must never be missed.  This is the single most
    # important knob: the sampling interval is derived from it, so it is the
    # user-facing question ("what is the shortest thing worth catching?")
    # rather than an opaque "sample every N seconds".
    shortest_event_s: float = 15.0

    # How many samples must fall inside the shortest event.  >= 2 guarantees
    # (by pigeonhole) that at least one sample lands inside it; 3 gives margin
    # for a sample landing on a weak first/last moment of the event.
    safety_factor: float = 3.0

    # Hard bounds so a silly shortest_event_s cannot produce an absurd plan.
    min_sample_interval_s: float = 0.10
    max_sample_interval_s: float = 5.00

    # Width the frame is downscaled to for scanning (height keeps the aspect
    # ratio).  320 px keeps a person-sized object several pixels wide while
    # costing ~1/20 of a 1080p frame to process.
    scan_width_px: int = 320

    # Fraction of ROI pixels that must change before a sample counts as
    # "active".  Used as an absolute floor underneath the adaptive threshold,
    # so a completely static camera does not fire on sensor noise.
    min_changed_area_ratio: float = 0.004

    # Adaptive thresholding: a sample is active when its change score exceeds
    # median + enter_sigma * MAD-derived-sigma of the whole trace, and stays
    # active while above median + exit_sigma * sigma (hysteresis).
    enter_sigma: float = 6.0
    exit_sigma: float = 3.0

    # 0 = least sensitive/fastest, 1 = most sensitive.  Scales the sigma pair
    # above; exposed in the UI as a single "speed vs sensitivity" slider.
    sensitivity: float = 0.5

    # Candidate windows shorter than this are discarded as noise before
    # refinement (Stage B is comparatively expensive).
    min_candidate_duration_s: float = 1.0

    # Gaps shorter than this between two active runs are bridged into one
    # candidate, so a brief pause mid-event does not split it in two.
    bridge_gap_s: float = 2.0

    # Safety valve for pathological footage (a strobing light, a shaking
    # camera): stop refining after this many candidates and say so in the log
    # and the UI rather than silently truncating.
    max_candidates: int = 200

    def validate(self) -> None:
        if self.shortest_event_s <= 0:
            raise ConfigurationError("The shortest event duration must be greater than zero.")
        if self.safety_factor < 2.0:
            raise ConfigurationError(
                "The sampling safety factor must be at least 2, otherwise an event "
                "can fall between two samples."
            )
        if not 0.0 <= self.sensitivity <= 1.0:
            raise ConfigurationError("Sensitivity must be between 0 and 1.")
        if self.min_sample_interval_s <= 0 or self.max_sample_interval_s <= 0:
            raise ConfigurationError("Sampling interval bounds must be positive.")
        if self.min_sample_interval_s > self.max_sample_interval_s:
            raise ConfigurationError("The minimum sampling interval exceeds the maximum.")
        if self.scan_width_px < 64:
            raise ConfigurationError("The scan width must be at least 64 pixels.")
        if self.exit_sigma > self.enter_sigma:
            raise ConfigurationError(
                "The hysteresis exit threshold must not be above the enter threshold."
            )
        if self.max_candidates < 1:
            raise ConfigurationError("At least one candidate window must be allowed.")


# --------------------------------------------------------------------------- #
# Stage B - candidate refinement
# --------------------------------------------------------------------------- #


@dataclass
class RefinementConfig:
    """Stage B: dense re-analysis of the neighbourhood of each candidate."""

    # How far back/forward from the coarse hit the dense pass looks.  Must be
    # at least one coarse sampling interval, otherwise the true start can lie
    # outside the refined window; the service enforces that at runtime.
    pre_roll_s: float = 5.0
    post_roll_s: float = 5.0

    # Dense sampling step.  0 means "every frame".  A small non-zero value is
    # useful on very high frame-rate footage where 1/2 frames is plenty.
    dense_step_s: float = 0.0

    # Resolution used during refinement.  Higher than the coarse pass because
    # accurate boundaries need more detail, but still not necessarily full res.
    refine_width_px: int = 640

    # An event must persist this long in the dense signal to be confirmed.
    # This is what removes single-frame flashes that survived Stage A.
    min_event_duration_s: float = 1.0

    # Activity gaps shorter than this inside a refined event are treated as
    # part of the same event (an arm pausing mid-cycle is not two events).
    merge_gap_s: float = 1.5

    # Events below this confidence are labelled "review recommended" in the UI
    # and the CSV instead of being presented as established fact.
    review_confidence: float = 0.60

    def validate(self) -> None:
        if self.pre_roll_s < 0 or self.post_roll_s < 0:
            raise ConfigurationError("Pre-roll and post-roll must not be negative.")
        if self.dense_step_s < 0:
            raise ConfigurationError("The dense sampling step must not be negative.")
        if self.refine_width_px < 64:
            raise ConfigurationError("The refinement width must be at least 64 pixels.")
        if self.min_event_duration_s < 0 or self.merge_gap_s < 0:
            raise ConfigurationError("Event duration and gap settings must not be negative.")
        if not 0.0 <= self.review_confidence <= 1.0:
            raise ConfigurationError("The review confidence must be between 0 and 1.")


# --------------------------------------------------------------------------- #
# Zahn cup
# --------------------------------------------------------------------------- #


@dataclass
class ZahnConfig:
    """Zahn cup efflux timing.

    Defaults are starting points chosen from the physics of the measurement,
    not from one demonstration video.  They are expected to be tuned against
    real cup footage; every one of them is exposed here for that reason.
    """

    # --- analysis region -------------------------------------------------- #
    # When the user clicks the outlet instead of drawing a box, the analysis
    # region is built from the click: this many multiples of the nominal
    # stream width to each side, extending downward by roi_height_fraction of
    # the frame height.
    default_roi_width_px: int = 60
    roi_height_fraction: float = 0.45

    # A thin band at the top of the ROI, immediately under the outlet.  The
    # stream must be present here for flow to count as "coming from the cup";
    # this is what stops a hand crossing the lower ROI from starting the clock.
    outlet_band_fraction: float = 0.22

    # --- what counts as liquid ------------------------------------------- #
    # Absolute grey-level difference floor.  The working threshold is
    # max(this, noise_sigma_multiplier * measured noise sigma), so a noisy or
    # heavily compressed camera raises its own bar automatically.
    min_abs_diff: int = 12
    noise_sigma_multiplier: float = 4.0

    # Morphological opening kernel (pixels) used to delete speckle noise
    # before shape analysis.  Larger removes more noise but also thin streams.
    open_kernel_px: int = 2

    # A stream is a *tall, thin, vertical* object.  A blob only counts as
    # liquid when it spans at least this fraction of the ROI height or is a
    # falling drop of at least min_drop_area_px.
    min_stream_height_fraction: float = 0.25
    max_stream_width_fraction: float = 0.55
    min_drop_area_px: int = 8

    # Score above which a frame is called "liquid present".  The score is the
    # fraction of ROI rows containing stream pixels, so 0.15 means the stream
    # is visible over at least 15% of the region's height.
    activity_threshold: float = 0.12

    # --- temporal persistence (seconds, never frames) --------------------- #
    # Liquid must be seen continuously for this long before flow start is
    # declared.  The reported start is the *first* frame of that run, not the
    # moment of confirmation.  0.25-0.4 s rejects reflections and single bad
    # frames while costing nothing in accuracy.
    flow_start_persistence_s: float = 0.30

    # Sustained absence of any liquid activity (stream *or* drops) before the
    # flow is declared finished.  The reported end is the last frame that had
    # activity.  The end of a Zahn run is stream -> weak stream -> drops, so
    # this must be long enough to bridge the gaps between final drops.
    flow_end_persistence_s: float = 0.50

    # Gaps in the stream longer than this during the run are recorded as
    # "breaks" and lower the confidence, because they make the end ambiguous.
    break_report_threshold_s: float = 0.20

    # --- disturbance rejection ------------------------------------------- #
    # A guard region (the ROI grown sideways, excluding the ROI itself) is
    # monitored for whole-scene change.  Above this fraction the frame is
    # marked "disturbed" - camera knock, hand, light switch - and can neither
    # start the clock nor end it.
    disturbance_area_ratio: float = 0.35
    guard_margin_px: int = 40

    # If the median pixel of the whole watched area differs from the reference
    # by this much, more than half the picture changed at once (the camera was
    # knocked, the light was switched, someone stepped in front of the cup).
    # Checked before thresholding, because such a frame also inflates the
    # measured noise and would otherwise hide behind its own threshold.
    disturbance_median_diff: float = 18.0

    # --- reporting -------------------------------------------------------- #
    # A Zahn cup #4 reading below roughly 20 s is outside the cup's useful
    # range, and anything of a couple of seconds is almost certainly a
    # mis-detection rather than a real measurement.  Results shorter than this
    # are capped in confidence and flagged for review rather than reported as
    # fact.
    min_plausible_efflux_s: float = 2.0

    # Below this confidence the result is presented as "review recommended"
    # instead of a plain number.
    review_confidence: float = 0.70
    # Below this the detection is reported as failed and *no* time is given.
    fail_confidence: float = 0.40

    def validate(self) -> None:
        if self.flow_start_persistence_s <= 0 or self.flow_end_persistence_s <= 0:
            raise ConfigurationError("Flow persistence values must be greater than zero.")
        if not 0.0 < self.activity_threshold < 1.0:
            raise ConfigurationError("The liquid activity threshold must be between 0 and 1.")
        if not 0.0 < self.outlet_band_fraction <= 1.0:
            raise ConfigurationError("The outlet band fraction must be between 0 and 1.")
        if not 0.0 < self.roi_height_fraction <= 1.0:
            raise ConfigurationError("The ROI height fraction must be between 0 and 1.")
        if self.min_abs_diff < 1:
            raise ConfigurationError("The minimum difference threshold must be at least 1.")
        if not 0.0 <= self.fail_confidence <= self.review_confidence <= 1.0:
            raise ConfigurationError("Confidence thresholds must satisfy 0 <= fail <= review <= 1.")


# --------------------------------------------------------------------------- #
# Application / server
# --------------------------------------------------------------------------- #

SUPPORTED_VIDEO_EXTENSIONS: tuple[str, ...] = (
    ".mp4",
    ".m4v",
    ".mov",
    ".avi",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".wmv",
    ".webm",
)


@dataclass
class AppConfig:
    """Server-side settings. Everything is local by default."""

    # Where uploads live.  Files are stored under generated names; the name
    # supplied by the browser is never used as a path component.
    data_dir: Path = field(default_factory=lambda: Path.home() / ".vision_analyzer")

    max_upload_mb: int = 4096  # 4 GB; a 12 h 1080p H.264 recording fits comfortably.
    allowed_extensions: tuple[str, ...] = SUPPORTED_VIDEO_EXTENSIONS

    host: str = "127.0.0.1"  # Loopback only. Binding elsewhere is an explicit choice.
    port: int = 8000
    debug: bool = False

    # Debug artefacts (ROI crops, masks, boundary frames) are written under
    # data_dir/diagnostics/<job id>/ when enabled.  Off by default because it
    # writes a lot of images.
    save_diagnostics: bool = False

    # Delete uploads and their job records after this many hours of inactivity.
    retention_hours: float = 24.0

    # How far a recording's presentation timestamps may drift from a single
    # frame rate, in seconds, before the upload is refused as unreliably timed
    # (see app.video.metadata.VFR_MAX_TIMING_ERROR_S for the mechanism). Tune
    # this down for a use case that needs tighter accuracy than the default
    # gives, or up if real footage is being refused for drift that does not
    # matter to the measurement being taken.
    max_timing_error_s: float = 0.05

    log_level: str = "INFO"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def diagnostics_dir(self) -> Path:
        return self.data_dir / "diagnostics"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def validate(self) -> None:
        if self.max_upload_mb < 1:
            raise ConfigurationError("The upload size limit must be at least 1 MB.")
        if not 1 <= self.port <= 65535:
            raise ConfigurationError("The port number is invalid.")
        if self.retention_hours <= 0:
            raise ConfigurationError("The retention period must be greater than zero.")
        if self.max_timing_error_s <= 0:
            raise ConfigurationError("The frame-timing error budget must be greater than zero.")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AppConfig:
        """Build a config from ``VISION_ANALYZER_*`` environment variables."""
        env = os.environ if env is None else env
        cfg = cls()
        if value := env.get("VISION_ANALYZER_DATA_DIR"):
            cfg = replace(cfg, data_dir=Path(value).expanduser())
        if value := env.get("VISION_ANALYZER_MAX_UPLOAD_MB"):
            cfg = replace(cfg, max_upload_mb=_int_env("VISION_ANALYZER_MAX_UPLOAD_MB", value))
        if value := env.get("VISION_ANALYZER_HOST"):
            cfg = replace(cfg, host=value)
        if value := env.get("VISION_ANALYZER_PORT"):
            cfg = replace(cfg, port=_int_env("VISION_ANALYZER_PORT", value))
        if value := env.get("VISION_ANALYZER_LOG_LEVEL"):
            cfg = replace(cfg, log_level=value.upper())
        if _bool_env(env.get("VISION_ANALYZER_DEBUG")):
            cfg = replace(cfg, debug=True)
        if _bool_env(env.get("VISION_ANALYZER_SAVE_DIAGNOSTICS")):
            cfg = replace(cfg, save_diagnostics=True)
        if value := env.get("VISION_ANALYZER_RETENTION_HOURS"):
            cfg = replace(cfg, retention_hours=_float_env("VISION_ANALYZER_RETENTION_HOURS", value))
        if value := env.get("VISION_ANALYZER_MAX_TIMING_ERROR_S"):
            cfg = replace(
                cfg,
                max_timing_error_s=_float_env("VISION_ANALYZER_MAX_TIMING_ERROR_S", value),
            )
        cfg.validate()
        return cfg


TConfig = Any


def apply_overrides(
    config: TConfig, params: Mapping[str, Any], *, ignore: set[str] | None = None
) -> TConfig:
    """Copy recognised keys from ``params`` onto a configuration dataclass.

    Unknown keys are ignored on purpose: the UI sends one flat parameter bag
    covering several configuration objects, and each object takes the keys it
    understands.  Values are coerced to the field's existing type so a string
    from a form field becomes a number rather than silently corrupting the
    configuration.
    """
    ignore = ignore or set()
    for key, value in params.items():
        if key in ignore or value is None or not hasattr(config, key):
            continue
        current = getattr(config, key)
        if isinstance(current, (dict, list, tuple, Path)):
            continue
        try:
            coerced = bool(value) if isinstance(current, bool) else type(current)(value)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"The setting '{key}' has an invalid value.", detail=repr(value)
            ) from exc
        setattr(config, key, coerced)
    return config


def _int_env(name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a whole number.", detail=raw) from exc


def _float_env(name: str, raw: str) -> float:
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number.", detail=raw) from exc


def _bool_env(raw: str | None) -> bool:
    return (raw or "").strip().lower() in {"1", "true", "yes", "on"}
