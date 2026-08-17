"""Video metadata probing.

Metadata from real-world files is frequently wrong or missing: surveillance
exports with no frame count, variable-frame-rate phone recordings, containers
whose reported FPS is 0.  Everything here therefore treats the container's
claims as a hypothesis to be checked, and records *how* each value was
obtained so the UI can warn the user when a value was estimated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from ..errors import (
    EmptyVideoError,
    VariableFrameRateError,
    VideoMetadataError,
    VideoOpenError,
)
from ..logging_setup import get_logger
from .sampling import MAX_PLAUSIBLE_FPS, MIN_PLAUSIBLE_FPS

logger = get_logger(__name__)

# Number of frames decoded when the container's FPS has to be re-derived from
# presentation timestamps.
FPS_ESTIMATION_FRAMES = 60

# --- variable-frame-rate detection ----------------------------------------- #
# A VFR file usually declares a plausible *average* frame rate, so it passes
# every other check and is then timed as if it were constant. These settings
# decide when the spacing between presentation timestamps is irregular enough
# that the constant-rate assumption must be refused.
#
# The evidence must be representative of the *whole* recording. Sampling only
# the opening seconds misses the common case of a file that starts at a steady
# cadence and changes later - a phone or surveillance clip whose rate drops
# once the scene goes quiet or the encoder falls behind. Several short windows
# spread across the duration are compared both internally and against each
# other, which catches an irregular later section and a later section that is
# internally steady but at a different rate.

# Windows sampled across the recording, and frames grabbed in each. Five
# windows of 40 frames is ~200 frames however long the file is: bounded, and
# a handful of keyframe seeks.
VFR_WINDOW_COUNT = 5
VFR_WINDOW_FRAMES = 40

# A window needs this many intervals before its cadence means anything.
VFR_MIN_INTERVALS = 20

# Within a window, an interval is irregular when it differs from that window's
# median by more than this fraction. 0.5 is far wider than the +/-1 ms
# quantisation of a 29.97 FPS file and far narrower than a real rate change.
VFR_INTERVAL_TOLERANCE = 0.5

# A window is irregular when more than this fraction of its intervals are.
# A tenth tolerates the occasional dropped frame.
VFR_MAX_IRREGULAR_RATIO = 0.10

# Between windows, the median interval may drift by at most this fraction.
# Millisecond quantisation moves a median by ~3%; a real rate change (30 -> 25
# FPS is 20%, 30 -> 15 is 100%) moves it far more.
VFR_MEDIAN_DRIFT_TOLERANCE = 0.15

# A recording this short cannot accumulate meaningful drift even if its cadence
# does vary, so it is accepted without cadence evidence rather than refused for
# being too short to judge.
VFR_NEGLIGIBLE_FRAMES = VFR_MIN_INTERVALS + 1

# How far the reported timing may drift from the recording's own presentation
# timestamps, in seconds. Tune this the way the other thresholds above are
# tuned; it is the one that decides accuracy rather than shape.
#
# The three rules above ask "does the cadence *look* irregular?". This one
# measures the harm directly: at each sampled window start the frame's real
# presentation timestamp is compared against `index / fps`, which is the only
# timing this application ever reports (see reader.py). The *spread* of those
# offsets is what corrupts a measured duration - a constant offset shifts every
# timestamp equally and cancels out of every interval, so a recording that
# simply starts at a non-zero PTS is not penalised.
#
# Measuring drift also covers what the shape rules cannot: the windows examine
# roughly a third of a long recording, and an irregular burst falling entirely
# between two of them leaves every window internally perfect and every median
# identical. The offset at the *next* window start still carries it, because it
# accounts for everything decoded before that frame.
#
# 0.05 s is an order of magnitude below the shortest persistence rule the
# analysis layer applies (0.30 s to start Zahn flow), so a recording inside the
# budget cannot move an event across a decision boundary.
VFR_MAX_TIMING_ERROR_S = 0.05

# Shown when the cadence could not be checked at all, as opposed to when it was
# checked and found to vary. Both are refusals - an unverified constant-rate
# assumption is what silently corrupts timestamps - but the user deserves to
# know which happened.
UNVERIFIABLE_TIMING_MESSAGE = (
    "The frame timing of this recording could not be verified, so it has been "
    "rejected rather than measured on an assumption that may be wrong. This "
    "usually means the file does not carry usable frame timestamps, or is too "
    "long for its length to be established. Re-saving it with a constant frame "
    "rate (most editors can do this, as can 'ffmpeg -vsync cfr') will fix it."
)

# Shown when the cadence looks steady everywhere it was sampled but the frame
# timing has still drifted away from a single frame rate by more than the
# accuracy budget. This is the case an operator is least likely to expect - the
# file looks completely normal - so the message says what was measured.
TIMING_DRIFT_MESSAGE = (
    "The frame timing of this recording drifts too far from a single frame "
    "rate, so its timestamps would be wrong by more than the accuracy this "
    "tool guarantees, and it has been rejected rather than measured. This "
    "usually means frames were dropped unevenly while recording. Re-save it "
    "with a constant frame rate (most editors can do this, as can "
    "'ffmpeg -vsync cfr') and upload it again."
)


@dataclass(frozen=True)
class VideoInfo:
    """Everything the analysis layer needs to know about a source file."""

    path: Path
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_s: float | None
    fourcc: str
    size_bytes: int
    fps_source: str  # "container" | "timestamps"
    duration_source: str  # "frame_count" | "timestamps" | "unknown"
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_variable_frame_rate_suspect(self) -> bool:
        return self.fps_source == "timestamps"

    def to_dict(self) -> dict[str, object]:
        return {
            "filename": self.path.name,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 4),
            "frame_count": self.frame_count,
            "duration_s": round(self.duration_s, 3) if self.duration_s is not None else None,
            "fourcc": self.fourcc,
            "size_bytes": self.size_bytes,
            "fps_source": self.fps_source,
            "duration_source": self.duration_source,
            "warnings": list(self.warnings),
        }


def _fourcc_to_str(value: float) -> str:
    code = int(value)
    if code <= 0:
        return "unknown"
    chars = [chr((code >> shift) & 0xFF) for shift in (0, 8, 16, 24)]
    text = "".join(c for c in chars if c.isprintable()).strip()
    return text or "unknown"


def _estimate_fps_from_timestamps(capture: cv2.VideoCapture) -> float | None:
    """Derive FPS from presentation timestamps of the first frames.

    Returns ``None`` when the backend does not expose usable timestamps.
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    first_ms: float | None = None
    frames_read = 0
    last_ms = 0.0
    for _ in range(FPS_ESTIMATION_FRAMES):
        ok = capture.grab()
        if not ok:
            break
        position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        if not math.isfinite(position_ms) or position_ms <= 0:
            # Some backends report 0 for every frame; that is unusable.
            if frames_read > 0 and position_ms == last_ms:
                return None
            continue
        if first_ms is None:
            first_ms = position_ms
        last_ms = position_ms
        frames_read += 1

    if first_ms is None or frames_read < 5 or last_ms <= first_ms:
        return None
    elapsed_s = (last_ms - first_ms) / 1000.0
    fps = (frames_read - 1) / elapsed_s
    if not MIN_PLAUSIBLE_FPS <= fps <= MAX_PLAUSIBLE_FPS:
        return None
    return fps


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


@dataclass(frozen=True)
class FrameTimingEvidence:
    """Cadence samples taken from across a recording."""

    windows: list[list[float]]  # frame intervals in ms, one list per window
    frame_count: int | None
    windows_requested: int

    # For each sampled window, how far that window's first frame sits from
    # where `index / fps` says it should be, in milliseconds. Empty when no
    # frame rate was supplied to compare against.
    offsets_ms: list[float] = field(default_factory=list)

    @property
    def usable_windows(self) -> list[list[float]]:
        return [window for window in self.windows if len(window) >= VFR_MIN_INTERVALS]

    @property
    def is_representative(self) -> bool:
        """Whether the samples say anything about the recording as a whole.

        Two windows from different parts of the file qualify. So does a single
        window on a file short enough for that window to cover most of it. A
        single opening window on a long file does not - that is precisely the
        blind spot that lets a file which changes cadence later slip through.
        """
        usable = self.usable_windows
        if not usable:
            return False
        if len(usable) >= 2:
            return True
        if not self.frame_count:
            return False
        return (len(usable[0]) + 1) >= 0.5 * self.frame_count


@dataclass(frozen=True)
class FrameTimingVerdict:
    """Whether the recording can be timed with a single frame rate."""

    reliable: bool
    reason: str
    detail: str


def _window_start_indices(frame_count: int, windows: int, window_frames: int) -> list[int]:
    """Evenly spaced window starts that all fit inside the recording."""
    last_start = max(0, frame_count - window_frames)
    if windows <= 1 or last_start == 0:
        return [0]
    step = last_start / (windows - 1)
    return sorted({int(round(index * step)) for index in range(windows)})


def collect_interval_windows(
    capture: cv2.VideoCapture,
    *,
    frame_count: int | None,
    window_count: int = VFR_WINDOW_COUNT,
    window_frames: int = VFR_WINDOW_FRAMES,
    fps: float | None = None,
) -> FrameTimingEvidence:
    """Sample frame intervals from several points across the recording.

    When ``fps`` is supplied, each window also records how far its first frame's
    presentation timestamp sits from ``index / fps`` — the timing this
    application actually reports. Those offsets are what
    :func:`assess_frame_timing` uses to bound the error directly instead of
    inferring it from the shape of the cadence.

    Each window is reached with a frame-index seek, and a window counts only
    when that seek is *positively* verified: the request must succeed and the
    backend must report landing near where it was sent. A backend that ignored
    the seek - or that cannot say where it is, reporting a non-finite position -
    would otherwise hand back the opening frames repeatedly and make a recording
    that changes cadence later look perfectly steady. Unverifiable is therefore
    treated as unusable, and the resulting shortage of windows is refused by the
    representativeness rule in :func:`assess_frame_timing`.
    """
    starts = _window_start_indices(frame_count, window_count, window_frames) if frame_count else [0]
    windows: list[list[float]] = []
    offsets_ms: list[float] = []
    can_measure_offset = fps is not None and math.isfinite(fps) and fps > 0

    for start in starts:
        seek_ok = bool(capture.set(cv2.CAP_PROP_POS_FRAMES, float(start)))
        landed = capture.get(cv2.CAP_PROP_POS_FRAMES)
        if not seek_ok or not math.isfinite(landed) or abs(landed - start) > window_frames:
            logger.debug(
                "Seek to frame %d not verified (accepted=%s, landed=%r); skipping that window",
                start,
                seek_ok,
                landed,
            )
            continue

        stamps: list[float] = []
        for _ in range(window_frames):
            if not capture.grab():
                break
            position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            if not math.isfinite(position_ms):
                break
            stamps.append(position_ms)

        # stamps[0] belongs to frame `start` itself, so it is directly
        # comparable with what the reader would report for that index.
        if stamps and can_measure_offset:
            assert fps is not None  # for type checkers; can_measure_offset proves it
            offsets_ms.append(stamps[0] - start * 1000.0 / fps)

        # Non-positive gaps mean the backend is not reporting real timestamps;
        # dropping them lets "no usable evidence" be recognised as such.
        intervals = [
            later - earlier
            for earlier, later in zip(stamps, stamps[1:], strict=False)
            if later - earlier > 0
        ]
        if intervals:
            windows.append(intervals)

    return FrameTimingEvidence(
        windows=windows,
        frame_count=frame_count,
        windows_requested=len(starts),
        offsets_ms=offsets_ms,
    )


def assess_frame_timing(
    evidence: FrameTimingEvidence,
    *,
    tolerance: float = VFR_INTERVAL_TOLERANCE,
    max_irregular_ratio: float = VFR_MAX_IRREGULAR_RATIO,
    drift_tolerance: float = VFR_MEDIAN_DRIFT_TOLERANCE,
    max_timing_error_s: float = VFR_MAX_TIMING_ERROR_S,
) -> FrameTimingVerdict:
    """Decide whether a single frame rate can describe the whole recording.

    Pure and free of OpenCV, so every branch can be tested against scripted
    cadence patterns. Refuses whenever reliability cannot be *established* -
    not only when variability is proven - because an unverified constant-rate
    assumption is exactly what produces silently wrong timestamps.
    """
    if evidence.frame_count and evidence.frame_count <= VFR_NEGLIGIBLE_FRAMES:
        return FrameTimingVerdict(
            True,
            "too_short_to_matter",
            f"{evidence.frame_count} frames; drift cannot accumulate",
        )

    usable = evidence.usable_windows
    if not usable:
        return FrameTimingVerdict(
            False,
            "no_usable_timestamps",
            f"{evidence.windows_requested} window(s) sampled, none yielded "
            f"{VFR_MIN_INTERVALS} usable presentation timestamps",
        )

    if not evidence.is_representative:
        return FrameTimingVerdict(
            False,
            "unrepresentative_sample",
            f"only one window covering {len(usable[0]) + 1} of {evidence.frame_count} "
            "frames could be sampled, so a later change in cadence would be missed",
        )

    medians = [_median(window) for window in usable]
    for position, (window, median) in enumerate(zip(usable, medians, strict=True)):
        irregular = sum(1 for value in window if abs(value - median) > tolerance * median)
        if irregular / len(window) > max_irregular_ratio:
            return FrameTimingVerdict(
                False,
                "irregular_within_window",
                f"window {position + 1} of {len(usable)}: {irregular}/{len(window)} intervals "
                f"deviate from its {median:.1f} ms median",
            )

    overall = _median(medians)
    drift = max(abs(median - overall) for median in medians)
    if overall > 0 and drift > drift_tolerance * overall:
        return FrameTimingVerdict(
            False,
            "rate_changes_between_windows",
            "window medians "
            + ", ".join(f"{median:.1f}" for median in medians)
            + f" ms differ by up to {drift / overall:.0%}",
        )

    # Finally, bound the error itself. A constant offset shifts every reported
    # timestamp by the same amount and cancels out of every duration, so only
    # the *spread* of the offsets is charged against the budget.
    offsets = evidence.offsets_ms
    if len(offsets) >= 2:
        spread_ms = max(offsets) - min(offsets)
        if spread_ms > max_timing_error_s * 1000.0:
            return FrameTimingVerdict(
                False,
                "timing_drift",
                "presentation timestamps drift "
                + ", ".join(f"{value:+.1f}" for value in offsets)
                + f" ms from index/fps across the sampled windows: a spread of "
                f"{spread_ms:.0f} ms against a {max_timing_error_s * 1000:.0f} ms budget",
            )

    return FrameTimingVerdict(
        True,
        "constant",
        f"{len(usable)} window(s) across the recording, median {overall:.1f} ms"
        + (
            f", timing drift within {max(offsets) - min(offsets):.1f} ms"
            if len(offsets) >= 2
            else ""
        ),
    )


def _refusal_message(reason: str) -> str | None:
    """The user-facing wording for a refusal, or ``None`` for the default.

    The three cases are genuinely different from the operator's point of view -
    "we could not check", "the timing drifted", and "the rate varies" - and a
    single message would send someone looking for the wrong problem.
    """
    if reason in {"no_usable_timestamps", "unrepresentative_sample"}:
        return UNVERIFIABLE_TIMING_MESSAGE
    if reason == "timing_drift":
        return TIMING_DRIFT_MESSAGE
    return None


def probe_video(path: Path, *, max_timing_error_s: float = VFR_MAX_TIMING_ERROR_S) -> VideoInfo:
    """Open ``path`` and return validated metadata.

    Args:
        path: the video file to probe.
        max_timing_error_s: how far presentation timestamps may drift from a
            single frame rate before the file is refused as unreliably timed
            (see :data:`VFR_MAX_TIMING_ERROR_S`). Defaults to the module
            constant so direct callers - a script, a test, a ``VideoReader``
            constructed without a probed ``VideoInfo`` - see unchanged
            behaviour. The upload path passes ``AppConfig.max_timing_error_s``
            explicitly so it can be tuned without editing source.

    Raises:
        VideoOpenError: the file cannot be opened/decoded at all.
        EmptyVideoError: the file opens but yields no frames.
        VideoMetadataError: the frame rate cannot be established.
        VariableFrameRateError: the frame rate is not constant, or drifts
            beyond ``max_timing_error_s``, so the file cannot be timed
            accurately and is refused.
    """
    path = Path(path)
    if not path.is_file():
        raise VideoOpenError("The video file could not be found.", detail=str(path))

    size_bytes = path.stat().st_size
    if size_bytes == 0:
        raise EmptyVideoError("The uploaded file is empty.")

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoOpenError(detail=f"cv2.VideoCapture could not open {path.name}")

    try:
        warnings: list[str] = []

        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise EmptyVideoError(detail=f"no decodable frames in {path.name}")
        height, width = first_frame.shape[:2]

        reported_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        reported_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if (reported_width, reported_height) != (width, height) and reported_width > 0:
            warnings.append(
                "The container reports a different resolution than the decoded frames; "
                "the decoded size is used."
            )

        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps_source = "container"
        if not math.isfinite(fps) or not MIN_PLAUSIBLE_FPS <= fps <= MAX_PLAUSIBLE_FPS:
            logger.warning("Implausible container FPS %r for %s; estimating", fps, path.name)
            estimated = _estimate_fps_from_timestamps(capture)
            if estimated is None:
                raise VideoMetadataError(detail=f"container fps={fps!r}, estimation failed")
            fps = estimated
            fps_source = "timestamps"
            warnings.append(
                "The frame rate was estimated from timestamps because the file does not "
                "declare a usable one. Timings may be less accurate."
            )

        raw_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        has_frame_count = math.isfinite(raw_count) and raw_count > 0
        frame_count: int | None = int(raw_count) if has_frame_count else None

        # Refuse any recording that cannot be timed with a single frame rate.
        # Every timing here is derived from one FPS value, so a file whose
        # cadence changes would be reported with confidently wrong timestamps -
        # the one outcome this application must never produce. The evidence is
        # gathered from windows spread across the whole file, because a file
        # that starts steady and changes later is the common case.
        # Full VFR support is future work.
        evidence = collect_interval_windows(capture, frame_count=frame_count, fps=fps)
        verdict = assess_frame_timing(evidence, max_timing_error_s=max_timing_error_s)
        if not verdict.reliable:
            logger.warning(
                "Rejecting %s: frame timing not reliable (%s; %s; container claims %.3f fps)",
                path.name,
                verdict.reason,
                verdict.detail,
                fps,
            )
            raise VariableFrameRateError(
                _refusal_message(verdict.reason),
                detail=f"{verdict.reason}: {verdict.detail}; declared fps {fps:.3f}",
            )
        logger.debug("Frame timing accepted for %s (%s)", path.name, verdict.detail)

        duration_s: float | None
        if frame_count:
            duration_s = frame_count / fps
            duration_source = "frame_count"
        else:
            duration_s = _duration_from_end_timestamp(capture)
            duration_source = "timestamps" if duration_s is not None else "unknown"
            if duration_s is None:
                warnings.append(
                    "The video length is unknown; progress reporting will be approximate."
                )

        info = VideoInfo(
            path=path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=duration_s,
            fourcc=_fourcc_to_str(capture.get(cv2.CAP_PROP_FOURCC)),
            size_bytes=size_bytes,
            fps_source=fps_source,
            duration_source=duration_source,
            warnings=tuple(warnings),
        )
        logger.info(
            "Probed %s: %dx%d @ %.3f fps (%s), %s frames, duration %s s, codec %s, %.1f MB",
            path.name,
            info.width,
            info.height,
            info.fps,
            info.fps_source,
            info.frame_count,
            f"{info.duration_s:.2f}" if info.duration_s else "unknown",
            info.fourcc,
            size_bytes / 1e6,
        )
        return info
    finally:
        capture.release()


def _duration_from_end_timestamp(capture: cv2.VideoCapture) -> float | None:
    """Best-effort duration when the container has no frame count."""
    try:
        # Seeking by relative position works on containers that know their
        # length even when they do not expose a frame count.
        capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1.0)
        position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if math.isfinite(position_ms) and position_ms > 0:
            return position_ms / 1000.0
    except cv2.error as exc:  # pragma: no cover - backend dependent
        logger.debug("Duration probing by seek failed: %s", exc)
    return None
