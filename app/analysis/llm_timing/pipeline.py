"""Coarse-to-fine orchestration: the one place that decides which frames a
provider gets to see, and what happens when it abstains.

This reproduces, as a deterministic and auditable pipeline, the two-pass
strategy that worked by hand against two real clips before this spike
existed (see ``diagnostics/llm_spike/DESIGN.md``): sample sparsely across
the whole clip first, then re-extract every frame inside the short windows
that pass suggested, and ask again for a precise answer.

Nothing here fabricates a result. A provider abstention, a parse failure, or
a coarse pass that finds no plausible window all converge on the same
outcome: no :class:`~app.analysis.base_detector.Event`, and the caller falls
back to the existing assisted/manual workflow - the "never confidently
wrong" contract this whole application is built under.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from ...errors import ConfigurationError
from ...video.metadata import VideoInfo
from ...video.reader import VideoReader, encode_jpeg
from ..base_detector import Event, EventStatus
from . import contact_sheet
from .pricing import PRICING_TABLE_VERSION, ModelPricing, estimate_cost_usd
from .prompts import (
    PROMPT_END_CASCADE_COARSE_V1,
    PROMPT_END_CASCADE_COARSE_V1_ID,
    PROMPT_END_CASCADE_REFINE_V1,
    PROMPT_END_CASCADE_REFINE_V1_ID,
    PROMPT_END_COARSE_V2,
    PROMPT_END_COARSE_V2_ID,
    PROMPT_START_REFINE_V1,
    PROMPT_START_REFINE_V1_ID,
)
from .provider import (
    ProviderRequest,
    RawProviderResponse,
    TimedFrame,
    TimingProvider,
    parse_raw_response,
)
from .schema import TimingStatus, TimingVerdict

# One outcome's worth of intermediate pipeline decisions - see
# PipelineOutcome.derived's own docstring for what each key means. A type
# alias only to avoid repeating this union at every one of its several call
# sites (PipelineOutcome.derived, _abstain_outcome, run_llm_timing's own
# local variable).
DerivedValue = float | list[float] | bool | str | None


@dataclass(frozen=True)
class PipelineConfig:
    """Tunables for the coarse-to-fine scan. All are in seconds unless noted.

    Attributes:
        coarse_step_s: spacing between frames in the coarse pass.
        fine_margin_s: how far past the coarse pass's start/end estimate the
            fine pass extends, each direction, to make sure the true
            transition is inside the fine window even if the coarse estimate
            was off.
        fine_max_span_s: a fine window wider than this is refused rather
            than silently sending an enormous frame batch - a coarse pass
            answer that is wildly uncertain should abstain, not balloon the
            next request.
        min_confidence: verdicts below this confidence become ABSTAIN (see
            ``provider.parse_raw_response``). Applies to both passes.
        review_confidence: a CONFIRMED verdict at or above ``min_confidence``
            but below this becomes an :class:`Event` with
            ``EventStatus.REVIEW`` instead of ``CONFIRMED`` - plausible, but
            a human should look, per the same vocabulary the classical
            detectors already use.
        max_uncertainty_s: cap on a CONFIRMED verdict's
            ``start_uncertainty_s``/``end_uncertainty_s``. A verdict can be
            internally consistent (``schema.TimingVerdict``'s own
            invariants) and still be too unsure of itself to trust as
            CONFIRMED - a provider that says "start_s=4.0 +/- 12s" has not
            actually located the boundary. Exceeding this cap converges on
            ABSTAIN (reason code "uncertainty_exceeds_cap"), same as every
            other grounding failure - see ``_validate_grounding``. Default
            is deliberately tight relative to the ±0.75s acceptance target
            (gate 1) - an uncertainty near that bound is not "probably
            fine", it is "probably outside the gate" - and is itself
            uncalibrated until the blinded real-clip set (gate 3) says
            otherwise.
        target_tolerance_s: the accuracy this pipeline is trying to hit
            (matches the supervisor's own gate-1 bound, ±0.75s). Used to
            derive how sparse the fine pass's frame sampling is allowed to
            get under the request-size budget below - see
            ``_min_frames_for_window`` - not to change the grounding checks
            themselves.
        max_frame_dimension_px: every extracted frame is downscaled (never
            upscaled) so its longer side is at most this many pixels,
            before JPEG encoding - deterministic, applied identically
            regardless of provider.
        jpeg_quality: passed straight to ``video.reader.encode_jpeg``.
        max_request_bytes: an explicit ceiling on one request's estimated
            serialized size (image bytes, base64-inflated, plus the prompt
            text) - see ``_estimated_request_bytes``. Default leaves
            headroom under Gemini's ~20MB inline-request limit; set this to
            match whichever provider is actually wired in. Frames are
            thinned deterministically (``_fit_frames_to_budget``) to fit,
            never silently below the precision floor
            ``_min_frames_for_window`` derives from ``target_tolerance_s`` -
            a request that still can't fit at that floor aborts with
            ``request_too_large`` rather than sending fewer frames than the
            gate needs, or a request the provider would reject anyway.
        end_validation_pre_s: how much clip, immediately before the
            end-coarse pass's nominated candidate, gets included in the
            single dense trend-validation window - see
            ``_build_validation_frames`` and ``run_llm_timing``'s
            trend-validation follow-up request below. Widened (from an
            original "~1s") after three consecutive real gate-1 reruns put
            the sparse end-coarse candidate anywhere from ~1s to ~4s away
            from the true break: a window built tightly around a
            *possibly-wrong* candidate can structurally exclude the truth
            regardless of what the candidate is (supervisor-directed, see
            diagnostics/llm_spike/DESIGN.md).
        end_validation_post_s: how much clip, immediately after the
            end-coarse pass's nominated candidate, gets included in the same
            window - sizing only. This is deliberately decoupled from the
            *required* future-evidence rule below (``end_validation_min_future_s``)
            - a candidate near the clip's own end still gets a window
            clipped at the clip boundary, and it is the refined onset's own
            trailing evidence that is actually checked, not this nominal
            sizing value.
        end_validation_max_span_s: hard ceiling on
            ``end_validation_pre_s + end_validation_post_s`` (enforced in
            ``validate()``, not at request time) - this window is still one
            coherent dense request, never an open-ended or multi-candidate
            search, however wide the observed coarse error range gets.
        end_validation_min_future_s: how much *actually submitted* evidence
            must follow the timestamp the dense validation pass reports
            (which may be a refinement, not the original candidate - see
            ``PROMPT_END_VALIDATE_V2``) before that onset is trusted, at all
            within the same window. Checked against the refined onset, not
            the nominal ``end_validation_post_s`` sizing above, so clipping
            the window at the clip's end still allows a refined onset with
            enough room after it. Supervisor-specified floor is "~2s"; this
            is that default.
        end_validation_min_trend_checkpoints: the fewest
            ``trend_checkpoint_timestamps_s`` entries a CONFIRMED
            end-validate verdict may cite - see
            ``_validate_trend_checkpoints``. A second real audited run
            confirmed a "sustained trend" from a single cluster of
            adjacent native-fps frames; at least two independently-spaced
            checkpoints are required to demonstrate persistence rather than
            one localized observation. Supervisor-specified floor is "at
            least two".
        end_validation_min_onset_gap_s: how far past the reported onset the
            *first* trend checkpoint must sit. Default 0.75s - the same
            floor as ``target_tolerance_s`` (gate 1's own accuracy bound) -
            chosen so this can never be satisfied by native-fps jitter
            (tens of milliseconds) while still being achievable well inside
            ``end_validation_post_s``/``end_validation_min_future_s``.
        end_validation_min_checkpoint_spacing_s: how far apart *consecutive*
            trend checkpoints must sit. Same default and rationale as
            ``end_validation_min_onset_gap_s`` - three checkpoints only
            ~33ms apart (a real audited run's exact shape) is exactly what
            this rejects.
    """

    coarse_step_s: float = 0.5
    fine_margin_s: float = 1.5
    fine_max_span_s: float = 6.0
    min_confidence: float = 0.5
    review_confidence: float = 0.75
    max_uncertainty_s: float = 0.5
    target_tolerance_s: float = 0.75
    max_frame_dimension_px: int = 768
    jpeg_quality: int = 80
    max_request_bytes: int = 18_000_000
    end_validation_pre_s: float = 4.0
    end_validation_post_s: float = 6.0
    end_validation_max_span_s: float = 10.0
    end_validation_min_future_s: float = 2.0
    end_validation_min_trend_checkpoints: int = 2
    end_validation_min_onset_gap_s: float = 0.75
    end_validation_min_checkpoint_spacing_s: float = 0.75

    def validate(self) -> None:
        if self.coarse_step_s <= 0:
            raise ConfigurationError("coarse_step_s must be positive.")
        if self.fine_margin_s < 0:
            raise ConfigurationError("fine_margin_s must not be negative.")
        if self.fine_max_span_s <= 0:
            raise ConfigurationError("fine_max_span_s must be positive.")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ConfigurationError("min_confidence must be between 0 and 1.")
        if not self.min_confidence <= self.review_confidence <= 1.0:
            raise ConfigurationError("review_confidence must be between min_confidence and 1.")
        if self.max_uncertainty_s <= 0:
            raise ConfigurationError("max_uncertainty_s must be positive.")
        if self.target_tolerance_s <= 0:
            raise ConfigurationError("target_tolerance_s must be positive.")
        if self.max_frame_dimension_px <= 0:
            raise ConfigurationError("max_frame_dimension_px must be positive.")
        if not 1 <= self.jpeg_quality <= 100:
            raise ConfigurationError("jpeg_quality must be between 1 and 100.")
        if self.end_validation_pre_s <= 0:
            raise ConfigurationError("end_validation_pre_s must be positive.")
        if self.end_validation_post_s <= 0:
            raise ConfigurationError("end_validation_post_s must be positive.")
        if self.end_validation_max_span_s <= 0:
            raise ConfigurationError("end_validation_max_span_s must be positive.")
        if self.end_validation_pre_s + self.end_validation_post_s > (
            self.end_validation_max_span_s + _BOUNDS_EPSILON_S
        ):
            raise ConfigurationError(
                "end_validation_pre_s + end_validation_post_s must not exceed "
                "end_validation_max_span_s - this window is a single coherent "
                "request, not an open-ended search."
            )
        if self.end_validation_min_future_s <= 0:
            raise ConfigurationError("end_validation_min_future_s must be positive.")
        if self.end_validation_min_future_s > self.end_validation_post_s + _BOUNDS_EPSILON_S:
            raise ConfigurationError(
                "end_validation_min_future_s must not exceed end_validation_post_s - "
                "a refined onset could never have enough trailing evidence otherwise."
            )
        if self.max_request_bytes <= 0:
            raise ConfigurationError("max_request_bytes must be positive.")
        if self.end_validation_min_trend_checkpoints < 2:
            raise ConfigurationError(
                "end_validation_min_trend_checkpoints must be at least 2 - a single "
                "checkpoint cannot demonstrate a sustained trend, only one observation."
            )
        if self.end_validation_min_onset_gap_s <= 0:
            raise ConfigurationError("end_validation_min_onset_gap_s must be positive.")
        if self.end_validation_min_checkpoint_spacing_s <= 0:
            raise ConfigurationError("end_validation_min_checkpoint_spacing_s must be positive.")


@dataclass
class PipelineOutcome:
    """Everything the caller and the audit log need from one run.

    ``event`` is ``None`` whenever ``verdict.status`` is ABSTAIN - the
    absence of an event *is* the abstention; callers must not infer an
    abstention from field values, only from this being unset.
    """

    verdict: TimingVerdict
    event: Event | None
    coarse_response: RawProviderResponse
    fine_response: RawProviderResponse | None
    end_coarse_response: RawProviderResponse | None = None
    """The single whole-clip (post-start), sparsely-sampled call that
    nominates one end candidate - see ``PROMPT_END_COARSE_V2``. ``None``
    for a run that never reached this phase (an abstain before it). This
    replaced a chronological multi-window scan (Codex/supervisor
    experiment, see ``diagnostics/llm_spike/DESIGN.md``): a real gate-1
    rerun showed that decomposing the end search into many narrow,
    isolated windows lost the temporal context needed to judge a
    *sustained* trend and produced a false rejection of the true break,
    while one coherent request with the same trend contract succeeded.
    Kept separate from ``fine_response`` so every pass's cost/latency/
    retries is individually inspectable, not collapsed into one number."""
    end_validation_response: RawProviderResponse | None = None
    """The DECISIVE contact-sheet cascade call for ``end_coarse_response``'s
    candidate - the 7-panel refine sheet's own response when one ran,
    otherwise the 9-panel coarse sheet's (see ``_run_end_validation_pass``).
    ``None`` for a run that never reached this phase. A rejected or
    unvalidatable candidate now converges directly on ABSTAIN (never
    confidently wrong) rather than searching for another candidate - see
    ``run_llm_timing``."""
    end_validation_coarse_cascade_response: RawProviderResponse | None = None
    """The coarse (9-panel) contact sheet's own response, kept separately
    inspectable ONLY when a refine sheet also ran and superseded it (i.e.
    ``end_validation_response`` above is the refine call's response, not
    this one) - so both calls' cost/latency/tokens stay individually
    auditable rather than the coarse call's own spend silently vanishing
    once a refine sheet supersedes its verdict. ``None`` whenever no
    refine sheet ran (the coarse sheet's response is already
    ``end_validation_response`` in that case)."""
    end_validation_conflict_response: RawProviderResponse | None = None
    """A second, differently-anchored cascade around the *coarse* pass's
    own independent end estimate - sent only when
    ``derived["candidate_conflict"]`` is true (the coarse and end-coarse
    passes disagreed badly enough that neither's own cascade window would
    contain the other's candidate). ``None`` whenever no conflict was
    detected, which is the common case - this is a bounded, at-most-
    one-extra-candidate check, never an open-ended search (supervisor-
    directed, see diagnostics/llm_spike/DESIGN.md)."""
    end_validation_conflict_coarse_cascade_response: RawProviderResponse | None = None
    """The conflict-path counterpart of
    ``end_validation_coarse_cascade_response`` - the coarse sheet built
    around the *coarse* pass's own independent end estimate, kept
    separately inspectable only when that anchor's own refine sheet also
    ran and superseded it."""
    pass_verdicts: dict[str, TimingVerdict] = field(default_factory=dict)
    """Every pass's own post-grounding verdict, keyed by pass name
    ("coarse", "fine", "end_coarse", "end_validate") - populated
    incrementally as each pass in ``run_llm_timing`` completes, so a run
    that abstains partway through still records what every pass *before*
    the abstain actually said. Distinct from ``verdict`` (the run's single
    final answer) and from the ``*_response`` fields above (the unparsed
    provider payload): this is each pass's own candidate/refined timestamp,
    evidence, reason codes, and raw_notes, after grounding - exactly the
    per-pass detail a real-clip failure needs to be diagnosed without
    another paid rerun (supervisor-directed, see
    diagnostics/llm_spike/DESIGN.md). ``TimingVerdict.raw_notes`` is already
    sanitized/bounded (see ``redaction.py``); nothing here ever carries
    image bytes or credentials."""
    pass_frames: dict[str, tuple[float, ...]] = field(default_factory=dict)
    """Every pass's *actual* submitted frame timestamps, keyed by the same
    pass names as ``pass_verdicts`` - captured at the point each
    ``ProviderRequest`` is built, so this reflects frames *after* any
    budget-driven thinning (``_fit_frames_to_budget``), never the
    theoretical extraction plan. Supervisor-directed auditability
    requirement: a caller must be able to show exactly what was sent to a
    provider, not reconstruct it later from config defaults, which could
    silently misrepresent a thinned/sparse request as a continuous range
    (see diagnostics/llm_spike/DESIGN.md). Only timestamps - never image
    bytes."""
    derived: dict[str, DerivedValue] = field(default_factory=dict)
    """The pipeline's own intermediate decisions, recorded as they are made
    - *not* only when the run ends CONFIRMED. Keys, populated incrementally
    (absent until the step that produces them runs): ``locked_start_s``/
    ``locked_start_uncertainty_s`` (the fine pass's confirmed start),
    ``start_evidence_s`` (the grounded evidence frame behind it),
    ``end_coarse_candidate_s`` (the sparse end-coarse pass's nominated
    break), ``end_validation_window_s`` (the ``[lo, hi]`` dense window built
    around that candidate - recorded the moment it is *decided*, even if
    the request that window implies never fits the byte budget or the
    validation pass rejects it), and ``end_evidence_s`` (the grounded
    evidence frame behind the validated end). Supervisor-directed audit
    requirement: a wrong or abstained result must still let a reader
    reconstruct the chain "sparse frames -> candidate -> validation window
    -> final result" from this one outcome, not only a CONFIRMED one -
    ``Event.details`` (CONFIRMED-only, pre-existing) duplicates a few of
    these keys for its own established consumers; this field is the one
    that is populated regardless of how the run ends."""

    def _all_responses(self) -> list[RawProviderResponse]:
        """Every provider call this outcome actually made, including a
        coarse cascade sheet superseded by its own refine sheet - so
        cost/latency/token accounting never silently drops a call that was
        genuinely sent (see ``end_validation_coarse_cascade_response``'s
        own docstring)."""
        return [
            response
            for response in (
                self.coarse_response,
                self.fine_response,
                self.end_coarse_response,
                self.end_validation_response,
                self.end_validation_coarse_cascade_response,
                self.end_validation_conflict_response,
                self.end_validation_conflict_coarse_cascade_response,
            )
            if response is not None
        ]

    @property
    def total_retries(self) -> int:
        return sum(response.retries for response in self._all_responses())

    @property
    def total_latency_s(self) -> float:
        return sum(response.latency_s for response in self._all_responses())

    @property
    def total_tokens(self) -> int | None:
        """``None`` (unknown) unless every pass that ran reported usage."""
        counts = []
        for response in self._all_responses():
            if response.prompt_tokens is None or response.completion_tokens is None:
                return None
            counts.append(response.prompt_tokens + response.completion_tokens)
        return sum(counts) if counts else None

    def estimated_cost_usd(self, *, table: dict[str, ModelPricing] | None = None) -> float | None:
        """Sum of each pass's estimated cost, or ``None`` if any is unknown.

        Deliberately not "sum the known ones and ignore the rest" - a
        partial total would understate cost silently. See ``pricing.py``.
        """
        total = 0.0
        for response in self._all_responses():
            cost = estimate_cost_usd(
                response.model_id,
                response.prompt_tokens,
                response.completion_tokens,
                table=table,
            )
            if cost is None:
                return None
            total += cost
        return total

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict.to_dict(),
            "event": self.event.to_dict() if self.event is not None else None,
            "coarse_model_id": self.coarse_response.model_id,
            "coarse_latency_s": self.coarse_response.latency_s,
            "coarse_retries": self.coarse_response.retries,
            "coarse_prompt_tokens": self.coarse_response.prompt_tokens,
            "coarse_completion_tokens": self.coarse_response.completion_tokens,
            "fine_model_id": self.fine_response.model_id if self.fine_response else None,
            "fine_latency_s": self.fine_response.latency_s if self.fine_response else None,
            "fine_retries": self.fine_response.retries if self.fine_response else None,
            "fine_prompt_tokens": self.fine_response.prompt_tokens if self.fine_response else None,
            "fine_completion_tokens": (
                self.fine_response.completion_tokens if self.fine_response else None
            ),
            "end_coarse_model_id": (
                self.end_coarse_response.model_id if self.end_coarse_response else None
            ),
            "end_coarse_latency_s": (
                self.end_coarse_response.latency_s if self.end_coarse_response else None
            ),
            "end_coarse_retries": (
                self.end_coarse_response.retries if self.end_coarse_response else None
            ),
            "end_validation_model_id": (
                self.end_validation_response.model_id if self.end_validation_response else None
            ),
            "end_validation_latency_s": (
                self.end_validation_response.latency_s if self.end_validation_response else None
            ),
            "end_validation_retries": (
                self.end_validation_response.retries if self.end_validation_response else None
            ),
            "end_validation_coarse_cascade_model_id": (
                self.end_validation_coarse_cascade_response.model_id
                if self.end_validation_coarse_cascade_response
                else None
            ),
            "end_validation_coarse_cascade_latency_s": (
                self.end_validation_coarse_cascade_response.latency_s
                if self.end_validation_coarse_cascade_response
                else None
            ),
            "end_validation_coarse_cascade_retries": (
                self.end_validation_coarse_cascade_response.retries
                if self.end_validation_coarse_cascade_response
                else None
            ),
            "end_validation_conflict_model_id": (
                self.end_validation_conflict_response.model_id
                if self.end_validation_conflict_response
                else None
            ),
            "end_validation_conflict_latency_s": (
                self.end_validation_conflict_response.latency_s
                if self.end_validation_conflict_response
                else None
            ),
            "end_validation_conflict_retries": (
                self.end_validation_conflict_response.retries
                if self.end_validation_conflict_response
                else None
            ),
            "end_validation_conflict_coarse_cascade_model_id": (
                self.end_validation_conflict_coarse_cascade_response.model_id
                if self.end_validation_conflict_coarse_cascade_response
                else None
            ),
            "end_validation_conflict_coarse_cascade_latency_s": (
                self.end_validation_conflict_coarse_cascade_response.latency_s
                if self.end_validation_conflict_coarse_cascade_response
                else None
            ),
            "end_validation_conflict_coarse_cascade_retries": (
                self.end_validation_conflict_coarse_cascade_response.retries
                if self.end_validation_conflict_coarse_cascade_response
                else None
            ),
            "total_latency_s": self.total_latency_s,
            "total_retries": self.total_retries,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": self.estimated_cost_usd(),
            "pricing_table_version": PRICING_TABLE_VERSION,
            "pass_verdicts": {name: v.to_dict() for name, v in self.pass_verdicts.items()},
            "pass_frames": {name: list(times) for name, times in self.pass_frames.items()},
            "derived": dict(self.derived),
        }


def _resize_for_encoding(image: np.ndarray, max_dimension_px: int) -> np.ndarray:
    """Downscale (never upscale) so the longer side is at most this many
    pixels - deterministic, applied before every frame is encoded, so a
    request's size never depends on hidden per-provider behaviour."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dimension_px:
        return image
    scale = max_dimension_px / longest
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def _extract_frames(
    reader: VideoReader,
    timestamps_s: list[float],
    *,
    max_dimension_px: int,
    jpeg_quality: int,
) -> list[TimedFrame]:
    frames = []
    for ts in timestamps_s:
        image = _resize_for_encoding(reader.frame_at(ts), max_dimension_px)
        frames.append(
            TimedFrame(timestamp_s=ts, image_bytes=encode_jpeg(image, quality=jpeg_quality))
        )
    return frames


# Roughly what base64 costs on top of raw bytes (RFC 4648: 4 output bytes
# per 3 input bytes) - used only to estimate a request's serialized size
# before sending it, not to actually encode anything.
_BASE64_INFLATION = 4 / 3


def _estimated_request_bytes(frames: list[TimedFrame], prompt_text: str) -> int:
    image_bytes = sum(len(frame.image_bytes) for frame in frames)
    return int(image_bytes * _BASE64_INFLATION) + len(prompt_text.encode("utf-8"))


def _min_frames_for_window(span_s: float, target_tolerance_s: float) -> int:
    """The sparsest sampling still "sufficient for the ±0.75s gate": at
    least one sample per half-tolerance, so consecutive samples are never
    farther apart than half of what the gate allows as an error - a
    deliberate safety margin, not sampling at exactly the tolerance itself.
    """
    if span_s <= 0:
        return 2
    step = max(target_tolerance_s / 2, 1e-6)
    return max(3, math.ceil(span_s / step) + 1)


def _min_coarse_frames(duration_s: float, fine_margin_s: float) -> int:
    """The sparsest coarse sampling that still guarantees *some* coarse
    sample lands within ``fine_margin_s`` of the true transition, wherever
    it actually is - so the fine window later built around the coarse
    estimate (+/- ``fine_margin_s``) is still guaranteed to cover the true
    boundary, even in the worst case.

    A flat floor (the previous ``min_frames=4``) can't make that guarantee
    for a long clip: with only 4 samples spread across a 60s video,
    consecutive coarse samples can be ~20s apart - far past any reasonable
    ``fine_margin_s`` - so the coarse estimate could be nowhere near a
    sample, and the resulting fine window could miss the true boundary
    entirely (Codex re-review round 3, finding 1).
    """
    if fine_margin_s <= 0:
        return 4
    return max(4, math.ceil(duration_s / fine_margin_s) + 1)


def _dense_timestamps(lo: float, hi: float, step_s: float) -> list[float]:
    """Every native-fps timestamp in ``[lo, hi]``, deterministically."""
    return sorted(
        {
            round(lo + i * step_s, 6)
            for i in range(int((hi - lo) / step_s) + 2)
            if lo + i * step_s <= hi
        }
    )


def _merge_frames_sorted(*groups: list[TimedFrame]) -> list[TimedFrame]:
    """Combine independently-thinned window frame lists into the one batch
    actually sent to the provider - sorted, and de-duplicated by timestamp
    in case two windows happen to overlap (a short true efflux time can put
    the start and end fine windows close enough to share native-fps
    timestamps)."""
    by_timestamp: dict[float, TimedFrame] = {}
    for group in groups:
        for frame in group:
            by_timestamp.setdefault(frame.timestamp_s, frame)
    return [by_timestamp[ts] for ts in sorted(by_timestamp)]


def _fit_fine_windows_to_budget(
    start_frames: list[TimedFrame],
    end_frames: list[TimedFrame],
    *,
    prompt_text: str,
    max_request_bytes: int,
    start_min_frames: int,
    end_min_frames: int,
) -> tuple[list[TimedFrame] | None, list[TimedFrame] | None]:
    """Thin the start and end fine-pass windows independently, each against
    its own half of the shared request byte budget.

    Independent, not "merge both windows then thin the combined list" (the
    previous approach): merging first lets whichever window happens to have
    larger or more numerous frames crowd out the other window's density
    during thinning, and - separately - is what made
    ``_effective_tolerance_s`` blow up to the empty space *between* the two
    windows rather than either window's own achieved sampling density (see
    ``_validate_grounding``/``_GroundingRegion``). Splitting the budget in
    half up front keeps each boundary's achieved precision fully
    determined by its own frames, regardless of the other window's size
    (Codex re-review round 3, finding 1).

    Each element of the returned pair is ``None`` independently if that
    window's floor still doesn't fit in its half of the budget - the caller
    decides how to report a fine-pass abstain from that, per-window.
    """
    window_budget = max_request_bytes // 2
    start_fit = _fit_frames_to_budget(
        start_frames,
        prompt_text=prompt_text,
        max_request_bytes=window_budget,
        min_frames=start_min_frames,
    )
    end_fit = _fit_frames_to_budget(
        end_frames,
        prompt_text=prompt_text,
        max_request_bytes=window_budget,
        min_frames=end_min_frames,
    )
    return start_fit, end_fit


def _fit_frames_to_budget(
    frames: list[TimedFrame], *, prompt_text: str, max_request_bytes: int, min_frames: int
) -> list[TimedFrame] | None:
    """Deterministically thin an already-extracted frame list until the
    estimated serialized request fits the budget, never dropping below
    ``min_frames``.

    Returns ``None`` - never a silent partial selection below that floor -
    if even the sparsest allowed selection still doesn't fit; the caller
    turns that into an abstain rather than sending an undersized or
    oversized request.
    """
    candidate = frames
    while _estimated_request_bytes(candidate, prompt_text) > max_request_bytes:
        if len(candidate) <= min_frames:
            return None
        target_count = max(min_frames, (len(candidate) + 1) // 2)
        if target_count >= len(candidate):
            target_count = len(candidate) - 1
        step = len(candidate) / target_count
        indices = sorted({min(len(candidate) - 1, int(i * step)) for i in range(target_count)})
        if len(indices) < 2:
            return None
        candidate = [candidate[i] for i in indices]
    return candidate


def _effective_tolerance_s(timestamps_s: list[float], fallback_step_s: float) -> float:
    """Grounding tolerance derived from the density actually achieved after
    budget-fitting, not the nominal (pre-thinning) sampling step - if the
    frame set had to be thinned, "near a submitted frame" and "near the
    claimed boundary" must widen to match, or a legitimately-cited frame
    would be rejected purely because thinning made it farther from its
    neighbours than the original dense step assumed.
    """
    if len(timestamps_s) < 2:
        return 2.0 * fallback_step_s
    ordered = sorted(timestamps_s)
    largest_gap = max(b - a for a, b in zip(ordered, ordered[1:], strict=False))
    return max(2.0 * fallback_step_s, largest_gap)


def _end_scan_windows(
    scan_from_s: float, duration_s: float, window_width_s: float, step_s: float
) -> list[tuple[float, float]]:
    """Bounded, overlapping windows scanning chronologically forward from
    ``scan_from_s`` to ``duration_s``.

    Overlap (``step_s < window_width_s``) so a break sitting near one
    window's boundary still falls fully inside the *next* window too,
    rather than being split across a boundary and visible nowhere in full.
    """
    windows = []
    lo = scan_from_s
    while lo < duration_s:
        hi = min(duration_s, lo + window_width_s)
        windows.append((lo, hi))
        if hi >= duration_s:
            break
        lo += step_s
    return windows


def _build_validation_frames(
    reader: VideoReader,
    *,
    candidate_ts: float,
    validation_lo: float,
    validation_hi: float,
    fine_step_s: float,
    max_dimension_px: int,
    jpeg_quality: int,
    prompt_text: str,
    max_request_bytes: int,
    target_tolerance_s: float,
) -> list[TimedFrame] | None:
    """Dense frames spanning one end-scan candidate's trend-validation
    window - baseline before it, evidence after it - with the candidate's
    own frame always present and marked (see ``TimedFrame.is_candidate``),
    even if budget-fitting would otherwise have thinned it away: a
    validation request with no labelled CANDIDATE frame at all is useless,
    not merely imprecise, since the prompt has nothing else to point the
    model at.

    Returns ``None`` - same contract as ``_fit_frames_to_budget`` - if even
    the sparsest allowed selection still doesn't fit the budget.
    """
    dense_times = sorted(
        set(_dense_timestamps(validation_lo, validation_hi, fine_step_s)) | {candidate_ts}
    )
    frames_dense = _extract_frames(
        reader, dense_times, max_dimension_px=max_dimension_px, jpeg_quality=jpeg_quality
    )
    min_frames = _min_frames_for_window(validation_hi - validation_lo, target_tolerance_s)
    frames = _fit_frames_to_budget(
        frames_dense,
        prompt_text=prompt_text,
        max_request_bytes=max_request_bytes,
        min_frames=min_frames,
    )
    if frames is None:
        return None
    if not any(frame.timestamp_s == candidate_ts for frame in frames):
        candidate_frame = next(f for f in frames_dense if f.timestamp_s == candidate_ts)
        frames = _merge_frames_sorted(frames, [candidate_frame])
    return [
        replace(frame, is_candidate=True) if frame.timestamp_s == candidate_ts else frame
        for frame in frames
    ]


def _coarse_timestamps(duration_s: float, step_s: float) -> list[float]:
    if duration_s <= 0:
        raise ConfigurationError(
            "Cannot run the coarse pass: the video has no known positive duration."
        )
    count = max(2, int(duration_s / step_s) + 1)
    return [min(i * step_s, duration_s) for i in range(count)]


# Pure float-rounding slack for "is this value inside the window we sized
# for it" - not a tolerance for the model being approximately right.
_BOUNDS_EPSILON_S = 1e-6


def _grounding_abstain(verdict: TimingVerdict, reason_code: str, detail: str) -> TimingVerdict:
    return TimingVerdict.abstain(
        reason_codes=(reason_code,),
        model_id=verdict.model_id,
        prompt_version=verdict.prompt_version,
        confidence=verdict.confidence,
        raw_notes=detail,
    )


def _nearest_grounded_evidence_ts(
    evidence_ts: tuple[float, ...],
    submitted_ts: tuple[float, ...],
    boundary_ts: float,
) -> float | None:
    """The audit-safe timestamp for a boundary's "evidence image": among a
    grounded verdict's own cited ``evidence_frame_timestamps_s`` (already
    validated by ``_validate_grounding`` to correspond to a frame actually
    submitted to the provider), the one nearest ``boundary_ts`` - then
    snapped to its own nearest entry in ``submitted_ts`` so the result is
    always an exact frame timestamp this pass actually sent, never an
    approximate or interpolated value a browser-supplied timestamp could
    otherwise be used to spoof. Returns ``None`` - never a fabricated
    guess - when there is no cited evidence or nothing was submitted for
    this pass at all (supervisor-directed auditability requirement, see
    diagnostics/llm_spike/DESIGN.md: "do not fake an image; show an
    explicit no-evidence state")."""
    if not evidence_ts or not submitted_ts:
        return None
    closest_evidence = min(evidence_ts, key=lambda ts: abs(ts - boundary_ts))
    return min(submitted_ts, key=lambda ts: abs(ts - closest_evidence))


@dataclass(frozen=True)
class _GroundingRegion:
    """One claimed boundary's own submitted-frame context.

    ``bounds`` is the window that boundary's claim must fall inside;
    ``submitted_timestamps_s``/``time_tolerance_s`` describe *only the
    frames actually sent for this boundary* and the tolerance derived from
    their own achieved density.

    Kept separate per boundary (rather than one shared set/tolerance across
    both) because the start and end fine windows can be many seconds apart:
    merging them first (as an earlier version of this pipeline did) lets
    the empty space *between* the two windows dominate
    ``_effective_tolerance_s``, which in turn makes "evidence corresponds
    to a frame actually sent" and "evidence is near the claimed boundary"
    far too permissive - a timestamp that was never sent to the provider at
    all (it falls in the dead zone between the windows) can end up
    "matching" a real frame purely because the inter-window gap inflated
    the tolerance, and can even "ground" *both* the start and end claims at
    once. Keeping each boundary's region separate closes that gap (Codex
    re-review round 3, finding 1) and also prevents evidence legitimately
    drawn from one window from grounding the other window's claim.
    """

    bounds: tuple[float, float]
    submitted_timestamps_s: tuple[float, ...]
    time_tolerance_s: float


def _validate_grounding(
    verdict: TimingVerdict,
    *,
    start_region: _GroundingRegion,
    end_region: _GroundingRegion,
    max_uncertainty_s: float,
) -> TimingVerdict:
    """Re-check a parsed CONFIRMED verdict against the request it answers.

    ``provider.parse_raw_response`` only validates a verdict's own internal
    shape - it has no idea what video or which frames were actually sent, so
    a well-formed, internally consistent, *fabricated* answer (a timestamp
    nowhere near the submitted frames, evidence that doesn't correspond to
    anything actually sent) parses cleanly. This is the check that catches
    that: every failure converges on ABSTAIN, same discipline as
    ``parse_raw_response``, just with request-level context it doesn't have.

    ``start_region``/``end_region`` are independent - see
    :class:`_GroundingRegion`. For the coarse pass (one shared frame batch,
    not yet split into a start/end window), pass the same region for both;
    its distinct-regions behaviour only matters once the fine pass has
    separate windows.

    A no-op for an already-ABSTAIN verdict (nothing to ground).
    """
    if verdict.status is not TimingStatus.CONFIRMED:
        return verdict
    assert verdict.start_s is not None and verdict.end_s is not None

    start_lo, start_hi = start_region.bounds
    end_lo, end_hi = end_region.bounds

    if not (start_lo - _BOUNDS_EPSILON_S <= verdict.start_s <= start_hi + _BOUNDS_EPSILON_S):
        return _grounding_abstain(
            verdict,
            "out_of_bounds",
            f"start_s={verdict.start_s} outside the submitted window "
            f"[{start_lo:.3f}, {start_hi:.3f}]",
        )
    if not (end_lo - _BOUNDS_EPSILON_S <= verdict.end_s <= end_hi + _BOUNDS_EPSILON_S):
        return _grounding_abstain(
            verdict,
            "out_of_bounds",
            f"end_s={verdict.end_s} outside the submitted window [{end_lo:.3f}, {end_hi:.3f}]",
        )

    if (
        verdict.start_uncertainty_s > max_uncertainty_s
        or verdict.end_uncertainty_s > max_uncertainty_s
    ):
        return _grounding_abstain(
            verdict,
            "uncertainty_exceeds_cap",
            f"start_uncertainty_s={verdict.start_uncertainty_s} "
            f"end_uncertainty_s={verdict.end_uncertainty_s} exceeds cap {max_uncertainty_s}",
        )

    if not verdict.evidence_frame_timestamps_s:
        return _grounding_abstain(
            verdict, "ungrounded_evidence", "no evidence_frame_timestamps_s given"
        )

    def _matches_region(ts: float, region: _GroundingRegion) -> bool:
        return any(
            abs(ts - sent) <= region.time_tolerance_s for sent in region.submitted_timestamps_s
        )

    ungrounded = [
        ts
        for ts in verdict.evidence_frame_timestamps_s
        if not _matches_region(ts, start_region) and not _matches_region(ts, end_region)
    ]
    if ungrounded:
        return _grounding_abstain(
            verdict,
            "ungrounded_evidence",
            f"evidence timestamps {ungrounded} do not correspond to any frame "
            f"actually submitted in this request",
        )

    # Evidence grounds a claim only if it both (a) is within that claim's
    # own region's tolerance of the claim, and (b) actually matches a frame
    # submitted *for that region* - so evidence legitimately drawn from the
    # end window (however close it happens to land, numerically, to
    # start_s) can never ground the start claim, and vice versa.
    near_start = any(
        abs(ts - verdict.start_s) <= start_region.time_tolerance_s
        and _matches_region(ts, start_region)
        for ts in verdict.evidence_frame_timestamps_s
    )
    near_end = any(
        abs(ts - verdict.end_s) <= end_region.time_tolerance_s and _matches_region(ts, end_region)
        for ts in verdict.evidence_frame_timestamps_s
    )
    if not (near_start and near_end):
        missing = "start_s" if not near_start else "end_s"
        claim = verdict.start_s if not near_start else verdict.end_s
        return _grounding_abstain(
            verdict,
            "evidence_far_from_claim",
            f"evidence_frame_timestamps_s={list(verdict.evidence_frame_timestamps_s)} has "
            f"nothing within its own window's tolerance of {missing}={claim}",
        )

    return verdict


def _abstain_outcome(
    verdict: TimingVerdict,
    coarse_response: RawProviderResponse,
    *,
    pass_verdicts: dict[str, TimingVerdict] | None = None,
    pass_frames: dict[str, tuple[float, ...]] | None = None,
    derived: dict[str, DerivedValue] | None = None,
) -> PipelineOutcome:
    return PipelineOutcome(
        verdict=verdict,
        event=None,
        coarse_response=coarse_response,
        fine_response=None,
        pass_verdicts=dict(pass_verdicts) if pass_verdicts else {},
        pass_frames=dict(pass_frames) if pass_frames else {},
        derived=dict(derived) if derived else {},
    )


def _unsent_response(reason: str) -> RawProviderResponse:
    """A placeholder for a pass that was never sent to the provider at all.

    Used only when frame-budget fitting fails before any network call was
    attempted (see ``request_too_large`` below) - ``PipelineOutcome``
    always needs a ``RawProviderResponse`` to report against, even when the
    honest answer is "we refused to send this."
    """
    return RawProviderResponse(model_id="", raw_text="", latency_s=0.0, error=reason)


def _oversized_abstain(prompt_version: str, pass_name: str, detail: str) -> TimingVerdict:
    return TimingVerdict.abstain(
        reason_codes=("request_too_large",),
        model_id="",
        prompt_version=prompt_version,
        raw_notes=f"{pass_name} pass: {detail}; never sent to the provider",
    )


# --------------------------------------------------------------------------- #
# End-of-clip validation: candidate-conflict detection and the (bounded, at
# most one extra call) dual-region check it can trigger.
#
# A real audited hands-on run exposed the failure this section exists to
# prevent: the whole-clip coarse pass independently estimated the end near
# 21.0s; the end-coarse pass instead nominated 5.5s; the dense validation
# window built only around 5.5s ([1.5, 11.5]s) never came anywhere near
# 21.0s; and the coarse pass's own estimate was silently discarded rather
# than cross-checked. See diagnostics/llm_spike/DESIGN.md for the full
# root-cause writeup.
# --------------------------------------------------------------------------- #

# A CONFIRMED end_validate verdict citing more evidence timestamps than this
# is not selective - it is "all of it", which satisfies grounding's mere
# presence check without showing which specific checkpoints demonstrate a
# sustained trend (the same real run cited ~300 of ~300 submitted frames).
_MAX_END_VALIDATE_EVIDENCE = 8

# A small, bounded number of widely-spaced checkpoints beyond the dense
# validation window - never a second dense scan - so a long plateau that
# looks locally sustained can still be checked against what the clip
# actually does much later, within the same request.
_MAX_SPARSE_FUTURE_CHECKPOINTS = 6

# Suggested spacing/count for the TREND CHECKPOINT frame labels - a
# labelling aid (see TimedFrame.is_trend_checkpoint), deliberately spaced
# wider than PipelineConfig.end_validation_min_onset_gap_s/
# end_validation_min_checkpoint_spacing_s's own enforced floor so the
# suggested anchors comfortably satisfy it, not just barely.
_TREND_CHECKPOINT_LABEL_SPACING_S = 1.0
_MAX_TREND_CHECKPOINT_LABELS = 4


def _validation_window_for(
    candidate_ts: float, locked_start_s: float, duration_s: float, cfg: PipelineConfig
) -> tuple[float, float]:
    lo = max(locked_start_s, candidate_ts - cfg.end_validation_pre_s)
    hi = min(duration_s, candidate_ts + cfg.end_validation_post_s)
    return lo, hi


def _candidates_conflict(
    a_ts: float, b_ts: float, locked_start_s: float, duration_s: float, cfg: PipelineConfig
) -> bool:
    """True when two independently-produced end candidates disagree badly
    enough that neither's own validation window would even contain the
    other - i.e. validating only one of them can structurally never reveal
    whether the other is (or is closer to) the true break. Symmetric: it
    does not matter which candidate is passed as ``a_ts``/``b_ts``."""
    a_lo, a_hi = _validation_window_for(a_ts, locked_start_s, duration_s, cfg)
    b_lo, b_hi = _validation_window_for(b_ts, locked_start_s, duration_s, cfg)
    a_in_b_window = b_lo - _BOUNDS_EPSILON_S <= a_ts <= b_hi + _BOUNDS_EPSILON_S
    b_in_a_window = a_lo - _BOUNDS_EPSILON_S <= b_ts <= a_hi + _BOUNDS_EPSILON_S
    return not a_in_b_window and not b_in_a_window


def _sparse_future_checkpoints(
    validation_hi: float, duration_s: float, *, max_count: int
) -> list[float]:
    """Up to ``max_count`` evenly-spaced timestamps strictly after
    ``validation_hi``, up to the end of the clip - deliberately sparse
    (never denser than the dense window itself) and bounded (never an
    open-ended or second dense scan). Empty when there is no clip left
    after the dense window, or when ``max_count`` is non-positive."""
    remaining = duration_s - validation_hi
    if remaining <= _BOUNDS_EPSILON_S or max_count <= 0:
        return []
    count = min(max_count, max(1, math.floor(remaining)))
    step = remaining / count
    return [round(validation_hi + step * (i + 1), 6) for i in range(count)]


def _bound_end_validate_evidence(verdict: TimingVerdict) -> TimingVerdict:
    """The evidence-shape check a real audited run's false confirmation
    exposed: a CONFIRMED end_validate verdict cited essentially every frame
    it looked at (~301 of ~301) as "evidence" of a sustained trend - citing
    everything is indistinguishable from citing nothing specific. A small,
    selective evidence list is enforced structurally here; the complementary
    half (that the *cited* checkpoints actually span a baseline and a later,
    continued-shortening point) is asked for in the prompt itself
    (``PROMPT_END_VALIDATE_V4``'s OUTPUT section) rather than enforced here,
    since it cannot be verified from timestamps alone - only the count can.
    Converges on ABSTAIN like every other grounding failure; a no-op for an
    already-ABSTAIN verdict."""
    if verdict.status is not TimingStatus.CONFIRMED:
        return verdict
    evidence = verdict.evidence_frame_timestamps_s
    if len(evidence) > _MAX_END_VALIDATE_EVIDENCE:
        return _grounding_abstain(
            verdict,
            "evidence_not_selective",
            f"cited {len(evidence)} evidence timestamps - more than "
            f"{_MAX_END_VALIDATE_EVIDENCE}, too broad to show which specific "
            f"checkpoints demonstrate a sustained trend",
        )
    return verdict


def _suggested_trend_checkpoints(
    onset_ts: float,
    available_ts: list[float],
    *,
    spacing_s: float,
    max_count: int,
) -> list[float]:
    """Up to ``max_count`` suggested trend-checkpoint anchors, roughly
    ``spacing_s`` apart starting just past ``onset_ts``, each snapped to
    the nearest timestamp actually in ``available_ts`` - a labelling aid
    only (see ``TimedFrame.is_trend_checkpoint``), not itself the
    enforcement mechanism (that is ``_validate_trend_checkpoints``, which
    accepts any grounded, sufficiently-spaced citation regardless of which
    frames carry this label). Deterministic and duplicate-free; empty if
    ``available_ts`` is empty."""
    if not available_ts:
        return []
    picked: list[float] = []
    target = onset_ts + spacing_s
    ceiling = max(available_ts)
    while target <= ceiling + _BOUNDS_EPSILON_S and len(picked) < max_count:
        nearest = min(available_ts, key=lambda ts: abs(ts - target))
        if nearest > onset_ts and nearest not in picked:
            picked.append(nearest)
        target += spacing_s
    return picked


def _validate_trend_checkpoints(
    verdict: TimingVerdict, *, region: _GroundingRegion, cfg: PipelineConfig
) -> TimingVerdict:
    """The temporal-spacing gate a second real audited run's false
    confirmation exposed: a CONFIRMED end_validate verdict cited three
    "trend checkpoints" only ~33ms apart - adjacent native-fps frames,
    sub-frame-interval jitter, not a multi-second sustained trend. Onset
    localization (a dense frame pinpointing *where* a break starts) and
    trend confirmation (specific checkpoints, spaced roughly a second
    apart, each showing further shortening) are different questions;
    prompt wording alone was not enough to keep a real model from
    conflating them, so this enforces the temporal contract structurally,
    in code, on ``TimingVerdict.trend_checkpoint_timestamps_s`` - a field
    kept separate from ``evidence_frame_timestamps_s`` for exactly this
    reason (see ``_bound_end_validate_evidence`` for the sibling check on
    that field). Converges on ABSTAIN like every other grounding failure;
    a no-op for an already-ABSTAIN verdict."""
    if verdict.status is not TimingStatus.CONFIRMED:
        return verdict
    assert verdict.end_s is not None
    checkpoints = verdict.trend_checkpoint_timestamps_s

    def _matches_region(ts: float) -> bool:
        return any(
            abs(ts - sent) <= region.time_tolerance_s for sent in region.submitted_timestamps_s
        )

    ungrounded = [ts for ts in checkpoints if not _matches_region(ts)]
    if ungrounded:
        return _grounding_abstain(
            verdict,
            "insufficient_trend_horizon",
            f"trend_checkpoint_timestamps_s {ungrounded} do not correspond to any frame "
            f"actually submitted in this request",
        )
    if len(checkpoints) < cfg.end_validation_min_trend_checkpoints:
        return _grounding_abstain(
            verdict,
            "insufficient_trend_horizon",
            f"cited only {len(checkpoints)} trend checkpoint(s) - at least "
            f"{cfg.end_validation_min_trend_checkpoints} are required to demonstrate a "
            f"sustained trend rather than a single onset-adjacent observation",
        )
    ordered = sorted(checkpoints)
    first_gap = ordered[0] - verdict.end_s
    if first_gap < cfg.end_validation_min_onset_gap_s - _BOUNDS_EPSILON_S:
        return _grounding_abstain(
            verdict,
            "insufficient_trend_horizon",
            f"first trend checkpoint at t={ordered[0]:.3f}s is only {first_gap:.3f}s after "
            f"the reported onset t={verdict.end_s:.3f}s - needs at least "
            f"{cfg.end_validation_min_onset_gap_s:.2f}s to be more than native-fps jitter",
        )
    for earlier, later in zip(ordered, ordered[1:], strict=False):
        gap = later - earlier
        if gap < cfg.end_validation_min_checkpoint_spacing_s - _BOUNDS_EPSILON_S:
            return _grounding_abstain(
                verdict,
                "insufficient_trend_horizon",
                f"trend checkpoints at t={earlier:.3f}s and t={later:.3f}s are only "
                f"{gap:.3f}s apart - needs at least "
                f"{cfg.end_validation_min_checkpoint_spacing_s:.2f}s between checkpoints to "
                f"be more than native-fps jitter",
            )
    return verdict


@dataclass(frozen=True)
class _EndValidationOutcome:
    """One candidate's full contact-sheet cascade result - the reusable unit
    both the primary (end-coarse's own candidate) and, when a conflict is
    detected, the secondary (the coarse pass's own independent end
    estimate) validation calls are built from.

    ``verdict``/``response``/``submitted_timestamps_s``/``window_s`` always
    describe the DECISIVE call - the 7-panel refine sheet's own result when
    one ran, otherwise the 9-panel coarse sheet's. ``coarse_cascade_*``
    fields are set only when a refine sheet actually ran, so the coarse
    sheet's own (superseded) verdict/response/panels stay individually
    inspectable for cost/audit purposes rather than silently discarded -
    see ``pipeline.PipelineOutcome.end_validation_coarse_cascade_response``.
    """

    verdict: TimingVerdict
    response: RawProviderResponse
    submitted_timestamps_s: tuple[float, ...]
    window_s: tuple[float, float]
    coarse_cascade_verdict: TimingVerdict | None = None
    coarse_cascade_response: RawProviderResponse | None = None
    coarse_cascade_submitted_timestamps_s: tuple[float, ...] | None = None


# --------------------------------------------------------------------------- #
# Candidate-centred contact-sheet cascade (supervisor-authorized, real
# gpt-5-mini experiment - see prompts.py's own module comment and
# diagnostics/llm_spike/DESIGN.md for the full write-up). Replaces the
# dense-individual-frames trend-validation request above with ONE composite
# grid image per call: a coarse 9-panel/0.5s/4s sheet centred on the
# candidate, and - only when that sheet abstains but points at a possible
# collapse - one denser 7-panel/0.25s/1.5s look-back refine sheet. Never
# more than two calls per candidate, matching the "bounded, never an
# open-ended search" discipline every other pass in this module follows.
# --------------------------------------------------------------------------- #

_CASCADE_COARSE_STEP_S = 0.5
_CASCADE_COARSE_COUNT = 9
_CASCADE_COARSE_COLUMNS = 3
_CASCADE_COARSE_ROWS = 3
_CASCADE_COARSE_HALF_SPAN_S = (_CASCADE_COARSE_COUNT - 1) * _CASCADE_COARSE_STEP_S / 2  # 2.0s

_CASCADE_REFINE_STEP_S = 0.25
_CASCADE_REFINE_COUNT = 7
_CASCADE_REFINE_COLUMNS = 4
_CASCADE_REFINE_ROWS = 2
_CASCADE_REFINE_LOOKBACK_S = (_CASCADE_REFINE_COUNT - 1) * _CASCADE_REFINE_STEP_S  # 1.5s


def _shift_window_into_bounds(
    lo: float, hi: float, bound_lo: float, bound_hi: float
) -> tuple[float, float]:
    """Slide a fixed-width ``[lo, hi]`` window so it fits inside
    ``[bound_lo, bound_hi]``, preserving its width whenever the bounds are
    wide enough to hold it - so a candidate near the start or end of the
    scannable region still gets a full-width, evenly-spaced panel sheet,
    just not perfectly centred on it. Clamped (width degrades) only when
    the bounds themselves are narrower than the window - a short clip or a
    candidate hard up against ``locked_start_s``."""
    if bound_hi - bound_lo <= (hi - lo) + _BOUNDS_EPSILON_S:
        return bound_lo, bound_hi
    if lo < bound_lo:
        shift = bound_lo - lo
        lo, hi = lo + shift, hi + shift
    if hi > bound_hi:
        shift = hi - bound_hi
        lo, hi = lo - shift, hi - shift
    return max(lo, bound_lo), min(hi, bound_hi)


def _cascade_coarse_timestamps(
    candidate_ts: float, bound_lo: float, bound_hi: float
) -> list[float]:
    """9 timestamps, 0.5s apart, centred on ``candidate_ts`` where the
    bounds allow it - see :func:`_shift_window_into_bounds`."""
    lo, hi = _shift_window_into_bounds(
        candidate_ts - _CASCADE_COARSE_HALF_SPAN_S,
        candidate_ts + _CASCADE_COARSE_HALF_SPAN_S,
        bound_lo,
        bound_hi,
    )
    return [
        round(min(lo + i * _CASCADE_COARSE_STEP_S, hi), 6) for i in range(_CASCADE_COARSE_COUNT)
    ]


def _cascade_refine_timestamps(collapse_ts: float, bound_lo: float, bound_hi: float) -> list[float]:
    """7 timestamps, 0.25s apart, looking back 1.5s from ``collapse_ts`` -
    see :func:`_shift_window_into_bounds`."""
    lo, hi = _shift_window_into_bounds(
        collapse_ts - _CASCADE_REFINE_LOOKBACK_S, collapse_ts, bound_lo, bound_hi
    )
    return [
        round(min(lo + i * _CASCADE_REFINE_STEP_S, hi), 6) for i in range(_CASCADE_REFINE_COUNT)
    ]


def _build_cascade_request(
    reader: VideoReader,
    panel_timestamps_s: list[float],
    *,
    crop_origin: tuple[int, int],
    columns: int,
    rows: int,
    prompt_version: str,
    prompt_text: str,
    jpeg_quality: int,
) -> ProviderRequest:
    """Render every panel, compose them into one contact sheet, and wrap it
    as a single-frame :class:`ProviderRequest` whose
    ``grounding_timestamps_s`` carries the *logical* per-panel timestamps -
    see ``TimedFrame.is_contact_sheet``/``ProviderRequest.grounding_timestamps_s``
    for why a single composite image still needs several grounded
    timestamps."""
    panels = [
        contact_sheet.render_panel(reader.frame_at(ts), timestamp_s=ts, crop_origin=crop_origin)
        for ts in panel_timestamps_s
    ]
    grid = contact_sheet.compose_grid(panels, columns=columns, rows=rows)
    image_bytes = encode_jpeg(grid, quality=jpeg_quality)
    anchor_ts = panel_timestamps_s[len(panel_timestamps_s) // 2]
    frame = TimedFrame(timestamp_s=anchor_ts, image_bytes=image_bytes, is_contact_sheet=True)
    return ProviderRequest(
        prompt_version=prompt_version,
        prompt_text=prompt_text,
        frames=(frame,),
        pass_name="end_validate",
        grounding_timestamps_s=tuple(panel_timestamps_s),
    )


def _validate_possible_collapse(
    verdict: TimingVerdict, *, region: _GroundingRegion
) -> TimingVerdict:
    """An ungrounded ``possible_collapse_s`` (not one of this pass's own
    submitted panels) is a fabrication like any other ungrounded citation -
    converges on ABSTAIN, same discipline as ``_validate_grounding``. A
    no-op when nothing was cited; applies to CONFIRMED and ABSTAIN alike,
    since ``possible_collapse_s`` is meaningful on either (unlike
    ``evidence_frame_timestamps_s``, which only a CONFIRMED verdict uses)."""
    if verdict.possible_collapse_s is None:
        return verdict
    ts = verdict.possible_collapse_s
    if not any(abs(ts - sent) <= region.time_tolerance_s for sent in region.submitted_timestamps_s):
        return _grounding_abstain(
            verdict,
            "ungrounded_evidence",
            f"possible_collapse_s={ts} does not correspond to any panel actually "
            f"submitted in this contact sheet",
        )
    return verdict


def _validate_cascade_trend_checkpoints(
    verdict: TimingVerdict, *, region: _GroundingRegion, min_count: int
) -> TimingVerdict:
    """The contact-sheet cascade's own trend-confirmation gate: a CONFIRMED
    verdict must cite at least ``min_count`` grounded panel timestamps,
    strictly after the reported onset, as ``trend_checkpoint_timestamps_s``
    - "later panels are mandatory confirmation" (supervisor's own wording).

    Deliberately does not reuse ``_validate_trend_checkpoints``'s numeric
    onset-gap/checkpoint-spacing floors (0.75s, calibrated for the old
    dense-native-fps-frame method's own jitter risk): a contact sheet's
    panels are already fixed, deliberately-spaced anchors (0.5s coarse /
    0.25s refine apart) - there is no native-fps density for the model to
    mistake as a trend, so the floor that problem needed does not apply
    here. Converges on ABSTAIN like every other grounding failure; a no-op
    for an already-ABSTAIN verdict."""
    if verdict.status is not TimingStatus.CONFIRMED:
        return verdict
    assert verdict.end_s is not None
    checkpoints = verdict.trend_checkpoint_timestamps_s

    def _matches_region(ts: float) -> bool:
        return any(
            abs(ts - sent) <= region.time_tolerance_s for sent in region.submitted_timestamps_s
        )

    ungrounded = [ts for ts in checkpoints if not _matches_region(ts)]
    if ungrounded:
        return _grounding_abstain(
            verdict,
            "insufficient_trend_horizon",
            f"trend_checkpoint_timestamps_s {ungrounded} do not correspond to any panel "
            f"actually shown in this contact sheet",
        )
    after_onset = [ts for ts in checkpoints if ts > verdict.end_s + _BOUNDS_EPSILON_S]
    if len(after_onset) < min_count:
        return _grounding_abstain(
            verdict,
            "insufficient_trend_horizon",
            f"cited only {len(after_onset)} trend checkpoint(s) strictly after the "
            f"reported onset t={verdict.end_s:.3f}s - at least {min_count} later panels "
            f"are required to confirm the trend persists",
        )
    return verdict


def _run_cascade_sheet(
    reader: VideoReader,
    panel_timestamps_s: list[float],
    *,
    crop_origin: tuple[int, int],
    columns: int,
    rows: int,
    prompt_version: str,
    prompt_text: str,
    cfg: PipelineConfig,
    provider: TimingProvider,
    on_stage: Callable[[str], None] | None,
) -> tuple[TimingVerdict, RawProviderResponse, tuple[float, ...]]:
    """Send one contact sheet (coarse or refine) and return its grounded
    verdict - the shared body both cascade stages use."""
    request = _build_cascade_request(
        reader,
        panel_timestamps_s,
        crop_origin=crop_origin,
        columns=columns,
        rows=rows,
        prompt_version=prompt_version,
        prompt_text=prompt_text,
        jpeg_quality=cfg.jpeg_quality,
    )
    if on_stage is not None:
        on_stage("end_validate")
    response = provider.analyze(request)
    verdict = parse_raw_response(
        response, prompt_version=prompt_version, min_confidence=cfg.min_confidence
    )
    submitted = tuple(panel_timestamps_s)
    region = _GroundingRegion(
        bounds=(submitted[0], submitted[-1]),
        submitted_timestamps_s=submitted,
        time_tolerance_s=2.0 * (submitted[1] - submitted[0] if len(submitted) > 1 else 0.5),
    )
    verdict = _validate_grounding(
        verdict, start_region=region, end_region=region, max_uncertainty_s=cfg.max_uncertainty_s
    )
    verdict = _bound_end_validate_evidence(verdict)
    verdict = _validate_cascade_trend_checkpoints(
        verdict, region=region, min_count=cfg.end_validation_min_trend_checkpoints
    )
    verdict = _validate_possible_collapse(verdict, region=region)
    return verdict, response, submitted


def _run_end_validation_pass(
    reader: VideoReader,
    *,
    candidate_ts: float,
    locked_start_s: float,
    duration_s: float,
    fine_step_s: float,
    cfg: PipelineConfig,
    provider: TimingProvider,
    on_stage: Callable[[str], None] | None,
) -> _EndValidationOutcome:
    """Candidate-centred contact-sheet cascade for one candidate end
    timestamp: a coarse 9-panel sheet first, and - only when it abstains
    but points at a possible collapse - one denser 7-panel refine sheet.
    Reuses the application's own existing broad/coarse pass's candidate as
    the sheet's centre; never invents a truth-centred window (supervisor
    authorization). ``fine_step_s`` is unused by the cascade itself (kept
    in the signature for call-site compatibility with the dense-frame
    method this replaced) but is not needed here - panel spacing is fixed
    by the cascade's own contract, not the clip's native frame rate.

    Unlike the dense-frame method this replaced, a contact sheet is a
    small, fixed-size composite image (9 or 7 tiles at a fixed resolution)
    - never subject to the per-request byte-budget thinning that could
    make the old method's own frame batch not fit, so there is no
    ``None``/oversized-abstain outcome to report here.
    """
    del fine_step_s  # unused by the cascade - see docstring
    bound_lo = max(0.0, locked_start_s)
    sample_frame = reader.frame_at(candidate_ts)
    crop_origin = contact_sheet.default_crop_origin(sample_frame.shape[1], sample_frame.shape[0])

    coarse_timestamps = _cascade_coarse_timestamps(candidate_ts, bound_lo, duration_s)
    coarse_verdict, coarse_response, coarse_submitted = _run_cascade_sheet(
        reader,
        coarse_timestamps,
        crop_origin=crop_origin,
        columns=_CASCADE_COARSE_COLUMNS,
        rows=_CASCADE_COARSE_ROWS,
        prompt_version=PROMPT_END_CASCADE_COARSE_V1_ID,
        prompt_text=PROMPT_END_CASCADE_COARSE_V1,
        cfg=cfg,
        provider=provider,
        on_stage=on_stage,
    )
    coarse_window_s = (coarse_submitted[0], coarse_submitted[-1])

    coarse_needs_refine = (
        coarse_verdict.status is not TimingStatus.CONFIRMED
        and coarse_verdict.possible_collapse_s is not None
    )
    if not coarse_needs_refine:
        # Either confirmed outright, or abstained with nothing to point a
        # refine sheet at - the coarse verdict's own reason codes (the
        # model's, e.g. "no_break_found", or a grounding failure's, e.g.
        # "ungrounded_evidence") are already the accurate, specific
        # terminal reason; nothing to relabel.
        return _EndValidationOutcome(
            verdict=coarse_verdict,
            response=coarse_response,
            submitted_timestamps_s=coarse_submitted,
            window_s=coarse_window_s,
        )

    assert coarse_verdict.possible_collapse_s is not None
    refine_timestamps = _cascade_refine_timestamps(
        coarse_verdict.possible_collapse_s, bound_lo, duration_s
    )
    refine_verdict, refine_response, refine_submitted = _run_cascade_sheet(
        reader,
        refine_timestamps,
        crop_origin=crop_origin,
        columns=_CASCADE_REFINE_COLUMNS,
        rows=_CASCADE_REFINE_ROWS,
        prompt_version=PROMPT_END_CASCADE_REFINE_V1_ID,
        prompt_text=PROMPT_END_CASCADE_REFINE_V1,
        cfg=cfg,
        provider=provider,
        on_stage=on_stage,
    )
    if refine_verdict.status is not TimingStatus.CONFIRMED:
        # The refine sheet was reached (the coarse sheet did point at a
        # possible collapse) and still could not confirm - tag the
        # cascade's own terminal reason alongside whatever specific reason
        # the refine stage itself already gave, rather than replacing it.
        merged_reasons = dict.fromkeys((*refine_verdict.reason_codes, "cascade_unconfirmed"))
        refine_verdict = replace(refine_verdict, reason_codes=tuple(merged_reasons))
    return _EndValidationOutcome(
        verdict=refine_verdict,
        response=refine_response,
        submitted_timestamps_s=refine_submitted,
        window_s=(refine_submitted[0], refine_submitted[-1]),
        coarse_cascade_verdict=coarse_verdict,
        coarse_cascade_response=coarse_response,
        coarse_cascade_submitted_timestamps_s=coarse_submitted,
    )


def run_llm_timing(
    video_path: Path,
    provider: TimingProvider,
    *,
    prompt_version: str,
    prompt_text: str,
    config: PipelineConfig | None = None,
    video_info: VideoInfo | None = None,
    on_stage: Callable[[str], None] | None = None,
) -> PipelineOutcome:
    """Run the coarse-then-fine timing pass and return a full outcome.

    Every branch that cannot produce a trustworthy answer returns an
    ABSTAIN verdict with ``event=None`` rather than raising - the caller is
    expected to treat that exactly like "the detector found nothing" and
    fall back to the assisted/manual workflow. Genuine programming errors
    (a bad config, an unreadable video) still raise, per the rest of this
    codebase's error conventions.

    ``on_stage``, if given, is called with the pass name ("coarse", "fine",
    "end_coarse", "end_validate") immediately before that pass's provider
    call is sent - never after, and never for a pass this run never reaches
    (an early abstain or budget failure). Two purposes, both additive and
    optional: staged progress for a caller with a UI to update, and
    cooperative cancellation - a caller that wants to abort between passes
    can raise from inside the callback, and that exception propagates
    naturally out of this function (no new exception type here; this
    module has no opinion on what "cancelled" means to its caller).
    """
    cfg = config or PipelineConfig()
    cfg.validate()
    pass_verdicts: dict[str, TimingVerdict] = {}
    pass_frames: dict[str, tuple[float, ...]] = {}
    derived: dict[str, DerivedValue] = {}

    with VideoReader(video_path, video_info).open() as reader:
        duration_s = reader.info.duration_s
        if duration_s is None:
            raise ConfigurationError(
                "Cannot run the LLM timing pipeline on a video with unknown duration."
            )

        coarse_times = _coarse_timestamps(duration_s, cfg.coarse_step_s)
        coarse_frames_dense = _extract_frames(
            reader,
            coarse_times,
            max_dimension_px=cfg.max_frame_dimension_px,
            jpeg_quality=cfg.jpeg_quality,
        )
        # The coarse pass only has to locate an approximate region, not hit
        # gate-level precision - but it must be dense enough that *some*
        # sample lands within fine_margin_s of wherever the true transition
        # actually is, or the fine window built around the coarse estimate
        # could miss it entirely. See _min_coarse_frames.
        coarse_min_frames = _min_coarse_frames(duration_s, cfg.fine_margin_s)
        coarse_frames = _fit_frames_to_budget(
            coarse_frames_dense,
            prompt_text=prompt_text,
            max_request_bytes=cfg.max_request_bytes,
            min_frames=coarse_min_frames,
        )
        if coarse_frames is None:
            abstain = _oversized_abstain(
                prompt_version,
                "coarse",
                f"{len(coarse_frames_dense)} frames still exceed the request byte budget "
                f"even at the sparsest sampling ({coarse_min_frames} frames) that "
                f"guarantees coverage within fine_margin_s of the true transition",
            )
            return _abstain_outcome(abstain, _unsent_response(abstain.raw_notes))

        coarse_request = ProviderRequest(
            prompt_version=prompt_version,
            prompt_text=prompt_text,
            frames=tuple(coarse_frames),
            pass_name="coarse",
        )
        if on_stage is not None:
            on_stage("coarse")
        coarse_response = provider.analyze(coarse_request)
        coarse_verdict = parse_raw_response(
            coarse_response, prompt_version=prompt_version, min_confidence=cfg.min_confidence
        )
        coarse_submitted = [frame.timestamp_s for frame in coarse_frames]
        pass_frames["coarse"] = tuple(coarse_submitted)
        # A single shared frame batch, not yet split into per-boundary
        # windows - the same region serves both start_s and end_s here.
        coarse_region = _GroundingRegion(
            bounds=(0.0, duration_s),
            submitted_timestamps_s=tuple(coarse_submitted),
            time_tolerance_s=_effective_tolerance_s(coarse_submitted, cfg.coarse_step_s),
        )
        coarse_verdict = _validate_grounding(
            coarse_verdict,
            start_region=coarse_region,
            end_region=coarse_region,
            max_uncertainty_s=cfg.max_uncertainty_s,
        )

        pass_verdicts["coarse"] = coarse_verdict
        if coarse_verdict.status is not TimingStatus.CONFIRMED:
            return _abstain_outcome(
                coarse_verdict,
                coarse_response,
                pass_verdicts=pass_verdicts,
                pass_frames=pass_frames,
                derived=derived,
            )

        assert coarse_verdict.start_s is not None and coarse_verdict.end_s is not None
        # The fine pass now asks about the start ONLY - a single window,
        # widened past fine_margin_s when the coarse pass itself reported
        # more start uncertainty than that. It used to also build and send
        # an end window in the same request (to refine both boundaries at
        # once), but that end answer was already fully discarded once the
        # chronological end-scan below was introduced - and a live gate-1
        # run showed why sending it anyway is actively unsafe: the fine
        # pass's own end_s can fall outside its (irrelevant) end window,
        # and _validate_grounding requires *both* claimed boundaries to
        # ground before confirming anything - so an invalid answer to a
        # question nobody needed could abstain the whole run before the
        # end-scan ever got to start (supervisor-directed fix, see
        # diagnostics/llm_spike/DESIGN.md). See PROMPT_START_REFINE_V1.
        start_margin = max(cfg.fine_margin_s, coarse_verdict.start_uncertainty_s)
        start_lo = max(0.0, coarse_verdict.start_s - start_margin)
        start_hi = min(duration_s, coarse_verdict.start_s + start_margin)

        if (start_hi - start_lo) > cfg.fine_max_span_s:
            abstain = TimingVerdict.abstain(
                reason_codes=("ambiguous_evidence",),
                model_id=coarse_response.model_id,
                prompt_version=prompt_version,
                raw_notes=(
                    f"coarse pass's own start uncertainty makes the fine start window "
                    f"too wide to scan densely ({start_hi - start_lo:.2f}s, "
                    f"cap={cfg.fine_max_span_s:.2f}s); coarse estimate was "
                    f"start_s={coarse_verdict.start_s}"
                ),
            )
            return _abstain_outcome(
                abstain,
                coarse_response,
                pass_verdicts=pass_verdicts,
                pass_frames=pass_frames,
                derived=derived,
            )

        fps = reader.info.fps
        fine_step_s = 1.0 / fps
        start_times = _dense_timestamps(start_lo, start_hi, fine_step_s)
        start_frames_dense = _extract_frames(
            reader,
            start_times,
            max_dimension_px=cfg.max_frame_dimension_px,
            jpeg_quality=cfg.jpeg_quality,
        )
        start_min_frames = _min_frames_for_window(start_hi - start_lo, cfg.target_tolerance_s)
        start_frames = _fit_frames_to_budget(
            start_frames_dense,
            prompt_text=PROMPT_START_REFINE_V1,
            max_request_bytes=cfg.max_request_bytes,
            min_frames=start_min_frames,
        )
        if start_frames is None:
            abstain = _oversized_abstain(
                PROMPT_START_REFINE_V1_ID,
                "fine",
                f"start window: {len(start_frames_dense)} frames still exceed the "
                f"request byte budget even at the sparsest sampling "
                f"({start_min_frames} frames) that meets the target tolerance",
            )
            return PipelineOutcome(
                verdict=abstain,
                event=None,
                coarse_response=coarse_response,
                fine_response=_unsent_response(abstain.raw_notes),
                pass_verdicts=dict(pass_verdicts),
                pass_frames=dict(pass_frames),
                derived=dict(derived),
            )

        fine_request = ProviderRequest(
            prompt_version=PROMPT_START_REFINE_V1_ID,
            prompt_text=PROMPT_START_REFINE_V1,
            frames=tuple(start_frames),
            pass_name="fine",
        )
        if on_stage is not None:
            on_stage("fine")
        fine_response = provider.analyze(fine_request)
        fine_verdict = parse_raw_response(
            fine_response,
            prompt_version=PROMPT_START_REFINE_V1_ID,
            min_confidence=cfg.min_confidence,
        )
        start_submitted = [frame.timestamp_s for frame in start_frames]
        pass_frames["fine"] = tuple(start_submitted)
        start_region = _GroundingRegion(
            bounds=(start_lo, start_hi),
            submitted_timestamps_s=tuple(start_submitted),
            time_tolerance_s=_effective_tolerance_s(start_submitted, fine_step_s),
        )
        # Single shared region for both arguments - this call only ever
        # claims one boundary (start_s == end_s, per PROMPT_START_REFINE_V1),
        # same pattern as the coarse pass and each end-scan window.
        fine_verdict = _validate_grounding(
            fine_verdict,
            start_region=start_region,
            end_region=start_region,
            max_uncertainty_s=cfg.max_uncertainty_s,
        )

        pass_verdicts["fine"] = fine_verdict
        if fine_verdict.status is not TimingStatus.CONFIRMED:
            return PipelineOutcome(
                verdict=fine_verdict,
                event=None,
                coarse_response=coarse_response,
                fine_response=fine_response,
                pass_verdicts=dict(pass_verdicts),
                pass_frames=dict(pass_frames),
                derived=dict(derived),
            )

        # The end is nominated by ONE whole-clip (post-start), sparsely
        # sampled coarse request, then confirmed or rejected by ONE dense
        # trend-validation follow-up - a bounded two-stage strategy
        # (2 calls, not N chronological scan windows). This replaced an
        # earlier chronological end-scan entirely: a real gate-1 experiment
        # found that decomposing the search into many narrow, isolated
        # windows lost the temporal context a model needs to judge a
        # *sustained* trend - a 13-window scan found the true break's
        # window but then wrongly rejected it, while one coherent request
        # covering the same span with the same trend contract correctly
        # confirmed it (787,182 tokens/~7 minutes for the 13-window scan,
        # versus ~16,000 tokens/~22s for the isolated single-request
        # equivalent of stage 2 below) - supervisor-directed replacement,
        # see diagnostics/llm_spike/DESIGN.md and PROMPT_END_COARSE_V2/
        # PROMPT_END_VALIDATE_V3. Failed validation now falls back directly
        # to ABSTAIN (assisted/manual workflow) rather than searching for
        # another candidate - single-shot, not an open-ended search (a
        # detected cross-pass conflict can still add one bounded second
        # validation call, never more - see _candidates_conflict below).
        #
        # The coarse candidate search must not start at locked_start_s
        # itself: a real gate-1 rerun (against the old chronological scan)
        # showed the first end-scan window built that way still contained
        # the onset transition (nothing visible -> stream visible), and a
        # model can misread that transition as a "break" - a false-
        # confirmed end just a fraction of a second after the true start.
        # The fix is structural, not a prompt-wording request, and applies
        # equally here: sampling begins no earlier than start_hi, the far
        # edge of the grounded start-refinement window, so no onset/
        # pre-flow frame is ever eligible to be submitted as a candidate
        # end timestamp in the first place (supervisor-directed fix, see
        # diagnostics/llm_spike/DESIGN.md).
        assert fine_verdict.start_s is not None
        locked_start_s: float = fine_verdict.start_s
        locked_start_uncertainty_s = fine_verdict.start_uncertainty_s
        start_side_evidence = fine_verdict.evidence_frame_timestamps_s
        derived["locked_start_s"] = locked_start_s
        derived["locked_start_uncertainty_s"] = locked_start_uncertainty_s
        derived["start_evidence_s"] = _nearest_grounded_evidence_ts(
            start_side_evidence, pass_frames.get("fine", ()), locked_start_s
        )

        scan_from_s = min(max(start_hi, locked_start_s), duration_s)

        end_coarse_times = _dense_timestamps(scan_from_s, duration_s, cfg.coarse_step_s)
        end_coarse_frames_dense = _extract_frames(
            reader,
            end_coarse_times,
            max_dimension_px=cfg.max_frame_dimension_px,
            jpeg_quality=cfg.jpeg_quality,
        )
        # Same coverage guarantee _min_coarse_frames already gives the
        # start-side coarse pass, applied to the (shorter) post-start span.
        end_coarse_min_frames = _min_coarse_frames(duration_s - scan_from_s, cfg.fine_margin_s)
        end_coarse_frames = _fit_frames_to_budget(
            end_coarse_frames_dense,
            prompt_text=PROMPT_END_COARSE_V2,
            max_request_bytes=cfg.max_request_bytes,
            min_frames=end_coarse_min_frames,
        )
        if end_coarse_frames is None:
            abstain = _oversized_abstain(
                PROMPT_END_COARSE_V2_ID,
                "end_coarse",
                f"{len(end_coarse_frames_dense)} frames still exceed the request byte "
                f"budget even at the sparsest sampling ({end_coarse_min_frames} frames) "
                f"that guarantees coverage within fine_margin_s of the true transition",
            )
            return PipelineOutcome(
                verdict=abstain,
                event=None,
                coarse_response=coarse_response,
                fine_response=fine_response,
                pass_verdicts=dict(pass_verdicts),
                pass_frames=dict(pass_frames),
                derived=dict(derived),
            )

        end_coarse_request = ProviderRequest(
            prompt_version=PROMPT_END_COARSE_V2_ID,
            prompt_text=PROMPT_END_COARSE_V2,
            frames=tuple(end_coarse_frames),
            pass_name="end_coarse",
        )
        if on_stage is not None:
            on_stage("end_coarse")
        end_coarse_response = provider.analyze(end_coarse_request)
        end_coarse_verdict = parse_raw_response(
            end_coarse_response,
            prompt_version=PROMPT_END_COARSE_V2_ID,
            min_confidence=cfg.min_confidence,
        )
        end_coarse_submitted = [frame.timestamp_s for frame in end_coarse_frames]
        pass_frames["end_coarse"] = tuple(end_coarse_submitted)
        end_coarse_region = _GroundingRegion(
            bounds=(scan_from_s, duration_s),
            submitted_timestamps_s=tuple(end_coarse_submitted),
            time_tolerance_s=_effective_tolerance_s(end_coarse_submitted, cfg.coarse_step_s),
        )
        end_coarse_verdict = _validate_grounding(
            end_coarse_verdict,
            start_region=end_coarse_region,
            end_region=end_coarse_region,
            max_uncertainty_s=cfg.max_uncertainty_s,
        )
        pass_verdicts["end_coarse"] = end_coarse_verdict
        if end_coarse_verdict.status is not TimingStatus.CONFIRMED:
            return PipelineOutcome(
                verdict=end_coarse_verdict,
                event=None,
                coarse_response=coarse_response,
                fine_response=fine_response,
                end_coarse_response=end_coarse_response,
                pass_verdicts=dict(pass_verdicts),
                pass_frames=dict(pass_frames),
                derived=dict(derived),
            )

        # A sparse coarse candidate is never trusted on its own - it must
        # still pass a dense, bounded trend-validation check before being
        # accepted (supervisor-directed generalization: a real break is a
        # *sustained* shortening trend, not a single frame that happens to
        # look shorter - see PROMPT_END_VALIDATE_V3 and
        # diagnostics/llm_spike/DESIGN.md).
        assert end_coarse_verdict.end_s is not None
        candidate_ts = end_coarse_verdict.end_s
        derived["end_coarse_candidate_s"] = candidate_ts

        # The whole-clip coarse pass (the very first request this run made)
        # already produced its own, independent end estimate - never
        # discarded from here on. A real audited run showed why that
        # matters: the coarse pass correctly estimated the end near 21.0s,
        # end-coarse instead nominated 5.5s, and only ever validating 5.5s's
        # own window ([1.5, 11.5]s) meant the true break near 21.0s was
        # structurally invisible to the run - a confidently wrong CONFIRMED
        # result, not merely an imprecise one (supervisor-directed, see
        # diagnostics/llm_spike/DESIGN.md).
        assert coarse_verdict.end_s is not None
        coarse_end_estimate_s = coarse_verdict.end_s
        derived["coarse_end_estimate_s"] = coarse_end_estimate_s
        conflict = _candidates_conflict(
            candidate_ts, coarse_end_estimate_s, locked_start_s, duration_s, cfg
        )
        derived["candidate_conflict"] = conflict

        # The candidate-centred contact-sheet cascade (coarse 9-panel sheet,
        # and - only when it abstains but points at a possible collapse -
        # one denser 7-panel refine sheet) is still one bounded check per
        # candidate, never an open-ended search (supervisor-directed, see
        # diagnostics/llm_spike/DESIGN.md and PROMPT_END_CASCADE_COARSE_V1/
        # PROMPT_END_CASCADE_REFINE_V1) - a conflict adds at most ONE extra
        # such (up to two-call) check, never more.
        primary = _run_end_validation_pass(
            reader,
            candidate_ts=candidate_ts,
            locked_start_s=locked_start_s,
            duration_s=duration_s,
            fine_step_s=fine_step_s,
            cfg=cfg,
            provider=provider,
            on_stage=on_stage,
        )

        pass_verdicts["end_validate"] = primary.verdict
        pass_frames["end_validate"] = primary.submitted_timestamps_s
        derived["end_validation_window_s"] = list(primary.window_s)
        if primary.coarse_cascade_verdict is not None:
            pass_verdicts["end_validate_coarse_cascade"] = primary.coarse_cascade_verdict
            assert primary.coarse_cascade_submitted_timestamps_s is not None
            pass_frames["end_validate_coarse_cascade"] = (
                primary.coarse_cascade_submitted_timestamps_s
            )
        # Computed regardless of whether validation confirmed - even a
        # rejected/abstained candidate should show which cited evidence (if
        # any) was closest to it, per the same "never fabricate, but always
        # show what's known" rule _nearest_grounded_evidence_ts follows.
        derived["end_evidence_s"] = _nearest_grounded_evidence_ts(
            primary.verdict.evidence_frame_timestamps_s,
            primary.submitted_timestamps_s,
            primary.verdict.end_s if primary.verdict.end_s is not None else candidate_ts,
        )

        secondary: _EndValidationOutcome | None = None
        if conflict:
            # Bounded: exactly one extra call, only when the two
            # independent candidates disagree badly enough that neither's
            # own window contains the other - never a search across more
            # than these two anchors.
            secondary = _run_end_validation_pass(
                reader,
                candidate_ts=coarse_end_estimate_s,
                locked_start_s=locked_start_s,
                duration_s=duration_s,
                fine_step_s=fine_step_s,
                cfg=cfg,
                provider=provider,
                on_stage=on_stage,
            )
            pass_verdicts["end_validate_conflict"] = secondary.verdict
            pass_frames["end_validate_conflict"] = secondary.submitted_timestamps_s
            derived["end_validation_conflict_window_s"] = list(secondary.window_s)
            derived["end_conflict_evidence_s"] = _nearest_grounded_evidence_ts(
                secondary.verdict.evidence_frame_timestamps_s,
                secondary.submitted_timestamps_s,
                (
                    secondary.verdict.end_s
                    if secondary.verdict.end_s is not None
                    else coarse_end_estimate_s
                ),
            )
            if secondary.coarse_cascade_verdict is not None:
                pass_verdicts["end_validate_conflict_coarse_cascade"] = (
                    secondary.coarse_cascade_verdict
                )
                assert secondary.coarse_cascade_submitted_timestamps_s is not None
                pass_frames["end_validate_conflict_coarse_cascade"] = (
                    secondary.coarse_cascade_submitted_timestamps_s
                )

        primary_confirmed = primary.verdict.status is TimingStatus.CONFIRMED
        secondary_confirmed = (
            secondary is not None and secondary.verdict.status is TimingStatus.CONFIRMED
        )

        # Resolution: never silently discard the coarse candidate, and
        # never confidently pick one of two independently-confirmed,
        # conflicting candidates - see diagnostics/llm_spike/DESIGN.md.
        chosen: _EndValidationOutcome | None
        if not conflict:
            resolution = "single_candidate"
            chosen = primary if primary_confirmed else None
        elif primary_confirmed and secondary_confirmed:
            resolution = "both_confirmed_conflict_abstain"
            chosen = None
        elif primary_confirmed:
            resolution = "end_coarse_candidate_confirmed"
            chosen = primary
        elif secondary_confirmed:
            resolution = "coarse_estimate_confirmed"
            chosen = secondary
        else:
            resolution = "neither_confirmed"
            chosen = None
        derived["conflict_resolution"] = resolution

        if chosen is None:
            if resolution == "both_confirmed_conflict_abstain":
                assert secondary is not None and secondary.verdict.end_s is not None
                assert primary.verdict.end_s is not None
                abstain_verdict = TimingVerdict.abstain(
                    reason_codes=("ambiguous_evidence",),
                    model_id=primary.verdict.model_id,
                    prompt_version=primary.verdict.prompt_version,
                    raw_notes=(
                        f"both the end-coarse candidate (confirmed at "
                        f"t={primary.verdict.end_s:.3f}s) and the coarse pass's own "
                        f"independent end estimate (confirmed at "
                        f"t={secondary.verdict.end_s:.3f}s) were independently "
                        f"validated as sustained trends - never confidently picking "
                        f"one over the other"
                    ),
                )
            else:
                # Rejected (trend didn't hold), couldn't be validated at all,
                # or - in the conflict case - neither anchor confirmed:
                # single-shot, fall back to the assisted/manual workflow
                # rather than hunting for yet another candidate
                # (supervisor-directed, see diagnostics/llm_spike/DESIGN.md).
                abstain_verdict = primary.verdict
            return PipelineOutcome(
                verdict=abstain_verdict,
                event=None,
                coarse_response=coarse_response,
                fine_response=fine_response,
                end_coarse_response=end_coarse_response,
                end_validation_response=primary.response,
                end_validation_coarse_cascade_response=primary.coarse_cascade_response,
                end_validation_conflict_response=(secondary.response if secondary else None),
                end_validation_conflict_coarse_cascade_response=(
                    secondary.coarse_cascade_response if secondary else None
                ),
                pass_verdicts=dict(pass_verdicts),
                pass_frames=dict(pass_frames),
                derived=dict(derived),
            )

        # Report the chosen cascade's own (possibly refined) onset, not
        # necessarily the end-coarse candidate that nominated the primary
        # sheet: the cascade is deliberately allowed to localize the true
        # onset anywhere within its own grounded panels,
        # and - when a conflict was resolved in the coarse estimate's
        # favor - the reported onset instead comes from that independently-
        # anchored window. The future-context guarantee for this exact
        # timestamp was already re-checked inside _run_end_validation_pass.
        assert chosen.verdict.end_s is not None
        derived["confirmed_end_source"] = (
            "end_coarse_candidate" if chosen is primary else "coarse_estimate"
        )
        final_confidence = min(
            fine_verdict.confidence, end_coarse_verdict.confidence, chosen.verdict.confidence
        )
        conflict_note = (
            (
                f"; conflicted with the coarse pass's own end estimate at "
                f"t={coarse_end_estimate_s:.3f}s (each outside the other's own "
                f"validation window), so both were independently validated"
            )
            if conflict
            else ""
        )
        source_note = (
            " (the coarse pass's own independent estimate, not the end-coarse candidate)"
            if chosen is not primary
            else ""
        )
        final_verdict = TimingVerdict(
            status=TimingStatus.CONFIRMED,
            start_s=locked_start_s,
            end_s=chosen.verdict.end_s,
            start_uncertainty_s=locked_start_uncertainty_s,
            end_uncertainty_s=chosen.verdict.end_uncertainty_s,
            confidence=final_confidence,
            reason_codes=tuple(
                sorted(
                    set(fine_verdict.reason_codes)
                    | set(end_coarse_verdict.reason_codes)
                    | set(chosen.verdict.reason_codes)
                )
            ),
            evidence_frame_timestamps_s=tuple(
                sorted(
                    set(start_side_evidence)
                    | set(end_coarse_verdict.evidence_frame_timestamps_s)
                    | set(chosen.verdict.evidence_frame_timestamps_s)
                )
            ),
            trend_checkpoint_timestamps_s=chosen.verdict.trend_checkpoint_timestamps_s,
            model_id=chosen.verdict.model_id,
            prompt_version=chosen.verdict.prompt_version,
            raw_notes=(
                f"start confirmed via {PROMPT_START_REFINE_V1_ID}; candidate break "
                f"nominated via {PROMPT_END_COARSE_V2_ID} at t={candidate_ts:.3f}s"
                f"{conflict_note}; refined and confirmed as a sustained trend via "
                f"{chosen.verdict.prompt_version} at t={chosen.verdict.end_s:.3f}s"
                f"{source_note}"
            ),
        )

        status = (
            EventStatus.CONFIRMED
            if final_verdict.confidence >= cfg.review_confidence
            else EventStatus.REVIEW
        )
        # The two decisive "evidence image" timestamps - see
        # _nearest_grounded_evidence_ts's own docstring. Computed from each
        # boundary's own grounded pass (fine for start, the chosen
        # validation pass for end), never fabricated: None when that pass
        # cited no evidence.
        start_evidence_s = _nearest_grounded_evidence_ts(
            start_side_evidence, pass_frames.get("fine", ()), locked_start_s
        )
        end_evidence_s = _nearest_grounded_evidence_ts(
            chosen.verdict.evidence_frame_timestamps_s,
            chosen.submitted_timestamps_s,
            chosen.verdict.end_s,
        )
        event = Event(
            label="Efflux (LLM spike)",
            start_s=locked_start_s,
            end_s=chosen.verdict.end_s,
            confidence=final_verdict.confidence,
            detector="llm_timing_spike",
            status=status,
            notes=final_verdict.reason_codes,
            details={
                "start_uncertainty_s": final_verdict.start_uncertainty_s,
                "end_uncertainty_s": final_verdict.end_uncertainty_s,
                "evidence_frame_timestamps_s": list(final_verdict.evidence_frame_timestamps_s),
                "trend_checkpoint_timestamps_s": list(final_verdict.trend_checkpoint_timestamps_s),
                "model_id": final_verdict.model_id,
                "start_prompt_version": PROMPT_START_REFINE_V1_ID,
                "end_coarse_prompt_version": PROMPT_END_COARSE_V2_ID,
                "end_validate_prompt_version": chosen.verdict.prompt_version,
                "coarse_start_s": coarse_verdict.start_s,
                "coarse_end_s": coarse_verdict.end_s,
                "end_coarse_candidate_s": candidate_ts,
                "coarse_end_estimate_s": coarse_end_estimate_s,
                "candidate_conflict": conflict,
                "conflict_resolution": resolution,
                "confirmed_end_source": derived["confirmed_end_source"],
                "end_validation_window_s": list(primary.window_s),
                "start_evidence_s": start_evidence_s,
                "end_evidence_s": end_evidence_s,
            },
        )
        return PipelineOutcome(
            verdict=final_verdict,
            event=event,
            coarse_response=coarse_response,
            fine_response=fine_response,
            end_coarse_response=end_coarse_response,
            end_validation_response=primary.response,
            end_validation_coarse_cascade_response=primary.coarse_cascade_response,
            end_validation_conflict_response=(secondary.response if secondary else None),
            end_validation_conflict_coarse_cascade_response=(
                secondary.coarse_cascade_response if secondary else None
            ),
            pass_verdicts=dict(pass_verdicts),
            pass_frames=dict(pass_frames),
            derived=dict(derived),
        )
