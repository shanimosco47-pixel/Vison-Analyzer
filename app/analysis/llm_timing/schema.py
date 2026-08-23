"""The strict contract an LLM timing provider must satisfy.

Per the supervisor's spike authorization (PR #4): conversational text is not
authoritative. A provider's raw output is untrusted until it has been parsed
into a :class:`TimingVerdict` and passed every invariant below - anything
that fails parsing or validation becomes an ``ABSTAIN`` verdict, never a
best-effort guess. This is the same "never confidently wrong" discipline the
classical tracker (``app/analysis/outlet_tracker.py``) was built under.

A verdict is data, not prose: nothing downstream should ever need to parse
free text to find out what the model concluded.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from ...errors import ConfigurationError


class TimingStatus(str, Enum):
    """How much trust this verdict is entitled to."""

    CONFIRMED = "confirmed"  # evidence met every bar; start_s/end_s are usable
    ABSTAIN = "abstain"  # provider failed, was malformed, or evidence was weak


# Known reason codes. This list documents the vocabulary the pipeline itself
# emits (schema/provider failures) and the codes the prompt asks the model to
# use. It is deliberately not a closed set enforced at parse time: an unknown
# code from the model is logged and kept, not treated as a parse failure, so
# a new failure mode surfaces as data instead of silently becoming "abstain,
# reason unknown".
PIPELINE_REASON_CODES = frozenset(
    {
        "provider_error",  # the API call itself failed (timeout, 5xx, network)
        "malformed_output",  # response did not parse against the schema
        "low_confidence",  # parsed fine, but confidence fell below the config floor
        "invalid_invariant",  # e.g. end_s < start_s, negative uncertainty
        # The four below are the request-grounding checks (Codex review,
        # finding 1): a verdict can satisfy every invariant above and still
        # be untethered from the actual video/frames it was asked about - a
        # model can hallucinate a finite-looking, internally consistent
        # answer that is simply about the wrong thing.
        "out_of_bounds",  # start_s/end_s outside the video, or outside the
        # window whose frames were actually sent for this pass
        "ungrounded_evidence",  # evidence_frame_timestamps_s empty, or none
        # of it corresponds to a frame actually submitted in this request
        "evidence_far_from_claim",  # evidence exists and is grounded, but
        # none of it sits near the claimed start_s/end_s
        "uncertainty_exceeds_cap",  # start/end uncertainty too large to
        # trust as CONFIRMED, even though it parsed and is non-negative
        "request_too_large",  # even the sparsest frame selection that still
        # meets the gate's precision floor would exceed the provider's
        # request-size budget; abstain rather than silently drop evidence
        # below that floor or send an oversized request the provider would
        # reject anyway (Codex review, round 2, finding 2)
        # The below is an end-validate-specific evidence-shape check (a real
        # audited run found a CONFIRMED end-validate verdict citing
        # essentially every submitted frame as "evidence" - a generic trend
        # assertion that satisfies grounding without showing which specific
        # checkpoints actually prove a sustained trend, see
        # diagnostics/llm_spike/DESIGN.md):
        "evidence_not_selective",  # more evidence timestamps cited than the
        # selective-checkpoint cap allows - citing "all of it" is not
        # grounding, it is the absence of grounding
    }
)

MODEL_REASON_CODES = frozenset(
    {
        "weak_contrast",  # stream/cup/background too close in intensity to be sure
        "camera_motion",  # handheld shake made the evidence window ambiguous
        "no_continuous_stream_found",  # never saw a qualifying continuous stream
        "no_break_found",  # stream never resolved to a clear break before video end
        "ambiguous_evidence",  # coarse+fine passes did not converge on one frame
        "resumed_flow",  # stream broke and reconnected; genuinely ambiguous per spec
        "missing_outlet",  # cup outlet not visible/identifiable in frame
    }
)


@dataclass(frozen=True)
class TimingVerdict:
    """A parsed, validated answer from a timing provider for one video.

    Attributes:
        status: CONFIRMED or ABSTAIN. Every other field is populated either
            way, so a caller does not need to branch on status just to log.
        start_s: video-relative timestamp flow was judged to start. ``None``
            only when ``status`` is ABSTAIN.
        end_s: video-relative timestamp flow was judged to end. ``None`` only
            when ``status`` is ABSTAIN.
        start_uncertainty_s: provider's own +/- bound on ``start_s``, seconds.
        end_uncertainty_s: provider's own +/- bound on ``end_s``, seconds.
        confidence: 0..1, the provider's self-reported confidence. Distinct
            from ``status``: a low-confidence CONFIRMED verdict is possible
            when explicitly requested; the pipeline is what turns "low
            confidence" into ABSTAIN via ``min_confidence``.
        reason_codes: machine-readable codes explaining the verdict. Always
            non-empty for ABSTAIN. May be empty for a clean CONFIRMED.
        evidence_frame_timestamps_s: timestamps of the frames the provider
            says it examined around its decision, for audit/replay.
        model_id: identifier of the model/version that produced this verdict.
        prompt_version: identifier of the prompt template used (see
            ``prompts.py``). Pinned so a later prompt edit cannot silently
            change what an old logged verdict means.
        raw_notes: free-text explanation, kept for the audit trail only.
            Never read programmatically - see the module docstring.
    """

    status: TimingStatus
    start_s: float | None
    end_s: float | None
    start_uncertainty_s: float
    end_uncertainty_s: float
    confidence: float
    reason_codes: tuple[str, ...]
    evidence_frame_timestamps_s: tuple[float, ...]
    model_id: str
    prompt_version: str
    raw_notes: str = ""

    def __post_init__(self) -> None:
        # Every numeric field is untrusted (it may have come straight from a
        # model's JSON output, which - unlike normal application input - can
        # contain NaN/Infinity: Python's json module accepts those as an
        # extension). A comparison against NaN is always False, so a naive
        # range check silently lets NaN through; explicit isfinite() checks
        # close that gap for every numeric field, not just confidence.
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ConfigurationError(
                "A timing verdict's confidence must be a finite number between 0 and 1.",
                detail=f"confidence={self.confidence!r}",
            )
        if not math.isfinite(self.start_uncertainty_s) or not math.isfinite(self.end_uncertainty_s):
            raise ConfigurationError(
                "A timing verdict's uncertainty bounds must be finite numbers.",
                detail=(
                    f"start_uncertainty_s={self.start_uncertainty_s!r} "
                    f"end_uncertainty_s={self.end_uncertainty_s!r}"
                ),
            )
        if self.start_uncertainty_s < 0.0 or self.end_uncertainty_s < 0.0:
            raise ConfigurationError(
                "A timing verdict's uncertainty bounds must not be negative.",
                detail=(
                    f"start_uncertainty_s={self.start_uncertainty_s!r} "
                    f"end_uncertainty_s={self.end_uncertainty_s!r}"
                ),
            )
        if self.status is TimingStatus.CONFIRMED:
            if self.start_s is None or self.end_s is None:
                raise ConfigurationError(
                    "A CONFIRMED timing verdict must have both start_s and end_s."
                )
            if not math.isfinite(self.start_s) or not math.isfinite(self.end_s):
                raise ConfigurationError(
                    "A CONFIRMED timing verdict's start_s and end_s must be finite numbers.",
                    detail=f"start_s={self.start_s!r} end_s={self.end_s!r}",
                )
            if self.end_s < self.start_s:
                raise ConfigurationError(
                    "A timing verdict cannot end before it starts.",
                    detail=f"start_s={self.start_s!r} end_s={self.end_s!r}",
                )
        else:  # ABSTAIN
            if not self.reason_codes:
                raise ConfigurationError(
                    "An ABSTAIN timing verdict must carry at least one reason code."
                )

    @property
    def duration_s(self) -> float | None:
        if self.status is not TimingStatus.CONFIRMED:
            return None
        assert self.start_s is not None and self.end_s is not None
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        return {
            "status": self.status.value,
            "start_s": self.start_s,
            "end_s": self.end_s,
            "duration_s": self.duration_s,
            "start_uncertainty_s": self.start_uncertainty_s,
            "end_uncertainty_s": self.end_uncertainty_s,
            "confidence": round(self.confidence, 4),
            "reason_codes": list(self.reason_codes),
            "evidence_frame_timestamps_s": list(self.evidence_frame_timestamps_s),
            "model_id": self.model_id,
            "prompt_version": self.prompt_version,
            "raw_notes": self.raw_notes,
        }

    @staticmethod
    def abstain(
        *,
        reason_codes: tuple[str, ...],
        model_id: str,
        prompt_version: str,
        confidence: float = 0.0,
        raw_notes: str = "",
    ) -> TimingVerdict:
        """Build the abstain verdict every failure path converges on."""
        return TimingVerdict(
            status=TimingStatus.ABSTAIN,
            start_s=None,
            end_s=None,
            start_uncertainty_s=0.0,
            end_uncertainty_s=0.0,
            confidence=confidence,
            reason_codes=reason_codes,
            evidence_frame_timestamps_s=(),
            model_id=model_id,
            prompt_version=prompt_version,
            raw_notes=raw_notes,
        )
