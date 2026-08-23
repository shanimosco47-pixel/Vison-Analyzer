"""Tests for the LLM timing spike (app/analysis/llm_timing).

This spike is not wired into the detector registry or the web layer (see the
package docstring and diagnostics/llm_spike/DESIGN.md), so these tests run
entirely offline against StubTimingProvider - no network, no API key. They
exist to prove the *scaffolding* is trustworthy: schema invariants hold,
every provider failure mode converges on ABSTAIN, and the coarse-to-fine
pipeline never fabricates an Event when it shouldn't.

They intentionally do NOT prove the spike measures real footage accurately -
that is evidence gate 1 in the design doc, and requires a real provider and
the two hand-verified real clips, neither of which exist in this repo.
"""

from __future__ import annotations

import pytest

from app.analysis.base_detector import EventStatus
from app.analysis.llm_timing.pipeline import (
    PipelineConfig,
    _nearest_grounded_evidence_ts,
    run_llm_timing,
)
from app.analysis.llm_timing.prompts import PROMPT_END_COARSE_V2_ID
from app.analysis.llm_timing.provider import (
    ProviderRequest,
    RawProviderResponse,
    StubTimingProvider,
    canned_json_response,
    parse_raw_response,
)
from app.analysis.llm_timing.schema import TimingStatus, TimingVerdict
from app.errors import ConfigurationError

PROMPT_VERSION = "test-prompt-v1"


def _candidate_timestamp(request: ProviderRequest) -> float:
    """The timestamp of the frame the pipeline marked as the trend-
    validation candidate (see ``TimedFrame.is_candidate``) - every
    "end_validate" request has exactly one."""
    for frame in request.frames:
        if frame.is_candidate:
            return frame.timestamp_s
    raise AssertionError("end_validate request missing a candidate-marked frame")


def _confirm_validation(
    request: ProviderRequest, *, confidence: float = 0.9
) -> RawProviderResponse:
    """A perfect trend-validation stub response: always validates whichever
    candidate the pipeline flagged. Good enough for tests whose synthetic
    fixture has a single genuine, sustained break with no recovery - the
    dedicated trend-validation regressions below script something more
    specific."""
    ts = _candidate_timestamp(request)
    return canned_json_response(
        start_s=ts, end_s=ts, confidence=confidence, evidence_frame_timestamps_s=(ts,)
    )


# --------------------------------------------------------------------------- #
# schema.TimingVerdict invariants
# --------------------------------------------------------------------------- #


def test_confirmed_verdict_requires_start_and_end():
    with pytest.raises(ConfigurationError):
        TimingVerdict(
            status=TimingStatus.CONFIRMED,
            start_s=None,
            end_s=5.0,
            start_uncertainty_s=0.1,
            end_uncertainty_s=0.1,
            confidence=0.9,
            reason_codes=(),
            evidence_frame_timestamps_s=(),
            model_id="m",
            prompt_version=PROMPT_VERSION,
        )


def test_confirmed_verdict_rejects_end_before_start():
    with pytest.raises(ConfigurationError):
        TimingVerdict(
            status=TimingStatus.CONFIRMED,
            start_s=5.0,
            end_s=1.0,
            start_uncertainty_s=0.1,
            end_uncertainty_s=0.1,
            confidence=0.9,
            reason_codes=(),
            evidence_frame_timestamps_s=(),
            model_id="m",
            prompt_version=PROMPT_VERSION,
        )


def test_abstain_verdict_requires_a_reason_code():
    with pytest.raises(ConfigurationError):
        TimingVerdict.abstain(reason_codes=(), model_id="m", prompt_version=PROMPT_VERSION)


def test_confidence_out_of_range_rejected():
    with pytest.raises(ConfigurationError):
        TimingVerdict(
            status=TimingStatus.ABSTAIN,
            start_s=None,
            end_s=None,
            start_uncertainty_s=0.0,
            end_uncertainty_s=0.0,
            confidence=1.5,
            reason_codes=("no_break_found",),
            evidence_frame_timestamps_s=(),
            model_id="m",
            prompt_version=PROMPT_VERSION,
        )


def test_duration_is_none_when_abstained():
    verdict = TimingVerdict.abstain(
        reason_codes=("no_break_found",), model_id="m", prompt_version=PROMPT_VERSION
    )
    assert verdict.duration_s is None


# --------------------------------------------------------------------------- #
# provider.parse_raw_response: every failure mode -> ABSTAIN, never a guess
# --------------------------------------------------------------------------- #


def test_parse_well_formed_confirmed_response():
    response = canned_json_response(start_s=4.0, end_s=20.6, confidence=0.9)
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.CONFIRMED
    assert verdict.start_s == 4.0
    assert verdict.end_s == 20.6
    assert verdict.duration_s == pytest.approx(16.6)


def test_parse_provider_error_abstains():
    response = RawProviderResponse(
        model_id="m", raw_text="", latency_s=0.0, error="timeout after 30s"
    )
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "provider_error" in verdict.reason_codes


def test_parse_non_json_abstains():
    response = RawProviderResponse(model_id="m", raw_text="not json at all", latency_s=0.0)
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "malformed_output" in verdict.reason_codes


def test_parse_missing_required_field_abstains():
    response = RawProviderResponse(
        model_id="m", raw_text='{"status": "confirmed", "confidence": 0.9}', latency_s=0.0
    )
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "malformed_output" in verdict.reason_codes


def test_parse_end_before_start_abstains_not_raises():
    response = RawProviderResponse(
        model_id="m",
        raw_text='{"status": "confirmed", "start_s": 10.0, "end_s": 2.0, "confidence": 0.9}',
        latency_s=0.0,
    )
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "invalid_invariant" in verdict.reason_codes


def test_parse_below_confidence_floor_abstains():
    response = canned_json_response(start_s=4.0, end_s=20.6, confidence=0.3)
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "low_confidence" in verdict.reason_codes


def test_parse_model_requested_abstain_preserves_reason_codes():
    response = RawProviderResponse(
        model_id="m",
        raw_text='{"status": "abstain", "reason_codes": ["weak_contrast", "camera_motion"]}',
        latency_s=0.0,
    )
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert set(verdict.reason_codes) == {"weak_contrast", "camera_motion"}


# --------------------------------------------------------------------------- #
# parse_raw_response must be total: NaN/Infinity/non-numeric never raise,
# in either a "confirmed" or an "abstain" payload (Codex review, finding 2)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw_text",
    [
        '{"status": "confirmed", "start_s": 4.0, "end_s": 20.6, "confidence": NaN}',
        '{"status": "confirmed", "start_s": 4.0, "end_s": 20.6, "confidence": Infinity}',
        '{"status": "confirmed", "start_s": NaN, "end_s": 20.6, "confidence": 0.9}',
        '{"status": "confirmed", "start_s": 4.0, "end_s": Infinity, "confidence": 0.9}',
        '{"status": "confirmed", "start_s": "not-a-number", "end_s": 20.6, "confidence": 0.9}',
        '{"status": "confirmed", "start_s": 4.0, "end_s": 20.6, "confidence": "high"}',
        '{"status": "confirmed", "start_s": 4.0, "end_s": 20.6, "confidence": 0.9, '
        '"start_uncertainty_s": NaN}',
        '{"status": "confirmed", "start_s": 4.0, "end_s": 20.6, "confidence": 0.9, '
        '"reason_codes": "not-a-list"}',
    ],
)
def test_parse_never_raises_on_confirmed_payload_garbage(raw_text):
    response = RawProviderResponse(model_id="m", raw_text=raw_text, latency_s=0.0)
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN


@pytest.mark.parametrize(
    "raw_text",
    [
        '{"status": "abstain", "confidence": NaN}',
        '{"status": "abstain", "confidence": Infinity}',
        '{"status": "abstain", "confidence": "not-a-number"}',
        '{"status": "abstain", "reason_codes": ["ok"], "confidence": -Infinity}',
    ],
)
def test_parse_never_raises_on_abstain_payload_garbage(raw_text):
    response = RawProviderResponse(model_id="m", raw_text=raw_text, latency_s=0.0)
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN


# --------------------------------------------------------------------------- #
# pipeline.run_llm_timing: coarse-to-fine orchestration against a real video
# --------------------------------------------------------------------------- #


def _stub_matching_truth(
    truth: dict[str, float], *, fine_confidence: float = 0.9, end_coarse_confidence: float = 0.9
):
    """A StubTimingProvider that answers accurately on every pass: the
    single whole-clip end-coarse call nominates the true break, and the
    trend-validation follow-up confirms it - the same shape a real model's
    answers take under the two-stage end-determination design."""

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=truth["flow_start_s"], end_s=truth["flow_end_s"], confidence=0.8
            )
        if request.pass_name == "fine":
            # The fine pass is start-only now: it reports the confirmed
            # start as a degenerate point (start_s == end_s), never the
            # (now-irrelevant) end.
            return canned_json_response(
                start_s=truth["flow_start_s"],
                end_s=truth["flow_start_s"],
                confidence=fine_confidence,
                evidence_frame_timestamps_s=(truth["flow_start_s"],),
            )
        if request.pass_name == "end_validate":
            return _confirm_validation(request)
        assert request.pass_name == "end_coarse"
        window_times = [f.timestamp_s for f in request.frames]
        end_s = truth["flow_end_s"]
        if window_times and min(window_times) <= end_s <= max(window_times):
            return canned_json_response(
                start_s=end_s,
                end_s=end_s,
                confidence=end_coarse_confidence,
                evidence_frame_timestamps_s=(end_s,),
            )
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    return StubTimingProvider(respond)


def test_pipeline_confirms_event_matching_ground_truth(zahn_video):
    provider = _stub_matching_truth(zahn_video.truth)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.status is EventStatus.CONFIRMED
    assert outcome.event.start_s == pytest.approx(zahn_video.truth["flow_start_s"])
    assert outcome.event.end_s == pytest.approx(zahn_video.truth["flow_end_s"])
    # coarse, fine (start), one whole-clip end-coarse candidate nomination,
    # one trend-validation confirmation - exactly four calls total, not N
    # chronological scan windows.
    assert [c.pass_name for c in provider.calls] == [
        "coarse",
        "fine",
        "end_coarse",
        "end_validate",
    ]
    assert len(provider.calls[1].frames) > len(provider.calls[0].frames)
    # The end-coarse request's frames actually contain the true break.
    end_coarse_times = [f.timestamp_s for f in provider.calls[2].frames]
    assert min(end_coarse_times) <= zahn_video.truth["flow_end_s"] <= max(end_coarse_times)
    assert outcome.end_coarse_response is not None
    assert outcome.end_validation_response is not None
    assert outcome.end_coarse_response.model_id == outcome.verdict.model_id


def test_pipeline_marks_low_but_passing_confidence_as_review(zahn_video):
    provider = _stub_matching_truth(zahn_video.truth, fine_confidence=0.6)
    config = PipelineConfig(min_confidence=0.5, review_confidence=0.75)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        config=config,
    )
    assert outcome.event is not None
    assert outcome.event.status is EventStatus.REVIEW


def test_pipeline_abstains_when_coarse_pass_abstains(zahn_video):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        return RawProviderResponse(
            model_id="m",
            raw_text='{"status": "abstain", "reason_codes": ["no_continuous_stream_found"]}',
            latency_s=0.0,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert outcome.fine_response is None  # never got that far
    assert len(provider.calls) == 1


def test_pipeline_abstains_when_fine_pass_abstains_after_confirmed_coarse(zahn_video):
    calls = {"n": 0}

    def respond(request: ProviderRequest) -> RawProviderResponse:
        calls["n"] += 1
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=zahn_video.truth["flow_start_s"],
                end_s=zahn_video.truth["flow_end_s"],
                confidence=0.8,
            )
        return RawProviderResponse(
            model_id="m",
            raw_text='{"status": "abstain", "reason_codes": ["ambiguous_evidence"]}',
            latency_s=0.0,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert outcome.fine_response is not None  # it did run, and then abstained
    assert calls["n"] == 2


def test_pipeline_abstains_on_malformed_provider_output_without_crashing(zahn_video):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        return RawProviderResponse(model_id="m", raw_text="<html>not json</html>", latency_s=0.0)

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "malformed_output" in outcome.verdict.reason_codes


def test_pipeline_abstains_when_coarse_window_is_implausibly_wide(zahn_video):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        # The coarse pass itself reports huge uncertainty on the start
        # boundary - too wide a window to fine-scan densely.
        return canned_json_response(
            start_s=zahn_video.truth["flow_start_s"],
            end_s=zahn_video.truth["flow_end_s"],
            start_uncertainty_s=15.0,
            confidence=0.8,
        )

    provider = StubTimingProvider(respond)
    # max_uncertainty_s raised so this test isolates the oversized-window
    # path from the separate uncertainty-cap path (see
    # test_pipeline_abstains_when_uncertainty_exceeds_cap).
    config = PipelineConfig(fine_max_span_s=6.0, max_uncertainty_s=20.0)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        config=config,
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "ambiguous_evidence" in outcome.verdict.reason_codes
    assert len(provider.calls) == 1  # never sent the (would-be enormous) fine batch


def test_pipeline_abstains_when_uncertainty_exceeds_cap(zahn_video):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        return canned_json_response(
            start_s=zahn_video.truth["flow_start_s"],
            end_s=zahn_video.truth["flow_end_s"],
            start_uncertainty_s=15.0,
            confidence=0.8,
        )

    provider = StubTimingProvider(respond)
    config = PipelineConfig(max_uncertainty_s=5.0)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        config=config,
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "uncertainty_exceeds_cap" in outcome.verdict.reason_codes
    assert len(provider.calls) == 1  # rejected at the coarse pass, no fine call


# --------------------------------------------------------------------------- #
# _validate_grounding via run_llm_timing: a CONFIRMED verdict that parses
# cleanly but isn't tethered to the actual request must still abstain
# (Codex review, finding 1)
# --------------------------------------------------------------------------- #


def _stub_coarse_ok_fine_custom(truth, fine_response: RawProviderResponse):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=truth["flow_start_s"], end_s=truth["flow_end_s"], confidence=0.8
            )
        return fine_response

    return StubTimingProvider(respond)


def test_pipeline_abstains_when_fine_start_is_outside_its_window(zahn_video):
    # The fine window is only ~fine_margin_s wide around the coarse
    # estimate (default 1.5s); 0.0 is nowhere near a true start_s of 4.0.
    fine_response = canned_json_response(start_s=0.0, end_s=0.0, confidence=0.9)
    provider = _stub_coarse_ok_fine_custom(zahn_video.truth, fine_response)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "out_of_bounds" in outcome.verdict.reason_codes


def test_pipeline_abstains_when_evidence_is_empty(zahn_video):
    fine_response = canned_json_response(
        start_s=zahn_video.truth["flow_start_s"],
        end_s=zahn_video.truth["flow_start_s"],
        confidence=0.9,
        evidence_frame_timestamps_s=(),
    )
    provider = _stub_coarse_ok_fine_custom(zahn_video.truth, fine_response)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "ungrounded_evidence" in outcome.verdict.reason_codes


def test_pipeline_abstains_when_evidence_matches_no_submitted_frame(zahn_video):
    fine_response = canned_json_response(
        start_s=zahn_video.truth["flow_start_s"],
        end_s=zahn_video.truth["flow_start_s"],
        confidence=0.9,
        # A timestamp nowhere near anything the pipeline actually extracted
        # and sent - a fabricated citation.
        evidence_frame_timestamps_s=(9999.0,),
    )
    provider = _stub_coarse_ok_fine_custom(zahn_video.truth, fine_response)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "ungrounded_evidence" in outcome.verdict.reason_codes


def test_pipeline_abstains_when_evidence_is_grounded_but_far_from_the_claim(zahn_video):
    # start_lo is roughly flow_start_s - fine_margin_s = 4.0 - 1.5 = 2.5;
    # this is a real, submittable frame timestamp, but nowhere near the
    # claimed start_s (4.0).
    fine_response = canned_json_response(
        start_s=zahn_video.truth["flow_start_s"],
        end_s=zahn_video.truth["flow_start_s"],
        confidence=0.9,
        evidence_frame_timestamps_s=(2.5,),
    )
    provider = _stub_coarse_ok_fine_custom(zahn_video.truth, fine_response)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "evidence_far_from_claim" in outcome.verdict.reason_codes


# --------------------------------------------------------------------------- #
# Two-stage end determination: one whole-clip (post-start), sparsely
# sampled end-coarse request nominates a single candidate, then one dense
# trend-validation follow-up confirms or rejects it - replacing an earlier
# chronological multi-window scan entirely (supervisor-directed, after a
# real gate-1 experiment showed the chronological scan's own decomposition
# into many narrow windows lost the temporal context needed to judge a
# sustained trend and wrongly rejected the true break; one coherent
# request with the same trend contract succeeded). Target: 2-3 total
# provider calls for the whole end determination, not N scan windows. A
# rejected/unvalidatable candidate now falls straight through to ABSTAIN
# (assisted/manual fallback) - single-shot, not a search.
# --------------------------------------------------------------------------- #


def test_pipeline_abstains_when_end_coarse_finds_no_candidate(zahn_video):
    """The stream stays continuous (or the model never sees a genuine
    break) - the single end-coarse request abstains, and the whole run
    converges on ABSTAIN immediately, never a fabricated end, and never
    even reaching the trend-validation call."""

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=21.5, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "no_break_found" in outcome.verdict.reason_codes
    assert [c.pass_name for c in provider.calls] == ["coarse", "fine", "end_coarse"]
    assert outcome.end_coarse_response is not None
    assert outcome.end_validation_response is None  # never reached - no candidate to validate
    # The single end-coarse request's sparse frames span all the way to the
    # end of the clip - nothing to find, so it never had to ask again.
    end_coarse_times = [f.timestamp_s for f in provider.calls[-1].frames]
    assert max(end_coarse_times) == pytest.approx(zahn_video.duration_s, abs=0.1)


def test_pipeline_start_confirmation_is_independent_of_a_stale_coarse_end_estimate(zahn_video):
    """Reproduces the shape of a real gate-1 failure: the coarse pass's own
    end estimate is stale/wrong, close to the end of the clip - exactly the
    shape that, under the old combined start+end fine-window design, built
    an unusable fine end window (e.g. ~[25.500, 27.031]s) whose invalid
    end_s could abstain the *whole* run before the end determination ever
    got to run, even though that end answer was never actually used
    downstream. The fine pass no longer builds or asks about an end window
    at all, so a bad coarse end estimate can't touch start confirmation,
    and the end-coarse/end-validate pair still runs and finds the true
    break (supervisor-directed fix, see diagnostics/llm_spike/DESIGN.md)."""
    stale_coarse_end_s = 25.9  # nowhere near the true end (21.5)

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=zahn_video.truth["flow_start_s"], end_s=stale_coarse_end_s, confidence=0.8
            )
        if request.pass_name == "fine":
            return canned_json_response(
                start_s=zahn_video.truth["flow_start_s"],
                end_s=zahn_video.truth["flow_start_s"],
                confidence=0.9,
            )
        if request.pass_name == "end_validate":
            return _confirm_validation(request)
        assert request.pass_name == "end_coarse"
        window_times = [f.timestamp_s for f in request.frames]
        true_end_s = zahn_video.truth["flow_end_s"]
        if window_times and min(window_times) <= true_end_s <= max(window_times):
            return canned_json_response(
                start_s=true_end_s,
                end_s=true_end_s,
                confidence=0.9,
                evidence_frame_timestamps_s=(true_end_s,),
            )
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.start_s == pytest.approx(zahn_video.truth["flow_start_s"])
    assert outcome.event.end_s == pytest.approx(zahn_video.truth["flow_end_s"])
    # The fine request never covered anything near the coarse pass's (bad)
    # end estimate - it only ever spans the start window.
    fine_call = provider.calls[1]
    assert fine_call.pass_name == "fine"
    fine_times = [f.timestamp_s for f in fine_call.frames]
    assert max(fine_times) < 10.0


def test_pipeline_end_coarse_never_considers_the_onset_transition_as_a_candidate_break(zahn_video):
    """Reproduces a real gate-1 failure: a third real rerun (commit
    ``894dfe6``) got a CONFIRMED start of ~4.033s (correct) but a
    false-CONFIRMED end of ~4.666s from the *first* chronological end-scan
    window, ``[4.03, 7.03]`` - the model misread the stream's own onset
    transition (nothing visible -> stream visible), right next to the
    confirmed start, as if it were a break. The same structural fix applies
    to the current single end-coarse request: its frames begin no earlier
    than ``start_hi``, the far edge of the grounded start-refinement
    window, so an onset timestamp is never even eligible to be submitted
    as a candidate end - a stub that is (unrealistically) willing to
    falsely confirm right next to the true start must never get the
    chance, and the later, true break must still be found."""
    false_onset_break_s = zahn_video.truth["flow_start_s"] + 0.633  # ~4.666s
    true_break_s = 15.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=zahn_video.truth["flow_start_s"], end_s=true_break_s, confidence=0.9
            )
        if request.pass_name == "fine":
            return canned_json_response(
                start_s=zahn_video.truth["flow_start_s"],
                end_s=zahn_video.truth["flow_start_s"],
                confidence=0.9,
            )
        if request.pass_name == "end_validate":
            return _confirm_validation(request)
        assert request.pass_name == "end_coarse"
        window_times = [f.timestamp_s for f in request.frames]
        for candidate in (false_onset_break_s, true_break_s):
            if window_times and min(window_times) <= candidate <= max(window_times):
                return canned_json_response(
                    start_s=candidate,
                    end_s=candidate,
                    confidence=0.9,
                    evidence_frame_timestamps_s=(candidate,),
                )
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.start_s == pytest.approx(zahn_video.truth["flow_start_s"])
    assert outcome.event.end_s == pytest.approx(true_break_s)
    # The end-coarse request never included the onset region - the false
    # break was never even a reachable candidate, not merely one that lost
    # to a later grounded confirmation.
    end_coarse_call = next(c for c in provider.calls if c.pass_name == "end_coarse")
    window_times = [f.timestamp_s for f in end_coarse_call.frames]
    assert min(window_times) > false_onset_break_s


def test_pipeline_abstains_when_the_only_candidate_fails_trend_validation(zahn_video):
    """A candidate break that looked plausible from the sparse end-coarse
    pass but fully recovers back toward the established baseline
    afterward - a visual/camera/contrast artifact, not a real break - is
    rejected by trend validation. The two-stage design is single-shot, not
    a search: a rejected candidate falls straight through to ABSTAIN (the
    assisted/manual fallback), it does not hunt for a different one."""
    false_candidate_s = 10.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=false_candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            # Shortened briefly, then recovered back to the baseline reach
            # and stayed there - reject.
            return RawProviderResponse(
                model_id="stub-model",
                raw_text='{"status": "abstain", "reason_codes": ["trend_not_sustained"]}',
                latency_s=0.01,
            )
        assert request.pass_name == "end_coarse"
        return canned_json_response(
            start_s=false_candidate_s,
            end_s=false_candidate_s,
            confidence=0.9,
            evidence_frame_timestamps_s=(false_candidate_s,),
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "trend_not_sustained" in outcome.verdict.reason_codes
    # Single-shot: exactly one end-coarse call and one validation call -
    # never a second attempt at a different candidate.
    assert [c.pass_name for c in provider.calls] == [
        "coarse",
        "fine",
        "end_coarse",
        "end_validate",
    ]


def test_pipeline_clamps_the_validation_window_to_locked_start_s_for_an_early_candidate(
    zahn_video,
):
    """A candidate close enough to the confirmed start that
    ``candidate_ts - end_validation_pre_s`` would fall before it must have
    its validation window clamped at ``locked_start_s``, never reaching
    back before the confirmed start of the stream itself."""
    # locked_start_s is 4.0 (default fine_margin_s=1.5 puts start_hi at
    # 5.5) - scan_from_s is 5.5, so the earliest reachable candidate is
    # just past that; end_validation_pre_s=4.0 would reach back to 1.5s,
    # before the confirmed start, without the clamp.
    early_candidate_s = 6.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=early_candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(
                start_s=4.0, end_s=4.0, confidence=0.9, evidence_frame_timestamps_s=(4.0,)
            )
        if request.pass_name == "end_validate":
            return _confirm_validation(request)
        assert request.pass_name == "end_coarse"
        return canned_json_response(
            start_s=early_candidate_s,
            end_s=early_candidate_s,
            confidence=0.9,
            evidence_frame_timestamps_s=(early_candidate_s,),
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    validate_call = next(c for c in provider.calls if c.pass_name == "end_validate")
    validate_times = [f.timestamp_s for f in validate_call.frames]
    assert min(validate_times) == pytest.approx(4.0, abs=0.1)  # clamped, not 6.0 - 4.0 = 2.0


def test_pipeline_clamps_the_validation_window_to_the_clip_end_and_still_confirms(zahn_video):
    """A candidate close to the end of the clip gets a validation window
    clamped at ``duration_s`` (never past the clip), but the request is
    still sent - unlike the earlier pre-flight design, clip-end clamping
    alone is no longer a reason to abstain before even asking. If the
    refined onset reported still has enough of *this clamped window's own*
    future evidence after it, the run confirms normally."""
    # duration_s is 26.0; end_validation_post_s defaults to 6.0, so an
    # unclamped window would reach to 29.0s - past the clip - and must be
    # clamped to 26.0 instead.
    near_end_candidate_s = 23.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=near_end_candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(
                start_s=4.0, end_s=4.0, confidence=0.9, evidence_frame_timestamps_s=(4.0,)
            )
        if request.pass_name == "end_validate":
            return _confirm_validation(request)
        assert request.pass_name == "end_coarse"
        return canned_json_response(
            start_s=near_end_candidate_s,
            end_s=near_end_candidate_s,
            confidence=0.9,
            evidence_frame_timestamps_s=(near_end_candidate_s,),
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.end_s == pytest.approx(near_end_candidate_s)
    validate_call = next(c for c in provider.calls if c.pass_name == "end_validate")
    validate_times = [f.timestamp_s for f in validate_call.frames]
    assert max(validate_times) == pytest.approx(zahn_video.duration_s, abs=0.1)  # clamped to 26.0


def test_pipeline_sends_validation_then_abstains_when_clip_end_clamping_leaves_no_future_room(
    zahn_video,
):
    """The mandatory future-evidence rule is now checked against the
    validation window's own actual (possibly clip-clamped) upper bound,
    after the response comes back - not as a pre-flight refusal to even
    ask, per the old design. A candidate near enough to the clip's own end
    that even the clamped window leaves no room for the refined onset's
    required future evidence must still abstain, but only after the
    validation request was actually sent."""
    # duration_s is 26.0 - a candidate at 25.0s clamps the window to
    # [21.0, 26.0]s (1.0s of nominal future room, well under the 2.0s
    # end_validation_min_future_s floor if the model reports the candidate
    # itself as the onset).
    near_end_candidate_s = 25.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=near_end_candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(
                start_s=4.0, end_s=4.0, confidence=0.9, evidence_frame_timestamps_s=(4.0,)
            )
        if request.pass_name == "end_validate":
            return _confirm_validation(request)
        assert request.pass_name == "end_coarse"
        return canned_json_response(
            start_s=near_end_candidate_s,
            end_s=near_end_candidate_s,
            confidence=0.9,
            evidence_frame_timestamps_s=(near_end_candidate_s,),
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "insufficient_future_context" in outcome.verdict.reason_codes
    # Unlike the old pre-flight design, the validation request WAS sent -
    # only the response's own reported onset was rejected.
    assert [c.pass_name for c in provider.calls] == [
        "coarse",
        "fine",
        "end_coarse",
        "end_validate",
    ]
    assert outcome.end_validation_response is not None


def test_pipeline_reaches_the_true_break_in_one_call_when_end_coarse_avoids_the_transient(
    zahn_video,
):
    """Reproduces the shape of a real whole-clip rerun (commit 27587f5):
    the end-coarse pass nominated a transient early shortening (18.5s)
    that recovered toward baseline, which trend validation correctly
    rejected, leaving the true break (~20.5s) unreachable in the
    single-shot design. PROMPT_END_COARSE_V2 fixes this at the prompt
    level only - no pipeline change was needed, since the whole sparse
    batch was already visible to the model in one request. This proves
    the plumbing itself needs nothing further: a single end-coarse call
    that reports the true, trend-confirmed candidate (not the transient)
    still reaches CONFIRMED in exactly one end-coarse call and one
    validation call, and the pipeline is wired to the live V2 prompt."""
    true_break_s = 20.5

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=true_break_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            return _confirm_validation(request)
        assert request.pass_name == "end_coarse"
        assert request.prompt_version == PROMPT_END_COARSE_V2_ID
        # A compliant model, per V2's rule, rejects the 18.5s transient
        # (its own later sparse checkpoints recover toward baseline) and
        # nominates the earliest trend-confirmed candidate instead.
        return canned_json_response(
            start_s=true_break_s,
            end_s=true_break_s,
            confidence=0.9,
            evidence_frame_timestamps_s=(true_break_s,),
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.end_s == pytest.approx(true_break_s)
    assert [c.pass_name for c in provider.calls] == [
        "coarse",
        "fine",
        "end_coarse",
        "end_validate",
    ]


def test_pipeline_accepts_a_validation_refined_onset_earlier_than_the_sparse_candidate(
    zahn_video,
):
    """Reproduces the shape of a real whole-clip rerun (commit 86c083d):
    the sparse end-coarse pass nominated 21.5s, a second too late; the
    dense validation window - which already reaches back to
    ``candidate - end_validation_pre_s`` by design - contained the true
    onset at 20.5s. PROMPT_END_VALIDATE_V2 is explicitly allowed to report
    that earlier, better-supported timestamp instead of only confirming
    or rejecting the sparse candidate verbatim, and the pipeline must
    report the validation pass's own (refined) answer, not the coarse
    candidate that only nominated the window to search."""
    sparse_candidate_s = 21.5
    refined_onset_s = 20.5  # comfortably within [17.5, 26.0] at the default 4.0s pre-margin

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=sparse_candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            return canned_json_response(
                start_s=refined_onset_s,
                end_s=refined_onset_s,
                confidence=0.9,
                evidence_frame_timestamps_s=(refined_onset_s,),
            )
        assert request.pass_name == "end_coarse"
        return canned_json_response(
            start_s=sparse_candidate_s,
            end_s=sparse_candidate_s,
            confidence=0.9,
            evidence_frame_timestamps_s=(sparse_candidate_s,),
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.end_s == pytest.approx(refined_onset_s)
    assert outcome.event.end_s != pytest.approx(sparse_candidate_s)
    # The original sparse nomination stays visible for audit, distinct
    # from the final refined answer.
    assert outcome.event.details["end_coarse_candidate_s"] == pytest.approx(sparse_candidate_s)


def test_pipeline_reaches_a_break_the_old_narrow_validation_window_could_not_reach(zahn_video):
    """Reproduces the exact shape of a real whole-clip rerun (commit
    b42bb9a): the sparse end-coarse pass nominated 16.5s - now ~4s
    *before* the true break (~20.5s), rather than after it. Under the old
    point-anchored window (``candidate_ts - 1.0s`` to
    ``candidate_ts + 2.0s``, i.e. ``[15.5, 18.5]``), the true break was
    structurally unreachable - not merely missed, but nowhere in the
    request at all - so even a clean CONFIRMED there would have been
    wrong by roughly 4 seconds. The wider default window
    (``end_validation_pre_s``/``end_validation_post_s`` = 4.0s/6.0s) now
    reaches ``[12.5, 22.5]``, which does contain 20.5s, and the dense
    validation pass - which scans its whole batch chronologically per
    PROMPT_END_VALIDATE_V2 - can refine forward from the sparse (wrong,
    too-early) candidate to the true, later, evidence-supported onset."""
    sparse_candidate_s = 16.5  # a real end-coarse nomination that was ~4s early
    true_break_s = 20.5

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=sparse_candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(
                start_s=4.0, end_s=4.0, confidence=0.9, evidence_frame_timestamps_s=(4.0,)
            )
        if request.pass_name == "end_validate":
            # A compliant model rejects the early transient near the
            # sparse candidate and reports the true, later, sustained
            # onset it can see later in this same wide batch.
            return canned_json_response(
                start_s=true_break_s,
                end_s=true_break_s,
                confidence=0.9,
                evidence_frame_timestamps_s=(true_break_s,),
            )
        assert request.pass_name == "end_coarse"
        return canned_json_response(
            start_s=sparse_candidate_s,
            end_s=sparse_candidate_s,
            confidence=0.9,
            evidence_frame_timestamps_s=(sparse_candidate_s,),
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.end_s == pytest.approx(true_break_s)
    assert outcome.event.details["end_coarse_candidate_s"] == pytest.approx(sparse_candidate_s)
    # The window actually sent reaches well past where the old 2.0s
    # horizon would have stopped (18.5s) - it must cover the true break.
    validate_call = next(c for c in provider.calls if c.pass_name == "end_validate")
    validate_times = [f.timestamp_s for f in validate_call.frames]
    assert max(validate_times) >= true_break_s


def test_pipeline_abstains_when_a_refined_onset_falls_outside_the_validation_window(zahn_video):
    """A validation response naming a timestamp outside its own submitted
    window's bounds must never be accepted, even under the new freedom to
    refine - grounding still enforces that every reported timestamp
    actually falls within the frames that were sent."""
    candidate_s = 10.0
    out_of_bounds_s = 50.0  # nowhere near the validation window, or the clip

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            return canned_json_response(
                start_s=out_of_bounds_s,
                end_s=out_of_bounds_s,
                confidence=0.9,
                evidence_frame_timestamps_s=(out_of_bounds_s,),
            )
        assert request.pass_name == "end_coarse"
        return canned_json_response(
            start_s=candidate_s,
            end_s=candidate_s,
            confidence=0.9,
            evidence_frame_timestamps_s=(candidate_s,),
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "out_of_bounds" in outcome.verdict.reason_codes


def test_pipeline_abstains_when_the_refined_onset_has_no_room_for_its_own_future_horizon(
    zahn_video,
):
    """The mandatory future-context rule is re-checked against whatever
    timestamp validation actually reports, not just the original sparse
    candidate: refining forward, toward the validation window's own edge,
    leaves less than the mandatory ``end_validation_min_future_s`` of
    submitted evidence after it - the pipeline must catch this
    structurally rather than trust the model's own compliance."""
    candidate_s = 10.0
    # Window is [6.0, 16.0] (locked_start_s=4.0 clamps the low side;
    # candidate_s + end_validation_post_s=6.0 sets the high side). A
    # refined onset at 15.0s only has 1.0s of submitted evidence after it
    # within this window - short of the 2.0s end_validation_min_future_s
    # floor.
    forward_point_s = 15.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            return canned_json_response(
                start_s=forward_point_s,
                end_s=forward_point_s,
                confidence=0.9,
                evidence_frame_timestamps_s=(forward_point_s,),
            )
        assert request.pass_name == "end_coarse"
        return canned_json_response(
            start_s=candidate_s,
            end_s=candidate_s,
            confidence=0.9,
            evidence_frame_timestamps_s=(candidate_s,),
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "insufficient_future_context" in outcome.verdict.reason_codes
    # Unlike the pre-flight check for the sparse candidate, this abstain
    # only happens after the validation call actually ran and answered.
    assert [c.pass_name for c in provider.calls] == [
        "coarse",
        "fine",
        "end_coarse",
        "end_validate",
    ]


def test_pipeline_accepts_a_candidate_with_sustained_per_second_shortening(zahn_video):
    """The straightforward positive case: a candidate break whose
    post-candidate frames show a clear, sustained shortening trend -
    evidence cited at the candidate itself plus later per-checkpoint
    frames - is accepted, no rejection needed."""
    candidate_s = 12.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            candidate_ts = next(f.timestamp_s for f in request.frames if f.is_candidate)
            later_frames = sorted(
                f.timestamp_s for f in request.frames if f.timestamp_s > candidate_ts
            )
            # Cite the candidate plus two later checkpoints - a stand-in for
            # "checked roughly per second through the validation horizon".
            checkpoints = (
                (candidate_ts, later_frames[len(later_frames) // 2], later_frames[-1])
                if later_frames
                else (candidate_ts,)
            )
            return canned_json_response(
                start_s=candidate_ts,
                end_s=candidate_ts,
                confidence=0.95,
                evidence_frame_timestamps_s=checkpoints,
            )
        assert request.pass_name == "end_coarse"
        return canned_json_response(
            start_s=candidate_s,
            end_s=candidate_s,
            confidence=0.9,
            evidence_frame_timestamps_s=(candidate_s,),
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.end_s == pytest.approx(candidate_s)
    assert outcome.end_validation_response is not None  # validated immediately, no rejects


def test_pipeline_abstains_with_request_too_large_when_the_wide_validation_window_cannot_fit(
    zahn_video,
):
    """The dense validation window is now up to ``end_validation_max_span_s``
    wide (10.0s by default) - far larger than the ~3s fine window or the
    sparse coarse/end-coarse batches - so a byte budget that comfortably
    fits every other pass can still be too small for the validation
    pass's own precision floor. The validation request must never be
    sent in that case, same "abstain, never send an undersized or
    oversized request" contract as every other budget failure (see
    ``_fit_frames_to_budget``)."""
    candidate_s = 15.0  # far enough from either clip edge that the window is the full 10.0s

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(
                start_s=4.0, end_s=4.0, confidence=0.9, evidence_frame_timestamps_s=(4.0,)
            )
        assert request.pass_name == "end_coarse"
        return canned_json_response(
            start_s=candidate_s,
            end_s=candidate_s,
            confidence=0.9,
            evidence_frame_timestamps_s=(candidate_s,),
        )

    provider = StubTimingProvider(respond)
    # A tight target_tolerance_s inflates every window's own precision
    # floor (frame count needed), but the validation window (~10s) is far
    # wider than the fine window (~3s), so its floor grows disproportion-
    # ately more - a budget sized to comfortably fit coarse/fine/end-coarse
    # at this tolerance still can't fit the validation pass's own floor.
    config = PipelineConfig(max_request_bytes=800_000, target_tolerance_s=0.1)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        config=config,
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "request_too_large" in outcome.verdict.reason_codes
    assert [c.pass_name for c in provider.calls] == ["coarse", "fine", "end_coarse"]
    assert outcome.end_validation_response is None  # never sent


def test_pipeline_calls_on_stage_before_each_pass_it_actually_reaches(zahn_video):
    """``on_stage`` is invoked with each pass name immediately before that
    pass's provider call is sent, in order, and never for a pass this run
    doesn't reach (this run aborts after "fine" never gets to end_coarse/
    end_validate)."""
    stages: list[str] = []

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=10.0, confidence=0.9)
        assert request.pass_name == "fine"
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["ambiguous_evidence"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        on_stage=stages.append,
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert stages == ["coarse", "fine"]


def test_pipeline_on_stage_can_abort_the_run_by_raising(zahn_video):
    """A caller that wants cooperative cancellation raises from inside
    ``on_stage``; the exception propagates straight out of
    ``run_llm_timing`` rather than being swallowed."""

    class _Cancelled(Exception):
        pass

    def respond(request: ProviderRequest) -> RawProviderResponse:
        raise AssertionError("the provider must never be called after on_stage raises")

    def on_stage(stage: str) -> None:
        if stage == "coarse":
            raise _Cancelled()

    provider = StubTimingProvider(respond)
    with pytest.raises(_Cancelled):
        run_llm_timing(
            zahn_video.path,
            provider,
            prompt_version=PROMPT_VERSION,
            prompt_text="irrelevant for a stub",
            on_stage=on_stage,
        )


# --------------------------------------------------------------------------- #
# Auditability: pass_frames (actual submitted timestamps) and grounded
# evidence-image timestamp selection - supervisor-directed requirement, see
# diagnostics/llm_spike/DESIGN.md.
# --------------------------------------------------------------------------- #


def test_pipeline_records_pass_frames_matching_the_actual_submitted_requests(zahn_video):
    """``PipelineOutcome.pass_frames`` must reflect exactly what was sent to
    the provider - not a reconstruction from config - for every pass that
    ran."""
    candidate_s = 12.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            ts = next(f.timestamp_s for f in request.frames if f.is_candidate)
            return canned_json_response(start_s=ts, end_s=ts, confidence=0.9)
        assert request.pass_name == "end_coarse"
        return canned_json_response(start_s=candidate_s, end_s=candidate_s, confidence=0.9)

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path, provider, prompt_version=PROMPT_VERSION, prompt_text="irrelevant"
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert set(outcome.pass_frames) == {"coarse", "fine", "end_coarse", "end_validate"}
    calls_by_pass = {c.pass_name: c for c in provider.calls}
    for pass_name, call in calls_by_pass.items():
        expected = tuple(sorted(f.timestamp_s for f in call.frames))
        actual = tuple(sorted(outcome.pass_frames[pass_name]))
        assert actual == expected, pass_name


def test_pipeline_pass_frames_omits_passes_that_never_ran(zahn_video):
    """A run that abstains before end_coarse/end_validate must not claim
    those passes sent anything - the frontend's "Not run" state depends on
    the key being absent, not present-with-an-empty-list."""

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=21.5, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path, provider, prompt_version=PROMPT_VERSION, prompt_text="irrelevant"
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert set(outcome.pass_frames) == {"coarse", "fine", "end_coarse"}
    assert "end_validate" not in outcome.pass_frames


def test_pipeline_thinned_pass_frames_match_the_frames_actually_sent(zahn_video):
    """Under a tight byte budget that forces thinning, ``pass_frames`` must
    reflect the *thinned* request, never the dense pre-thinning extraction
    plan - the exact failure mode this feature exists to prevent."""
    candidate_s = 12.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(
                start_s=4.0, end_s=4.0, confidence=0.9, evidence_frame_timestamps_s=(4.0,)
            )
        if request.pass_name == "end_validate":
            ts = next(f.timestamp_s for f in request.frames if f.is_candidate)
            return canned_json_response(start_s=ts, end_s=ts, confidence=0.9)
        assert request.pass_name == "end_coarse"
        return canned_json_response(start_s=candidate_s, end_s=candidate_s, confidence=0.9)

    provider = StubTimingProvider(respond)
    config = PipelineConfig(max_request_bytes=300_000)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant",
        config=config,
    )
    fine_call = next(c for c in provider.calls if c.pass_name == "fine")
    # The tight budget must actually have forced thinning for this
    # assertion to mean anything - native-fps density over the fine window
    # would be dozens of frames.
    assert len(fine_call.frames) < 30
    expected = tuple(sorted(f.timestamp_s for f in fine_call.frames))
    assert tuple(sorted(outcome.pass_frames["fine"])) == expected


def test_pipeline_confirmed_result_exposes_grounded_evidence_timestamps(zahn_video):
    """A CONFIRMED result's Event.details carries start/end evidence
    timestamps that are themselves drawn from what was actually submitted
    to the deciding pass - never an arbitrary or interpolated value."""
    candidate_s = 12.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            ts = next(f.timestamp_s for f in request.frames if f.is_candidate)
            return canned_json_response(start_s=ts, end_s=ts, confidence=0.9)
        assert request.pass_name == "end_coarse"
        return canned_json_response(start_s=candidate_s, end_s=candidate_s, confidence=0.9)

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path, provider, prompt_version=PROMPT_VERSION, prompt_text="irrelevant"
    )
    assert outcome.event is not None
    start_evidence_s = outcome.event.details["start_evidence_s"]
    end_evidence_s = outcome.event.details["end_evidence_s"]
    assert start_evidence_s is not None
    assert end_evidence_s is not None
    assert start_evidence_s in outcome.pass_frames["fine"]
    assert end_evidence_s in outcome.pass_frames["end_validate"]


def test_pipeline_abstain_outcome_has_no_event_and_therefore_no_evidence(zahn_video):
    """ABSTAIN must never present fabricated evidence - the absence of
    ``event`` (already the contract for every other field) covers this
    too: there is nowhere for a frontend to even find an evidence
    timestamp when the run abstained."""

    def respond(request: ProviderRequest) -> RawProviderResponse:
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path, provider, prompt_version=PROMPT_VERSION, prompt_text="irrelevant"
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None


class TestNearestGroundedEvidenceTs:
    def test_picks_the_evidence_timestamp_closest_to_the_boundary(self):
        result = _nearest_grounded_evidence_ts(
            evidence_ts=(2.0, 4.0, 6.0), submitted_ts=(2.0, 4.0, 6.0), boundary_ts=5.5
        )
        assert result == 6.0

    def test_snaps_to_the_nearest_actually_submitted_timestamp(self):
        # The evidence value itself (4.0) is not among the submitted
        # timestamps - the returned value must still come from what was
        # actually sent, never the raw (unsnapped) evidence value.
        result = _nearest_grounded_evidence_ts(
            evidence_ts=(4.0,), submitted_ts=(3.96, 8.0), boundary_ts=4.0
        )
        assert result == 3.96

    def test_returns_none_when_there_is_no_evidence(self):
        assert _nearest_grounded_evidence_ts((), (1.0, 2.0), 1.5) is None

    def test_returns_none_when_nothing_was_submitted(self):
        assert _nearest_grounded_evidence_ts((1.0,), (), 1.0) is None


def test_pipeline_config_rejects_invalid_bounds():
    with pytest.raises(ConfigurationError):
        PipelineConfig(min_confidence=0.9, review_confidence=0.5).validate()


def test_pipeline_config_rejects_end_validation_span_exceeding_the_hard_cap():
    with pytest.raises(ConfigurationError):
        PipelineConfig(end_validation_pre_s=8.0, end_validation_post_s=8.0).validate()


def test_pipeline_config_rejects_min_future_exceeding_the_post_margin():
    with pytest.raises(ConfigurationError):
        PipelineConfig(end_validation_post_s=1.0, end_validation_min_future_s=2.0).validate()
