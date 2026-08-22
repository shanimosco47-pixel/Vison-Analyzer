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
from app.analysis.llm_timing.pipeline import PipelineConfig, run_llm_timing
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
# pipeline.run_llm_timing: coarse-to-fine orchestration against a real video
# --------------------------------------------------------------------------- #


def _stub_matching_truth(truth: dict[str, float], *, fine_confidence: float = 0.9):
    """A StubTimingProvider that answers accurately on both passes."""

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=truth["flow_start_s"], end_s=truth["flow_end_s"], confidence=0.8)
        return canned_json_response(
            start_s=truth["flow_start_s"], end_s=truth["flow_end_s"], confidence=fine_confidence
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
    # Both passes actually ran, and the fine pass saw dense native-fps frames.
    assert len(provider.calls) == 2
    assert provider.calls[0].pass_name == "coarse"
    assert provider.calls[1].pass_name == "fine"
    assert len(provider.calls[1].frames) > len(provider.calls[0].frames)


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
    config = PipelineConfig(fine_max_span_s=6.0)
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


def test_pipeline_config_rejects_invalid_bounds():
    with pytest.raises(ConfigurationError):
        PipelineConfig(min_confidence=0.9, review_confidence=0.5).validate()
