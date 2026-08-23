"""Tests for pricing.py and PipelineOutcome's cost/usage reporting (Codex
review, finding 5: the evaluation contract needs real cost/token evidence).
"""

from __future__ import annotations

import pytest

from app.analysis.llm_timing.pipeline import PipelineConfig, run_llm_timing
from app.analysis.llm_timing.pricing import ModelPricing, estimate_cost_usd
from app.analysis.llm_timing.provider import (
    ProviderRequest,
    RawProviderResponse,
    StubTimingProvider,
    canned_json_response,
)
from app.errors import ConfigurationError

PROMPT_VERSION = "test-prompt-v1"


def test_estimate_cost_usd_is_none_for_an_unpriced_model():
    assert estimate_cost_usd("unknown-model", 1000, 1000) is None


def test_gemini_flash_lite_has_a_versioned_default_pricing_entry():
    # 1,000,000 input tokens @ $0.10/1M + 1,000,000 output tokens @ $0.40/1M
    cost = estimate_cost_usd("gemini-2.5-flash-lite", 1_000_000, 1_000_000)
    assert cost == pytest.approx(0.10 + 0.40)


def test_estimate_cost_usd_is_none_when_usage_is_unknown():
    table = {"m": ModelPricing(input_usd_per_1k_tokens=1.0, output_usd_per_1k_tokens=2.0)}
    assert estimate_cost_usd("m", None, 500, table=table) is None
    assert estimate_cost_usd("m", 500, None, table=table) is None


def test_estimate_cost_usd_computes_input_and_output_separately():
    table = {"m": ModelPricing(input_usd_per_1k_tokens=1.0, output_usd_per_1k_tokens=2.0)}
    # 2000 input tokens @ $1/1k = $2.00; 500 output tokens @ $2/1k = $1.00
    cost = estimate_cost_usd("m", 2000, 500, table=table)
    assert cost == 3.0


def _respond_confirming_every_pass(
    *, model_id: str, latency_s: float, retries: int, prompt_tokens: int, completion_tokens: int
):
    """Coarse/fine confirm against the manifest truth as usual; the single
    whole-clip end-coarse call immediately confirms against whichever of its
    own submitted frames sits closest to the coarse pass's own end estimate
    (so the two passes agree and no cross-pass conflict is triggered), and
    the one resulting trend-validation call immediately confirms the
    flagged candidate too - so exactly one end-coarse call and one
    end-validate call happen, deterministic, so these cost/usage tests
    don't depend on any real model behaviour.
    """
    coarse_end_estimate_s = 21.5

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "end_coarse":
            ts = min(
                (f.timestamp_s for f in request.frames),
                key=lambda t: abs(t - coarse_end_estimate_s),
            )
            base = canned_json_response(
                start_s=ts, end_s=ts, confidence=0.9, evidence_frame_timestamps_s=(ts,)
            )
        elif request.pass_name == "end_validate":
            ts = next(f.timestamp_s for f in request.frames if f.is_candidate)
            base = canned_json_response(
                start_s=ts, end_s=ts, confidence=0.9, evidence_frame_timestamps_s=(ts,)
            )
        elif request.pass_name == "fine":
            # Start-only pass: reports the confirmed start as a degenerate
            # point, never the (discarded) end.
            base = canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        else:
            base = canned_json_response(start_s=4.0, end_s=coarse_end_estimate_s, confidence=0.9)
        return RawProviderResponse(
            model_id=model_id,
            raw_text=base.raw_text,
            latency_s=latency_s,
            retries=retries,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    return respond


def test_pipeline_outcome_reports_none_cost_when_model_unpriced(zahn_video):
    provider = StubTimingProvider(
        _respond_confirming_every_pass(
            model_id="unpriced-stub",
            latency_s=1.5,
            retries=1,
            prompt_tokens=1000,
            completion_tokens=200,
        )
    )
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.event is not None
    assert len(provider.calls) == 4  # coarse, fine, one end-coarse call, one validation call
    # Usage is reported even though cost can't be estimated (no pricing entry).
    assert outcome.total_tokens == (1000 + 200) * 4
    assert outcome.total_retries == 4
    assert outcome.total_latency_s == 6.0
    assert outcome.estimated_cost_usd() is None
    payload = outcome.to_dict()
    assert payload["estimated_cost_usd"] is None
    assert payload["total_tokens"] == outcome.total_tokens
    assert payload["pricing_table_version"]
    assert payload["end_coarse_model_id"] == "unpriced-stub"
    assert payload["end_coarse_latency_s"] == 1.5
    assert payload["end_validation_model_id"] == "unpriced-stub"
    assert payload["end_validation_latency_s"] == 1.5


def test_pipeline_outcome_estimates_cost_with_an_injected_table(zahn_video):
    provider = StubTimingProvider(
        _respond_confirming_every_pass(
            model_id="priced-stub",
            latency_s=1.0,
            retries=0,
            prompt_tokens=1000,
            completion_tokens=1000,
        )
    )
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    table = {"priced-stub": ModelPricing(input_usd_per_1k_tokens=0.5, output_usd_per_1k_tokens=1.5)}
    # Each pass: 1000 in @ $0.5/1k + 1000 out @ $1.5/1k = $2.00; four passes
    # (coarse, fine, one end-coarse call, one validation call) = $8.00
    assert outcome.estimated_cost_usd(table=table) == 8.0


def test_pipeline_config_max_uncertainty_s_must_be_positive():
    with pytest.raises(ConfigurationError):
        PipelineConfig(max_uncertainty_s=0.0).validate()
