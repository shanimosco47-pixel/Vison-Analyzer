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


def test_estimate_cost_usd_is_none_when_usage_is_unknown():
    table = {"m": ModelPricing(input_usd_per_1k_tokens=1.0, output_usd_per_1k_tokens=2.0)}
    assert estimate_cost_usd("m", None, 500, table=table) is None
    assert estimate_cost_usd("m", 500, None, table=table) is None


def test_estimate_cost_usd_computes_input_and_output_separately():
    table = {"m": ModelPricing(input_usd_per_1k_tokens=1.0, output_usd_per_1k_tokens=2.0)}
    # 2000 input tokens @ $1/1k = $2.00; 500 output tokens @ $2/1k = $1.00
    cost = estimate_cost_usd("m", 2000, 500, table=table)
    assert cost == 3.0


def test_pipeline_outcome_reports_none_cost_when_model_unpriced(zahn_video):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        base = canned_json_response(
            start_s=zahn_video.truth["flow_start_s"],
            end_s=zahn_video.truth["flow_end_s"],
            confidence=0.9,
        )
        return RawProviderResponse(
            model_id="unpriced-stub",
            raw_text=base.raw_text,
            latency_s=1.5,
            retries=1,
            prompt_tokens=1000,
            completion_tokens=200,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.event is not None
    # Usage is reported even though cost can't be estimated (no pricing entry).
    assert outcome.total_tokens == (1000 + 200) * 2  # coarse + fine, both stubbed the same
    assert outcome.total_retries == 2
    assert outcome.total_latency_s == 3.0
    assert outcome.estimated_cost_usd() is None
    payload = outcome.to_dict()
    assert payload["estimated_cost_usd"] is None
    assert payload["total_tokens"] == outcome.total_tokens
    assert payload["pricing_table_version"]


def test_pipeline_outcome_estimates_cost_with_an_injected_table(zahn_video):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        base = canned_json_response(
            start_s=zahn_video.truth["flow_start_s"],
            end_s=zahn_video.truth["flow_end_s"],
            confidence=0.9,
        )
        return RawProviderResponse(
            model_id="priced-stub",
            raw_text=base.raw_text,
            latency_s=1.0,
            prompt_tokens=1000,
            completion_tokens=1000,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    table = {"priced-stub": ModelPricing(input_usd_per_1k_tokens=0.5, output_usd_per_1k_tokens=1.5)}
    # Each pass: 1000 in @ $0.5/1k + 1000 out @ $1.5/1k = $2.00; two passes = $4.00
    assert outcome.estimated_cost_usd(table=table) == 4.0


def test_pipeline_config_max_uncertainty_s_must_be_positive():
    with pytest.raises(ConfigurationError):
        PipelineConfig(max_uncertainty_s=0.0).validate()
