"""Tests for the multi-engine configuration contract and orchestrator.

Backend-only: no Settings UI, no web-layer wiring exists yet (see
diagnostics/llm_spike/DESIGN.md §10). These tests check the two properties
the supervisor's authorization actually cares about - per-engine isolation
(one engine failing never blocks the others) and no aggregation (every
engine's result is returned separately, never merged or ranked) - plus the
credential_ref safety guard.
"""

from __future__ import annotations

import pytest

from app.analysis.llm_timing.engine_config import (
    EngineConfig,
    EngineRunStatus,
    run_llm_timing_for_engines,
)
from app.analysis.llm_timing.provider import StubTimingProvider, canned_json_response
from app.analysis.llm_timing.schema import TimingStatus
from app.errors import ConfigurationError

PROMPT_VERSION = "test-prompt-v1"


def _config(engine_id: str, *, enabled: bool = True, credential_ref: str = "ENGINE_API_KEY"):
    return EngineConfig(
        engine_id=engine_id,
        provider_name="stub-vendor",
        model_id="stub-model-1",
        credential_ref=credential_ref,
        enabled=enabled,
    )


# --------------------------------------------------------------------------- #
# EngineConfig validation
# --------------------------------------------------------------------------- #


def test_engine_config_rejects_empty_fields():
    with pytest.raises(ConfigurationError):
        EngineConfig(engine_id="", provider_name="p", model_id="m", credential_ref="REF")
    with pytest.raises(ConfigurationError):
        EngineConfig(engine_id="e1", provider_name="", model_id="m", credential_ref="REF")
    with pytest.raises(ConfigurationError):
        EngineConfig(engine_id="e1", provider_name="p", model_id="", credential_ref="REF")
    with pytest.raises(ConfigurationError):
        EngineConfig(engine_id="e1", provider_name="p", model_id="m", credential_ref="")


def test_engine_config_rejects_a_raw_looking_secret_as_credential_ref():
    with pytest.raises(ConfigurationError):
        EngineConfig(
            engine_id="e1",
            provider_name="p",
            model_id="m",
            credential_ref="sk-abcdefghijklmnopqrstuvwxyz123456",
        )


def test_engine_config_accepts_a_plausible_env_var_reference():
    cfg = EngineConfig(
        engine_id="e1", provider_name="openai", model_id="gpt-x", credential_ref="OPENAI_API_KEY"
    )
    assert cfg.credential_ref == "OPENAI_API_KEY"


def test_engine_config_to_dict_never_exposes_credential_ref():
    cfg = EngineConfig(
        engine_id="e1", provider_name="openai", model_id="gpt-x", credential_ref="OPENAI_API_KEY"
    )
    dto = cfg.to_dict()
    assert "credential_ref" not in dto
    assert dto["credential_configured"] is True


# --------------------------------------------------------------------------- #
# run_llm_timing_for_engines: isolation and no-aggregation
# --------------------------------------------------------------------------- #


def _perfect_provider(zahn_video):
    def respond(request):
        return canned_json_response(
            start_s=zahn_video.truth["flow_start_s"],
            end_s=zahn_video.truth["flow_end_s"],
            confidence=0.9,
        )

    return StubTimingProvider(respond)


def test_disabled_engine_is_skipped_and_never_calls_the_factory(zahn_video):
    engines = [_config("e1", enabled=False)]
    factory_calls = []

    def factory(engine):
        factory_calls.append(engine.engine_id)
        return _perfect_provider(zahn_video)

    results = run_llm_timing_for_engines(
        zahn_video.path,
        engines,
        factory,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant",
    )
    assert len(results) == 1
    assert results[0].status is EngineRunStatus.SKIPPED
    assert results[0].outcome is None
    assert factory_calls == []


def test_one_engine_erroring_does_not_block_the_others(zahn_video):
    engines = [_config("good-1"), _config("bad"), _config("good-2")]

    def factory(engine):
        if engine.engine_id == "bad":
            raise RuntimeError("connection refused")
        return _perfect_provider(zahn_video)

    results = run_llm_timing_for_engines(
        zahn_video.path,
        engines,
        factory,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant",
    )

    # Order preserved, one entry per engine, nothing merged.
    assert [r.engine_id for r in results] == ["good-1", "bad", "good-2"]

    good1, bad, good2 = results
    assert good1.status is EngineRunStatus.RAN
    assert good1.outcome.verdict.status is TimingStatus.CONFIRMED
    assert good2.status is EngineRunStatus.RAN
    assert good2.outcome.verdict.status is TimingStatus.CONFIRMED

    assert bad.status is EngineRunStatus.ENGINE_ERROR
    assert bad.outcome is None
    assert "connection refused" in bad.error_message


def test_engine_error_message_is_redacted_when_it_contains_a_secret_shape(zahn_video):
    engines = [_config("bad")]

    def factory(engine):
        raise RuntimeError("auth failed for key sk-abcdefghijklmnopqrstuvwxyz123456")

    results = run_llm_timing_for_engines(
        zahn_video.path,
        engines,
        factory,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant",
    )
    assert results[0].status is EngineRunStatus.ENGINE_ERROR
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in results[0].error_message
    assert "[redacted]" in results[0].error_message


def test_results_are_never_aggregated_even_when_engines_disagree(zahn_video):
    """Two engines confirm different, mutually exclusive answers - both must
    survive untouched in the output, with no averaging or winner picked."""
    engines = [_config("early"), _config("late")]

    def factory(engine):
        if engine.engine_id == "early":

            def respond(request):
                return canned_json_response(start_s=4.0, end_s=10.0, confidence=0.9)
        else:

            def respond(request):
                return canned_json_response(start_s=4.0, end_s=21.5, confidence=0.9)

        return StubTimingProvider(respond)

    results = run_llm_timing_for_engines(
        zahn_video.path,
        engines,
        factory,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant",
    )
    early, late = results
    assert early.outcome.verdict.end_s == pytest.approx(10.0)
    assert late.outcome.verdict.end_s == pytest.approx(21.5)
    # No combined/averaged field exists anywhere on the return value - the
    # type itself (list[EngineOutcome]) is the guarantee here.
    assert isinstance(results, list)
