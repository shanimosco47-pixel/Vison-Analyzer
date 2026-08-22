"""Offline tests for scripts/llm_timing_eval.py's ``--provider gemini`` wiring.

These verify routing only: that selecting "gemini" builds a
``GeminiTimingProvider`` with the right model ID, via whichever client
factory is injected. None of this needs the real ``google-genai`` SDK
installed or a ``GEMINI_API_KEY`` set - ``_build_provider`` takes the client
factory as a parameter specifically so this is testable without either (see
its own docstring in the script). Real-provider behaviour (retries, parsing,
grounding, ...) is already covered by ``test_llm_timing_gemini_provider.py``
and ``test_llm_timing_pipeline.py`` - this file only proves the CLI/harness
plumbing routes to it correctly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from app.analysis.llm_timing.gemini_provider import (
    DEFAULT_GEMINI_MODEL_ID,
    GeminiCallResult,
    GeminiTimingProvider,
)
from app.analysis.llm_timing.provider import ProviderRequest, TimedFrame

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "llm_timing_eval.py"
_spec = importlib.util.spec_from_file_location("llm_timing_eval", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
llm_timing_eval = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = llm_timing_eval
_spec.loader.exec_module(llm_timing_eval)

_CONFIRMED_JSON = json.dumps(
    {
        "status": "confirmed",
        "start_s": 1.0,
        "end_s": 2.0,
        "start_uncertainty_s": 0.1,
        "end_uncertainty_s": 0.1,
        "confidence": 0.9,
        "reason_codes": [],
        "evidence_frame_timestamps_s": [1.0, 2.0],
    }
)


class _RecordingFakeClient:
    """Stands in for the real google-genai-backed client - records which
    model each call requested, no network or SDK involved."""

    def __init__(self, response_text: str) -> None:
        self._response_text = response_text
        self.calls: list[str] = []

    def generate_content(self, *, model, parts, generation_config):
        self.calls.append(model)
        return GeminiCallResult(text=self._response_text)


def _dummy_request() -> ProviderRequest:
    return ProviderRequest(
        prompt_version="test-v1",
        prompt_text="analyze this",
        frames=(TimedFrame(timestamp_s=0.0, image_bytes=b"\xff\xd8\xff\xe0fake"),),
        pass_name="coarse",
    )


def test_build_provider_gemini_routes_to_geminitimingprovider_with_default_model_id():
    client = _RecordingFakeClient(_CONFIRMED_JSON)
    provider = llm_timing_eval._build_provider(
        "gemini", {}, model_id=None, gemini_client_factory=lambda: client
    )
    assert isinstance(provider, GeminiTimingProvider)
    response = provider.analyze(_dummy_request())
    assert response.model_id == DEFAULT_GEMINI_MODEL_ID
    assert client.calls == [DEFAULT_GEMINI_MODEL_ID]


def test_build_provider_gemini_honors_an_explicit_model_id():
    client = _RecordingFakeClient(_CONFIRMED_JSON)
    provider = llm_timing_eval._build_provider(
        "gemini", {}, model_id="gemini-2.5-pro", gemini_client_factory=lambda: client
    )
    response = provider.analyze(_dummy_request())
    assert response.model_id == "gemini-2.5-pro"
    assert client.calls == ["gemini-2.5-pro"]


def test_build_provider_gemini_calls_the_injected_factory_lazily_not_the_real_one():
    # No GEMINI_API_KEY is set in this test environment and google-genai may
    # not even be installed - if this routed to the real
    # build_default_gemini_client it would raise before this test could
    # observe anything. It must not.
    calls = {"n": 0}

    def fake_factory():
        calls["n"] += 1
        return _RecordingFakeClient(_CONFIRMED_JSON)

    llm_timing_eval._build_provider("gemini", {}, model_id=None, gemini_client_factory=fake_factory)
    assert calls["n"] == 1


def test_build_provider_stub_perfect_is_unaffected():
    entry = {"true_start_s": 4.0, "true_end_s": 21.5}
    provider = llm_timing_eval._build_provider("stub-perfect", entry)
    response = provider.analyze(_dummy_request())
    payload = json.loads(response.raw_text)
    assert payload["start_s"] == 4.0
    assert payload["end_s"] == 21.5


def test_build_provider_rejects_unknown_provider_name():
    with pytest.raises(ValueError, match="gemini"):
        llm_timing_eval._build_provider("not-a-real-provider", {})


def test_evaluate_clip_routes_through_the_full_harness_offline_with_gemini(zahn_video):
    """End-to-end proof the wiring works through the actual harness path
    (_evaluate_clip -> run_llm_timing -> GeminiTimingProvider), entirely
    offline against the existing synthetic zahn_video fixture."""
    entry = {
        "clip_id": "offline-routing-check",
        "video_path": str(zahn_video.path),
        "true_start_s": zahn_video.truth["flow_start_s"],
        "true_end_s": zahn_video.truth["flow_end_s"],
    }
    client = _RecordingFakeClient(_CONFIRMED_JSON)
    result = llm_timing_eval._evaluate_clip(
        entry,
        "gemini",
        llm_timing_eval.PipelineConfig(),
        model_id=None,
        gemini_client_factory=lambda: client,
    )
    assert result.coarse_model_id == DEFAULT_GEMINI_MODEL_ID
    assert result.fine_model_id == DEFAULT_GEMINI_MODEL_ID
    assert client.calls == [DEFAULT_GEMINI_MODEL_ID, DEFAULT_GEMINI_MODEL_ID]
