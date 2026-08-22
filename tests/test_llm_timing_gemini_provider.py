"""Tests for GeminiTimingProvider - the first real TimingProvider adapter.

Entirely offline: every test injects a fake GeminiClient (plain Python
object, no SDK, no network). build_default_gemini_client (the only place
that imports the real google.generativeai SDK) is never called here - see
its own docstring for why, and diagnostics/llm_spike/DESIGN.md for the
standing rule that nothing in this spike makes a live call yet.
"""

from __future__ import annotations

import json

import pytest

from app.analysis.llm_timing.gemini_provider import (
    DEFAULT_GEMINI_MODEL_ID,
    GeminiCallResult,
    GeminiTimingProvider,
)
from app.analysis.llm_timing.provider import ProviderRequest, TimedFrame, parse_raw_response
from app.analysis.llm_timing.schema import TimingStatus
from app.errors import ConfigurationError

CONFIRMED_JSON = json.dumps(
    {
        "status": "confirmed",
        "start_s": 4.0,
        "end_s": 20.6,
        "start_uncertainty_s": 0.1,
        "end_uncertainty_s": 0.1,
        "confidence": 0.9,
        "reason_codes": [],
        "evidence_frame_timestamps_s": [4.0, 20.6],
    }
)


def _frame(ts: float) -> TimedFrame:
    return TimedFrame(
        timestamp_s=ts, image_bytes=b"\xff\xd8\xff\xe0fakejpeg", media_type="image/jpeg"
    )


def _request(pass_name: str = "fine") -> ProviderRequest:
    return ProviderRequest(
        prompt_version="test-v1",
        prompt_text="analyze this",
        frames=(_frame(4.0), _frame(20.6)),
        pass_name=pass_name,
    )


class _FakeClient:
    """Records every call it receives and returns a scripted result."""

    def __init__(self, respond) -> None:
        self._respond = respond
        self.calls: list[dict] = []

    def generate_content(self, *, model, parts, generation_config):
        self.calls.append({"model": model, "parts": parts, "generation_config": generation_config})
        return self._respond(model, parts, generation_config)


def test_uses_the_pinned_default_model_id_by_default():
    client = _FakeClient(lambda model, parts, cfg: GeminiCallResult(text=CONFIRMED_JSON))
    provider = GeminiTimingProvider(client)
    response = provider.analyze(_request())
    assert response.model_id == DEFAULT_GEMINI_MODEL_ID
    assert client.calls[0]["model"] == DEFAULT_GEMINI_MODEL_ID


def test_model_id_is_configurable_and_pinned_explicitly():
    client = _FakeClient(lambda model, parts, cfg: GeminiCallResult(text=CONFIRMED_JSON))
    provider = GeminiTimingProvider(client, model_id="gemini-2.5-pro")
    response = provider.analyze(_request())
    assert response.model_id == "gemini-2.5-pro"
    assert client.calls[0]["model"] == "gemini-2.5-pro"


def test_sends_one_text_and_one_image_part_per_frame_plus_the_prompt():
    client = _FakeClient(lambda model, parts, cfg: GeminiCallResult(text=CONFIRMED_JSON))
    provider = GeminiTimingProvider(client)
    provider.analyze(_request())
    parts = client.calls[0]["parts"]
    # prompt text + (label + image) per frame, for 2 frames = 1 + 2*2 = 5
    assert len(parts) == 5
    assert parts[0] == {"text": "analyze this"}
    assert "4.000" in parts[1]["text"]
    assert "inline_data" in parts[2]
    assert parts[2]["inline_data"]["mime_type"] == "image/jpeg"


def test_response_round_trips_through_parse_raw_response_to_confirmed():
    client = _FakeClient(lambda model, parts, cfg: GeminiCallResult(text=CONFIRMED_JSON))
    provider = GeminiTimingProvider(client)
    response = provider.analyze(_request())
    verdict = parse_raw_response(response, prompt_version="test-v1", min_confidence=0.5)
    assert verdict.status is TimingStatus.CONFIRMED
    assert verdict.start_s == 4.0
    assert verdict.end_s == 20.6


def test_usage_is_propagated_when_the_client_reports_it():
    client = _FakeClient(
        lambda model, parts, cfg: GeminiCallResult(
            text=CONFIRMED_JSON, prompt_tokens=1200, completion_tokens=80
        )
    )
    provider = GeminiTimingProvider(client)
    response = provider.analyze(_request())
    assert response.prompt_tokens == 1200
    assert response.completion_tokens == 80
    assert response.retries == 0


def test_client_exception_becomes_an_error_response_not_an_exception():
    def respond(model, parts, cfg):
        raise RuntimeError("upstream 503")

    client = _FakeClient(respond)
    provider = GeminiTimingProvider(client, max_retries=0)
    response = provider.analyze(_request())
    assert response.error is not None
    assert "upstream 503" in response.error
    assert response.retries == 1  # one attempt made, then gave up


def test_client_exception_is_retried_up_to_max_retries_then_succeeds():
    attempts = {"n": 0}

    def respond(model, parts, cfg):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise RuntimeError("transient")
        return GeminiCallResult(text=CONFIRMED_JSON)

    client = _FakeClient(respond)
    provider = GeminiTimingProvider(client, max_retries=3)
    response = provider.analyze(_request())
    assert response.error is None
    assert response.retries == 2  # two failures before the third attempt succeeded
    assert attempts["n"] == 3


def test_exhausting_retries_still_returns_a_response_not_a_raised_exception():
    def respond(model, parts, cfg):
        raise RuntimeError("always fails")

    client = _FakeClient(respond)
    provider = GeminiTimingProvider(client, max_retries=2)
    response = provider.analyze(_request())
    assert response.error is not None
    assert response.retries == 3  # 1 initial + 2 retries, all failed

    # This must still parse to a clean ABSTAIN, never raise, all the way
    # through the same parser every other provider uses.
    verdict = parse_raw_response(response, prompt_version="test-v1", min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "provider_error" in verdict.reason_codes


def test_rejects_empty_model_id():
    client = _FakeClient(lambda model, parts, cfg: GeminiCallResult(text=CONFIRMED_JSON))
    with pytest.raises(ConfigurationError):
        GeminiTimingProvider(client, model_id="  ")


def test_rejects_negative_max_retries():
    client = _FakeClient(lambda model, parts, cfg: GeminiCallResult(text=CONFIRMED_JSON))
    with pytest.raises(ConfigurationError):
        GeminiTimingProvider(client, max_retries=-1)
