"""Tests for GeminiTimingProvider - the first real TimingProvider adapter.

Entirely offline: every test injects a fake GeminiClient (plain Python
object, no SDK, no network) and a fake sleep function (no real backoff
delay). build_default_gemini_client (the only place that imports the real
google-genai SDK) is never called here - see its own docstring for why, and
diagnostics/llm_spike/DESIGN.md for the standing rule that nothing in this
spike makes a live call yet.
"""

from __future__ import annotations

import json

import pytest

from app.analysis.llm_timing.gemini_provider import (
    DEFAULT_GEMINI_MODEL_ID,
    GeminiCallResult,
    GeminiTimingProvider,
)
from app.analysis.llm_timing.provider import (
    PermanentProviderError,
    ProviderRequest,
    TimedFrame,
    TransientProviderError,
    parse_raw_response,
)
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


class _RecordingSleep:
    """Fake sleep_fn: records requested delays instead of actually sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.delays.append(seconds)


def test_uses_the_pinned_default_model_id_by_default():
    client = _FakeClient(lambda model, parts, cfg: GeminiCallResult(text=CONFIRMED_JSON))
    provider = GeminiTimingProvider(client, sleep_fn=_RecordingSleep())
    response = provider.analyze(_request())
    assert response.model_id == DEFAULT_GEMINI_MODEL_ID
    assert client.calls[0]["model"] == DEFAULT_GEMINI_MODEL_ID


def test_model_id_is_configurable_and_pinned_explicitly():
    client = _FakeClient(lambda model, parts, cfg: GeminiCallResult(text=CONFIRMED_JSON))
    provider = GeminiTimingProvider(client, model_id="gemini-2.5-pro", sleep_fn=_RecordingSleep())
    response = provider.analyze(_request())
    assert response.model_id == "gemini-2.5-pro"
    assert client.calls[0]["model"] == "gemini-2.5-pro"


def test_sends_one_text_and_one_image_part_per_frame_plus_the_prompt():
    client = _FakeClient(lambda model, parts, cfg: GeminiCallResult(text=CONFIRMED_JSON))
    provider = GeminiTimingProvider(client, sleep_fn=_RecordingSleep())
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
    provider = GeminiTimingProvider(client, sleep_fn=_RecordingSleep())
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
    provider = GeminiTimingProvider(client, sleep_fn=_RecordingSleep())
    response = provider.analyze(_request())
    assert response.prompt_tokens == 1200
    assert response.completion_tokens == 80
    assert response.retries == 0


# --------------------------------------------------------------------------- #
# Retry policy: only transient failures are retried, retries counts retries
# *after* the initial attempt (Codex re-review, finding 3)
# --------------------------------------------------------------------------- #


def test_an_unclassified_exception_is_not_retried_by_default():
    """A bare, unrecognised exception is treated as permanent - retrying
    something the adapter can't classify (e.g. an auth failure the client
    forgot to wrap) risks retrying a failure that will never succeed."""

    def respond(model, parts, cfg):
        raise RuntimeError("upstream 503")

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = GeminiTimingProvider(client, max_retries=3, sleep_fn=sleep)
    response = provider.analyze(_request())
    assert response.error is not None
    assert "upstream 503" in response.error
    assert response.retries == 0  # no retry attempted - not classified as retryable
    assert sleep.delays == []


def test_permanent_provider_error_is_never_retried():
    def respond(model, parts, cfg):
        raise PermanentProviderError("invalid API key")

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = GeminiTimingProvider(client, max_retries=3, sleep_fn=sleep)
    response = provider.analyze(_request())
    assert response.error is not None
    assert response.retries == 0
    assert sleep.delays == []


def test_transient_provider_error_is_retried_with_backoff_then_succeeds():
    attempts = {"n": 0}

    def respond(model, parts, cfg):
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise TransientProviderError("rate limited")
        return GeminiCallResult(text=CONFIRMED_JSON)

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = GeminiTimingProvider(
        client, max_retries=3, backoff_base_s=1.0, backoff_max_s=10.0, sleep_fn=sleep
    )
    response = provider.analyze(_request())
    assert response.error is None
    assert response.retries == 2  # two retries after the initial failed attempt
    assert attempts["n"] == 3
    # Exponential backoff: 1.0 * 2**0, then 1.0 * 2**1
    assert sleep.delays == [1.0, 2.0]


def test_bare_timeout_and_connection_errors_are_retried_like_transient():
    attempts = {"n": 0}

    def respond(model, parts, cfg):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("deadline exceeded")
        if attempts["n"] == 2:
            raise ConnectionError("reset by peer")
        return GeminiCallResult(text=CONFIRMED_JSON)

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = GeminiTimingProvider(client, max_retries=3, sleep_fn=sleep)
    response = provider.analyze(_request())
    assert response.error is None
    assert response.retries == 2


def test_max_retries_zero_means_zero_retries_on_immediate_failure():
    def respond(model, parts, cfg):
        raise TransientProviderError("rate limited")

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = GeminiTimingProvider(client, max_retries=0, sleep_fn=sleep)
    response = provider.analyze(_request())
    assert response.error is not None
    assert response.retries == 0
    assert sleep.delays == []


def test_backoff_is_capped_at_backoff_max_s():
    def respond(model, parts, cfg):
        raise TransientProviderError("still failing")

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = GeminiTimingProvider(
        client, max_retries=5, backoff_base_s=1.0, backoff_max_s=3.0, sleep_fn=sleep
    )
    provider.analyze(_request())
    # 1, 2, 4->capped 3, 8->capped 3, 16->capped 3
    assert sleep.delays == [1.0, 2.0, 3.0, 3.0, 3.0]


def test_exhausting_retries_still_returns_a_response_not_a_raised_exception():
    def respond(model, parts, cfg):
        raise TransientProviderError("always fails")

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = GeminiTimingProvider(client, max_retries=2, sleep_fn=sleep)
    response = provider.analyze(_request())
    assert response.error is not None
    assert response.retries == 2  # 2 retries after the initial attempt, all failed

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


def test_rejects_negative_backoff():
    client = _FakeClient(lambda model, parts, cfg: GeminiCallResult(text=CONFIRMED_JSON))
    with pytest.raises(ConfigurationError):
        GeminiTimingProvider(client, backoff_base_s=-1.0)
