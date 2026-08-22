"""Tests for OpenAITimingProvider - the second real TimingProvider adapter.

Entirely offline: every test injects a fake OpenAIClient (plain Python
object, no SDK, no network) and a fake sleep function (no real backoff
delay). build_default_openai_client (the only place that imports the real
openai SDK) is never called here - see its own docstring for why, and
diagnostics/llm_spike/DESIGN.md for the standing rule that nothing in this
spike makes a live call yet. Mirrors test_llm_timing_gemini_provider.py's
structure closely.
"""

from __future__ import annotations

import json

import pytest

from app.analysis.llm_timing.openai_provider import (
    DEFAULT_OPENAI_MODEL_ID,
    OpenAICallResult,
    OpenAITimingProvider,
    _classify_status_error,
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
    client = _FakeClient(lambda model, parts, cfg: OpenAICallResult(text=CONFIRMED_JSON))
    provider = OpenAITimingProvider(client, sleep_fn=_RecordingSleep())
    response = provider.analyze(_request())
    assert response.model_id == DEFAULT_OPENAI_MODEL_ID
    assert client.calls[0]["model"] == DEFAULT_OPENAI_MODEL_ID


def test_model_id_is_configurable_and_pinned_explicitly():
    client = _FakeClient(lambda model, parts, cfg: OpenAICallResult(text=CONFIRMED_JSON))
    provider = OpenAITimingProvider(client, model_id="gpt-5", sleep_fn=_RecordingSleep())
    response = provider.analyze(_request())
    assert response.model_id == "gpt-5"
    assert client.calls[0]["model"] == "gpt-5"


def test_sends_one_text_and_one_image_part_per_frame_plus_the_prompt():
    client = _FakeClient(lambda model, parts, cfg: OpenAICallResult(text=CONFIRMED_JSON))
    provider = OpenAITimingProvider(client, sleep_fn=_RecordingSleep())
    provider.analyze(_request())
    parts = client.calls[0]["parts"]
    # prompt text + (label + image) per frame, for 2 frames = 1 + 2*2 = 5
    assert len(parts) == 5
    assert parts[0] == {"text": "analyze this"}
    assert "4.000" in parts[1]["text"]
    assert "inline_data" in parts[2]
    assert parts[2]["inline_data"]["mime_type"] == "image/jpeg"


def test_response_round_trips_through_parse_raw_response_to_confirmed():
    client = _FakeClient(lambda model, parts, cfg: OpenAICallResult(text=CONFIRMED_JSON))
    provider = OpenAITimingProvider(client, sleep_fn=_RecordingSleep())
    response = provider.analyze(_request())
    verdict = parse_raw_response(response, prompt_version="test-v1", min_confidence=0.5)
    assert verdict.status is TimingStatus.CONFIRMED
    assert verdict.start_s == 4.0
    assert verdict.end_s == 20.6


def test_usage_is_propagated_when_the_client_reports_it():
    client = _FakeClient(
        lambda model, parts, cfg: OpenAICallResult(
            text=CONFIRMED_JSON, prompt_tokens=1200, completion_tokens=80
        )
    )
    provider = OpenAITimingProvider(client, sleep_fn=_RecordingSleep())
    response = provider.analyze(_request())
    assert response.prompt_tokens == 1200
    assert response.completion_tokens == 80
    assert response.retries == 0


# --------------------------------------------------------------------------- #
# Retry policy: only transient failures are retried, retries counts retries
# *after* the initial attempt - same discipline as the Gemini adapter.
# --------------------------------------------------------------------------- #


def test_an_unclassified_exception_is_not_retried_by_default():
    def respond(model, parts, cfg):
        raise RuntimeError("upstream 503")

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = OpenAITimingProvider(client, max_retries=3, sleep_fn=sleep)
    response = provider.analyze(_request())
    assert response.error is not None
    assert "upstream 503" in response.error
    assert response.retries == 0
    assert sleep.delays == []


def test_permanent_provider_error_is_never_retried():
    def respond(model, parts, cfg):
        raise PermanentProviderError("bad request")

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = OpenAITimingProvider(client, max_retries=3, sleep_fn=sleep)
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
        return OpenAICallResult(text=CONFIRMED_JSON)

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = OpenAITimingProvider(
        client, max_retries=3, backoff_base_s=1.0, backoff_max_s=10.0, sleep_fn=sleep
    )
    response = provider.analyze(_request())
    assert response.error is None
    assert response.retries == 2
    assert attempts["n"] == 3
    assert sleep.delays == [1.0, 2.0]


def test_bare_timeout_and_connection_errors_are_retried_like_transient():
    attempts = {"n": 0}

    def respond(model, parts, cfg):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("deadline exceeded")
        if attempts["n"] == 2:
            raise ConnectionError("reset by peer")
        return OpenAICallResult(text=CONFIRMED_JSON)

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = OpenAITimingProvider(client, max_retries=3, sleep_fn=sleep)
    response = provider.analyze(_request())
    assert response.error is None
    assert response.retries == 2


def test_max_retries_zero_means_zero_retries_on_immediate_failure():
    def respond(model, parts, cfg):
        raise TransientProviderError("rate limited")

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = OpenAITimingProvider(client, max_retries=0, sleep_fn=sleep)
    response = provider.analyze(_request())
    assert response.error is not None
    assert response.retries == 0
    assert sleep.delays == []


def test_backoff_is_capped_at_backoff_max_s():
    def respond(model, parts, cfg):
        raise TransientProviderError("still failing")

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = OpenAITimingProvider(
        client, max_retries=5, backoff_base_s=1.0, backoff_max_s=3.0, sleep_fn=sleep
    )
    provider.analyze(_request())
    assert sleep.delays == [1.0, 2.0, 3.0, 3.0, 3.0]


def test_exhausting_retries_still_returns_a_response_not_a_raised_exception():
    def respond(model, parts, cfg):
        raise TransientProviderError("always fails")

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = OpenAITimingProvider(client, max_retries=2, sleep_fn=sleep)
    response = provider.analyze(_request())
    assert response.error is not None
    assert response.retries == 2

    verdict = parse_raw_response(response, prompt_version="test-v1", min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "provider_error" in verdict.reason_codes


def test_rejects_empty_model_id():
    client = _FakeClient(lambda model, parts, cfg: OpenAICallResult(text=CONFIRMED_JSON))
    with pytest.raises(ConfigurationError):
        OpenAITimingProvider(client, model_id="  ")


def test_rejects_negative_max_retries():
    client = _FakeClient(lambda model, parts, cfg: OpenAICallResult(text=CONFIRMED_JSON))
    with pytest.raises(ConfigurationError):
        OpenAITimingProvider(client, max_retries=-1)


def test_rejects_negative_backoff():
    client = _FakeClient(lambda model, parts, cfg: OpenAICallResult(text=CONFIRMED_JSON))
    with pytest.raises(ConfigurationError):
        OpenAITimingProvider(client, backoff_base_s=-1.0)


# --------------------------------------------------------------------------- #
# _classify_status_error: 429 transient, every other status code permanent -
# confirmed directly against openai-python's actual exception hierarchy (see
# the module docstring), not a best-effort guess.
# --------------------------------------------------------------------------- #


class _FakeStatusError(Exception):
    """Stands in for an openai.APIStatusError subclass, carrying a
    status_code attribute the same way the real hierarchy does."""

    def __init__(self, status_code: int, message: str = "status error") -> None:
        super().__init__(message)
        self.status_code = status_code
        self._message = message

    def __str__(self) -> str:
        return self._message


class _FakeHeaders(dict):
    """dict subclass so hasattr(headers, "get") is True, like httpx.Headers."""


class _FakeResponse:
    def __init__(self, headers: dict) -> None:
        self.headers = _FakeHeaders(headers)


class _FakeStatusErrorWithRetryAfterHeader(_FakeStatusError):
    def __init__(self, status_code: int, retry_after_header: str) -> None:
        super().__init__(status_code)
        self.response = _FakeResponse({"Retry-After": retry_after_header})


def test_429_classifies_as_transient():
    classified = _classify_status_error(_FakeStatusError(429))
    assert isinstance(classified, TransientProviderError)


@pytest.mark.parametrize("code", [400, 401, 403, 404, 409, 422])
def test_other_status_codes_classify_as_permanent(code):
    classified = _classify_status_error(_FakeStatusError(code))
    assert isinstance(classified, PermanentProviderError)


def test_an_exception_with_no_status_code_classifies_as_permanent():
    classified = _classify_status_error(RuntimeError("shape unknown"))
    assert isinstance(classified, PermanentProviderError)


def test_retry_after_is_captured_from_a_response_header_on_429():
    exc = _FakeStatusErrorWithRetryAfterHeader(429, retry_after_header="7")
    classified = _classify_status_error(exc)
    assert isinstance(classified, TransientProviderError)
    assert classified.retry_after_s == 7.0


def test_retry_after_is_none_when_nothing_matches():
    classified = _classify_status_error(_FakeStatusError(429))
    assert isinstance(classified, TransientProviderError)
    assert classified.retry_after_s is None


def test_analyze_honors_a_server_suggested_retry_after_over_computed_backoff():
    attempts = {"n": 0}

    def respond(model, parts, cfg):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _classify_status_error(
                _FakeStatusErrorWithRetryAfterHeader(429, retry_after_header="9")
            )
        return OpenAICallResult(text=CONFIRMED_JSON)

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = OpenAITimingProvider(
        client, max_retries=2, backoff_base_s=1.0, backoff_max_s=10.0, sleep_fn=sleep
    )
    response = provider.analyze(_request())
    assert response.error is None
    assert sleep.delays == [9.0]
