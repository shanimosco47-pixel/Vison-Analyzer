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
    _classify_client_error,
    _client_error_retry_after_s,
    _client_error_status_code,
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


# --------------------------------------------------------------------------- #
# 408/429 client-error classification (Codex re-review round 3, finding 2):
# a ClientError-shaped fake exception is unit-tested directly, no real SDK
# involved - see _classify_client_error's own docstring for why.
# --------------------------------------------------------------------------- #


class _FakeClientErrorWithCode:
    """Stands in for google.genai.errors.ClientError, carrying a status code
    via the ``code`` attribute (one of the two attribute names checked)."""

    def __init__(self, code: int, message: str = "client error") -> None:
        self.code = code
        super_message = message
        self._message = super_message

    def __str__(self) -> str:
        return self._message


class _FakeClientErrorWithStatusCode:
    """Same idea, but via the other checked attribute name: status_code."""

    def __init__(self, status_code: int, message: str = "client error") -> None:
        self.status_code = status_code
        self._message = message

    def __str__(self) -> str:
        return self._message


class _FakeClientErrorWithRetryAfter(_FakeClientErrorWithCode):
    """A 429 that also carries a direct retry_after attribute."""

    def __init__(self, code: int, retry_after: float, message: str = "rate limited") -> None:
        super().__init__(code, message)
        self.retry_after = retry_after


class _FakeHeaders(dict):
    """dict subclass so hasattr(headers, "get") is True, like a real
    requests/httpx headers mapping."""


class _FakeResponse:
    def __init__(self, headers: dict) -> None:
        self.headers = _FakeHeaders(headers)


class _FakeClientErrorWithRetryAfterHeader(_FakeClientErrorWithCode):
    """A 429 with no direct retry_after attribute, but a response.headers
    Retry-After entry - the other shape _client_error_retry_after_s checks."""

    def __init__(self, code: int, retry_after_header: str, message: str = "rate limited") -> None:
        super().__init__(code, message)
        self.response = _FakeResponse({"Retry-After": retry_after_header})


@pytest.mark.parametrize("code", [429, 408])
def test_429_and_408_client_errors_classify_as_transient(code):
    exc = _FakeClientErrorWithCode(code)
    classified = _classify_client_error(exc)
    assert isinstance(classified, TransientProviderError)


@pytest.mark.parametrize("code", [400, 401, 403, 413, 422, 500])
def test_other_status_codes_classify_as_permanent(code):
    exc = _FakeClientErrorWithCode(code)
    classified = _classify_client_error(exc)
    assert isinstance(classified, PermanentProviderError)


def test_an_exception_with_no_recognisable_status_code_classifies_as_permanent():
    """No code/status_code attribute at all - the safe default: never retry
    what can't be classified."""
    exc = RuntimeError("something went wrong, shape unknown")
    classified = _classify_client_error(exc)
    assert isinstance(classified, PermanentProviderError)


def test_status_code_is_read_from_either_code_or_status_code_attribute():
    assert _client_error_status_code(_FakeClientErrorWithCode(429)) == 429
    assert _client_error_status_code(_FakeClientErrorWithStatusCode(408)) == 408
    assert _client_error_status_code(RuntimeError("no code here")) is None


def test_retry_after_is_captured_from_a_direct_attribute():
    exc = _FakeClientErrorWithRetryAfter(429, retry_after=12.5)
    classified = _classify_client_error(exc)
    assert isinstance(classified, TransientProviderError)
    assert classified.retry_after_s == 12.5


def test_retry_after_is_captured_from_a_response_header():
    exc = _FakeClientErrorWithRetryAfterHeader(429, retry_after_header="7")
    assert _client_error_retry_after_s(exc) == 7.0
    classified = _classify_client_error(exc)
    assert isinstance(classified, TransientProviderError)
    assert classified.retry_after_s == 7.0


def test_retry_after_is_none_when_nothing_matches():
    exc = _FakeClientErrorWithCode(429)
    assert _client_error_retry_after_s(exc) is None
    classified = _classify_client_error(exc)
    assert isinstance(classified, TransientProviderError)
    assert classified.retry_after_s is None


def test_analyze_honors_a_server_suggested_retry_after_over_computed_backoff():
    """When the raised TransientProviderError carries a retry_after_s (as
    _classify_client_error produces for a real 429), the retry loop must
    sleep that long instead of its own exponential backoff guess."""
    attempts = {"n": 0}

    def respond(model, parts, cfg):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _classify_client_error(_FakeClientErrorWithRetryAfter(429, retry_after=9.0))
        return GeminiCallResult(text=CONFIRMED_JSON)

    client = _FakeClient(respond)
    sleep = _RecordingSleep()
    provider = GeminiTimingProvider(
        client, max_retries=2, backoff_base_s=1.0, backoff_max_s=10.0, sleep_fn=sleep
    )
    response = provider.analyze(_request())
    assert response.error is None
    assert sleep.delays == [9.0]  # server-suggested delay, not the 1.0 backoff guess
