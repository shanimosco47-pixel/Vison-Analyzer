"""Tests for build_default_openai_client / build_default_gemini_client -
the two real vendor client factories.

Both import their SDK lazily, inside their own function body, and neither
SDK (``openai``, ``google-genai``) is installed in this environment - see
each function's own docstring. These tests exploit that laziness rather
than skip it: a fake module is injected into ``sys.modules`` before the
factory is called, so ``from openai import (...)``/``from google import
genai`` bind to the fake instead of attempting a real import. That proves
two Codex-review-driven properties neither factory's other tests can
reach without a real SDK installed:

1. a resolved secret passed via the new ``api_key`` keyword argument goes
   straight to the SDK client constructor and is never written to
   ``os.environ`` (the original design's bug, caught by a Codex review of
   the app-integration slice - see ``app/services/llm_engine_store.py``'s
   module docstring);
2. the new ``timeout_s`` argument reaches the SDK client constructor, so a
   stalled request is bounded rather than hanging indefinitely.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

from app.errors import ConfigurationError

# --------------------------------------------------------------------------- #
# OpenAI
# --------------------------------------------------------------------------- #


def _install_fake_openai_module(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    class _FakeOpenAI:
        def __init__(self, *, api_key, max_retries, timeout):
            captured["api_key"] = api_key
            captured["max_retries"] = max_retries
            captured["timeout"] = timeout

    fake_module = types.ModuleType("openai")
    fake_module.OpenAI = _FakeOpenAI
    fake_module.APIConnectionError = type("APIConnectionError", (Exception,), {})
    fake_module.APITimeoutError = type("APITimeoutError", (fake_module.APIConnectionError,), {})
    fake_module.APIStatusError = type("APIStatusError", (Exception,), {})
    fake_module.InternalServerError = type("InternalServerError", (Exception,), {})
    monkeypatch.setitem(sys.modules, "openai", fake_module)


def test_openai_factory_passes_a_given_api_key_directly_to_the_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.analysis.llm_timing.openai_provider import build_default_openai_client

    captured: dict = {}
    _install_fake_openai_module(monkeypatch, captured)
    env_snapshot_before = dict(os.environ)

    client = build_default_openai_client(api_key="sk-test-value")

    assert client is not None
    assert captured["api_key"] == "sk-test-value"
    # Never written to the process environment.
    assert dict(os.environ) == env_snapshot_before


def test_openai_factory_passes_the_configured_timeout_to_the_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.analysis.llm_timing.openai_provider import build_default_openai_client

    captured: dict = {}
    _install_fake_openai_module(monkeypatch, captured)

    build_default_openai_client(api_key="sk-test-value", timeout_s=45.0)

    assert captured["timeout"] == 45.0


def test_openai_factory_still_reads_the_named_env_var_when_no_api_key_is_given(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.analysis.llm_timing.openai_provider import build_default_openai_client

    captured: dict = {}
    _install_fake_openai_module(monkeypatch, captured)
    monkeypatch.setenv("MY_OPENAI_KEY", "raw-dev-key")

    build_default_openai_client("MY_OPENAI_KEY")

    assert captured["api_key"] == "raw-dev-key"


def test_openai_client_reports_the_configured_timeout_in_its_failure_message(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.analysis.llm_timing.openai_provider import build_default_openai_client
    from app.analysis.llm_timing.provider import TransientProviderError

    fake_module = types.ModuleType("openai")
    api_connection_error = type("APIConnectionError", (Exception,), {})
    api_timeout_error = type("APITimeoutError", (api_connection_error,), {})
    fake_module.APIConnectionError = api_connection_error
    fake_module.APITimeoutError = api_timeout_error
    fake_module.APIStatusError = type("APIStatusError", (Exception,), {})
    fake_module.InternalServerError = type("InternalServerError", (Exception,), {})

    class _FakeResponses:
        def create(self, **kwargs):
            raise api_timeout_error("Request timed out.")

    class _FakeOpenAI:
        def __init__(self, **kwargs):
            self.responses = _FakeResponses()

    fake_module.OpenAI = _FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)

    client = build_default_openai_client(api_key="sk-test-value", timeout_s=7.0)
    with pytest.raises(TransientProviderError) as exc_info:
        client.generate_content(
            model="gpt-5-mini",
            parts=[{"text": "hi"}],
            generation_config={"pass_name": "coarse", "frame_timestamps_s": []},
        )
    assert "7s" in str(exc_info.value)


def test_openai_factory_rejects_a_missing_credential_before_touching_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.analysis.llm_timing.openai_provider import build_default_openai_client

    monkeypatch.delitem(sys.modules, "openai", raising=False)
    monkeypatch.delenv("THIS_VAR_IS_NEVER_SET", raising=False)

    with pytest.raises(ConfigurationError):
        build_default_openai_client("THIS_VAR_IS_NEVER_SET")


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #


def _install_fake_google_genai_module(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    class _FakeHttpOptions:
        def __init__(self, *, timeout):
            captured["http_options_timeout"] = timeout

    class _FakeClient:
        def __init__(self, *, api_key, http_options):
            captured["api_key"] = api_key
            captured["http_options"] = http_options

    class _FakeGenerateContentConfig:
        def __init__(self, **kwargs):
            pass

    class _FakePart:
        @staticmethod
        def from_text(*, text):
            return {"text": text}

        @staticmethod
        def from_bytes(*, data, mime_type):
            return {"data": data, "mime_type": mime_type}

    genai_module = types.ModuleType("google.genai")
    genai_module.Client = _FakeClient

    errors_module = types.ModuleType("google.genai.errors")
    errors_module.ClientError = type("ClientError", (Exception,), {})
    errors_module.ServerError = type("ServerError", (Exception,), {})

    types_module = types.ModuleType("google.genai.types")
    types_module.HttpOptions = _FakeHttpOptions
    types_module.GenerateContentConfig = _FakeGenerateContentConfig
    types_module.Part = _FakePart

    google_module = types.ModuleType("google")
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.errors", errors_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)


def test_gemini_factory_passes_a_given_api_key_directly_to_the_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.analysis.llm_timing.gemini_provider import build_default_gemini_client

    captured: dict = {}
    _install_fake_google_genai_module(monkeypatch, captured)
    env_snapshot_before = dict(os.environ)

    client = build_default_gemini_client(api_key="gm-test-value")

    assert client is not None
    assert captured["api_key"] == "gm-test-value"
    assert dict(os.environ) == env_snapshot_before


def test_gemini_factory_passes_the_configured_timeout_to_the_sdk_client(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.analysis.llm_timing.gemini_provider import build_default_gemini_client

    captured: dict = {}
    _install_fake_google_genai_module(monkeypatch, captured)

    build_default_gemini_client(api_key="gm-test-value", timeout_s=30.0)

    # HttpOptions.timeout is documented (unverified) as milliseconds.
    assert captured["http_options_timeout"] == 30_000


def test_gemini_factory_still_reads_the_named_env_var_when_no_api_key_is_given(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.analysis.llm_timing.gemini_provider import build_default_gemini_client

    captured: dict = {}
    _install_fake_google_genai_module(monkeypatch, captured)
    monkeypatch.setenv("MY_GEMINI_KEY", "raw-dev-key")

    build_default_gemini_client("MY_GEMINI_KEY")

    assert captured["api_key"] == "raw-dev-key"


def test_gemini_client_reports_the_configured_timeout_in_its_failure_message(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.analysis.llm_timing.gemini_provider import build_default_gemini_client
    from app.analysis.llm_timing.provider import TransientProviderError

    class _ReadTimeout(Exception):
        pass

    class _FakeModels:
        def generate_content(self, **kwargs):
            raise _ReadTimeout("timed out")

    class _FakeClient:
        def __init__(self, **kwargs):
            self.models = _FakeModels()

    genai_module = types.ModuleType("google.genai")
    genai_module.Client = _FakeClient
    errors_module = types.ModuleType("google.genai.errors")
    errors_module.ClientError = type("ClientError", (Exception,), {})
    errors_module.ServerError = type("ServerError", (Exception,), {})
    types_module = types.ModuleType("google.genai.types")
    types_module.HttpOptions = lambda **kwargs: None
    types_module.GenerateContentConfig = lambda **kwargs: None
    types_module.Part = type("Part", (), {"from_text": staticmethod(lambda **k: k)})
    google_module = types.ModuleType("google")
    google_module.genai = genai_module

    monkeypatch.setitem(sys.modules, "google", google_module)
    monkeypatch.setitem(sys.modules, "google.genai", genai_module)
    monkeypatch.setitem(sys.modules, "google.genai.errors", errors_module)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_module)

    client = build_default_gemini_client(api_key="gm-test-value", timeout_s=9.0)
    with pytest.raises(TransientProviderError) as exc_info:
        client.generate_content(
            model="gemini-2.5-flash", parts=[{"text": "hi"}], generation_config={}
        )
    assert "9s" in str(exc_info.value)


def test_gemini_factory_rejects_a_missing_credential_before_touching_the_sdk(
    monkeypatch: pytest.MonkeyPatch,
):
    from app.analysis.llm_timing.gemini_provider import build_default_gemini_client

    monkeypatch.delitem(sys.modules, "google", raising=False)
    monkeypatch.delenv("THIS_VAR_IS_NEVER_SET", raising=False)

    with pytest.raises(ConfigurationError):
        build_default_gemini_client("THIS_VAR_IS_NEVER_SET")


def test_looks_like_a_timeout_matches_common_timeout_exception_shapes():
    from app.analysis.llm_timing.gemini_provider import _looks_like_a_timeout

    class ReadTimeout(Exception):
        pass

    class ConnectTimeout(Exception):
        pass

    assert _looks_like_a_timeout(ReadTimeout()) is True
    assert _looks_like_a_timeout(ConnectTimeout()) is True
    assert _looks_like_a_timeout(TimeoutError()) is True
    assert _looks_like_a_timeout(ValueError()) is False
    assert _looks_like_a_timeout(ConnectionError()) is False
