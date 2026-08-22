"""Tests for app/services/llm_engine_store.py.

Uses a real ``SecretStore`` backed by an in-memory fake keyring backend
(same pattern as ``tests/test_secret_store.py``) - no real OS secret
service or network call anywhere here.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.config import AppConfig
from app.errors import AnalyzerError, NotFoundError
from app.services.llm_engine_store import LLMEngineStore
from app.services.secret_store import SecretStore


class _FakeKeyringBackend:
    def __init__(self) -> None:
        self._values: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._values[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._values.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        self._values.pop((service, username), None)


@pytest.fixture
def secret_store() -> SecretStore:
    backend = _FakeKeyringBackend()
    return SecretStore(backend_factory=lambda: backend)


@pytest.fixture
def config(tmp_path: Path) -> AppConfig:
    return replace(AppConfig(), data_dir=tmp_path / "data")


@pytest.fixture
def store(config: AppConfig, secret_store: SecretStore) -> LLMEngineStore:
    return LLMEngineStore(config, secret_store)


# --------------------------------------------------------------------------- #
# create()
# --------------------------------------------------------------------------- #


def test_create_with_an_api_key_saves_it_via_the_secret_store(
    store: LLMEngineStore, secret_store: SecretStore
):
    engine = store.create(
        provider_name="openai",
        model_id="gpt-5-mini",
        display_name="OpenAI mini",
        api_key="sk-abc123",
    )
    assert engine.credential_ref == f"secret:{engine.engine_id}"
    assert secret_store.resolve(engine.engine_id) == "sk-abc123"


def test_create_with_an_env_var_never_touches_the_secret_store(
    store: LLMEngineStore, secret_store: SecretStore
):
    engine = store.create(
        provider_name="gemini", model_id="gemini-2.5-flash", env_var="MY_GEMINI_KEY"
    )
    assert engine.credential_ref == "env:MY_GEMINI_KEY"
    assert secret_store.resolve(engine.engine_id) is None


def test_create_rejects_both_api_key_and_env_var(store: LLMEngineStore):
    with pytest.raises(AnalyzerError):
        store.create(
            provider_name="openai",
            model_id="gpt-5-mini",
            api_key="sk-abc",
            env_var="OPENAI_API_KEY",
        )


def test_create_rejects_neither_api_key_nor_env_var(store: LLMEngineStore):
    with pytest.raises(AnalyzerError):
        store.create(provider_name="openai", model_id="gpt-5-mini")


def test_create_rejects_an_invalid_env_var_name(store: LLMEngineStore):
    with pytest.raises(AnalyzerError):
        store.create(provider_name="openai", model_id="gpt-5-mini", env_var="not a valid name!")


def test_create_rejects_an_unsupported_provider(store: LLMEngineStore):
    with pytest.raises(AnalyzerError):
        store.create(provider_name="anthropic", model_id="claude", env_var="ANTHROPIC_API_KEY")


def test_create_rejects_an_empty_model_id(store: LLMEngineStore):
    with pytest.raises(AnalyzerError):
        store.create(provider_name="openai", model_id="   ", env_var="OPENAI_API_KEY")


def test_created_engine_to_dict_never_includes_the_credential_ref(store: LLMEngineStore):
    engine = store.create(provider_name="openai", model_id="gpt-5-mini", api_key="sk-abc123")
    payload = engine.to_dict()
    assert "credential_ref" not in payload
    assert "sk-abc123" not in json.dumps(payload)
    assert payload["credential_configured"] is True


def test_the_persisted_file_never_contains_the_raw_api_key(
    store: LLMEngineStore, config: AppConfig
):
    store.create(provider_name="openai", model_id="gpt-5-mini", api_key="sk-super-secret-value")
    raw = (config.data_dir / "llm_engines.json").read_text()
    assert "sk-super-secret-value" not in raw
    assert "secret:" in raw


# --------------------------------------------------------------------------- #
# list()/get()/persistence
# --------------------------------------------------------------------------- #


def test_multiple_engines_are_listed_independently(store: LLMEngineStore):
    a = store.create(
        provider_name="openai", model_id="gpt-5-mini", display_name="A", env_var="A_KEY"
    )
    b = store.create(
        provider_name="gemini", model_id="gemini-2.5-flash", display_name="B", env_var="B_KEY"
    )
    listed = {e.engine_id: e for e in store.list()}
    assert set(listed) == {a.engine_id, b.engine_id}
    assert listed[a.engine_id].display_name == "A"
    assert listed[b.engine_id].display_name == "B"


def test_get_of_an_unknown_engine_raises_not_found(store: LLMEngineStore):
    with pytest.raises(NotFoundError):
        store.get("does-not-exist")


def test_entries_survive_a_reload_from_the_same_data_dir(
    config: AppConfig, secret_store: SecretStore
):
    store = LLMEngineStore(config, secret_store)
    engine = store.create(provider_name="openai", model_id="gpt-5-mini", env_var="OPENAI_API_KEY")

    reloaded = LLMEngineStore(config, secret_store)
    assert reloaded.get(engine.engine_id).model_id == "gpt-5-mini"


# --------------------------------------------------------------------------- #
# update()
# --------------------------------------------------------------------------- #


def test_update_changing_only_display_name_leaves_the_credential_untouched(
    store: LLMEngineStore, secret_store: SecretStore
):
    engine = store.create(provider_name="openai", model_id="gpt-5-mini", api_key="sk-abc123")
    updated = store.update(engine.engine_id, display_name="Renamed")
    assert updated.display_name == "Renamed"
    assert updated.credential_ref == engine.credential_ref
    assert secret_store.resolve(engine.engine_id) == "sk-abc123"


def test_update_rotating_the_api_key_deletes_the_old_secret(
    store: LLMEngineStore, secret_store: SecretStore
):
    engine = store.create(provider_name="openai", model_id="gpt-5-mini", api_key="sk-old-value")
    store.update(engine.engine_id, api_key="sk-new-value")
    assert secret_store.resolve(engine.engine_id) == "sk-new-value"


def test_update_switching_from_api_key_to_env_var_deletes_the_saved_secret(
    store: LLMEngineStore, secret_store: SecretStore
):
    engine = store.create(provider_name="openai", model_id="gpt-5-mini", api_key="sk-old-value")
    updated = store.update(engine.engine_id, env_var="OPENAI_API_KEY")
    assert updated.credential_ref == "env:OPENAI_API_KEY"
    assert secret_store.resolve(engine.engine_id) is None


def test_update_of_an_unknown_engine_raises_not_found(store: LLMEngineStore):
    with pytest.raises(NotFoundError):
        store.update("does-not-exist", display_name="x")


# --------------------------------------------------------------------------- #
# delete()
# --------------------------------------------------------------------------- #


def test_delete_removes_the_engine_and_its_secret(store: LLMEngineStore, secret_store: SecretStore):
    engine = store.create(provider_name="openai", model_id="gpt-5-mini", api_key="sk-abc123")
    store.delete(engine.engine_id)
    with pytest.raises(NotFoundError):
        store.get(engine.engine_id)
    assert secret_store.resolve(engine.engine_id) is None


def test_delete_of_an_unknown_engine_raises_not_found(store: LLMEngineStore):
    with pytest.raises(NotFoundError):
        store.delete("does-not-exist")


def test_delete_of_an_env_var_engine_does_not_touch_the_secret_store(
    store: LLMEngineStore, secret_store: SecretStore, monkeypatch: pytest.MonkeyPatch
):
    engine = store.create(provider_name="openai", model_id="gpt-5-mini", env_var="OPENAI_API_KEY")
    calls = []
    monkeypatch.setattr(secret_store, "delete", lambda key: calls.append(key))
    store.delete(engine.engine_id)
    assert calls == []


# --------------------------------------------------------------------------- #
# build_provider() / test() - credential materialization, frames-only clients
# --------------------------------------------------------------------------- #


def test_build_provider_materializes_a_saved_secret_into_a_unique_env_var_then_builds(
    store: LLMEngineStore, monkeypatch: pytest.MonkeyPatch
):
    engine = store.create(provider_name="openai", model_id="gpt-5-mini", api_key="sk-abc123")

    seen_env_var: dict[str, str] = {}

    class _FakeClient:
        pass

    def fake_build_default_openai_client(api_key_env_var: str):
        import os

        seen_env_var["name"] = api_key_env_var
        seen_env_var["value"] = os.environ.get(api_key_env_var)
        return _FakeClient()

    monkeypatch.setattr(
        "app.analysis.llm_timing.openai_provider.build_default_openai_client",
        fake_build_default_openai_client,
    )

    provider = store.build_provider(engine)
    assert provider is not None
    assert seen_env_var["value"] == "sk-abc123"
    assert engine.engine_id.upper() in seen_env_var["name"]


def test_build_provider_uses_the_operators_own_env_var_for_the_env_reference(
    store: LLMEngineStore, monkeypatch: pytest.MonkeyPatch
):
    engine = store.create(
        provider_name="gemini", model_id="gemini-2.5-flash", env_var="MY_GEMINI_KEY"
    )
    monkeypatch.setenv("MY_GEMINI_KEY", "raw-dev-key")

    seen = {}

    class _FakeClient:
        pass

    def fake_build_default_gemini_client(api_key_env_var: str):
        seen["name"] = api_key_env_var
        return _FakeClient()

    monkeypatch.setattr(
        "app.analysis.llm_timing.gemini_provider.build_default_gemini_client",
        fake_build_default_gemini_client,
    )

    store.build_provider(engine)
    assert seen["name"] == "MY_GEMINI_KEY"


def test_build_provider_raises_a_safe_error_when_the_saved_secret_is_missing(
    store: LLMEngineStore, secret_store: SecretStore
):
    engine = store.create(provider_name="openai", model_id="gpt-5-mini", api_key="sk-abc123")
    secret_store.delete(engine.engine_id)  # simulate it having disappeared from the OS store
    with pytest.raises(AnalyzerError):
        store.build_provider(engine)


def test_test_endpoint_helper_reports_ok_when_the_client_builds(
    store: LLMEngineStore, monkeypatch: pytest.MonkeyPatch
):
    engine = store.create(provider_name="openai", model_id="gpt-5-mini", api_key="sk-abc123")

    class _FakeClient:
        pass

    monkeypatch.setattr(
        "app.analysis.llm_timing.openai_provider.build_default_openai_client",
        lambda api_key_env_var: _FakeClient(),
    )
    result = store.test(engine.engine_id)
    assert result["ok"] is True


def test_test_endpoint_helper_reports_a_safe_message_when_resolution_fails(
    store: LLMEngineStore, secret_store: SecretStore
):
    engine = store.create(provider_name="openai", model_id="gpt-5-mini", api_key="sk-abc123")
    secret_store.delete(engine.engine_id)
    result = store.test(engine.engine_id)
    assert result["ok"] is False
    assert "sk-abc123" not in result["message"]
