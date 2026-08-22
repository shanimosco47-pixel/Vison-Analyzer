"""Backend registry for the Experimental LLM analysis section's configured
engines.

Persists :class:`~app.analysis.llm_timing.engine_config.EngineConfig`
metadata (provider, model, display name, enabled) to one JSON file under
``AppConfig.data_dir`` - **never** a secret value, only a ``credential_ref``
(see that class's own docstring for why). This module is where a
``credential_ref`` is actually resolved back into a usable
``TimingProvider``, and it recognises exactly two reference shapes:

- ``"secret:<key>"`` - a real API key the user saved through the UI, stored
  OS-protected via :class:`~app.services.secret_store.SecretStore`.
- ``"env:<VAR_NAME>"`` - a pre-set environment variable the operator
  manages themselves (the documented development fallback); nothing is
  stored server-side for this path beyond the variable's own name.

Building a real provider still goes through the existing
``build_default_openai_client``/``build_default_gemini_client`` factories
unchanged - both read a credential from a *named* environment variable
only, a deliberate pre-existing choice (never accept a literal key through
a function argument that might end up in a traceback or a repr). A
resolved secret is therefore materialized into a per-engine, unique
process environment variable immediately before constructing the client -
see :func:`_env_var_for_secret` - so two engines' credentials, even
resolved concurrently by two independent runs, can never collide on the
same variable name.
"""

from __future__ import annotations

import json
import os
import re
import threading
import uuid

from ..analysis.llm_timing.engine_config import EngineConfig
from ..analysis.llm_timing.provider import TimingProvider
from ..analysis.llm_timing.redaction import sanitize_untrusted_text
from ..config import AppConfig
from ..errors import AnalyzerError, ConfigurationError, NotFoundError
from ..logging_setup import get_logger
from .secret_store import SecretStore, SecretStoreUnavailable

logger = get_logger(__name__)

SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "gemini")

_SECRET_REF_PREFIX = "secret:"
_ENV_REF_PREFIX = "env:"

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _looks_like_an_env_var_name(value: str) -> bool:
    return bool(_ENV_VAR_NAME_RE.match(value))


def _env_var_for_secret(engine_id: str) -> str:
    """A per-engine, deterministic, unique process environment variable
    name - purely an implementation detail of handing a resolved secret to
    the existing vendor client factories, which read a *named* environment
    variable only. Never persisted, never shown to a client; only ever set
    in this process's own ``os.environ`` right before constructing a
    client for this specific engine."""
    return f"VISION_ANALYZER_LLM_ENGINE_SECRET_{engine_id.upper()}"


class LLMEngineStore:
    """CRUD for configured LLM engines, plus resolving one into a real
    :class:`~app.analysis.llm_timing.provider.TimingProvider`."""

    def __init__(self, config: AppConfig, secret_store: SecretStore) -> None:
        self._path = config.data_dir / "llm_engines.json"
        self._secret_store = secret_store
        self._lock = threading.RLock()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, EngineConfig] = self._load()

    # -- persistence ---------------------------------------------------- #

    def _load(self) -> dict[str, EngineConfig]:
        if not self._path.is_file():
            return {}
        try:
            raw = json.loads(self._path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning(
                "Could not read %s, starting with no configured engines: %s", self._path, exc
            )
            return {}
        entries: dict[str, EngineConfig] = {}
        for item in raw if isinstance(raw, list) else []:
            try:
                entry = EngineConfig(
                    engine_id=item["engine_id"],
                    provider_name=item["provider_name"],
                    model_id=item["model_id"],
                    credential_ref=item["credential_ref"],
                    enabled=bool(item.get("enabled", True)),
                    display_name=item.get("display_name", ""),
                )
            except (KeyError, TypeError, ConfigurationError) as exc:
                logger.warning("Skipping a malformed saved engine entry: %s", exc)
                continue
            entries[entry.engine_id] = entry
        return entries

    def _save_locked(self) -> None:
        """Caller must already hold ``self._lock``."""
        payload = [
            {
                "engine_id": e.engine_id,
                "provider_name": e.provider_name,
                "model_id": e.model_id,
                "credential_ref": e.credential_ref,
                "enabled": e.enabled,
                "display_name": e.display_name,
            }
            for e in self._entries.values()
        ]
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self._path)  # atomic on both POSIX and Windows

    # -- CRUD ------------------------------------------------------------ #

    def list(self) -> list[EngineConfig]:
        with self._lock:
            return list(self._entries.values())

    def get(self, engine_id: str) -> EngineConfig:
        with self._lock:
            entry = self._entries.get(engine_id)
        if entry is None:
            raise NotFoundError("That engine configuration no longer exists.")
        return entry

    def create(
        self,
        *,
        provider_name: str,
        model_id: str,
        display_name: str = "",
        api_key: str | None = None,
        env_var: str | None = None,
        enabled: bool = True,
    ) -> EngineConfig:
        provider_name = _validated_provider_name(provider_name)
        model_id = (model_id or "").strip()
        if not model_id:
            raise AnalyzerError("Choose a model ID for this engine.")

        engine_id = uuid.uuid4().hex
        credential_ref = self._new_credential_ref(
            api_key=api_key, env_var=env_var, engine_id=engine_id
        )
        engine = EngineConfig(
            engine_id=engine_id,
            provider_name=provider_name,
            model_id=model_id,
            credential_ref=credential_ref,
            enabled=enabled,
            display_name=(display_name or "").strip(),
        )
        with self._lock:
            self._entries[engine_id] = engine
            self._save_locked()
        logger.info("Configured a new LLM engine %s (%s/%s)", engine_id, provider_name, model_id)
        return engine

    def update(
        self,
        engine_id: str,
        *,
        provider_name: str | None = None,
        model_id: str | None = None,
        display_name: str | None = None,
        enabled: bool | None = None,
        api_key: str | None = None,
        env_var: str | None = None,
    ) -> EngineConfig:
        with self._lock:
            existing = self._entries.get(engine_id)
            if existing is None:
                raise NotFoundError("That engine configuration no longer exists.")

            credential_ref = existing.credential_ref
            if api_key is not None or env_var is not None:
                new_ref = self._new_credential_ref(
                    api_key=api_key, env_var=env_var, engine_id=engine_id
                )
                old_secret_key = _secret_key_of(existing.credential_ref)
                old_ref = f"{_SECRET_REF_PREFIX}{old_secret_key}"
                if old_secret_key is not None and old_ref != new_ref:
                    self._secret_store.delete(old_secret_key)
                credential_ref = new_ref

            new_display_name = (
                existing.display_name if display_name is None else display_name.strip()
            )
            updated = EngineConfig(
                engine_id=engine_id,
                provider_name=_validated_provider_name(provider_name or existing.provider_name),
                model_id=(model_id or existing.model_id).strip() or existing.model_id,
                credential_ref=credential_ref,
                enabled=existing.enabled if enabled is None else enabled,
                display_name=new_display_name,
            )
            self._entries[engine_id] = updated
            self._save_locked()
        logger.info("Updated LLM engine %s", engine_id)
        return updated

    def delete(self, engine_id: str) -> None:
        with self._lock:
            entry = self._entries.pop(engine_id, None)
            if entry is None:
                raise NotFoundError("That engine configuration no longer exists.")
            self._save_locked()
        secret_key = _secret_key_of(entry.credential_ref)
        if secret_key is not None:
            self._secret_store.delete(secret_key)
        logger.info("Deleted LLM engine %s", engine_id)

    # -- credentials ------------------------------------------------------ #

    def _new_credential_ref(
        self, *, api_key: str | None, env_var: str | None, engine_id: str
    ) -> str:
        if bool(api_key) == bool(env_var):
            raise AnalyzerError(
                "Provide exactly one of an API key to save, or the name of an "
                "environment variable that already holds it."
            )
        if env_var is not None:
            env_var = env_var.strip()
            if not env_var or not _looks_like_an_env_var_name(env_var):
                raise AnalyzerError(
                    "That does not look like a valid environment variable name "
                    "(letters, digits, underscores only, must not start with a digit)."
                )
            return f"{_ENV_REF_PREFIX}{env_var}"

        assert api_key is not None
        api_key = api_key.strip()
        if not api_key:
            raise AnalyzerError("The API key was empty.")
        try:
            self._secret_store.save(engine_id, api_key)
        except SecretStoreUnavailable as exc:
            raise AnalyzerError(
                "This machine has no OS-protected place to save an API key. "
                "Set it as an environment variable instead and reference its "
                "name here.",
                detail=str(exc),
            ) from exc
        return f"{_SECRET_REF_PREFIX}{engine_id}"

    def build_provider(self, engine: EngineConfig) -> TimingProvider:
        """Resolve ``engine``'s credential and construct a real
        ``TimingProvider`` for it. Raises :class:`AnalyzerError` (a message
        safe to show a user - never the credential's value) if the
        credential can't be resolved or the provider is unsupported."""
        api_key_env_var = self._materialize_credential(engine)
        if engine.provider_name == "openai":
            from ..analysis.llm_timing.openai_provider import (
                OpenAITimingProvider,
                build_default_openai_client,
            )

            openai_client = build_default_openai_client(api_key_env_var)
            return OpenAITimingProvider(openai_client, model_id=engine.model_id)
        if engine.provider_name == "gemini":
            from ..analysis.llm_timing.gemini_provider import (
                GeminiTimingProvider,
                build_default_gemini_client,
            )

            gemini_client = build_default_gemini_client(api_key_env_var)
            return GeminiTimingProvider(gemini_client, model_id=engine.model_id)
        raise AnalyzerError(f"Unsupported provider '{engine.provider_name}'.")  # pragma: no cover

    def _materialize_credential(self, engine: EngineConfig) -> str:
        ref = engine.credential_ref
        if ref.startswith(_ENV_REF_PREFIX):
            return ref[len(_ENV_REF_PREFIX) :]
        if ref.startswith(_SECRET_REF_PREFIX):
            secret_key = ref[len(_SECRET_REF_PREFIX) :]
            value = self._secret_store.resolve(secret_key)
            if not value:
                raise AnalyzerError(
                    "This engine's saved API key could not be found. Re-enter "
                    "it in the engine's settings."
                )
            env_var = _env_var_for_secret(engine.engine_id)
            os.environ[env_var] = value
            return env_var
        raise AnalyzerError("This engine's credential reference is not valid.")  # pragma: no cover

    def test(self, engine_id: str) -> dict:
        """Best-effort readiness check for the "Test" button: resolves the
        credential and constructs a real provider client. Deliberately does
        NOT make a live API call - that would cost real time/money on every
        click - so ``ok: True`` means "this key/model is configured and the
        client builds", not "a request to the vendor round-tripped"."""
        engine = self.get(engine_id)
        try:
            self.build_provider(engine)
        except AnalyzerError as exc:
            return {"ok": False, "message": exc.user_message}
        except Exception as exc:  # e.g. the vendor SDK package isn't installed
            return {"ok": False, "message": sanitize_untrusted_text(str(exc))}
        return {
            "ok": True,
            "message": "Credential resolved and the client constructed successfully. "
            "This does not confirm a live request would succeed.",
        }


def _validated_provider_name(provider_name: str) -> str:
    provider_name = (provider_name or "").strip().lower()
    if provider_name not in SUPPORTED_PROVIDERS:
        raise AnalyzerError(
            f"Unsupported provider '{provider_name}'. Choose one of: "
            + ", ".join(SUPPORTED_PROVIDERS)
        )
    return provider_name


def _secret_key_of(credential_ref: str) -> str | None:
    if credential_ref.startswith(_SECRET_REF_PREFIX):
        return credential_ref[len(_SECRET_REF_PREFIX) :]
    return None
