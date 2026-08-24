"""OS-protected storage for LLM engine API keys.

Supervisor-approved requirement (PR #5 comment thread, app-integration
authorization): a saved API key must live in an OS-protected secret
mechanism - Windows Credential Manager (backed by DPAPI) is the concrete
target platform - never in the browser, the repository, logs, or
diagnostics. This module is the one place that ever touches a real secret
*value*; every other module in this application only ever holds a
``credential_ref`` (see ``app/analysis/llm_timing/engine_config.py``), a
reference this module resolves.

Backed by the third-party ``keyring`` package (an optional dependency, see
``requirements-llm-spike.txt``), which auto-selects the right OS backend -
Windows Credential Manager on Windows, Keychain on macOS, Secret Service on
Linux desktops - without this module needing to know which platform it is
running on. Nothing here imports ``keyring`` at module load time: like
every other optional vendor dependency in this codebase (see
``gemini_provider.build_default_gemini_client`` /
``openai_provider.build_default_openai_client``), the import is lazy and
scoped to one function, so nothing that doesn't actually save or resolve a
secret needs the package installed.

A machine with no usable OS secret backend at all (a minimal headless
Linux box, for instance) degrades to "secrets can't be saved here" rather
than silently falling back to something less protected - the caller is
expected to use an environment-variable reference instead for that case
(see ``llm_engine_store.py``'s "development fallback" path), which never
touches this module at all.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from typing import Protocol

_SERVICE_NAME = "vision-analyzer-llm-engines"


class SecretStoreUnavailable(Exception):
    """No usable OS-protected secret backend on this machine.

    Raised only by :meth:`SecretStore.save` - the one operation that
    cannot silently no-op, since the caller specifically asked for a
    secret to be protected and it wasn't. ``resolve``/``delete`` degrade
    quietly (``None`` / no-op) since there is nothing to protect in either
    case.
    """


class _KeyringBackend(Protocol):
    """The three ``keyring`` module-level functions this module calls,
    narrowed to a Protocol so a test can inject a fake backend without
    the real package installed."""

    def set_password(self, service: str, username: str, password: str) -> None: ...
    def get_password(self, service: str, username: str) -> str | None: ...
    def delete_password(self, service: str, username: str) -> None: ...


def _default_backend() -> _KeyringBackend:
    import keyring  # optional dependency - see requirements-llm-spike.txt

    return keyring  # the module itself already exposes this exact shape


class SecretStore:
    """Save/resolve/delete a secret by an opaque key, never logging or
    otherwise exposing the value itself.

    ``key`` is caller-chosen (this module has no opinion on its shape) -
    ``llm_engine_store.py`` uses one per configured engine, keyed by that
    engine's own ``engine_id``, so deleting an engine cleanly deletes
    exactly its own secret and nothing else's.
    """

    def __init__(
        self,
        *,
        backend_factory: Callable[[], _KeyringBackend] = _default_backend,
        service_name: str = _SERVICE_NAME,
    ) -> None:
        self._backend_factory = backend_factory
        self._service_name = service_name
        self._backend: _KeyringBackend | None = None
        self._probed = False

    def _backend_or_none(self) -> _KeyringBackend | None:
        # Probed once per SecretStore instance, not once per call - the
        # underlying backend (an installed package, an OS service) doesn't
        # change mid-process, and re-importing on every save/resolve/delete
        # would be wasted work on every request.
        if not self._probed:
            try:
                self._backend = self._backend_factory()
            except Exception:
                self._backend = None
            self._probed = True
        return self._backend

    @property
    def available(self) -> bool:
        """Best-effort only: a truthy result means the backend *imported*,
        not that every operation against it will succeed - an OS backend
        can still reject a specific save (see :meth:`save`)."""
        return self._backend_or_none() is not None

    def save(self, key: str, secret: str) -> None:
        """Store ``secret`` under ``key``. Raises :class:`SecretStoreUnavailable`
        - never a raw ``keyring`` exception - if there is no usable backend
        or the backend rejects the write."""
        backend = self._backend_or_none()
        if backend is None:
            raise SecretStoreUnavailable(
                "No OS-protected secret backend is available on this machine. "
                "Use an environment-variable reference instead (see the "
                "'Advanced: use an environment variable' option)."
            )
        try:
            backend.set_password(self._service_name, key, secret)
        except Exception as exc:
            raise SecretStoreUnavailable(
                "The OS secret store rejected the request; the key was not saved."
            ) from exc

    def resolve(self, key: str) -> str | None:
        """The stored secret, or ``None`` if there is no backend or no
        secret saved under ``key`` - never raises."""
        backend = self._backend_or_none()
        if backend is None:
            return None
        try:
            return backend.get_password(self._service_name, key)
        except Exception:
            return None

    def delete(self, key: str) -> None:
        """Remove the secret saved under ``key``, if any - never raises,
        including when nothing was ever saved under this key."""
        backend = self._backend_or_none()
        if backend is None:
            return
        with contextlib.suppress(Exception):
            backend.delete_password(self._service_name, key)  # "already gone" is not a failure
