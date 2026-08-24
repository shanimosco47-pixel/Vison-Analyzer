"""Tests for app/services/secret_store.py.

No real OS keyring is touched here - a fake backend (matching the
``keyring`` module's own set_password/get_password/delete_password shape)
is injected via ``backend_factory``, so these run identically whether or
not the ``keyring`` package (or an OS secret service) is actually
installed on the machine running the tests.
"""

from __future__ import annotations

import pytest

from app.services.secret_store import SecretStore, SecretStoreUnavailable


class _FakeKeyringBackend:
    """In-memory stand-in for the real ``keyring`` module's three
    functions - good enough to prove SecretStore's own logic without a
    real OS secret service."""

    def __init__(self) -> None:
        self._values: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._values[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._values.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        try:
            del self._values[(service, username)]
        except KeyError as exc:
            raise LookupError("not found") from exc


class _AlwaysBrokenBackend:
    def set_password(self, service: str, username: str, password: str) -> None:
        raise RuntimeError("OS secret service refused the write")

    def get_password(self, service: str, username: str) -> str | None:
        raise RuntimeError("OS secret service is down")

    def delete_password(self, service: str, username: str) -> None:
        raise RuntimeError("OS secret service is down")


def test_available_is_true_when_the_backend_factory_succeeds():
    store = SecretStore(backend_factory=_FakeKeyringBackend)
    assert store.available is True


def test_available_is_false_when_the_backend_factory_raises():
    def factory():
        raise ImportError("keyring is not installed")

    store = SecretStore(backend_factory=factory)
    assert store.available is False


def test_save_then_resolve_round_trips_the_secret():
    store = SecretStore(backend_factory=_FakeKeyringBackend)
    store.save("engine-1", "sk-super-secret-value")
    assert store.resolve("engine-1") == "sk-super-secret-value"


def test_resolve_returns_none_for_an_unknown_key():
    store = SecretStore(backend_factory=_FakeKeyringBackend)
    assert store.resolve("never-saved") is None


def test_delete_removes_the_secret():
    store = SecretStore(backend_factory=_FakeKeyringBackend)
    store.save("engine-1", "sk-super-secret-value")
    store.delete("engine-1")
    assert store.resolve("engine-1") is None


def test_delete_of_an_unknown_key_does_not_raise():
    store = SecretStore(backend_factory=_FakeKeyringBackend)
    store.delete("never-saved")  # no exception


def test_save_raises_secret_store_unavailable_when_no_backend_exists():
    def factory():
        raise ImportError("keyring is not installed")

    store = SecretStore(backend_factory=factory)
    with pytest.raises(SecretStoreUnavailable):
        store.save("engine-1", "sk-super-secret-value")


def test_resolve_returns_none_when_no_backend_exists_rather_than_raising():
    def factory():
        raise ImportError("keyring is not installed")

    store = SecretStore(backend_factory=factory)
    assert store.resolve("engine-1") is None


def test_save_raises_secret_store_unavailable_when_the_backend_rejects_the_write():
    store = SecretStore(backend_factory=_AlwaysBrokenBackend)
    with pytest.raises(SecretStoreUnavailable):
        store.save("engine-1", "sk-super-secret-value")


def test_resolve_returns_none_when_the_backend_raises_rather_than_propagating():
    store = SecretStore(backend_factory=_AlwaysBrokenBackend)
    assert store.resolve("engine-1") is None


def test_delete_does_not_raise_when_the_backend_raises():
    store = SecretStore(backend_factory=_AlwaysBrokenBackend)
    store.delete("engine-1")  # no exception


def test_secret_store_unavailable_error_message_never_contains_the_secret():
    store = SecretStore(backend_factory=_FakeKeyringBackend)
    secret = "sk-this-must-never-appear-in-any-message"
    try:
        # Force a failure path that still had the secret in scope, to prove
        # the exception text is a fixed string, not built from the value.
        broken = SecretStore(backend_factory=_AlwaysBrokenBackend)
        broken.save("engine-1", secret)
    except SecretStoreUnavailable as exc:
        assert secret not in str(exc)
    else:  # pragma: no cover - defensive, should always raise here
        pytest.fail("expected SecretStoreUnavailable")
    # And the working store never leaked it into __repr__/str either.
    store.save("engine-1", secret)
    assert secret not in repr(store)
