"""Tests for app/version.py: the running backend's build identifier.

Supervisor-directed operational requirement: a clearly visible application
version, generated automatically from the backend's own commit, never
hand-edited and never derived from client-supplied data.
"""

from __future__ import annotations

import subprocess

from app.version import app_version


def test_app_version_matches_the_real_git_short_sha():
    app_version.cache_clear()
    expected = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=".",
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert app_version() == expected


def test_app_version_is_cached_not_recomputed_every_call():
    app_version.cache_clear()
    first = app_version()
    info_before = app_version.cache_info()
    second = app_version()
    info_after = app_version.cache_info()
    assert first == second
    assert info_before.hits == 0
    assert info_after.hits == 1


def test_app_version_falls_back_to_unknown_when_git_is_unavailable(monkeypatch):
    app_version.cache_clear()

    def _boom(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(subprocess, "run", _boom)
    assert app_version() == "unknown"
    app_version.cache_clear()


def test_app_version_falls_back_to_unknown_when_git_command_fails(monkeypatch):
    app_version.cache_clear()

    def _fail(*args, **kwargs):
        raise subprocess.CalledProcessError(128, ["git"])

    monkeypatch.setattr(subprocess, "run", _fail)
    assert app_version() == "unknown"
    app_version.cache_clear()
