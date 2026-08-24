"""This build's identifying version string.

Supervisor-directed operational requirement: the running backend must show
a version, generated automatically from its own commit/build identifier
(a short Git SHA is fine), so the UI, a downloaded audit report, and the
actual running backend can be matched against each other. It must change
automatically on each code revision - never hand-edited - and must never
be derived from anything a client supplies (the URL query string
explicitly does not count as a version).
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def app_version() -> str:
    """The running backend's own short git commit SHA - resolved once per
    process (git is a subprocess call, not something to repeat on every
    request) directly from this checkout, never from request data. Falls
    back to ``"unknown"`` rather than raising when git isn't available
    (e.g. a deployment that copied the source without its ``.git``
    directory) - a missing version is still an honest answer, not a
    fabricated one.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    sha = result.stdout.strip()
    return sha if sha else "unknown"
