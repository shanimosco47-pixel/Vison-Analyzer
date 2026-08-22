"""Centralized, bounded sanitization for any free text that came from a
provider or model and might reach logs, diagnostics, ``to_dict()`` output,
or a future UI.

Nothing in this package should put untrusted free text (a provider error
message, a model's own ``raw_notes``) into a returned object without routing
it through :func:`sanitize_untrusted_text` first. Two properties matter:

*   **Redaction**: anything credential-shaped is masked. This is a heuristic
    safety net (see the module docstring in ``engine_config.py`` for the same
    caveat) - it catches the easy case of a vendor SDK's exception text or a
    model's own output echoing back something that looks like a key, but it
    is not a substitute for provider implementations raising clean errors,
    or for secret scanning elsewhere in the toolchain.
*   **Bounding**: untrusted text is truncated to a fixed maximum length
    before it can bloat a log line, a diagnostics payload, or a UI panel.
"""

from __future__ import annotations

import re

MAX_UNTRUSTED_TEXT_LENGTH = 2000

# Unanchored, with word boundaries, so these can be found anywhere inside a
# larger free-text string (an error message, a model's raw_notes) rather
# than only matching a string that is *nothing but* a secret-shaped token.
_RAW_SECRET_SCAN_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\b[A-Za-z0-9_-]{32,}\b"),
)


def sanitize_untrusted_text(text: str, *, max_length: int = MAX_UNTRUSTED_TEXT_LENGTH) -> str:
    """Redact anything credential-shaped, then bound the result's length."""
    redacted = text
    for pattern in _RAW_SECRET_SCAN_PATTERNS:
        redacted = pattern.sub("[redacted]", redacted)
    if len(redacted) > max_length:
        redacted = redacted[:max_length] + "...[truncated]"
    return redacted
