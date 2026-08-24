"""Tests for redaction.sanitize_untrusted_text - in particular the masked/
fingerprinted-credential gap found in the first live OpenAI gate-1 run
(security review): a vendor SDK's own error text can echo a *partially
masked* key back, e.g. "sk-proj-AbCd********WxYz" - visible prefix, a run
of literal asterisks, visible suffix. Neither pre-existing pattern (each
requires one *contiguous* run of alnum characters) catches this shape,
because the asterisks split the token into two short halves.
"""

from __future__ import annotations

from app.analysis.llm_timing.redaction import sanitize_untrusted_text

# A synthetic masked-key error shaped like the real one observed - not a
# real credential, never has been.
_MASKED_KEY_PREFIX = "sk-proj-AbCd1234"
_MASKED_KEY_SUFFIX = "WxYz9876"
_MASKED_KEY_ERROR = (
    f"Error code: 401 - invalid_api_key: Incorrect API key provided: "
    f"{_MASKED_KEY_PREFIX}********{_MASKED_KEY_SUFFIX}. You can find your "
    f"API key at https://platform.openai.com/api-keys."
)


def test_masked_credential_prefix_and_suffix_are_both_redacted():
    sanitized = sanitize_untrusted_text(_MASKED_KEY_ERROR)
    assert _MASKED_KEY_PREFIX not in sanitized
    assert _MASKED_KEY_SUFFIX not in sanitized
    assert "[redacted]" in sanitized
    # The surrounding, non-secret context survives - this isn't a blunt
    # "redact the whole string" fix.
    assert "invalid_api_key" in sanitized
    assert "platform.openai.com/api-keys" in sanitized


def test_masked_credential_with_short_prefix_and_suffix_is_still_redacted():
    # Some vendors show as few as 2-4 visible characters on each side.
    sanitized = sanitize_untrusted_text("key: AB********yz end")
    assert "AB" not in sanitized
    assert "yz" not in sanitized
    assert "[redacted]" in sanitized


def test_plain_contiguous_secret_redaction_still_works():
    # Regression coverage: the fix must not weaken the two pre-existing
    # patterns (a full, unmasked sk- key; any other 32+ char contiguous
    # alnum/dash/underscore token).
    sanitized = sanitize_untrusted_text("key=sk-abcdefghijklmnopqrstuvwxyz123456")
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in sanitized
    assert "[redacted]" in sanitized

    sanitized2 = sanitize_untrusted_text("token AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 leaked")
    assert "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789" not in sanitized2
    assert "[redacted]" in sanitized2


def test_ordinary_text_without_any_secret_shape_is_left_alone():
    text = "the coarse pass abstained: no continuous stream found in the sampled frames"
    assert sanitize_untrusted_text(text) == text
