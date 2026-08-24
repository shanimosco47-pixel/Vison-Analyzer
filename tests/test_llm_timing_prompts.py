"""Content assertions for prompts.py - a category this spike didn't need
until now: every prior round could be proven by pipeline/provider-level
regressions alone, because the fix was in *code*. This round's fix (a real
whole-clip rerun on commit 27587f5 nominated a transient early shortening
at 18.5s instead of the true break at ~20.5s) is a *wording* fix -
PROMPT_END_COARSE_V1's own instructions told the model to report its best
candidate even without confirmation and to prefer an earlier one when
torn, exactly the bias that picks a transient over a later, better-
supported candidate. No pipeline code needed to change (the single
end-coarse/end-validate call structure is unchanged - the whole sparse
batch was already visible to the model in one request), so the only way
to pin the fix down at all is to assert on the prompt text itself.
"""

from __future__ import annotations

from app.analysis.llm_timing.prompts import (
    PROMPT_END_COARSE_V1,
    PROMPT_END_COARSE_V1_ID,
    PROMPT_END_COARSE_V2,
    PROMPT_END_COARSE_V2_ID,
    PROMPTS,
)


def test_end_coarse_v1_is_preserved_unchanged_as_the_historical_record():
    # V1 is retired but never mutated - the real rerun it was scored
    # against stays reproducible from its own prompt_version alone.
    assert "report your single best candidate even if you cannot fully" in PROMPT_END_COARSE_V1
    assert "prefer the EARLIER" in PROMPT_END_COARSE_V1


def test_end_coarse_v2_removes_the_early_bias_that_caused_the_real_failure():
    # The exact phrasing identified as the root cause of the 18.5s
    # transient being nominated over the true ~20.5s break must be gone.
    assert "report your single best candidate even if you cannot fully" not in PROMPT_END_COARSE_V2
    assert "prefer the EARLIER" not in PROMPT_END_COARSE_V2


def test_end_coarse_v2_instructs_rejecting_a_transient_and_continuing_chronologically():
    lowered = PROMPT_END_COARSE_V2.lower()
    assert "transient" in lowered
    assert "reject" in lowered
    # Must scan the whole batch, not stop at the first shorter-looking frame.
    assert "whole batch" in lowered
    assert "keep scanning" in lowered


def test_end_coarse_v2_requires_later_checkpoints_to_support_the_candidate():
    lowered = PROMPT_END_COARSE_V2.lower()
    assert "later frames" in lowered or "later checkpoints" in lowered
    assert "recover" in lowered


def test_end_coarse_v2_still_abstains_when_every_candidate_is_a_transient():
    lowered = PROMPT_END_COARSE_V2.lower()
    assert "no_break_found" in PROMPT_END_COARSE_V2
    assert "abstain" in lowered


def test_end_coarse_v2_still_uses_the_timestamp_copy_rule_and_degenerate_point_shape():
    # Unrelated disciplines from V1 must survive the rewrite unchanged.
    assert "TIMESTAMP RULE" in PROMPT_END_COARSE_V2
    assert "COPIED" in PROMPT_END_COARSE_V2 and "EXACTLY" in PROMPT_END_COARSE_V2
    assert '"start_s": <float seconds, copied exactly' in PROMPT_END_COARSE_V2
    assert '"end_s": <float seconds, copied exactly' in PROMPT_END_COARSE_V2


def test_both_end_coarse_versions_are_registered_under_their_own_ids():
    assert PROMPTS[PROMPT_END_COARSE_V1_ID] is PROMPT_END_COARSE_V1
    assert PROMPTS[PROMPT_END_COARSE_V2_ID] is PROMPT_END_COARSE_V2
    assert PROMPT_END_COARSE_V1_ID != PROMPT_END_COARSE_V2_ID
