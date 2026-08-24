"""Versioned, injected model pricing - never hard-coded invisibly in a
provider adapter.

Per the supervisor's authorization: the evaluation contract needs real
per-analysis cost evidence before any live gate. A provider adapter reports
usage (``RawProviderResponse.prompt_tokens``/``completion_tokens``); this
module is the one place that turns usage into an estimated dollar cost, so
the rate a given model was priced at is always visible and swappable - never
buried inline in adapter code.

``PRICING_TABLE`` is empty by default: no model is priced until an entry is
added here (or injected via :func:`estimate_cost_usd`'s ``table`` parameter),
so an unpriced model's cost is reported as ``None`` (unknown), never a
silent zero or a guess.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICING_TABLE_VERSION = "2026-08-22-v2"


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1,000 tokens, input and output priced separately."""

    input_usd_per_1k_tokens: float
    output_usd_per_1k_tokens: float


# Empty by default - see the module docstring. Populate as real models are
# wired in (one entry per pinned model_id, not per provider), and bump
# PRICING_TABLE_VERSION whenever a rate changes so old logged estimates
# remain interpretable against the table version that produced them.
#
# gemini-2.5-flash-lite: $0.10 / 1M input tokens, $0.40 / 1M output tokens.
#
# SOURCING: this sandbox's network egress to ai.google.dev is blocked, so
# this number was never fetched directly here - it was first entered from
# third-party aggregators, then confirmed correct (and re-typed as-is) per
# the supervisor's review of this PR, who reported reading it directly off
# the official page:
#   https://ai.google.dev/gemini-api/docs/pricing
# No shutdown/retirement date is listed for this model on the official
# deprecations page either, per the same review:
#   https://ai.google.dev/gemini-api/docs/deprecations
# (An earlier version of this comment claimed an 2026-10-16 retirement
# date from third-party sources - that claim was wrong, likely confusion
# with a different preview model, and has been removed.) Anyone who can
# reach those pages directly should still treat this as worth a periodic
# recheck, not a permanently-settled fact.
#
# gpt-5-mini: $0.25 / 1M input tokens, $2.00 / 1M output tokens, per the
# supervisor's authorization for the OpenAI adapter (official model record:
# https://developers.openai.com/api/docs/models/gpt-5-mini) - this figure
# was supplied directly by the supervisor, not independently fetched here
# (this sandbox's network egress to developers.openai.com is blocked, same
# as ai.google.dev above).
PRICING_TABLE: dict[str, ModelPricing] = {
    "gemini-2.5-flash-lite": ModelPricing(
        input_usd_per_1k_tokens=0.0001, output_usd_per_1k_tokens=0.0004
    ),
    "gpt-5-mini": ModelPricing(input_usd_per_1k_tokens=0.00025, output_usd_per_1k_tokens=0.002),
}


def estimate_cost_usd(
    model_id: str,
    prompt_tokens: int | None,
    completion_tokens: int | None,
    *,
    table: dict[str, ModelPricing] | None = None,
) -> float | None:
    """Estimate USD cost for one call, or ``None`` if it can't be estimated.

    Returns ``None`` - never ``0.0`` - when the model isn't in the pricing
    table or usage wasn't reported, so "unknown cost" is never
    indistinguishable from "this call was free".
    """
    pricing_table = table if table is not None else PRICING_TABLE
    pricing = pricing_table.get(model_id)
    if pricing is None or prompt_tokens is None or completion_tokens is None:
        return None
    return (
        prompt_tokens / 1000 * pricing.input_usd_per_1k_tokens
        + completion_tokens / 1000 * pricing.output_usd_per_1k_tokens
    )
