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

PRICING_TABLE_VERSION = "2026-08-22-v1"


@dataclass(frozen=True)
class ModelPricing:
    """USD per 1,000 tokens, input and output priced separately."""

    input_usd_per_1k_tokens: float
    output_usd_per_1k_tokens: float


# Empty by default - see the module docstring. Populate as real models are
# wired in (one entry per pinned model_id, not per provider), and bump
# PRICING_TABLE_VERSION whenever a rate changes so old logged estimates
# remain interpretable against the table version that produced them.
PRICING_TABLE: dict[str, ModelPricing] = {}


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
