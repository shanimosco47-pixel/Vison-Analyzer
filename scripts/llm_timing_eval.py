#!/usr/bin/env python3
"""Evaluation harness for the LLM timing spike.

Runs ``app.analysis.llm_timing.pipeline.run_llm_timing`` over a manifest of
labelled clips and reports the metrics the supervisor's evidence gates ask
for: per-clip start/end/duration error, abstention, "false confident" (a
CONFIRMED verdict that was actually wrong by more than the tolerance), and
latency. See ``diagnostics/llm_spike/DESIGN.md`` for gates 1-3 in full and
how this script's output maps onto them.

``--provider stub-perfect`` answers with the manifest's own ground truth and
exists ONLY to prove the harness's plumbing works end-to-end. Its numbers are
not evidence of anything about real accuracy and the report says so loudly.

``--provider gemini`` and ``--provider openai`` are real providers: each
lazily constructs its own SDK-backed client
(``gemini_provider.build_default_gemini_client()`` /
``openai_provider.build_default_openai_client()`` - see
``requirements-llm-spike.txt`` for both optional SDK dependencies) only when
actually selected, reading ``GEMINI_API_KEY``/``OPENAI_API_KEY`` from the
environment only - never a CLI argument, a file, or anything that could end
up logged or committed. Pass ``--model-id`` to override either one's pinned
default (``gemini_provider.DEFAULT_GEMINI_MODEL_ID`` /
``openai_provider.DEFAULT_OPENAI_MODEL_ID``).

Usage:
    python scripts/llm_timing_eval.py manifest.json --provider stub-perfect
    python scripts/llm_timing_eval.py manifest.json --provider gemini
    python scripts/llm_timing_eval.py manifest.json --provider openai
    python scripts/llm_timing_eval.py manifest.json --provider openai --model-id gpt-5 \
        --json out.json

Manifest format (JSON array):
    [
      {"clip_id": "20260820_184144", "video_path": "/path/to/clip.mp4",
       "true_start_s": 3.8, "true_end_s": 20.6}
    ]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.analysis.llm_timing.pipeline import PipelineConfig, run_llm_timing  # noqa: E402
from app.analysis.llm_timing.prompts import PROMPT_V1, PROMPT_V1_ID  # noqa: E402
from app.analysis.llm_timing.provider import (  # noqa: E402
    ProviderRequest,
    RawProviderResponse,
    TimingProvider,
    canned_json_response,
)
from app.analysis.llm_timing.schema import TimingStatus  # noqa: E402

DEFAULT_TOLERANCE_S = 0.75  # the supervisor's gate-1 pass/fail bound


@dataclass
class ClipResult:
    clip_id: str
    status: str  # "confirmed" | "abstain"
    true_start_s: float
    true_end_s: float
    predicted_start_s: float | None
    predicted_end_s: float | None
    start_error_s: float | None
    end_error_s: float | None
    duration_error_s: float | None
    within_tolerance: bool | None  # None when abstained (not applicable)
    false_confident: bool  # CONFIRMED but wrong by more than tolerance
    reason_codes: list[str]
    raw_notes: str  # sanitized, bounded (TimingVerdict.raw_notes) - see redaction.py
    harness_latency_s: float  # end-to-end wall time (frame extraction + calls + parsing)
    coarse_model_id: str
    fine_model_id: str | None
    provider_latency_s: float  # sum of the passes' own reported latency_s
    total_retries: int
    total_tokens: int | None  # None unless every pass that ran reported usage
    estimated_cost_usd: float | None  # None unless every pass's model is priced


def _load_manifest(path: Path) -> list[dict]:
    entries = json.loads(path.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: manifest must be a non-empty JSON array")
    for entry in entries:
        for key in ("clip_id", "video_path", "true_start_s", "true_end_s"):
            if key not in entry:
                raise ValueError(f"manifest entry missing required key {key!r}: {entry}")
    return entries


class _PerfectStubProvider:
    """Answers with the manifest's own ground truth - harness self-test only.

    Deliberately named so it cannot be mistaken for evidence: any report row
    produced with this provider is a check that the pipeline plumbing works,
    not a measurement of anything about real-world accuracy.
    """

    def __init__(self, true_start_s: float, true_end_s: float) -> None:
        self._true_start_s = true_start_s
        self._true_end_s = true_end_s

    def analyze(self, request: ProviderRequest) -> RawProviderResponse:
        return canned_json_response(
            start_s=self._true_start_s, end_s=self._true_end_s, confidence=0.95
        )


def _build_provider(
    name: str,
    entry: dict,
    *,
    model_id: str | None = None,
    gemini_client_factory: Callable[[], Any] | None = None,
    openai_client_factory: Callable[[], Any] | None = None,
) -> TimingProvider:
    """Construct the provider for one clip.

    ``gemini_client_factory``/``openai_client_factory`` default to the real
    ``gemini_provider.build_default_gemini_client`` /
    ``openai_provider.build_default_openai_client`` - imported lazily, here,
    inside their own branch, so nothing in this script (or its tests)
    requires ``google-genai``/``openai`` to be installed or
    ``GEMINI_API_KEY``/``OPENAI_API_KEY`` to be set unless that provider is
    actually selected. Tests inject a fake factory to verify this routing
    without a real key or SDK - see
    ``tests/test_llm_timing_eval_provider_wiring.py``.
    """
    if name == "stub-perfect":
        return _PerfectStubProvider(entry["true_start_s"], entry["true_end_s"])
    if name == "gemini":
        from app.analysis.llm_timing.gemini_provider import (
            DEFAULT_GEMINI_MODEL_ID,
            GeminiTimingProvider,
            build_default_gemini_client,
        )

        factory = gemini_client_factory or build_default_gemini_client
        client = factory()
        return GeminiTimingProvider(client, model_id=model_id or DEFAULT_GEMINI_MODEL_ID)
    if name == "openai":
        from app.analysis.llm_timing.openai_provider import (
            DEFAULT_OPENAI_MODEL_ID,
            OpenAITimingProvider,
            build_default_openai_client,
        )

        factory = openai_client_factory or build_default_openai_client
        client = factory()
        return OpenAITimingProvider(client, model_id=model_id or DEFAULT_OPENAI_MODEL_ID)
    # Another real provider (Anthropic / ...) goes here behind the same
    # TimingProvider.analyze(ProviderRequest) -> RawProviderResponse
    # contract - not implemented in this spike.
    raise ValueError(
        f"Unknown or not-yet-implemented provider {name!r}. "
        "'stub-perfect' (harness self-test), 'gemini', and 'openai' (real providers) exist."
    )


def _evaluate_clip(
    entry: dict,
    provider_name: str,
    config: PipelineConfig,
    *,
    model_id: str | None = None,
    gemini_client_factory: Callable[[], Any] | None = None,
    openai_client_factory: Callable[[], Any] | None = None,
) -> ClipResult:
    provider = _build_provider(
        provider_name,
        entry,
        model_id=model_id,
        gemini_client_factory=gemini_client_factory,
        openai_client_factory=openai_client_factory,
    )
    started = time.monotonic()
    outcome = run_llm_timing(
        Path(entry["video_path"]),
        provider,
        prompt_version=PROMPT_V1_ID,
        prompt_text=PROMPT_V1,
        config=config,
    )
    harness_latency_s = time.monotonic() - started

    true_start_s = float(entry["true_start_s"])
    true_end_s = float(entry["true_end_s"])
    verdict = outcome.verdict
    coarse_model_id = outcome.coarse_response.model_id
    fine_model_id = outcome.fine_response.model_id if outcome.fine_response else None
    provider_latency_s = outcome.total_latency_s
    total_retries = outcome.total_retries
    total_tokens = outcome.total_tokens
    estimated_cost_usd = outcome.estimated_cost_usd()

    if verdict.status is not TimingStatus.CONFIRMED:
        return ClipResult(
            clip_id=entry["clip_id"],
            status="abstain",
            true_start_s=true_start_s,
            true_end_s=true_end_s,
            predicted_start_s=None,
            predicted_end_s=None,
            start_error_s=None,
            end_error_s=None,
            duration_error_s=None,
            within_tolerance=None,
            false_confident=False,
            reason_codes=list(verdict.reason_codes),
            raw_notes=verdict.raw_notes,
            harness_latency_s=harness_latency_s,
            coarse_model_id=coarse_model_id,
            fine_model_id=fine_model_id,
            provider_latency_s=provider_latency_s,
            total_retries=total_retries,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

    assert verdict.start_s is not None and verdict.end_s is not None
    start_error = verdict.start_s - true_start_s
    end_error = verdict.end_s - true_end_s
    duration_error = (verdict.end_s - verdict.start_s) - (true_end_s - true_start_s)
    within_tolerance = (
        abs(start_error) <= DEFAULT_TOLERANCE_S and abs(end_error) <= DEFAULT_TOLERANCE_S
    )

    return ClipResult(
        clip_id=entry["clip_id"],
        status="confirmed",
        true_start_s=true_start_s,
        true_end_s=true_end_s,
        predicted_start_s=verdict.start_s,
        predicted_end_s=verdict.end_s,
        start_error_s=start_error,
        end_error_s=end_error,
        duration_error_s=duration_error,
        within_tolerance=within_tolerance,
        false_confident=not within_tolerance,
        reason_codes=list(verdict.reason_codes),
        raw_notes=verdict.raw_notes,
        harness_latency_s=harness_latency_s,
        coarse_model_id=coarse_model_id,
        fine_model_id=fine_model_id,
        provider_latency_s=provider_latency_s,
        total_retries=total_retries,
        total_tokens=total_tokens,
        estimated_cost_usd=estimated_cost_usd,
    )


def _summarize(results: list[ClipResult]) -> dict:
    confirmed = [r for r in results if r.status == "confirmed"]
    abstained = [r for r in results if r.status == "abstain"]
    false_confident = [r for r in confirmed if r.false_confident]

    def _stats(values: list[float]) -> dict:
        if not values:
            return {"mean": None, "median": None, "max": None}
        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "max": max(values, key=abs),
        }

    return {
        "n_clips": len(results),
        "n_confirmed": len(confirmed),
        "n_abstained": len(abstained),
        "abstention_rate": len(abstained) / len(results) if results else None,
        "n_false_confident": len(false_confident),
        "false_confident_rate": len(false_confident) / len(results) if results else None,
        "start_error_s": _stats(
            [r.start_error_s for r in confirmed if r.start_error_s is not None]
        ),
        "end_error_s": _stats([r.end_error_s for r in confirmed if r.end_error_s is not None]),
        "duration_error_s": _stats(
            [r.duration_error_s for r in confirmed if r.duration_error_s is not None]
        ),
        "harness_latency_s": _stats([r.harness_latency_s for r in results]),
        "provider_latency_s": _stats([r.provider_latency_s for r in results]),
        "total_retries": sum(r.total_retries for r in results),
        "total_tokens": (
            sum(t for r in results if (t := r.total_tokens) is not None)
            if any(r.total_tokens is not None for r in results)
            else None
        ),
        "total_cost_usd": (
            sum(c for r in results if (c := r.estimated_cost_usd) is not None)
            if any(r.estimated_cost_usd is not None for r in results)
            else None
        ),
        "n_clips_with_unknown_cost": sum(1 for r in results if r.estimated_cost_usd is None),
        "tolerance_s": DEFAULT_TOLERANCE_S,
    }


def _fmt(value: float | None) -> str:
    return f"{value:+.3f}" if value is not None else "-"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("manifest", type=Path, help="JSON manifest of labelled clips")
    parser.add_argument(
        "--provider",
        default="stub-perfect",
        choices=("stub-perfect", "gemini", "openai"),
        help="provider to evaluate: 'stub-perfect' (harness self-test), 'gemini', or 'openai'",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="override the pinned default model ID (--provider gemini/openai only)",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write full results as JSON")
    args = parser.parse_args()

    entries = _load_manifest(args.manifest)
    config = PipelineConfig()

    if args.provider == "stub-perfect":
        print(
            "WARNING: --provider stub-perfect answers with the manifest's own ground "
            "truth. This is a harness self-test, not an accuracy measurement. Do not "
            "cite these numbers as evidence for any gate.\n",
            file=sys.stderr,
        )

    results = [
        _evaluate_clip(entry, args.provider, config, model_id=args.model_id) for entry in entries
    ]
    summary = _summarize(results)

    header = (
        f"{'clip':<24} {'status':<10} {'start_err':>10} {'end_err':>10} {'dur_err':>10} {'ok':>4}"
    )
    print(header)
    for r in results:
        ok = "-" if r.within_tolerance is None else ("Y" if r.within_tolerance else "N")
        print(
            f"{r.clip_id:<24} {r.status:<10} {_fmt(r.start_error_s):>10} "
            f"{_fmt(r.end_error_s):>10} {_fmt(r.duration_error_s):>10} {ok:>4}"
        )

    notes = [r for r in results if r.raw_notes]
    if notes:
        # Already sanitized/bounded (TimingVerdict.raw_notes - see
        # redaction.py) before it ever reached this ClipResult, so it's safe
        # to print - this is the diagnostic a provider_error/malformed_output
        # abstain needs to be distinguishable (auth vs. quota vs. request
        # shape vs. model availability) instead of just a reason code.
        print("\nNotes:")
        for r in notes:
            print(f"  {r.clip_id} ({r.status}): {r.raw_notes}")

    print("\nSummary:")
    print(json.dumps(summary, indent=2, default=str))

    if args.json:
        payload = {"results": [asdict(r) for r in results], "summary": summary}
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nFull results written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
