#!/usr/bin/env python3
"""Evaluation harness for the LLM timing spike.

Runs ``app.analysis.llm_timing.pipeline.run_llm_timing`` over a manifest of
labelled clips and reports the metrics the supervisor's evidence gates ask
for: per-clip start/end/duration error, abstention, "false confident" (a
CONFIRMED verdict that was actually wrong by more than the tolerance), and
latency. See ``diagnostics/llm_spike/DESIGN.md`` for gates 1-3 in full and
how this script's output maps onto them.

No real provider is implemented in this spike (needs a vendor API key,
explicitly deferred - see the design doc). Right now the only selectable
provider is ``--provider stub-perfect``, which answers with the manifest's
own ground truth and exists ONLY to prove the harness's plumbing works
end-to-end. Its numbers are not evidence of anything about real accuracy and
the report says so loudly. Wiring in a real provider is a matter of adding
one more branch to ``_build_provider`` behind the same ``TimingProvider``
interface - nothing else in this file should need to change.

Usage:
    python scripts/llm_timing_eval.py manifest.json --provider stub-perfect
    python scripts/llm_timing_eval.py manifest.json --provider stub-perfect --json out.json

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
from dataclasses import asdict, dataclass
from pathlib import Path

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


def _build_provider(name: str, entry: dict) -> TimingProvider:
    if name == "stub-perfect":
        return _PerfectStubProvider(entry["true_start_s"], entry["true_end_s"])
    # A real provider (OpenAI / Anthropic / Google) goes here behind the same
    # TimingProvider.analyze(ProviderRequest) -> RawProviderResponse contract.
    # Not implemented in this spike: needs an API key (deferred - see the
    # design doc) and must not depend on an interactive chat/Code Interpreter
    # session per the supervisor's authorization.
    raise ValueError(
        f"Unknown or not-yet-implemented provider {name!r}. "
        "Only 'stub-perfect' (harness self-test) exists in this spike."
    )


def _evaluate_clip(entry: dict, provider_name: str, config: PipelineConfig) -> ClipResult:
    provider = _build_provider(provider_name, entry)
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
        help="provider to evaluate (only 'stub-perfect' exists until a real one is wired in)",
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

    results = [_evaluate_clip(entry, args.provider, config) for entry in entries]
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

    print("\nSummary:")
    print(json.dumps(summary, indent=2, default=str))

    if args.json:
        payload = {"results": [asdict(r) for r in results], "summary": summary}
        args.json.write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nFull results written to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
