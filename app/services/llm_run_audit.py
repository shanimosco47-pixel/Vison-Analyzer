"""Durable, redacted, per-run audit trail for the Experimental LLM analysis
section.

Supervisor-directed requirement: "The user must not need DevTools, console
commands, or manual JSON extraction to understand a wrong result." Two
things follow from that:

*   The record must *survive* the moment the browser tab closes -
      ``LLMRunService``'s in-memory job map (see ``llm_run_service.py``) is
      exactly that: in-memory, gone on restart, and already subject to its
      own 6-hour retention sweep. This module writes one JSON file per run,
      atomically, under ``AppConfig.data_dir`` - readable after a page
      refresh *or* a server restart.
*   The record must let a reader reconstruct *why* a result (right or
    wrong) came out the way it did - every pass's actual submitted frames,
    its own parsed verdict, and the pipeline's own intermediate decisions
    (locked start, end-coarse candidate, the validation window built
    around it) - not just the final answer. See
    ``pipeline.PipelineOutcome.derived`` for the latter; this module's
    :func:`build_audit_record` is what assembles both into one artifact.

Redaction: this module never touches ``EngineConfig.credential_ref``, a
video's filesystem path, or a ``TimedFrame``'s image bytes - none of those
are passed into :func:`build_audit_record` in the first place, so there is
no code path here that could leak them. Every free-text field
(``raw_notes``, the top-level ``error``) is routed through
``sanitize_untrusted_text`` again at build time, even though its two
sources (``schema.TimingVerdict.raw_notes`` and
``LLMRunJob.error``/``AnalyzerError.user_message``) already sanitize their
own text - defense in depth, not reliance on a single call site staying
correct forever.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from ..analysis.llm_timing.pipeline import PipelineOutcome
from ..analysis.llm_timing.pricing import estimate_cost_usd
from ..analysis.llm_timing.provider import RawProviderResponse
from ..analysis.llm_timing.redaction import sanitize_untrusted_text
from ..logging_setup import get_logger

logger = get_logger(__name__)

AUDIT_SCHEMA_VERSION = 2
"""Bumped from 1: added ``possible_collapse_s`` to every pass entry, and the
``end_validate_coarse_cascade``/``end_validate_conflict_coarse_cascade``
pass names (see PASS_ORDER's own docstring) - the candidate-centred
contact-sheet cascade round."""

# Mirrors PipelineOutcome's own per-pass field names - the four passes every
# run reaches, in the order they run, plus:
# - "end_validate_coarse_cascade": the candidate-centred cascade's own
#   9-panel coarse contact sheet, kept separately auditable only when a
#   denser 7-panel refine sheet also ran and superseded it as "end_validate"
#   (see pipeline.py's _run_end_validation_pass/_EndValidationOutcome);
# - "end_validate_conflict": a second, independently-anchored cascade that
#   only exists when a real run's end-coarse candidate and the coarse
#   pass's own end estimate conflicted badly enough to need one (see
#   pipeline.py's _candidates_conflict);
# - "end_validate_conflict_coarse_cascade": that second cascade's own
#   coarse sheet, kept separately auditable the same way as the primary's.
# Every one of these is absent from ``passes`` like any other pass that
# never ran, per _pass_audit's own contract - see
# diagnostics/llm_spike/DESIGN.md for the full write-up.
PASS_ORDER: tuple[str, ...] = (
    "coarse",
    "fine",
    "end_coarse",
    "end_validate",
    "end_validate_coarse_cascade",
    "end_validate_conflict",
    "end_validate_conflict_coarse_cascade",
)

_PASS_RESPONSE_ATTR: dict[str, str] = {
    "coarse": "coarse_response",
    "fine": "fine_response",
    "end_coarse": "end_coarse_response",
    "end_validate": "end_validation_response",
    "end_validate_coarse_cascade": "end_validation_coarse_cascade_response",
    "end_validate_conflict": "end_validation_conflict_response",
    "end_validate_conflict_coarse_cascade": "end_validation_conflict_coarse_cascade_response",
}

# Safe as a filename component: exactly what uuid.uuid4().hex produces (the
# only thing LLMRunService ever assigns as a run_id) plus a little slack for
# any future ID scheme - never a path separator or a ".." segment. Enforced
# on every path this module builds from a caller-supplied run_id, since one
# of them (get()) is reachable from a URL path parameter - "safe run IDs
# only" is an explicit part of the supervisor's requirement, not just an
# implementation detail.
_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _pass_audit(name: str, outcome: PipelineOutcome) -> dict[str, Any] | None:
    """One pass's full audit entry, or ``None`` if this pass never ran -
    mirrors ``pass_frames``'s own absent-key-means-not-run contract."""
    verdict = outcome.pass_verdicts.get(name)
    if verdict is None:
        return None
    submitted = sorted(outcome.pass_frames.get(name, ()))
    response: RawProviderResponse | None = getattr(outcome, _PASS_RESPONSE_ATTR[name])
    cost_usd = None
    if response is not None and response.model_id:
        cost_usd = estimate_cost_usd(
            response.model_id, response.prompt_tokens, response.completion_tokens
        )
    return {
        "prompt_version": verdict.prompt_version,
        "submitted_timestamps_s": submitted,
        "submitted_count": len(submitted),
        "submitted_range_s": [submitted[0], submitted[-1]] if submitted else None,
        "status": verdict.status.value,
        "start_s": verdict.start_s,
        "end_s": verdict.end_s,
        "start_uncertainty_s": verdict.start_uncertainty_s,
        "end_uncertainty_s": verdict.end_uncertainty_s,
        "confidence": verdict.confidence,
        "reason_codes": list(verdict.reason_codes),
        "raw_notes": sanitize_untrusted_text(verdict.raw_notes),
        "evidence_frame_timestamps_s": list(verdict.evidence_frame_timestamps_s),
        "trend_checkpoint_timestamps_s": list(verdict.trend_checkpoint_timestamps_s),
        "possible_collapse_s": verdict.possible_collapse_s,
        "model_id": response.model_id if response is not None else None,
        "latency_s": response.latency_s if response is not None else None,
        "retries": response.retries if response is not None else None,
        "prompt_tokens": response.prompt_tokens if response is not None else None,
        "completion_tokens": response.completion_tokens if response is not None else None,
        "estimated_cost_usd": cost_usd,
    }


def build_audit_record(
    *,
    run_id: str,
    video_id: str,
    engine_id: str,
    engine_display_name: str,
    provider_name: str,
    model_id: str,
    status: str,
    created_at: float,
    started_at: float | None,
    finished_at: float | None,
    error: str | None,
    outcome: PipelineOutcome | None,
    app_version: str = "unknown",
) -> dict[str, Any]:
    """Assemble one run's full audit record.

    ``outcome`` is ``None`` for a run that never reached the pipeline at
    all (e.g. credential resolution failed before any provider call) - the
    record is still built, with an empty ``passes``/``derived`` and
    whatever ``error`` says, so even that failure mode leaves a durable,
    inspectable trace (supervisor requirement: "persist safe partial
    information for ABSTAIN, failure and cancellation too").

    ``app_version`` is the running backend's own build identifier (see
    ``app.version.app_version``) - passed in explicitly, never resolved
    here, so this function stays a pure record-assembler with no
    subprocess/filesystem access of its own. Lets a downloaded audit be
    matched against the exact backend build that produced it, per the
    supervisor's operational requirement.
    """
    passes: dict[str, Any] = {}
    derived: dict[str, Any] = {}
    final_verdict: dict[str, Any] | None = None
    event: dict[str, Any] | None = None
    if outcome is not None:
        for name in PASS_ORDER:
            entry = _pass_audit(name, outcome)
            if entry is not None:
                passes[name] = entry
        derived = dict(outcome.derived)
        final_verdict = outcome.verdict.to_dict()
        event = outcome.event.to_dict() if outcome.event is not None else None

    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "app_version": app_version,
        "run_id": run_id,
        "video_id": video_id,
        "engine_id": engine_id,
        "engine_display_name": engine_display_name,
        "provider_name": provider_name,
        "model_id": model_id,
        "status": status,
        "created_at": created_at,
        "started_at": started_at,
        "finished_at": finished_at,
        "error": sanitize_untrusted_text(error) if error else None,
        "passes": passes,
        "derived": derived,
        "final_verdict": final_verdict,
        "event": event,
    }


class LLMRunAuditStore:
    """One JSON file per run under ``data_dir/llm_run_audits/``, written
    atomically (write-to-temp, then rename - atomic on both POSIX and
    Windows, same pattern as ``llm_engine_store.py``'s own persistence).

    Deliberately not one big file: many independent small writes (one per
    finished run) never risk corrupting any other run's record, and
    cleanup is just deleting files, not rewriting a shared one.
    """

    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir / "llm_run_audits"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path_for(self, run_id: str) -> Path:
        if not _SAFE_RUN_ID_RE.match(run_id):
            # Never reachable through normal use (run_id is always
            # uuid.uuid4().hex) - this is the explicit safety net for the
            # one place a run_id arrives from a URL path parameter
            # (llm_engine_routes.get_llm_run_audit).
            raise ValueError(f"Refusing to use an unsafe run id as a filename: {run_id!r}")
        return self._dir / f"{run_id}.json"

    def save(self, record: dict[str, Any]) -> None:
        path = self._path_for(record["run_id"])
        tmp = path.with_suffix(".json.tmp")
        with self._lock:
            tmp.write_text(json.dumps(record, indent=2))
            tmp.replace(path)

    def get(self, run_id: str) -> dict[str, Any] | None:
        try:
            path = self._path_for(run_id)
        except ValueError:
            return None
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            logger.warning("Could not read audit record %s: %s", path, exc)
            return None

    def latest_for_engine(self, engine_id: str) -> dict[str, Any] | None:
        """The most recently finished audit for ``engine_id``, or ``None``
        if none is on disk - so "the latest audit for each configured
        engine remains discoverable" after a refresh or restart without
        the caller needing to already know a ``run_id``.

        A plain directory scan, not a maintained index file: at this
        application's scale (bounded retention, a handful of concurrent
        runs) that is simpler and cannot itself drift out of sync with
        what is actually on disk, which a separate pointer file could.
        """
        latest: dict[str, Any] | None = None
        latest_ts = float("-inf")
        for path in self._dir.glob("*.json"):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if record.get("engine_id") != engine_id:
                continue
            timestamp = record.get("finished_at") or record.get("created_at") or 0.0
            if timestamp > latest_ts:
                latest = record
                latest_ts = timestamp
        return latest

    def purge_before(self, cutoff_epoch_s: float) -> int:
        """Delete every audit record that finished (or, if it never
        finished, was created) before ``cutoff_epoch_s``. Bounded cleanup
        tied to the caller's own retention policy - see
        ``LLMRunService.RUN_RETENTION_S``, which calls this with the same
        cutoff it already applies to its in-memory job map, so on-disk
        audits and in-memory jobs expire together rather than the durable
        copy growing forever once the in-memory one is gone."""
        removed = 0
        with self._lock:
            for path in self._dir.glob("*.json"):
                try:
                    record = json.loads(path.read_text())
                except (OSError, ValueError):
                    # Unreadable either way - not recoverable, so it counts
                    # as expired rather than accumulating forever.
                    path.unlink(missing_ok=True)
                    removed += 1
                    continue
                timestamp = record.get("finished_at") or record.get("created_at") or 0.0
                if timestamp < cutoff_epoch_s:
                    path.unlink(missing_ok=True)
                    removed += 1
        return removed
