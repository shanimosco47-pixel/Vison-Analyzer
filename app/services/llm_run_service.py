"""Job orchestration for one Experimental LLM engine run.

Mirrors ``app/services/analysis_service.py``'s shape and discipline
(background worker thread, staged progress, cooperative cancellation,
retention) for the classical detectors, but for a single
:class:`~app.analysis.llm_timing.engine_config.EngineConfig` run against
one uploaded video. Deliberately calls
``app.analysis.llm_timing.pipeline.run_llm_timing`` directly rather than
going through ``engine_config.run_llm_timing_for_engines``: that
function's per-engine ``try/except`` would catch a user-initiated
cancellation and misreport it as an engine crash, which this module needs
to keep distinct from a real failure (see :class:`RunCancelled`).

Every result produced here is explicitly experimental - see
``diagnostics/llm_spike/DESIGN.md`` - never a substitute for
``AnalysisService``'s classical detectors, and the web layer marks it as
such in every response.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from ..analysis.llm_timing.engine_config import EngineConfig
from ..analysis.llm_timing.pipeline import PipelineConfig, run_llm_timing
from ..analysis.llm_timing.prompts import PROMPT_V1, PROMPT_V1_ID
from ..analysis.llm_timing.redaction import sanitize_untrusted_text
from ..analysis.llm_timing.schema import TimingStatus
from ..errors import AnalyzerError, NotFoundError
from ..logging_setup import get_logger
from .llm_engine_store import LLMEngineStore
from .storage import VideoRecord

logger = get_logger(__name__)

# A handful of concurrent runs is plenty for "several engines against one
# clip, compared side by side" - the point of this feature - without
# risking every configured engine's request racing every other one's rate
# limit at once.
MAX_CONCURRENT_RUNS = 3

RUN_RETENTION_S = 6 * 3600
RETENTION_SWEEP_INTERVAL_S = 300.0

# One line per pipeline stage, shown as the run's live progress message -
# matches the stage names run_llm_timing's on_stage callback reports.
_STAGE_MESSAGES = {
    "coarse": "Scanning the whole clip for an approximate window",
    "fine": "Pinpointing the exact start",
    "end_coarse": "Scanning the rest of the clip for a candidate break",
    "end_validate": "Confirming the candidate break",
}


class RunCancelled(Exception):
    """Raised from inside the ``on_stage`` callback when the user cancels
    a run between pipeline passes - see ``run_llm_timing``'s own
    docstring for why this lives here and not in the pipeline module."""


@dataclass
class LLMRunJob:
    """One engine's run of one video, and its live state."""

    run_id: str
    video_id: str
    engine_id: str
    engine_display_name: str
    provider_name: str
    model_id: str
    status: str = "queued"  # queued | running | complete | failed | cancelled
    stage: str = "queued"
    message: str = "Waiting to start"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    outcome_dict: dict[str, Any] | None = None  # PipelineOutcome.to_dict() - set only on complete
    event_dict: dict[str, Any] | None = None  # Event.to_dict(), or None if abstained
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    @property
    def start_evidence_s(self) -> float | None:
        """The grounded, actually-submitted timestamp nearest the reported
        start boundary - see ``pipeline._nearest_grounded_evidence_ts`` -
        or ``None`` when unavailable (abstained, or no grounded evidence).
        Read from ``event_dict`` (the single source of truth already
        produced by the pipeline) rather than stored separately, so there
        is no way for this to drift out of sync with it."""
        if self.event_dict is None:
            return None
        return self.event_dict.get("details", {}).get("start_evidence_s")

    @property
    def end_evidence_s(self) -> float | None:
        """The end-boundary counterpart of :attr:`start_evidence_s`."""
        if self.event_dict is None:
            return None
        return self.event_dict.get("details", {}).get("end_evidence_s")

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "video_id": self.video_id,
            "engine_id": self.engine_id,
            "engine_display_name": self.engine_display_name,
            "provider_name": self.provider_name,
            "model_id": self.model_id,
            "status": self.status,
            "stage": self.stage,
            "message": self.message,
            "error": self.error,
            "elapsed_s": round(
                (self.finished_at or time.time()) - (self.started_at or self.created_at), 2
            ),
            "outcome": self.outcome_dict,
            "event": self.event_dict,
            "start_evidence_s": self.start_evidence_s,
            "end_evidence_s": self.end_evidence_s,
            "experimental": True,
        }


class LLMRunService:
    """Runs one engine against one video in the background and keeps its
    result, mirroring ``AnalysisService``'s polling/cancel/retention
    contract so the browser can reuse the same patterns for both."""

    def __init__(
        self,
        engine_store: LLMEngineStore,
        *,
        sweep_interval_s: float = RETENTION_SWEEP_INTERVAL_S,
    ) -> None:
        self._engine_store = engine_store
        self._jobs: dict[str, LLMRunJob] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_RUNS, thread_name_prefix="llm-run"
        )
        self._sweep_interval_s = sweep_interval_s
        self._stop_sweeper = threading.Event()
        self._sweeper = threading.Thread(
            target=self._sweep_loop, name="llm-run-retention-sweeper", daemon=True
        )
        self._sweeper.start()

    # -- submission ----------------------------------------------------- #

    def submit(self, record: VideoRecord, engine: EngineConfig) -> LLMRunJob:
        if not engine.enabled:
            raise AnalyzerError("This engine is disabled. Enable it before running.")

        job = LLMRunJob(
            run_id=uuid.uuid4().hex,
            video_id=record.video_id,
            engine_id=engine.engine_id,
            engine_display_name=engine.display_name or engine.model_id,
            provider_name=engine.provider_name,
            model_id=engine.model_id,
        )
        with self._lock:
            self._jobs[job.run_id] = job
        logger.info(
            "Queued LLM run %s: engine=%s video=%s", job.run_id, engine.engine_id, record.video_id
        )
        self._executor.submit(self._run_job, job, record, engine)
        return job

    # -- inspection ------------------------------------------------------- #

    def get(self, run_id: str) -> LLMRunJob:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            raise NotFoundError("That run is no longer available. Please run it again.")
        return job

    def cancel(self, run_id: str) -> LLMRunJob:
        """Request cancellation. Honest about what this can and can't stop:
        cancellation is checked cooperatively, only *between* pipeline
        passes (see ``on_stage`` below) - a provider request already in
        flight when Cancel is clicked keeps running until it returns, and
        only the *next* pass is prevented from starting. A still-queued
        run stops immediately, with nothing ever having been sent; a
        running one gets an honest "stopping after the current request"
        message instead of implying an in-flight vendor call was aborted
        (a Codex review flagged the earlier version of this message as
        overclaiming immediacy)."""
        job = self.get(run_id)
        job._cancel.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
            job.stage = "cancelled"
            job.message = "Cancelled before it started"
        elif job.status == "running":
            job.message = (
                "Cancel requested - stopping after the current provider "
                "request finishes (it cannot be interrupted mid-request)"
            )
        logger.info("Cancellation requested for LLM run %s", run_id)
        return job

    def purge_expired(self) -> int:
        cutoff = time.time() - RUN_RETENTION_S
        with self._lock:
            expired = [
                run_id
                for run_id, job in self._jobs.items()
                if job.finished_at is not None and job.finished_at < cutoff
            ]
            for run_id in expired:
                self._jobs.pop(run_id, None)
        return len(expired)

    # -- retention --------------------------------------------------------- #

    def _sweep_loop(self) -> None:
        while not self._stop_sweeper.wait(self._sweep_interval_s):
            try:
                removed = self.purge_expired()
            except Exception:  # noqa: BLE001 - the sweeper must outlive one bad sweep
                logger.exception("LLM run retention sweep failed; will retry next interval")
            else:
                if removed:
                    logger.info("LLM run retention sweep removed %d finished run(s)", removed)

    def shutdown(self) -> None:
        self._stop_sweeper.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- worker ------------------------------------------------------------- #

    def _run_job(self, job: LLMRunJob, record: VideoRecord, engine: EngineConfig) -> None:
        if job.cancelled:
            return
        job.status = "running"
        job.started_at = time.time()
        job.stage = "starting"
        job.message = "Resolving the engine's credential"

        def on_stage(stage: str) -> None:
            if job.cancelled:
                raise RunCancelled()
            job.stage = stage
            job.message = _STAGE_MESSAGES.get(stage, stage)

        try:
            if job.cancelled:
                raise RunCancelled()
            provider = self._engine_store.build_provider(engine)
            # Frames-only, by construction: run_llm_timing only ever extracts
            # individual JPEG frames from the video (see
            # pipeline._extract_frames) and sends those in each
            # ProviderRequest - the original video file is never opened by,
            # or passed to, the provider at all.
            outcome = run_llm_timing(
                record.path,
                provider,
                prompt_version=PROMPT_V1_ID,
                prompt_text=PROMPT_V1,
                config=PipelineConfig(),
                video_info=record.info,
                on_stage=on_stage,
            )
            job.outcome_dict = outcome.to_dict()
            job.event_dict = outcome.event.to_dict() if outcome.event is not None else None
            job.status = "complete"
            job.stage = "complete"
            job.message = _completion_message(outcome.verdict)
            logger.info(
                "LLM run %s complete in %.1fs: status=%s",
                job.run_id,
                time.time() - (job.started_at or time.time()),
                outcome.verdict.status.value,
            )
        except RunCancelled:
            job.status = "cancelled"
            job.stage = "cancelled"
            job.message = "Run cancelled"
            logger.info("LLM run %s cancelled by the user", job.run_id)
        except AnalyzerError as exc:
            job.status = "failed"
            job.error = exc.user_message
            job.stage = "failed"
            job.message = exc.user_message
            logger.error("LLM run %s failed: %s (%s)", job.run_id, exc.user_message, exc.detail)
        except Exception as exc:  # noqa: BLE001 - last line of defence for the worker
            job.status = "failed"
            job.error = (
                "The run stopped because of an unexpected internal error. "
                "The technical details are in the server log."
            )
            job.stage = "failed"
            job.message = job.error
            logger.error(
                "LLM run %s crashed: %s\n%s",
                job.run_id,
                sanitize_untrusted_text(str(exc)),
                traceback.format_exc(),
            )
        finally:
            job.finished_at = time.time()
            record.touch()


def _completion_message(verdict) -> str:
    if verdict.status is TimingStatus.CONFIRMED:
        assert verdict.start_s is not None and verdict.end_s is not None
        duration = verdict.end_s - verdict.start_s
        return f"Confirmed: {duration:.2f}s (experimental - review required)"
    return "No confident result - the engine abstained (experimental)"
