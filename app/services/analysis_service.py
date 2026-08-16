"""Job orchestration.

Analysis runs in a background worker thread so the browser gets progress
instead of a frozen page.  The service owns:

*   the job records and their progress state (thread-safe);
*   turning a detector's output into an :class:`~app.services.event_log.EventLog`;
*   translating failures into messages a non-programmer can act on, while the
    technical detail goes to the log;
*   cancellation and retention of finished jobs.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..analysis.base_detector import DetectorResult
from ..analysis.registry import create_detector
from ..config import AppConfig
from ..errors import AnalyzerError, NotFoundError
from ..logging_setup import get_logger
from ..video.reader import VideoReader
from .diagnostics import save_event_boundary_frames
from .event_log import EventLog, build_event_log, summarise
from .storage import VideoRecord, VideoStore

logger = get_logger(__name__)

# Two concurrent analyses: enough that a second video can be queued behind the
# first without the two starving each other of CPU on a normal workstation.
MAX_CONCURRENT_JOBS = 2

# Finished jobs are kept this long so the browser can still fetch results
# after a refresh.
JOB_RETENTION_S = 6 * 3600


class JobCancelled(Exception):
    """Raised inside a worker when the user cancels the job."""


@dataclass
class JobProgress:
    stage: str = "queued"
    fraction: float = 0.0
    message: str = "Waiting to start"


@dataclass
class AnalysisJob:
    """One analysis request and its state."""

    job_id: str
    video_id: str
    mode: str
    params: dict[str, Any]
    status: str = "queued"  # queued | running | complete | failed | cancelled
    progress: JobProgress = field(default_factory=JobProgress)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    plan: dict[str, Any] = field(default_factory=dict)
    result: DetectorResult | None = None
    event_log: EventLog | None = None
    diagnostics_paths: list[str] = field(default_factory=list)
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def to_dict(self, *, include_result: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "job_id": self.job_id,
            "video_id": self.video_id,
            "mode": self.mode,
            "status": self.status,
            "stage": self.progress.stage,
            "progress": round(self.progress.fraction, 4),
            "message": self.progress.message,
            "error": self.error,
            "plan": self.plan,
            "elapsed_s": round(
                (self.finished_at or time.time()) - (self.started_at or self.created_at), 2
            ),
        }
        if include_result and self.result is not None:
            payload["result"] = self.result.to_dict()
            payload["summary"] = self.result.summary
            payload["counts"] = summarise(self.result.events)
        if include_result and self.event_log is not None:
            payload["event_log"] = self.event_log.to_dict()
        if self.diagnostics_paths:
            payload["diagnostics_files"] = self.diagnostics_paths
        return payload


class AnalysisService:
    """Runs detectors in the background and keeps their results."""

    def __init__(self, config: AppConfig, store: VideoStore) -> None:
        self.config = config
        self.store = store
        self._jobs: dict[str, AnalysisJob] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_CONCURRENT_JOBS, thread_name_prefix="analysis"
        )

    # -- submission --------------------------------------------------------- #

    def submit(
        self,
        record: VideoRecord,
        mode: str,
        params: Mapping[str, Any],
        *,
        recording_start: datetime | None = None,
    ) -> AnalysisJob:
        """Validate the request, then queue it.

        The detector is constructed here, on the request thread, so an invalid
        ROI or setting is reported immediately as a normal error instead of
        surfacing later as a mysteriously failed job.
        """
        detector = create_detector(mode, record.info, params)
        plan = detector.describe()

        job = AnalysisJob(
            job_id=uuid.uuid4().hex,
            video_id=record.video_id,
            mode=mode,
            params=dict(params),
            plan=plan,
        )
        with self._lock:
            self._jobs[job.job_id] = job

        logger.info(
            "Queued job %s: mode=%s video=%s plan=%s",
            job.job_id,
            mode,
            record.original_name,
            plan,
        )
        self._executor.submit(self._run_job, job, record, recording_start)
        return job

    # -- inspection --------------------------------------------------------- #

    def get(self, job_id: str) -> AnalysisJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise NotFoundError("That analysis is no longer available. Please run it again.")
        return job

    def cancel(self, job_id: str) -> AnalysisJob:
        job = self.get(job_id)
        job._cancel.set()
        if job.status == "queued":
            job.status = "cancelled"
            job.finished_at = time.time()
            job.progress = JobProgress("cancelled", 0.0, "Cancelled before it started")
        logger.info("Cancellation requested for job %s", job_id)
        return job

    def purge_expired(self) -> int:
        cutoff = time.time() - JOB_RETENTION_S
        with self._lock:
            expired = [
                job_id
                for job_id, job in self._jobs.items()
                if job.finished_at is not None and job.finished_at < cutoff
            ]
            for job_id in expired:
                self._jobs.pop(job_id, None)
        return len(expired)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    # -- worker ------------------------------------------------------------- #

    def _run_job(
        self,
        job: AnalysisJob,
        record: VideoRecord,
        recording_start: datetime | None,
    ) -> None:
        if job.cancelled:
            return
        job.status = "running"
        job.started_at = time.time()
        job.progress = JobProgress("starting", 0.0, "Opening the video")

        def report(*, stage: str, fraction: float, message: str) -> None:
            if job.cancelled:
                raise JobCancelled()
            job.progress = JobProgress(stage, max(0.0, min(1.0, fraction)), message)

        try:
            detector = create_detector(job.mode, record.info, job.params)
            with VideoReader(record.path, record.info) as reader:
                result = detector.run(reader, report)
                if self.config.save_diagnostics and result.events:
                    job.diagnostics_paths = [
                        str(path)
                        for path in save_event_boundary_frames(
                            reader,
                            result.events,
                            self.config.diagnostics_dir / job.job_id,
                            roi=getattr(detector, "roi", None),
                        )
                    ]

            job.result = result
            job.event_log = build_event_log(
                result.events,
                record.info,
                recording_start=recording_start,
                mode=job.mode,
            )
            job.status = "complete"
            job.progress = JobProgress("complete", 1.0, _completion_message(result))
            logger.info(
                "Job %s complete in %.1fs: %d event(s); summary=%s",
                job.job_id,
                time.time() - (job.started_at or time.time()),
                len(result.events),
                result.summary,
            )
        except JobCancelled:
            job.status = "cancelled"
            job.progress = JobProgress("cancelled", job.progress.fraction, "Analysis cancelled")
            logger.info("Job %s cancelled by the user", job.job_id)
        except AnalyzerError as exc:
            job.status = "failed"
            job.error = exc.user_message
            job.progress = JobProgress("failed", job.progress.fraction, exc.user_message)
            logger.error("Job %s failed: %s (%s)", job.job_id, exc.user_message, exc.detail)
        except Exception as exc:  # noqa: BLE001 - last line of defence for the worker
            job.status = "failed"
            job.error = (
                "The analysis stopped because of an unexpected internal error. "
                "The technical details are in the server log."
            )
            job.progress = JobProgress("failed", job.progress.fraction, job.error)
            logger.error("Job %s crashed: %s\n%s", job.job_id, exc, traceback.format_exc())
        finally:
            job.finished_at = time.time()
            record.touch()


def _completion_message(result: DetectorResult) -> str:
    """The one-line outcome shown when the progress bar reaches the end."""
    if result.summary.get("mode") == "zahn_cup":
        status = result.summary.get("status")
        if status == "confirmed":
            return f"Efflux time measured: {result.summary.get('efflux_seconds')} s"
        if status == "review":
            return "A time was measured but needs review"
        return "No reliable efflux time could be measured"
    count = len(result.events)
    if count == 0:
        return "Analysis complete - no events detected"
    return f"Analysis complete - {count} event(s) detected"
