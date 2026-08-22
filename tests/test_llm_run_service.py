"""Tests for app/services/llm_run_service.py.

Runs against a real ``VideoRecord`` (probed from the shared synthetic
``zahn_video`` fixture) and a stubbed provider (``LLMEngineStore.build_provider``
monkeypatched to return a ``StubTimingProvider``) - no network, no real
vendor SDK, no real API key anywhere in this file.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from app.analysis.llm_timing.provider import (
    ProviderRequest,
    RawProviderResponse,
    StubTimingProvider,
    canned_json_response,
)
from app.config import AppConfig
from app.errors import AnalyzerError
from app.services.llm_engine_store import LLMEngineStore
from app.services.llm_run_service import LLMRunService
from app.services.secret_store import SecretStore
from app.services.storage import VideoRecord
from app.video.metadata import probe_video


class _FakeKeyringBackend:
    def __init__(self) -> None:
        self._values: dict[tuple[str, str], str] = {}

    def set_password(self, service: str, username: str, password: str) -> None:
        self._values[(service, username)] = password

    def get_password(self, service: str, username: str) -> str | None:
        return self._values.get((service, username))

    def delete_password(self, service: str, username: str) -> None:
        self._values.pop((service, username), None)


@pytest.fixture
def engine_store(tmp_path: Path) -> LLMEngineStore:
    config = replace(AppConfig(), data_dir=tmp_path / "data")
    backend = _FakeKeyringBackend()
    return LLMEngineStore(config, SecretStore(backend_factory=lambda: backend))


@pytest.fixture
def run_service(engine_store: LLMEngineStore):
    service = LLMRunService(engine_store, sweep_interval_s=0.05)
    try:
        yield service
    finally:
        service.shutdown()


@pytest.fixture
def record(zahn_video) -> VideoRecord:
    info = probe_video(zahn_video.path)
    return VideoRecord(
        video_id="test-video", path=zahn_video.path, original_name="zahn.mp4", info=info
    )


def _confirmed_stub_provider() -> StubTimingProvider:
    candidate_s = 12.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            ts = next(f.timestamp_s for f in request.frames if f.is_candidate)
            return canned_json_response(start_s=ts, end_s=ts, confidence=0.9)
        assert request.pass_name == "end_coarse"
        return canned_json_response(start_s=candidate_s, end_s=candidate_s, confidence=0.9)

    return StubTimingProvider(respond)


def _wait_for(run_service: LLMRunService, run_id: str, timeout_s: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        payload = run_service.get(run_id).to_dict()
        if payload["status"] in {"complete", "failed", "cancelled"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Run {run_id} did not finish within {timeout_s}s")


def test_a_run_completes_and_reports_a_confirmed_outcome(
    engine_store: LLMEngineStore, run_service: LLMRunService, record: VideoRecord, monkeypatch
):
    engine = engine_store.create(
        provider_name="openai", model_id="gpt-5-mini", env_var="OPENAI_API_KEY"
    )
    monkeypatch.setattr(engine_store, "build_provider", lambda e: _confirmed_stub_provider())

    job = run_service.submit(record, engine)
    payload = _wait_for(run_service, job.run_id)

    assert payload["status"] == "complete"
    assert payload["experimental"] is True
    assert payload["outcome"]["verdict"]["status"] == "confirmed"
    assert payload["event"] is not None


def test_a_run_reports_abstain_without_crashing(
    engine_store: LLMEngineStore, run_service: LLMRunService, record: VideoRecord, monkeypatch
):
    engine = engine_store.create(
        provider_name="openai", model_id="gpt-5-mini", env_var="OPENAI_API_KEY"
    )

    def respond(request: ProviderRequest) -> RawProviderResponse:
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    monkeypatch.setattr(engine_store, "build_provider", lambda e: StubTimingProvider(respond))

    job = run_service.submit(record, engine)
    payload = _wait_for(run_service, job.run_id)

    assert payload["status"] == "complete"
    assert payload["outcome"]["verdict"]["status"] == "abstain"
    assert payload["event"] is None


def test_a_missing_credential_fails_the_run_with_a_safe_message(
    engine_store: LLMEngineStore, run_service: LLMRunService, record: VideoRecord
):
    # env_var references a variable that was never actually set - build_provider
    # (the real one, not mocked) will fail resolving it.
    engine = engine_store.create(
        provider_name="openai", model_id="gpt-5-mini", env_var="THIS_VAR_IS_NEVER_SET"
    )

    job = run_service.submit(record, engine)
    payload = _wait_for(run_service, job.run_id)

    assert payload["status"] == "failed"
    assert payload["error"]
    assert "THIS_VAR_IS_NEVER_SET" not in payload["error"] or "OPENAI" not in payload["error"]


def test_disabled_engine_is_rejected_before_any_run_is_queued(
    engine_store: LLMEngineStore, run_service: LLMRunService, record: VideoRecord
):
    engine = engine_store.create(
        provider_name="openai", model_id="gpt-5-mini", env_var="OPENAI_API_KEY", enabled=False
    )
    with pytest.raises(AnalyzerError):
        run_service.submit(record, engine)


def test_cancel_before_the_run_starts_marks_it_cancelled_immediately(
    engine_store: LLMEngineStore, run_service: LLMRunService, record: VideoRecord, monkeypatch
):
    engine = engine_store.create(
        provider_name="openai", model_id="gpt-5-mini", env_var="OPENAI_API_KEY"
    )

    # Block the executor with an unrelated slow task so our real job stays
    # "queued" long enough to cancel before it ever starts.
    import threading

    gate = threading.Event()
    run_service._executor.submit(gate.wait)
    run_service._executor.submit(gate.wait)
    run_service._executor.submit(gate.wait)  # fill all MAX_CONCURRENT_RUNS workers

    monkeypatch.setattr(engine_store, "build_provider", lambda e: _confirmed_stub_provider())
    job = run_service.submit(record, engine)
    cancelled = run_service.cancel(job.run_id)
    gate.set()  # release the blocked workers so the executor can shut down cleanly

    assert cancelled.status == "cancelled"
    assert cancelled.to_dict()["message"] == "Cancelled before it started"


def test_cancel_mid_run_stops_before_the_next_pass_completes(
    engine_store: LLMEngineStore, run_service: LLMRunService, record: VideoRecord, monkeypatch
):
    """A cancellation requested while the coarse pass is in flight is
    honoured before the fine pass ever starts - proven by asserting the
    fine pass's request never arrives at the provider."""
    engine = engine_store.create(
        provider_name="openai", model_id="gpt-5-mini", env_var="OPENAI_API_KEY"
    )

    seen_passes: list[str] = []

    def respond(request: ProviderRequest) -> RawProviderResponse:
        seen_passes.append(request.pass_name)
        if request.pass_name == "coarse":
            # By the time the coarse response is being built, the job
            # object already exists (submit() returned it synchronously) -
            # cancel it now so the *next* on_stage call (for "fine") raises.
            run_service.cancel(job_holder["run_id"])
        return canned_json_response(start_s=4.0, end_s=12.0, confidence=0.9)

    monkeypatch.setattr(engine_store, "build_provider", lambda e: StubTimingProvider(respond))

    job_holder: dict[str, str] = {}
    job = run_service.submit(record, engine)
    job_holder["run_id"] = job.run_id

    payload = _wait_for(run_service, job.run_id)

    assert payload["status"] == "cancelled"
    assert "fine" not in seen_passes


def test_cancel_of_a_running_job_reports_an_honest_stopping_message(
    engine_store: LLMEngineStore, run_service: LLMRunService, record: VideoRecord, monkeypatch
):
    """Cancelling a job whose coarse request is already in flight must not
    claim the vendor call itself was stopped - only that the *next* pass
    won't start. The message right after cancel() (before the run has
    actually finished) says so, distinct from "Cancelled before it
    started" (the queued case) and from the final "Run cancelled"."""
    engine = engine_store.create(
        provider_name="openai", model_id="gpt-5-mini", env_var="OPENAI_API_KEY"
    )
    seen_message_at_cancel_time: dict[str, str] = {}

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            cancelled_job = run_service.cancel(job_holder["run_id"])
            seen_message_at_cancel_time["message"] = cancelled_job.message
            assert cancelled_job.status == "running"  # not yet actually stopped
        return canned_json_response(start_s=4.0, end_s=12.0, confidence=0.9)

    monkeypatch.setattr(engine_store, "build_provider", lambda e: StubTimingProvider(respond))

    job_holder: dict[str, str] = {}
    job = run_service.submit(record, engine)
    job_holder["run_id"] = job.run_id
    _wait_for(run_service, job.run_id)

    assert "stopping after the current" in seen_message_at_cancel_time["message"]
    assert "current" in seen_message_at_cancel_time["message"]


def test_a_frames_only_provider_request_never_carries_the_video_file(
    engine_store: LLMEngineStore, run_service: LLMRunService, record: VideoRecord, monkeypatch
):
    """The provider only ever receives extracted JPEG frames - never the
    original uploaded video file or its path."""
    seen_requests: list[ProviderRequest] = []

    def respond(request: ProviderRequest) -> RawProviderResponse:
        seen_requests.append(request)
        if request.pass_name == "end_validate":
            candidate_ts = next(f.timestamp_s for f in request.frames if f.is_candidate)
            return canned_json_response(start_s=candidate_ts, end_s=candidate_ts, confidence=0.9)
        return canned_json_response(start_s=4.0, end_s=12.0, confidence=0.9)

    engine = engine_store.create(
        provider_name="openai", model_id="gpt-5-mini", env_var="OPENAI_API_KEY"
    )
    monkeypatch.setattr(engine_store, "build_provider", lambda e: StubTimingProvider(respond))

    job = run_service.submit(record, engine)
    _wait_for(run_service, job.run_id)

    assert seen_requests
    for request in seen_requests:
        assert not hasattr(request, "video_path")
        assert not hasattr(request, "video_bytes")
        for frame in request.frames:
            assert isinstance(frame.image_bytes, (bytes, bytearray))
            assert str(record.path) not in repr(frame)
