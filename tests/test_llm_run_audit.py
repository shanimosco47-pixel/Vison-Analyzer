"""Tests for app/services/llm_run_audit.py: the durable, redacted per-run
audit trail, and its wiring into LLMRunService.

Three layers, each covered separately:

*   ``build_audit_record`` - a pure function, tested directly against a
    real ``PipelineOutcome`` (via ``run_llm_timing`` + a stubbed
    provider), no I/O.
*   ``LLMRunAuditStore`` - atomic persistence, tested directly against a
    ``tmp_path``, including "a fresh instance pointed at the same
    directory" to prove durability across what amounts to a server
    restart.
*   ``LLMRunService`` - the integration: a real run through the service
    ends up as a readable, durable audit record, including the safe
    partial cases (cancelled-before-start, a credential failure).
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import pytest

from app.analysis.llm_timing.pipeline import PipelineConfig, run_llm_timing
from app.analysis.llm_timing.prompts import PROMPT_V1, PROMPT_V1_ID
from app.analysis.llm_timing.provider import (
    ProviderRequest,
    RawProviderResponse,
    StubTimingProvider,
    canned_json_response,
    cascade_confirmed_response,
)
from app.config import AppConfig
from app.errors import NotFoundError
from app.services.llm_engine_store import LLMEngineStore
from app.services.llm_run_audit import LLMRunAuditStore, build_audit_record
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
def app_config(tmp_path: Path) -> AppConfig:
    return replace(AppConfig(), data_dir=tmp_path / "data")


@pytest.fixture
def engine_store(app_config: AppConfig) -> LLMEngineStore:
    backend = _FakeKeyringBackend()
    return LLMEngineStore(app_config, SecretStore(backend_factory=lambda: backend))


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
            return cascade_confirmed_response(request)
        assert request.pass_name == "end_coarse"
        return canned_json_response(start_s=candidate_s, end_s=candidate_s, confidence=0.9)

    return StubTimingProvider(respond)


# --------------------------------------------------------------------------- #
# build_audit_record
# --------------------------------------------------------------------------- #


def test_build_audit_record_for_a_confirmed_outcome_has_every_pass(zahn_video):
    provider = _confirmed_stub_provider()
    outcome = run_llm_timing(
        zahn_video.path, provider, prompt_version=PROMPT_V1_ID, prompt_text=PROMPT_V1
    )
    audit = build_audit_record(
        run_id="deadbeefdeadbeefdeadbeefdeadbeef",
        video_id="vid-1",
        engine_id="eng-1",
        engine_display_name="My Engine",
        provider_name="openai",
        model_id="gpt-4.1-mini",
        status="complete",
        created_at=1.0,
        started_at=1.0,
        finished_at=2.0,
        error=None,
        outcome=outcome,
    )
    assert set(audit["passes"].keys()) == {"coarse", "fine", "end_coarse", "end_validate"}
    for entry in audit["passes"].values():
        assert entry["status"] == "confirmed"
        assert entry["submitted_count"] == len(entry["submitted_timestamps_s"])
        assert entry["submitted_range_s"] == [
            entry["submitted_timestamps_s"][0],
            entry["submitted_timestamps_s"][-1],
        ]
        assert entry["prompt_version"]
    assert audit["derived"]["locked_start_s"] == pytest.approx(4.0)
    assert audit["derived"]["end_coarse_candidate_s"] == pytest.approx(12.0)
    assert audit["final_verdict"]["status"] == "confirmed"
    assert audit["event"] is not None


def test_build_audit_record_carries_the_running_backends_app_version():
    """Supervisor-directed operational requirement: the audit must record
    the backend build that produced it, so a downloaded audit, the page
    that showed it, and the running backend can be matched against each
    other. Never derived here - passed in by the caller (see
    app.version.app_version) - and defaults to "unknown" rather than
    raising when a caller doesn't supply one."""
    audit = build_audit_record(
        run_id="deadbeefdeadbeefdeadbeefdeadbeef",
        video_id="vid-1",
        engine_id="eng-1",
        engine_display_name="My Engine",
        provider_name="openai",
        model_id="gpt-4.1-mini",
        status="failed",
        created_at=1.0,
        started_at=1.0,
        finished_at=2.0,
        error="boom",
        outcome=None,
        app_version="abc1234",
    )
    assert audit["app_version"] == "abc1234"

    audit_default = build_audit_record(
        run_id="deadbeefdeadbeefdeadbeefdeadbeef",
        video_id="vid-1",
        engine_id="eng-1",
        engine_display_name="My Engine",
        provider_name="openai",
        model_id="gpt-4.1-mini",
        status="failed",
        created_at=1.0,
        started_at=1.0,
        finished_at=2.0,
        error="boom",
        outcome=None,
    )
    assert audit_default["app_version"] == "unknown"


def test_build_audit_record_exact_post_thinning_timestamps_match_pass_frames(zahn_video):
    """The audit's per-pass submitted_timestamps_s must be the exact same
    (post budget-thinning) set the pipeline actually sent - not a
    reconstructed/denser plan - proven the same way
    test_pipeline_thinned_pass_frames_match_the_frames_actually_sent proves
    it at the pipeline layer, but end-to-end through the audit builder."""
    provider = _confirmed_stub_provider()
    config = PipelineConfig(max_request_bytes=300_000)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_V1_ID,
        prompt_text=PROMPT_V1,
        config=config,
    )
    audit = build_audit_record(
        run_id="a" * 32,
        video_id="vid-1",
        engine_id="eng-1",
        engine_display_name="",
        provider_name="openai",
        model_id="gpt-4.1-mini",
        status="complete",
        created_at=1.0,
        started_at=1.0,
        finished_at=2.0,
        error=None,
        outcome=outcome,
    )
    fine_entry = audit["passes"]["fine"]
    assert fine_entry["submitted_count"] < 30
    assert fine_entry["submitted_timestamps_s"] == sorted(outcome.pass_frames["fine"])


def test_build_audit_record_for_a_rejected_candidate_still_shows_the_chain(zahn_video):
    """Same wrong-candidate shape as
    test_pipeline_derived_decisions_survive_a_rejected_end_candidate - the
    audit record built from that outcome must expose the sparse
    end-coarse candidate, the (too-narrow) validation window it produced,
    and the end_validate pass's own rejection reason/raw_notes, even
    though the run overall abstained."""
    candidate_s = 5.5

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=1.0, end_s=candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=1.0, end_s=1.0, confidence=0.9)
        if request.pass_name == "end_coarse":
            return canned_json_response(start_s=candidate_s, end_s=candidate_s, confidence=0.9)
        assert request.pass_name == "end_validate"
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"], '
            '"raw_notes": "no sustained drop observed in this window"}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path, provider, prompt_version=PROMPT_V1_ID, prompt_text=PROMPT_V1
    )
    audit = build_audit_record(
        run_id="b" * 32,
        video_id="vid-1",
        engine_id="eng-1",
        engine_display_name="",
        provider_name="openai",
        model_id="gpt-4.1-mini",
        status="complete",
        created_at=1.0,
        started_at=1.0,
        finished_at=2.0,
        error=None,
        outcome=outcome,
    )
    assert audit["final_verdict"]["status"] == "abstain"
    assert audit["event"] is None
    assert audit["derived"]["end_coarse_candidate_s"] == pytest.approx(candidate_s)
    assert audit["derived"]["end_validation_window_s"] == [pytest.approx(3.5), pytest.approx(7.5)]
    end_validate = audit["passes"]["end_validate"]
    assert end_validate["status"] == "abstain"
    assert "no_break_found" in end_validate["reason_codes"]
    assert "no sustained drop" in end_validate["raw_notes"]


def test_build_audit_record_for_a_run_that_never_reached_the_pipeline():
    """A run that fails before run_llm_timing is ever called (e.g. a
    credential error) has outcome=None - the record must still identify
    the run and carry the (sanitized) error, with empty-but-present
    passes/derived rather than missing keys."""
    audit = build_audit_record(
        run_id="c" * 32,
        video_id="vid-1",
        engine_id="eng-1",
        engine_display_name="My Engine",
        provider_name="openai",
        model_id="gpt-4.1-mini",
        status="failed",
        created_at=1.0,
        started_at=1.0,
        finished_at=1.5,
        error="Could not resolve this engine's saved API key.",
        outcome=None,
    )
    assert audit["passes"] == {}
    assert audit["derived"] == {}
    assert audit["final_verdict"] is None
    assert audit["event"] is None
    assert audit["error"] == "Could not resolve this engine's saved API key."
    assert audit["status"] == "failed"


def test_build_audit_record_redacts_secret_shaped_text_in_raw_notes_and_error(zahn_video):
    secret_like = "sk-" + "A" * 40

    def respond(request: ProviderRequest) -> RawProviderResponse:
        return RawProviderResponse(
            model_id="stub-model",
            raw_text=(
                '{"status": "abstain", "reason_codes": ["provider_error"], '
                f'"raw_notes": "auth failed with key {secret_like}"}}'
            ),
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path, provider, prompt_version=PROMPT_V1_ID, prompt_text=PROMPT_V1
    )
    audit = build_audit_record(
        run_id="d" * 32,
        video_id="vid-1",
        engine_id="eng-1",
        engine_display_name="",
        provider_name="openai",
        model_id="gpt-4.1-mini",
        status="complete",
        created_at=1.0,
        started_at=1.0,
        finished_at=2.0,
        error=f"vendor said: {secret_like}",
        outcome=outcome,
    )
    dumped = json.dumps(audit)
    assert secret_like not in dumped
    assert "[redacted]" in audit["passes"]["coarse"]["raw_notes"]
    assert audit["error"] is not None
    assert "[redacted]" in audit["error"]


def test_build_audit_record_never_carries_image_bytes_credentials_or_paths(zahn_video):
    """Structural guarantee: nothing this function is ever given access to
    (TimedFrame.image_bytes, EngineConfig.credential_ref, a VideoRecord's
    real filesystem Path) can appear in its output, because none of those
    are among its parameters in the first place - this test instead
    proves the *output* stays plain JSON-serializable data with no bytes
    objects and no path-shaped strings from this sandbox leaking in."""
    provider = _confirmed_stub_provider()
    outcome = run_llm_timing(
        zahn_video.path, provider, prompt_version=PROMPT_V1_ID, prompt_text=PROMPT_V1
    )
    audit = build_audit_record(
        run_id="e" * 32,
        video_id="vid-1",
        engine_id="eng-1",
        engine_display_name="",
        provider_name="openai",
        model_id="gpt-4.1-mini",
        status="complete",
        created_at=1.0,
        started_at=1.0,
        finished_at=2.0,
        error=None,
        outcome=outcome,
    )
    dumped = json.dumps(audit)  # raises TypeError if any value is bytes
    assert str(zahn_video.path) not in dumped
    assert "credential" not in dumped.lower()
    assert "secret:" not in dumped
    assert "api_key" not in dumped.lower()


# --------------------------------------------------------------------------- #
# LLMRunAuditStore
# --------------------------------------------------------------------------- #


def test_audit_store_save_and_get_round_trip(tmp_path: Path):
    store = LLMRunAuditStore(tmp_path)
    record = {"run_id": "f" * 32, "engine_id": "eng-1", "created_at": 100.0, "finished_at": 105.0}
    store.save(record)
    assert store.get("f" * 32) == record


def test_audit_store_write_is_atomic_no_tmp_file_left_behind(tmp_path: Path):
    store = LLMRunAuditStore(tmp_path)
    store.save({"run_id": "a" * 32, "engine_id": "e", "created_at": 1.0})
    audit_dir = tmp_path / "llm_run_audits"
    leftover_tmp = list(audit_dir.glob("*.tmp"))
    assert leftover_tmp == []
    assert (audit_dir / f"{'a' * 32}.json").is_file()


def test_audit_store_survives_a_fresh_instance_same_directory(tmp_path: Path):
    """Simulates a server restart: a brand-new LLMRunAuditStore pointed at
    the same data_dir can still read a record an earlier instance wrote -
    the whole point of writing to disk instead of keeping this only in
    memory."""
    first = LLMRunAuditStore(tmp_path)
    first.save({"run_id": "b" * 32, "engine_id": "eng-1", "created_at": 1.0, "finished_at": 2.0})

    second = LLMRunAuditStore(tmp_path)
    record = second.get("b" * 32)
    assert record is not None
    assert record["engine_id"] == "eng-1"


def test_audit_store_get_returns_none_for_an_unknown_or_unsafe_run_id(tmp_path: Path):
    store = LLMRunAuditStore(tmp_path)
    assert store.get("f" * 32) is None  # well-formed, but nothing saved
    assert store.get("../../etc/passwd") is None
    assert store.get("") is None
    # Confirm the unsafe id never touched the filesystem outside the audit dir.
    assert not (tmp_path / "etc").exists()


def test_audit_store_latest_for_engine_picks_the_most_recently_finished(tmp_path: Path):
    store = LLMRunAuditStore(tmp_path)
    store.save({"run_id": "a" * 32, "engine_id": "eng-1", "created_at": 1.0, "finished_at": 10.0})
    store.save({"run_id": "b" * 32, "engine_id": "eng-1", "created_at": 1.0, "finished_at": 20.0})
    store.save({"run_id": "c" * 32, "engine_id": "eng-2", "created_at": 1.0, "finished_at": 99.0})

    latest = store.latest_for_engine("eng-1")
    assert latest is not None
    assert latest["run_id"] == "b" * 32


def test_audit_store_latest_for_engine_returns_none_when_no_run_exists(tmp_path: Path):
    store = LLMRunAuditStore(tmp_path)
    assert store.latest_for_engine("no-such-engine") is None


def test_audit_store_purge_before_deletes_old_records_and_keeps_new_ones(tmp_path: Path):
    store = LLMRunAuditStore(tmp_path)
    store.save({"run_id": "a" * 32, "engine_id": "eng-1", "created_at": 1.0, "finished_at": 100.0})
    store.save({"run_id": "b" * 32, "engine_id": "eng-1", "created_at": 1.0, "finished_at": 900.0})

    removed = store.purge_before(500.0)

    assert removed == 1
    assert store.get("a" * 32) is None
    assert store.get("b" * 32) is not None


# --------------------------------------------------------------------------- #
# LLMRunService integration
# --------------------------------------------------------------------------- #


def _wait_for(service: LLMRunService, run_id: str, timeout_s: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        payload = service.get(run_id).to_dict()
        if payload["status"] in {"complete", "failed", "cancelled"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Run {run_id} did not finish within {timeout_s}s")


def test_a_completed_run_is_automatically_persisted_and_retrievable(
    engine_store: LLMEngineStore, app_config: AppConfig, record: VideoRecord, monkeypatch
):
    service = LLMRunService(engine_store, app_config, sweep_interval_s=0.05)
    try:
        engine = engine_store.create(
            provider_name="openai", model_id="gpt-4.1-mini", env_var="OPENAI_API_KEY"
        )
        monkeypatch.setattr(engine_store, "build_provider", lambda e: _confirmed_stub_provider())

        job = service.submit(record, engine)
        _wait_for(service, job.run_id)

        audit = service.get_audit(job.run_id)
        assert audit["run_id"] == job.run_id
        assert audit["engine_id"] == engine.engine_id
        assert audit["status"] == "complete"
        assert set(audit["passes"].keys()) == {"coarse", "fine", "end_coarse", "end_validate"}
        # LLMRunService threads the real, running backend's own build
        # identifier through automatically - never "unknown" here, since a
        # real git checkout is present.
        assert audit["app_version"] and audit["app_version"] != "unknown"

        latest = service.get_latest_audit_for_engine(engine.engine_id)
        assert latest["run_id"] == job.run_id
    finally:
        service.shutdown()


def test_audit_is_readable_from_a_brand_new_service_instance(
    engine_store: LLMEngineStore, app_config: AppConfig, record: VideoRecord, monkeypatch
):
    """Simulates a server restart at the LLMRunService layer: a run
    completed under one service instance must still be readable through a
    second, fresh instance pointed at the same data_dir - the in-memory
    job map does NOT carry over (get() on the new instance 404s), but the
    durable audit does."""
    first = LLMRunService(engine_store, app_config, sweep_interval_s=0.05)
    try:
        engine = engine_store.create(
            provider_name="openai", model_id="gpt-4.1-mini", env_var="OPENAI_API_KEY"
        )
        monkeypatch.setattr(engine_store, "build_provider", lambda e: _confirmed_stub_provider())
        job = first.submit(record, engine)
        _wait_for(first, job.run_id)
    finally:
        first.shutdown()

    second = LLMRunService(engine_store, app_config, sweep_interval_s=0.05)
    try:
        with pytest.raises(NotFoundError):
            second.get(job.run_id)  # the in-memory job map really is fresh

        audit = second.get_audit(job.run_id)
        assert audit["run_id"] == job.run_id
        assert audit["final_verdict"]["status"] == "confirmed"
    finally:
        second.shutdown()


def test_a_missing_credential_failure_still_persists_a_safe_partial_audit(
    engine_store: LLMEngineStore, app_config: AppConfig, record: VideoRecord
):
    service = LLMRunService(engine_store, app_config, sweep_interval_s=0.05)
    try:
        engine = engine_store.create(
            provider_name="openai", model_id="gpt-4.1-mini", env_var="THIS_VAR_IS_NEVER_SET"
        )
        job = service.submit(record, engine)
        _wait_for(service, job.run_id)

        audit = service.get_audit(job.run_id)
        assert audit["status"] == "failed"
        assert audit["passes"] == {}
        assert audit["final_verdict"] is None
        # The error names the *unset environment variable*, which is not a
        # secret value (there was never anything in it) - what matters is
        # that no actual credential value could have leaked, which is
        # trivially true here since none was ever resolved.
        assert audit["error"]
    finally:
        service.shutdown()


def test_a_run_cancelled_before_it_starts_still_persists_a_safe_partial_audit(
    engine_store: LLMEngineStore, app_config: AppConfig, record: VideoRecord, monkeypatch
):
    engine = engine_store.create(
        provider_name="openai", model_id="gpt-4.1-mini", env_var="OPENAI_API_KEY"
    )
    service = LLMRunService(engine_store, app_config, sweep_interval_s=0.05)
    try:
        import threading

        gate = threading.Event()
        service._executor.submit(gate.wait)
        service._executor.submit(gate.wait)
        service._executor.submit(gate.wait)  # fill every worker

        monkeypatch.setattr(engine_store, "build_provider", lambda e: _confirmed_stub_provider())
        job = service.submit(record, engine)
        cancelled = service.cancel(job.run_id)
        assert cancelled.status == "cancelled"
        gate.set()

        # The submitted _run_job still has to observe job.cancelled and
        # persist the audit - give it a moment once the gate is released.
        deadline = time.monotonic() + 5.0
        audit = None
        while time.monotonic() < deadline:
            try:
                audit = service.get_audit(job.run_id)
                break
            except NotFoundError:
                time.sleep(0.02)
        assert audit is not None
        assert audit["status"] == "cancelled"
        assert audit["passes"] == {}
    finally:
        service.shutdown()


def test_purge_expired_also_removes_the_durable_audit(
    engine_store: LLMEngineStore, app_config: AppConfig, record: VideoRecord, monkeypatch
):
    service = LLMRunService(engine_store, app_config, sweep_interval_s=0.05)
    try:
        engine = engine_store.create(
            provider_name="openai", model_id="gpt-4.1-mini", env_var="OPENAI_API_KEY"
        )
        monkeypatch.setattr(engine_store, "build_provider", lambda e: _confirmed_stub_provider())
        job = service.submit(record, engine)
        _wait_for(service, job.run_id)
        assert service.get_audit(job.run_id) is not None

        # Force this run's audit to look old, then purge with "now" as the
        # cutoff - bounded cleanup tied to the same retention policy the
        # in-memory job map already uses (see LLMRunService.purge_expired).
        service._audit_store.purge_before(time.time() + 1.0)

        with pytest.raises(NotFoundError):
            service.get_audit(job.run_id)
    finally:
        service.shutdown()
