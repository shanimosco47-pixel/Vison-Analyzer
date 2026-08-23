"""Web layer tests for the Experimental LLM analysis section
(app/web/llm_engine_routes.py).

Uses Flask's test client against the real application (same pattern as
tests/test_web.py), with a temporary data directory and, where an actual
"saved API key" round trip is being tested, a fake in-memory keyring
backend injected onto the real ``SecretStore`` instance the app already
built - no real OS secret service, network call, or vendor SDK anywhere
in this file.
"""

from __future__ import annotations

import io
import json
import time
from pathlib import Path

import pytest

from app.analysis.llm_timing.provider import (
    ProviderRequest,
    RawProviderResponse,
    StubTimingProvider,
    canned_json_response,
)


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
def working_secret_store(app):
    """Swap in a fake, always-available keyring backend for this app's
    SecretStore, so the "save a real API key" path can be tested without
    the real ``keyring`` package or an OS secret service."""
    backend = _FakeKeyringBackend()
    app.extensions["secret_store"]._backend_factory = lambda: backend
    return backend


def _confirmed_stub_provider():
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


def upload(client, path: Path):
    return client.post(
        "/api/videos",
        data={"file": (io.BytesIO(path.read_bytes()), path.name)},
        content_type="multipart/form-data",
    )


def _wait_for_run(client, run_id: str, timeout_s: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        payload = client.get(f"/api/llm-runs/{run_id}").get_json()
        if payload["status"] in {"complete", "failed", "cancelled"}:
            return payload
        time.sleep(0.02)
    raise AssertionError(f"Run {run_id} did not finish within {timeout_s}s")


# --------------------------------------------------------------------------- #
# Engine CRUD
# --------------------------------------------------------------------------- #


class TestEngineCrud:
    def test_list_is_empty_initially(self, client):
        assert client.get("/api/llm-engines").get_json() == {"engines": []}

    def test_create_with_an_env_var_reference(self, client):
        response = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "display_name": "My OpenAI engine",
                "env_var": "OPENAI_API_KEY",
            },
        )
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["provider_name"] == "openai"
        assert payload["model_id"] == "gpt-4.1-mini"
        assert payload["credential_configured"] is True
        assert "credential_ref" not in payload
        assert "env_var" not in payload
        assert "api_key" not in payload

    def test_create_without_a_credential_is_rejected(self, client):
        response = client.post(
            "/api/llm-engines", json={"provider_name": "openai", "model_id": "gpt-4.1-mini"}
        )
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_create_with_an_unsupported_provider_is_rejected(self, client):
        response = client.post(
            "/api/llm-engines",
            json={"provider_name": "anthropic", "model_id": "claude", "env_var": "X"},
        )
        assert response.status_code == 400

    def test_create_with_an_api_key_when_no_secret_backend_exists_fails_safely(self, client):
        """This test's app has no fake keyring backend injected, so the
        real (unavailable-in-CI) SecretStore path is exercised: the
        request must fail with a clear, safe message - never a crash,
        never leaking the submitted key."""
        response = client.post(
            "/api/llm-engines",
            json={"provider_name": "openai", "model_id": "gpt-4.1-mini", "api_key": "sk-abc123"},
        )
        assert response.status_code == 400
        body = response.get_json()
        assert "sk-abc123" not in json.dumps(body)

    def test_create_with_an_api_key_saves_it_write_only(self, client, working_secret_store):
        response = client.post(
            "/api/llm-engines",
            json={"provider_name": "openai", "model_id": "gpt-4.1-mini", "api_key": "sk-abc123"},
        )
        assert response.status_code == 200
        body = response.get_json()
        assert body["credential_configured"] is True
        assert "sk-abc123" not in json.dumps(body)
        # And the key never appears anywhere else the client can read back.
        listed = client.get("/api/llm-engines").get_json()
        assert "sk-abc123" not in json.dumps(listed)

    def test_the_saved_api_key_never_appears_in_the_persisted_metadata_file(
        self, app, client, working_secret_store
    ):
        client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "api_key": "sk-super-secret",
            },
        )
        raw = (app.extensions["app_config"].data_dir / "llm_engines.json").read_text()
        assert "sk-super-secret" not in raw

    def test_multiple_engines_are_configured_independently(self, client):
        a = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "display_name": "A",
                "env_var": "A_KEY",
            },
        ).get_json()
        b = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "gemini",
                "model_id": "gemini-3.5-flash",
                "display_name": "B",
                "env_var": "B_KEY",
            },
        ).get_json()
        listed = client.get("/api/llm-engines").get_json()["engines"]
        ids = {e["engine_id"] for e in listed}
        assert {a["engine_id"], b["engine_id"]} == ids
        by_id = {e["engine_id"]: e for e in listed}
        assert by_id[a["engine_id"]]["display_name"] == "A"
        assert by_id[b["engine_id"]]["display_name"] == "B"
        assert by_id[a["engine_id"]]["provider_name"] == "openai"
        assert by_id[b["engine_id"]]["provider_name"] == "gemini"

    def test_update_changes_the_display_name(self, client):
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()
        response = client.put(
            f"/api/llm-engines/{engine['engine_id']}", json={"display_name": "Renamed"}
        )
        assert response.status_code == 200
        assert response.get_json()["display_name"] == "Renamed"

    def test_update_of_an_unknown_engine_is_404(self, client):
        response = client.put("/api/llm-engines/does-not-exist", json={"display_name": "x"})
        assert response.status_code == 404

    def test_delete_removes_the_engine(self, client):
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()
        response = client.delete(f"/api/llm-engines/{engine['engine_id']}")
        assert response.status_code == 200
        assert response.get_json() == {"deleted": True}
        assert client.get("/api/llm-engines").get_json()["engines"] == []

    def test_delete_of_an_unknown_engine_is_404(self, client):
        assert client.delete("/api/llm-engines/does-not-exist").status_code == 404

    def test_test_endpoint_reports_a_safe_failure_for_an_unresolvable_credential(self, client):
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "THIS_VAR_IS_NEVER_SET",
            },
        ).get_json()
        response = client.post(f"/api/llm-engines/{engine['engine_id']}/test")
        assert response.status_code == 200
        body = response.get_json()
        assert body["ok"] is False
        assert isinstance(body["message"], str) and body["message"]


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


class TestLLMRuns:
    def test_run_completes_with_a_confirmed_result(self, app, client, zahn_video, monkeypatch):
        video = upload(client, zahn_video.path).get_json()
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()

        monkeypatch.setattr(
            app.extensions["llm_engine_store"],
            "build_provider",
            lambda e: _confirmed_stub_provider(),
        )

        started = client.post(
            f"/api/videos/{video['video_id']}/llm-runs", json={"engine_id": engine["engine_id"]}
        )
        assert started.status_code == 200
        run = started.get_json()
        assert run["status"] in {"queued", "running", "complete"}
        assert run["experimental"] is True

        payload = _wait_for_run(client, run["run_id"])
        assert payload["status"] == "complete"
        assert payload["outcome"]["verdict"]["status"] == "confirmed"
        assert payload["event"] is not None

    def test_run_with_a_missing_credential_fails_with_a_safe_message(self, client, zahn_video):
        video = upload(client, zahn_video.path).get_json()
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "THIS_VAR_IS_NEVER_SET",
            },
        ).get_json()

        started = client.post(
            f"/api/videos/{video['video_id']}/llm-runs", json={"engine_id": engine["engine_id"]}
        ).get_json()
        payload = _wait_for_run(client, started["run_id"])
        assert payload["status"] == "failed"
        assert payload["error"]

    def test_run_against_an_unknown_video_is_404(self, client):
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()
        response = client.post(
            "/api/videos/does-not-exist/llm-runs", json={"engine_id": engine["engine_id"]}
        )
        assert response.status_code == 404

    def test_run_with_an_unknown_engine_is_404(self, client, zahn_video):
        video = upload(client, zahn_video.path).get_json()
        response = client.post(
            f"/api/videos/{video['video_id']}/llm-runs", json={"engine_id": "does-not-exist"}
        )
        assert response.status_code == 404

    def test_cancel_a_run(self, app, client, zahn_video, monkeypatch):
        video = upload(client, zahn_video.path).get_json()
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()

        # Fill every worker slot so the new run stays queued long enough to cancel.
        import threading

        run_service = app.extensions["llm_run_service"]
        gate = threading.Event()
        for _ in range(3):
            run_service._executor.submit(gate.wait)

        monkeypatch.setattr(
            app.extensions["llm_engine_store"],
            "build_provider",
            lambda e: _confirmed_stub_provider(),
        )
        started = client.post(
            f"/api/videos/{video['video_id']}/llm-runs", json={"engine_id": engine["engine_id"]}
        ).get_json()
        cancelled = client.post(f"/api/llm-runs/{started['run_id']}/cancel")
        gate.set()

        assert cancelled.status_code == 200
        assert cancelled.get_json()["status"] == "cancelled"

    def test_status_of_an_unknown_run_is_404(self, client):
        assert client.get("/api/llm-runs/does-not-exist").status_code == 404

    def test_confirmed_run_exposes_pass_frames_and_evidence_timestamps(
        self, app, client, zahn_video, monkeypatch
    ):
        video = upload(client, zahn_video.path).get_json()
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()
        monkeypatch.setattr(
            app.extensions["llm_engine_store"],
            "build_provider",
            lambda e: _confirmed_stub_provider(),
        )
        started = client.post(
            f"/api/videos/{video['video_id']}/llm-runs", json={"engine_id": engine["engine_id"]}
        ).get_json()
        payload = _wait_for_run(client, started["run_id"])

        assert set(payload["outcome"]["pass_frames"]) == {
            "coarse",
            "fine",
            "end_coarse",
            "end_validate",
        }
        assert payload["start_evidence_s"] is not None
        assert payload["end_evidence_s"] is not None
        assert payload["start_evidence_s"] in payload["outcome"]["pass_frames"]["fine"]
        assert payload["end_evidence_s"] in payload["outcome"]["pass_frames"]["end_validate"]

    def test_evidence_image_endpoint_serves_a_jpeg_for_a_confirmed_run(
        self, app, client, zahn_video, monkeypatch
    ):
        video = upload(client, zahn_video.path).get_json()
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()
        monkeypatch.setattr(
            app.extensions["llm_engine_store"],
            "build_provider",
            lambda e: _confirmed_stub_provider(),
        )
        started = client.post(
            f"/api/videos/{video['video_id']}/llm-runs", json={"engine_id": engine["engine_id"]}
        ).get_json()
        _wait_for_run(client, started["run_id"])

        for boundary in ("start", "end"):
            response = client.get(f"/api/llm-runs/{started['run_id']}/evidence/{boundary}")
            assert response.status_code == 200
            assert response.mimetype == "image/jpeg"
            assert len(response.data) > 0

    def test_evidence_image_endpoint_uses_the_server_stored_timestamp_not_a_client_supplied_one(
        self, app, client, zahn_video, monkeypatch
    ):
        """The endpoint must never let a caller relabel an arbitrary frame
        as "evidence" by supplying its own ``t`` - it always serves the
        timestamp this run itself selected and grounded."""
        video = upload(client, zahn_video.path).get_json()
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()
        monkeypatch.setattr(
            app.extensions["llm_engine_store"],
            "build_provider",
            lambda e: _confirmed_stub_provider(),
        )
        started = client.post(
            f"/api/videos/{video['video_id']}/llm-runs", json={"engine_id": engine["engine_id"]}
        ).get_json()
        payload = _wait_for_run(client, started["run_id"])
        assert payload["start_evidence_s"] is not None

        import app.web.llm_engine_routes as routes_module

        real_video_reader = routes_module.VideoReader
        seen_timestamps: list[float] = []

        class _SpyVideoReader:
            def __init__(self, path, info):
                self._inner = real_video_reader(path, info)

            def __enter__(self):
                self._reader = self._inner.__enter__()
                return self

            def __exit__(self, *exc_info):
                return self._inner.__exit__(*exc_info)

            def frame_at(self, timestamp_s):
                seen_timestamps.append(timestamp_s)
                return self._reader.frame_at(timestamp_s)

        monkeypatch.setattr(routes_module, "VideoReader", _SpyVideoReader)

        response = client.get(f"/api/llm-runs/{started['run_id']}/evidence/start?t=999.0&t=-5")
        assert response.status_code == 200
        assert seen_timestamps == [payload["start_evidence_s"]]

    def test_evidence_endpoint_rejects_an_unknown_boundary(self, client, zahn_video):
        video = upload(client, zahn_video.path).get_json()
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()
        started = client.post(
            f"/api/videos/{video['video_id']}/llm-runs", json={"engine_id": engine["engine_id"]}
        ).get_json()
        response = client.get(f"/api/llm-runs/{started['run_id']}/evidence/middle")
        assert response.status_code == 400

    def test_evidence_endpoint_404s_rather_than_fabricating_for_an_abstained_run(
        self, app, client, zahn_video, monkeypatch
    ):
        video = upload(client, zahn_video.path).get_json()
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()

        def respond(request: ProviderRequest) -> RawProviderResponse:
            return RawProviderResponse(
                model_id="stub-model",
                raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
                latency_s=0.01,
            )

        monkeypatch.setattr(
            app.extensions["llm_engine_store"],
            "build_provider",
            lambda e: StubTimingProvider(respond),
        )
        started = client.post(
            f"/api/videos/{video['video_id']}/llm-runs", json={"engine_id": engine["engine_id"]}
        ).get_json()
        payload = _wait_for_run(client, started["run_id"])
        assert payload["status"] == "complete"
        assert payload["outcome"]["verdict"]["status"] == "abstain"

        for boundary in ("start", "end"):
            response = client.get(f"/api/llm-runs/{started['run_id']}/evidence/{boundary}")
            assert response.status_code == 404


class TestLLMModelOptions:
    def test_model_options_endpoint_returns_the_server_side_allowlist(self, client):
        response = client.get("/api/llm-model-options")
        assert response.status_code == 200
        body = response.get_json()
        assert "openai" in body and "gemini" in body
        assert "gpt-4.1-mini" in body["openai"]
        # The list is not empty for either provider and every model in it
        # is actually accepted by engine creation - proven indirectly by
        # the allowlist tests in test_llm_engine_store.py; here we just
        # confirm the wire shape.
        assert all(isinstance(models, list) and models for models in body.values())


class TestLLMRunAudit:
    """The durable audit trail's HTTP surface: fetching one run's record,
    finding an engine's latest one, the download's Content-Disposition
    header, and that the evidence-image endpoint keeps working from the
    durable record after the in-memory job is gone (simulating a server
    restart without actually restarting the process)."""

    def _run_a_confirmed_engine(self, app, client, zahn_video, monkeypatch):
        video = upload(client, zahn_video.path).get_json()
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "display_name": "Audit Test Engine",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()
        monkeypatch.setattr(
            app.extensions["llm_engine_store"],
            "build_provider",
            lambda e: _confirmed_stub_provider(),
        )
        started = client.post(
            f"/api/videos/{video['video_id']}/llm-runs", json={"engine_id": engine["engine_id"]}
        ).get_json()
        _wait_for_run(client, started["run_id"])
        return engine, started["run_id"]

    def test_audit_endpoint_returns_the_full_record(self, app, client, zahn_video, monkeypatch):
        engine, run_id = self._run_a_confirmed_engine(app, client, zahn_video, monkeypatch)

        response = client.get(f"/api/llm-runs/{run_id}/audit")

        assert response.status_code == 200
        assert "attachment" in response.headers.get("Content-Disposition", "")
        assert run_id in response.headers["Content-Disposition"]
        body = response.get_json()
        assert body["run_id"] == run_id
        assert body["engine_id"] == engine["engine_id"]
        assert set(body["passes"]) == {"coarse", "fine", "end_coarse", "end_validate"}
        assert body["final_verdict"]["status"] == "confirmed"
        assert body["derived"]["locked_start_s"] is not None

    def test_audit_endpoint_is_404_for_an_unknown_run(self, client):
        response = client.get("/api/llm-runs/no-such-run/audit")
        assert response.status_code == 404

    def test_latest_audit_for_engine_endpoint(self, app, client, zahn_video, monkeypatch):
        engine, run_id = self._run_a_confirmed_engine(app, client, zahn_video, monkeypatch)

        response = client.get(f"/api/llm-engines/{engine['engine_id']}/latest-audit")

        assert response.status_code == 200
        assert response.get_json()["run_id"] == run_id

    def test_latest_audit_for_engine_endpoint_404s_before_any_run(self, client):
        engine = client.post(
            "/api/llm-engines",
            json={
                "provider_name": "openai",
                "model_id": "gpt-4.1-mini",
                "env_var": "OPENAI_API_KEY",
            },
        ).get_json()
        response = client.get(f"/api/llm-engines/{engine['engine_id']}/latest-audit")
        assert response.status_code == 404

    def test_evidence_endpoint_falls_back_to_the_durable_audit_after_the_job_is_forgotten(
        self, app, client, zahn_video, monkeypatch
    ):
        """Simulates the in-memory job map being gone (server restart, or
        past the 6-hour retention window) without actually restarting the
        process: the evidence image must still be servable purely from
        the durable audit record on disk."""
        engine, run_id = self._run_a_confirmed_engine(app, client, zahn_video, monkeypatch)

        # Confirm both boundaries work while the job is still in memory,
        # then forget it and confirm they still work from disk alone.
        assert client.get(f"/api/llm-runs/{run_id}/evidence/start").status_code == 200
        app.extensions["llm_run_service"]._jobs.pop(run_id, None)

        for boundary in ("start", "end"):
            response = client.get(f"/api/llm-runs/{run_id}/evidence/{boundary}")
            assert response.status_code == 200
            assert response.mimetype == "image/jpeg"
            assert len(response.data) > 0

    def test_audit_record_over_the_wire_never_carries_credentials_or_the_video_path(
        self, app, client, zahn_video, monkeypatch
    ):
        _engine, run_id = self._run_a_confirmed_engine(app, client, zahn_video, monkeypatch)

        response = client.get(f"/api/llm-runs/{run_id}/audit")
        raw_body = response.get_data(as_text=True)

        assert str(zahn_video.path) not in raw_body
        assert "credential" not in raw_body.lower()
        assert "secret:" not in raw_body
        assert "OPENAI_API_KEY" not in raw_body
        assert "api_key" not in raw_body.lower()
