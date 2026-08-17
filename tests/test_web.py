"""Web layer: upload, analysis jobs, CSV export and error handling.

These use Flask's test client against the real application, including the real
storage and job service, with a temporary data directory.
"""

from __future__ import annotations

import csv
import io
import time
from dataclasses import replace
from pathlib import Path

import pytest

from app.config import AppConfig
from app.services.analysis_service import AnalysisService
from app.services.storage import VideoStore
from app.web.routes import create_app


@pytest.fixture
def fast_sweeper(tmp_path: Path):
    """A service whose retention sweep runs on a test-sized interval.

    Built directly rather than through create_app: the interval has to be set
    before the thread starts its first wait, which is exactly what the
    constructor argument is for.
    """
    config = replace(AppConfig(), data_dir=tmp_path / "sweep", log_level="WARNING")
    store = VideoStore(config)
    service = AnalysisService(config, store, sweep_interval_s=0.05)
    try:
        yield config, store, service
    finally:
        service.shutdown()


def upload(client, path: Path, filename: str | None = None):
    return client.post(
        "/api/videos",
        data={"file": (io.BytesIO(path.read_bytes()), filename or path.name)},
        content_type="multipart/form-data",
    )


def wait_for_job(client, job_id: str, timeout_s: float = 90.0) -> dict:
    """Poll a job the way the browser does, until it finishes."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        payload = client.get(f"/api/analyses/{job_id}").get_json()
        if payload["status"] in {"complete", "failed", "cancelled"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Job {job_id} did not finish within {timeout_s} s")


class TestPageAndHealth:
    def test_index_renders_the_five_steps(self, client):
        response = client.get("/")
        assert response.status_code == 200
        body = response.get_data(as_text=True)
        for text in ("Upload a video", "Choose what to analyse", "Configure", "Analyse", "Results"):
            assert text in body

    def test_index_lists_the_modes(self, client):
        body = client.get("/").get_data(as_text=True)
        assert "Zahn cup viscosity" in body
        assert "Robot / machine activity" in body

    def test_health(self, client):
        assert client.get("/api/health").get_json() == {"status": "ok"}

    def test_modes_endpoint(self, client):
        modes = client.get("/api/modes").get_json()["modes"]
        assert {mode["name"] for mode in modes} == {"zahn_cup", "robot_activity", "motion_scan"}

    def test_security_headers_are_set(self, client):
        response = client.get("/")
        assert "default-src 'self'" in response.headers["Content-Security-Policy"]
        assert response.headers["X-Content-Type-Options"] == "nosniff"


class TestUpload:
    def test_upload_returns_metadata(self, client, motion_video):
        payload = upload(client, motion_video.path).get_json()
        assert payload["width"] == motion_video.width
        assert payload["fps"] == pytest.approx(motion_video.fps, abs=0.01)
        assert payload["video_id"]
        assert payload["original_name"] == "motion.mp4"

    def test_upload_without_a_file_is_rejected(self, client):
        response = client.post("/api/videos", data={}, content_type="multipart/form-data")
        assert response.status_code == 400
        assert "error" in response.get_json()

    def test_unsupported_extension_is_rejected(self, client, motion_video):
        response = upload(client, motion_video.path, filename="payload.exe")
        assert response.status_code == 400
        assert "supported" in response.get_json()["error"].lower()

    def test_a_file_that_is_not_a_video_is_rejected(self, client, broken_video):
        response = upload(client, broken_video)
        assert response.status_code == 400
        assert response.get_json()["error"]

    def test_no_traceback_is_ever_returned(self, client, broken_video):
        body = upload(client, broken_video).get_data(as_text=True)
        assert "Traceback" not in body and 'File "' not in body

    def test_path_traversal_filename_cannot_escape_the_upload_folder(
        self, client, motion_video, app
    ):
        response = upload(client, motion_video.path, filename="../../../evil.mp4")
        assert response.status_code == 200
        upload_dir = app.extensions["app_config"].upload_dir
        stored = list(upload_dir.glob("*"))
        assert len(stored) == 1
        assert stored[0].parent == upload_dir
        assert "evil" not in stored[0].name  # the client name is never a path

    def test_oversized_upload_is_rejected(self, client, app, motion_video):
        app.config["MAX_CONTENT_LENGTH"] = 1024
        response = upload(client, motion_video.path)
        assert response.status_code == 413
        assert "larger than" in response.get_json()["error"]

    def test_metadata_can_be_fetched_again(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        assert client.get(f"/api/videos/{video_id}").get_json()["video_id"] == video_id

    def test_unknown_video_returns_404(self, client):
        response = client.get("/api/videos/does-not-exist")
        assert response.status_code == 404
        assert response.get_json()["error"]


class TestFrameAndMedia:
    def test_frame_endpoint_returns_a_jpeg(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        response = client.get(f"/api/videos/{video_id}/frame?t=6.0")
        assert response.status_code == 200
        assert response.mimetype == "image/jpeg"
        assert response.get_data()[:2] == b"\xff\xd8"  # JPEG magic bytes

    def test_frame_endpoint_can_draw_the_region(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        response = client.get(
            f"/api/videos/{video_id}/frame?t=1&roi_x=10&roi_y=10&roi_w=100&roi_h=80"
        )
        assert response.status_code == 200

    def test_invalid_frame_time_is_rejected(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        response = client.get(f"/api/videos/{video_id}/frame?t=abc")
        assert response.status_code == 400

    def test_media_supports_range_requests(self, client, motion_video):
        """Without ranges the preview player cannot seek in a long recording."""
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        response = client.get(f"/api/videos/{video_id}/media", headers={"Range": "bytes=0-1023"})
        assert response.status_code == 206
        assert len(response.get_data()) == 1024


class TestAnalysisJobs:
    def test_motion_analysis_end_to_end(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        started = client.post(
            "/api/analyses",
            json={
                "video_id": video_id,
                "mode": "motion_scan",
                "params": {"shortest_event_s": 3.0, "min_event_duration_s": 1.0},
            },
        ).get_json()
        assert started["status"] in {"queued", "running"}
        assert started["plan"]["sampling_interval_s"] > 0

        finished = wait_for_job(client, started["job_id"])
        assert finished["status"] == "complete"
        assert finished["counts"]["total"] == 2
        assert finished["progress"] == 1.0
        assert finished["event_log"]["rows"]

    def test_zahn_analysis_end_to_end(self, client, zahn_video):
        video_id = upload(client, zahn_video.path).get_json()["video_id"]
        started = client.post(
            "/api/analyses",
            json={
                "video_id": video_id,
                "mode": "zahn_cup",
                "params": {
                    "outlet": {
                        "x": int(zahn_video.truth["outlet_x"]),
                        "y": int(zahn_video.truth["outlet_y"]),
                    }
                },
            },
        ).get_json()
        finished = wait_for_job(client, started["job_id"])

        assert finished["status"] == "complete"
        summary = finished["summary"]
        assert summary["efflux_seconds"] == pytest.approx(zahn_video.truth["efflux_s"], abs=0.5)
        assert summary["status"] in {"confirmed", "review"}

    def test_wall_clock_columns_appear_when_a_start_time_is_given(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        started = client.post(
            "/api/analyses",
            json={
                "video_id": video_id,
                "mode": "motion_scan",
                "params": {"shortest_event_s": 3.0},
                "recording_start": "2026-03-04T07:43:00",
            },
        ).get_json()
        finished = wait_for_job(client, started["job_id"])
        assert finished["event_log"]["has_wall_clock"] is True
        assert finished["event_log"]["rows"][0]["start_wall_clock"].startswith("2026-03-04")

    def test_video_relative_only_without_a_start_time(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        started = client.post(
            "/api/analyses",
            json={"video_id": video_id, "mode": "motion_scan", "params": {"shortest_event_s": 3.0}},
        ).get_json()
        finished = wait_for_job(client, started["job_id"])
        assert finished["event_log"]["has_wall_clock"] is False
        assert finished["event_log"]["rows"][0]["start_wall_clock"] == ""

    def test_unparseable_recording_start_is_rejected(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        response = client.post(
            "/api/analyses",
            json={
                "video_id": video_id,
                "mode": "motion_scan",
                "params": {},
                "recording_start": "some time yesterday",
            },
        )
        assert response.status_code == 400

    def test_zahn_without_an_outlet_is_rejected_immediately(self, client, zahn_video):
        video_id = upload(client, zahn_video.path).get_json()["video_id"]
        response = client.post(
            "/api/analyses", json={"video_id": video_id, "mode": "zahn_cup", "params": {}}
        )
        assert response.status_code == 400
        assert "outlet" in response.get_json()["error"].lower()

    def test_invalid_roi_is_rejected_immediately(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        response = client.post(
            "/api/analyses",
            json={
                "video_id": video_id,
                "mode": "motion_scan",
                "params": {"roi": {"x": 0, "y": 0, "width": 9000, "height": 9000}},
            },
        )
        assert response.status_code == 400

    def test_unknown_mode_is_rejected(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        response = client.post(
            "/api/analyses", json={"video_id": video_id, "mode": "deep_magic", "params": {}}
        )
        assert response.status_code == 400

    def test_missing_video_is_reported(self, client):
        response = client.post(
            "/api/analyses", json={"video_id": "nope", "mode": "motion_scan", "params": {}}
        )
        assert response.status_code == 404

    def test_unknown_job_is_reported(self, client):
        assert client.get("/api/analyses/nope").status_code == 404

    def test_repeated_analysis_of_the_same_video(self, client, motion_video):
        """Re-running after a browser refresh must work and not interfere."""
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        job_ids = []
        for _ in range(2):
            started = client.post(
                "/api/analyses",
                json={
                    "video_id": video_id,
                    "mode": "motion_scan",
                    "params": {"shortest_event_s": 6.0},
                },
            ).get_json()
            job_ids.append(started["job_id"])
        results = [wait_for_job(client, job_id) for job_id in job_ids]
        assert all(result["status"] == "complete" for result in results)
        assert results[0]["counts"]["total"] == results[1]["counts"]["total"]
        assert job_ids[0] != job_ids[1]

    def test_results_survive_a_browser_refresh(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        started = client.post(
            "/api/analyses",
            json={"video_id": video_id, "mode": "motion_scan", "params": {"shortest_event_s": 6.0}},
        ).get_json()
        wait_for_job(client, started["job_id"])
        # A refreshed page asks for the same job again.
        again = client.get(f"/api/analyses/{started['job_id']}").get_json()
        assert again["status"] == "complete"
        assert again["counts"]["total"] >= 1

    def test_cancelling_a_job(self, client, zahn_video):
        video_id = upload(client, zahn_video.path).get_json()["video_id"]
        started = client.post(
            "/api/analyses",
            json={
                "video_id": video_id,
                "mode": "zahn_cup",
                "params": {"outlet": {"x": 240, "y": 180}},
            },
        ).get_json()
        client.post(f"/api/analyses/{started['job_id']}/cancel")
        finished = wait_for_job(client, started["job_id"])
        assert finished["status"] in {"cancelled", "complete"}


class TestCsvDownload:
    def test_csv_download(self, client, motion_video):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        started = client.post(
            "/api/analyses",
            json={"video_id": video_id, "mode": "motion_scan", "params": {"shortest_event_s": 3.0}},
        ).get_json()
        wait_for_job(client, started["job_id"])

        response = client.get(f"/api/analyses/{started['job_id']}/events.csv")
        assert response.status_code == 200
        assert response.mimetype == "text/csv"
        assert "attachment" in response.headers["Content-Disposition"]

        rows = list(csv.DictReader(io.StringIO(response.get_data(as_text=True))))
        assert len(rows) == 2
        assert rows[0]["event"] == "Motion"
        assert float(rows[0]["duration_s"]) > 0

    def test_csv_for_an_unknown_job_is_reported(self, client):
        assert client.get("/api/analyses/nope/events.csv").status_code == 404


class TestCleanup:
    def test_deleting_a_video_removes_the_file(self, client, motion_video, app):
        video_id = upload(client, motion_video.path).get_json()["video_id"]
        upload_dir = app.extensions["app_config"].upload_dir
        assert len(list(upload_dir.glob("*"))) == 1

        client.delete(f"/api/videos/{video_id}")
        assert list(upload_dir.glob("*")) == []
        assert client.get(f"/api/videos/{video_id}").status_code == 404

    def test_orphan_files_are_cleared_on_start_up(self, tmp_path: Path):
        config = replace(AppConfig(), data_dir=tmp_path / "data", log_level="WARNING")
        config.upload_dir.mkdir(parents=True)
        leftover = config.upload_dir / "leftover.mp4"
        leftover.write_bytes(b"stale file from a previous run")

        application = create_app(config)
        try:
            assert not leftover.exists()
        finally:
            application.extensions["analysis_service"].shutdown()

    def test_expired_uploads_are_purged(self, client, motion_video, app):
        upload(client, motion_video.path)
        store = app.extensions["video_store"]
        for record in store.list_records():
            record.last_used_at = 0.0  # pretend it has been idle for ever
        assert store.purge_expired() == 1
        assert list(app.extensions["app_config"].upload_dir.glob("*")) == []


class TestRetentionSweeper:
    """The retention policy must be enforced while the server runs.

    Regression: `purge_expired` existed on both the store and the service but
    was called only from tests, so the configured 24 h retention never removed
    anything during a session - the one cleanup that ran was the orphan sweep
    at start-up, which fires when the disk is still empty.
    """

    def test_the_sweeper_thread_is_running(self, app):
        service = app.extensions["analysis_service"]
        assert service._sweeper.is_alive()
        assert service._sweeper.daemon

    def test_a_sweep_removes_expired_uploads_and_job_records(self, client, motion_video, app):
        upload(client, motion_video.path)
        service = app.extensions["analysis_service"]
        upload_dir = app.extensions["app_config"].upload_dir
        assert len(list(upload_dir.glob("*"))) == 1

        for record in app.extensions["video_store"].list_records():
            record.last_used_at = 0.0  # aged past the retention window

        uploads_removed, _ = service.run_retention_sweep()
        assert uploads_removed == 1
        assert list(upload_dir.glob("*")) == []

    def test_the_sweeper_runs_without_any_request_traffic(self, fast_sweeper, motion_video):
        """The disk-filling case is an upload followed by an idle server."""
        config, store, _service = fast_sweeper
        with motion_video.path.open("rb") as handle:
            store.save_upload(handle, "idle.mp4")
        assert len(list(config.upload_dir.glob("*"))) == 1

        for record in store.list_records():
            record.last_used_at = 0.0  # aged past the retention window

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and list(config.upload_dir.glob("*")):
            time.sleep(0.05)
        assert list(config.upload_dir.glob("*")) == [], "the idle sweeper never ran"

    def test_a_failing_sweep_does_not_kill_the_sweeper(self, fast_sweeper, monkeypatch):
        """One transient error must not silently disable retention for good."""
        _config, store, service = fast_sweeper
        calls: list[int] = []

        def exploding_purge() -> int:
            calls.append(1)
            raise OSError("disk hiccup")

        monkeypatch.setattr(store, "purge_expired", exploding_purge)
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and len(calls) < 2:
            time.sleep(0.05)
        assert len(calls) >= 2, "the sweeper stopped after the first failure"
        assert service._sweeper.is_alive()

    def test_the_sweeper_stops_on_shutdown(self, fast_sweeper):
        _config, _store, service = fast_sweeper
        assert service._sweeper.is_alive()

        service.shutdown()
        service._sweeper.join(timeout=5.0)
        assert not service._sweeper.is_alive(), "the sweeper outlived shutdown()"
