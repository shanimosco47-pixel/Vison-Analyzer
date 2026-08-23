"""Standalone HTTP server for the browser regression suite.

Not part of the pytest suite - invoked as a subprocess by
``llm_flow.test.js`` (Node's built-in test runner, driving Playwright).
Builds the real Flask app (the same ``create_app()`` the product ships)
against a throwaway data directory, then patches
``LLMEngineStore.build_provider`` at the class level so every engine's
provider is a deterministic, offline ``StubTimingProvider`` (see
``app/analysis/llm_timing/provider.py``) - no network call, no real vendor
SDK, no paid API required to exercise the full UI flow end to end. This
mirrors the same stubbing pattern already used in
``tests/test_llm_run_service.py``'s ``_confirmed_stub_provider``, just
served over a real socket instead of ``app.test_client()``, because
Playwright needs an actual HTTP server to drive a real browser against.

Also writes a small synthetic video into the same throwaway directory so
the browser test can upload a real, valid file without committing a binary
fixture to the repository.

Usage: ``python e2e_server.py``. Prints exactly one line to stdout once
ready:

    LISTENING <port> <video_path>

then serves forever until the process is killed (the Node harness manages
its lifetime).
"""

from __future__ import annotations

import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from werkzeug.serving import make_server

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from app.analysis.llm_timing.provider import (  # noqa: E402
    ProviderRequest,
    RawProviderResponse,
    StubTimingProvider,
    canned_json_response,
)
from app.config import AppConfig  # noqa: E402
from app.services.llm_engine_store import LLMEngineStore  # noqa: E402
from app.web.routes import create_app  # noqa: E402


def _make_video(path: Path) -> None:
    """A short, cheap-to-generate but valid MP4 - its content is irrelevant
    here since the stub provider supplies the verdict directly; only its
    duration and frame timestamps matter to the pipeline's windowing."""
    width, height, fps, duration_s = 320, 240, 6.0, 26.0
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("This OpenCV build cannot write MP4 files")
    rng = np.random.default_rng(7)
    total_frames = int(fps * duration_s)
    for _ in range(total_frames):
        frame = np.full((height, width, 3), 200, dtype=np.uint8)
        noise = rng.integers(-3, 3, frame.shape, dtype=np.int16)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        cv2.circle(frame, (160, 120), 40, (80, 80, 220), -1)
        writer.write(frame)
    writer.release()


def _stub_respond(request: ProviderRequest) -> RawProviderResponse:
    """Deterministic CONFIRMED result for every pass - mirrors
    ``tests/test_llm_run_service.py``'s ``_confirmed_stub_provider``, a
    recipe already proven to reach a CONFIRMED verdict through the real
    pipeline against a real (probed) video."""
    candidate_s = 12.0
    if request.pass_name == "coarse":
        return canned_json_response(start_s=4.0, end_s=candidate_s, confidence=0.9)
    if request.pass_name == "fine":
        return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
    if request.pass_name == "end_validate":
        ts = next(f.timestamp_s for f in request.frames if f.is_candidate)
        return canned_json_response(start_s=ts, end_s=ts, confidence=0.9)
    assert request.pass_name == "end_coarse"
    return canned_json_response(start_s=candidate_s, end_s=candidate_s, confidence=0.9)


def main() -> None:
    tmp_dir = Path(tempfile.mkdtemp(prefix="llm-e2e-"))
    video_path = tmp_dir / "sample.mp4"
    _make_video(video_path)

    config = replace(AppConfig(), data_dir=tmp_dir / "data")
    app = create_app(config)

    # Applies to every LLMEngineStore instance for the rest of this
    # process's life, including the one create_app() built for itself -
    # so every engine configured through the UI runs against the stub,
    # regardless of which provider/model the browser test picks.
    LLMEngineStore.build_provider = lambda self, engine: StubTimingProvider(_stub_respond)

    server = make_server("127.0.0.1", 0, app)
    port = server.server_port
    print(f"LISTENING {port} {video_path}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
