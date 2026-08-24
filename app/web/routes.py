"""HTTP layer.

A deliberately small JSON API plus one page.  Rules:

*   the browser never receives a Python traceback - errors are translated to a
    short sentence, while the detail goes to the server log;
*   long work never happens in a request: uploads are streamed, analysis is
    queued and polled;
*   nothing leaves the machine.  There are no external scripts, fonts, CDNs or
    analytics anywhere in the page.
"""

from __future__ import annotations

import mimetypes
from typing import Any

from flask import (
    Blueprint,
    Flask,
    Response,
    current_app,
    jsonify,
    render_template,
    request,
    send_file,
)
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from ..analysis.registry import available_modes
from ..config import ROI, AppConfig
from ..errors import AnalyzerError, InvalidROIError, UploadTooLargeError
from ..logging_setup import configure_logging, get_logger
from ..services.analysis_service import AnalysisService
from ..services.diagnostics import draw_overlay
from ..services.event_log import parse_recording_start
from ..services.llm_engine_store import LLMEngineStore
from ..services.llm_run_service import LLMRunService
from ..services.secret_store import SecretStore
from ..services.storage import VideoStore
from ..version import app_version
from ..video.reader import VideoReader, encode_jpeg
from .llm_engine_routes import llm_engines

logger = get_logger(__name__)

api = Blueprint("api", __name__)
pages = Blueprint("pages", __name__)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _store() -> VideoStore:
    return current_app.extensions["video_store"]


def _service() -> AnalysisService:
    return current_app.extensions["analysis_service"]


def _json_error(message: str, status: int) -> Response:
    response = jsonify({"error": message})
    response.status_code = status
    return response


def _float_arg(name: str, default: float | None = None) -> float | None:
    raw = request.args.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise AnalyzerError(f"The value of '{name}' must be a number.", detail=raw) from exc


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


@pages.get("/")
def index() -> str:
    config: AppConfig = current_app.extensions["app_config"]
    return render_template(
        "index.html",
        modes=available_modes(),
        max_upload_mb=config.max_upload_mb,
        allowed_extensions=", ".join(config.allowed_extensions),
        app_version=app_version(),
    )


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


@api.get("/health")
def health() -> Response:
    return jsonify({"status": "ok"})


@api.get("/version")
def version() -> Response:
    """The running backend's own build identifier - see
    ``app.version.app_version``. Never derived from anything the request
    supplies (no query string, no header); a client-facing peer to the
    same value already rendered into the page and stored in every LLM run
    audit, so the three can be matched against each other."""
    return jsonify({"app_version": app_version()})


@api.get("/modes")
def modes() -> Response:
    return jsonify({"modes": available_modes()})


@api.post("/videos")
def upload_video() -> Response:
    if "file" not in request.files:
        raise AnalyzerError("No video file was included in the upload.")
    uploaded = request.files["file"]
    if not uploaded.filename:
        raise AnalyzerError("No video file was selected.")

    # The browser-declared MIME type is intentionally ignored: the extension is
    # checked against the allowlist and the content is validated by decoding.
    record = _store().save_upload(uploaded.stream, uploaded.filename)
    return jsonify(record.to_dict())


@api.get("/videos/<video_id>")
def video_metadata(video_id: str) -> Response:
    return jsonify(_store().get(video_id).to_dict())


@api.delete("/videos/<video_id>")
def delete_video(video_id: str) -> Response:
    _store().delete(video_id)
    return jsonify({"deleted": True})


@api.get("/videos/<video_id>/media")
def video_media(video_id: str) -> Response:
    """Serve the original file for the preview player, with range support."""
    record = _store().get(video_id)
    mimetype = mimetypes.guess_type(record.path.name)[0] or "application/octet-stream"
    # conditional=True makes Flask honour Range requests, which is what lets
    # the user scrub the preview without downloading the whole recording.
    return send_file(record.path, mimetype=mimetype, conditional=True)


@api.get("/videos/<video_id>/frame")
def video_frame(video_id: str) -> Response:
    """A single JPEG frame, optionally with the analysis region drawn on it."""
    record = _store().get(video_id)
    timestamp = _float_arg("t", 0.0) or 0.0

    roi = None
    if request.args.get("roi_x") is not None:
        try:
            roi = ROI(
                x=int(float(request.args["roi_x"])),
                y=int(float(request.args["roi_y"])),
                width=int(float(request.args["roi_w"])),
                height=int(float(request.args["roi_h"])),
            ).clipped_to(record.info.width, record.info.height)
        except (KeyError, ValueError) as exc:
            raise InvalidROIError(detail=str(exc)) from exc

    with VideoReader(record.path, record.info) as reader:
        frame = reader.frame_at(timestamp)
    if roi is not None:
        frame = draw_overlay(frame, roi=roi, label=f"Analysis region @ {timestamp:.2f}s")

    response = Response(encode_jpeg(frame), mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store"
    return response


@api.post("/analyses")
def start_analysis() -> Response:
    payload: dict[str, Any] = request.get_json(silent=True) or {}
    video_id = payload.get("video_id")
    mode = payload.get("mode")
    if not video_id or not mode:
        raise AnalyzerError("Choose a video and an analysis mode before starting.")

    record = _store().get(str(video_id))
    params = payload.get("params") or {}
    if not isinstance(params, dict):
        raise AnalyzerError("The analysis settings are not in the expected format.")

    recording_start = parse_recording_start(payload.get("recording_start"))
    if payload.get("recording_start") and recording_start is None:
        raise AnalyzerError(
            "The recording start time could not be understood. Use the date and "
            "time picker, or leave it empty to report video-relative times only."
        )

    job = _service().submit(record, str(mode), params, recording_start=recording_start)
    return jsonify(job.to_dict(include_result=False))


@api.get("/analyses/<job_id>")
def analysis_status(job_id: str) -> Response:
    job = _service().get(job_id)
    return jsonify(job.to_dict())


@api.post("/analyses/<job_id>/cancel")
def cancel_analysis(job_id: str) -> Response:
    job = _service().cancel(job_id)
    return jsonify(job.to_dict(include_result=False))


@api.get("/analyses/<job_id>/events.csv")
def download_events(job_id: str) -> Response:
    job = _service().get(job_id)
    if job.event_log is None:
        raise AnalyzerError("This analysis has no results to export yet.")
    csv_text = job.event_log.to_csv()
    response = Response(csv_text, mimetype="text/csv; charset=utf-8")
    response.headers["Content-Disposition"] = (
        f'attachment; filename="{job.event_log.suggested_filename()}"'
    )
    return response


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #


def create_app(config: AppConfig | None = None) -> Flask:
    """Build the Flask application."""
    config = config or AppConfig.from_env()
    config.validate()
    configure_logging(config.log_level, log_dir=config.data_dir)

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_upload_bytes
    app.config["JSON_SORT_KEYS"] = False

    store = VideoStore(config)
    store.purge_orphans()
    service = AnalysisService(config, store)

    secret_store = SecretStore()
    engine_store = LLMEngineStore(config, secret_store)
    llm_run_service = LLMRunService(engine_store, config)

    app.extensions["app_config"] = config
    app.extensions["video_store"] = store
    app.extensions["analysis_service"] = service
    app.extensions["secret_store"] = secret_store
    app.extensions["llm_engine_store"] = engine_store
    app.extensions["llm_run_service"] = llm_run_service

    app.register_blueprint(pages)
    app.register_blueprint(api, url_prefix="/api")
    app.register_blueprint(llm_engines, url_prefix="/api")

    @app.errorhandler(AnalyzerError)
    def handle_analyzer_error(error: AnalyzerError) -> Response:
        logger.warning("Request failed: %s (%s)", error.user_message, error.detail)
        return _json_error(error.user_message, error.http_status)

    @app.errorhandler(RequestEntityTooLarge)
    def handle_too_large(error: RequestEntityTooLarge) -> Response:
        message = UploadTooLargeError(
            f"The video is larger than the {config.max_upload_mb} MB limit."
        ).user_message
        logger.warning("Rejected an oversized upload: %s", error)
        return _json_error(message, 413)

    @app.errorhandler(HTTPException)
    def handle_http_error(error: HTTPException) -> Response:
        if request.path.startswith("/api/"):
            return _json_error(error.description or "Request failed.", error.code or 500)
        return error  # type: ignore[return-value]

    @app.errorhandler(Exception)
    def handle_unexpected(error: Exception) -> Response:
        logger.exception("Unhandled error while serving %s", request.path)
        return _json_error(
            "Something went wrong on the server. The details are in the server log.", 500
        )

    @app.after_request
    def security_headers(response: Response) -> Response:
        # Local-only application: lock the page down to its own resources so a
        # crafted filename or ROI value can never pull in remote content.
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
            "script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response

    logger.info(
        "Application ready (data dir %s, upload limit %d MB, diagnostics %s)",
        config.data_dir,
        config.max_upload_mb,
        "on" if config.save_diagnostics else "off",
    )
    return app
