"""HTTP layer for the Experimental LLM analysis section.

Kept in its own blueprint/module, separate from ``routes.py``'s classical-
detector endpoints, so that file's existing behaviour and tests stay
untouched. Registered onto the same ``/api`` prefix by
``routes.create_app`` and shares that module's app-wide error handling
(``AnalyzerError`` -> a short JSON message, technical detail to the log
only - see ``routes.create_app``'s ``errorhandler`` registrations, which
apply regardless of which blueprint raised the error).

Every response here is part of a feature explicitly marked Experimental in
the UI - see ``diagnostics/llm_spike/DESIGN.md``. Two rules specific to
this module, beyond the app's existing ones:

*   a request body may include a raw API key (``api_key``) - write-only,
    all the way through: it is handed straight to
    ``LLMEngineStore.create``/``update`` and never appears in any response,
    any log line, or the persisted engine-metadata file (see
    ``llm_engine_store.py``);
*   the video store's existing frame/media endpoints are reused as-is for
    the preview; this module never serves the original video file to a
    provider - see ``llm_run_service.py``'s frames-only guarantee.
"""

from __future__ import annotations

from typing import Any

from flask import Blueprint, Response, current_app, jsonify, request

from ..errors import AnalyzerError, NotFoundError
from ..services.llm_engine_store import SUPPORTED_MODELS, LLMEngineStore
from ..services.llm_run_service import LLMRunService
from ..video.reader import VideoReader, encode_jpeg

llm_engines = Blueprint("llm_engines", __name__)


def _engine_store() -> LLMEngineStore:
    return current_app.extensions["llm_engine_store"]


def _llm_run_service() -> LLMRunService:
    return current_app.extensions["llm_run_service"]


def _video_store():
    return current_app.extensions["video_store"]


def _json_body() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise AnalyzerError("The request body must be a JSON object.")
    return payload


# --------------------------------------------------------------------------- #
# Engine configuration
# --------------------------------------------------------------------------- #


@llm_engines.get("/llm-model-options")
def llm_model_options() -> Response:
    """The single server-side source of truth for which model IDs are
    selectable per provider - see ``llm_engine_store.SUPPORTED_MODELS``'s
    own docstring for how this list is curated. The frontend's Model
    dropdown is populated from this, but ``LLMEngineStore.create``/
    ``update`` enforce the same allowlist independently either way - this
    endpoint only saves the browser from hard-coding a duplicate copy."""
    return jsonify({provider: list(models) for provider, models in SUPPORTED_MODELS.items()})


@llm_engines.get("/llm-engines")
def list_engines() -> Response:
    return jsonify({"engines": [e.to_dict() for e in _engine_store().list()]})


@llm_engines.post("/llm-engines")
def create_engine() -> Response:
    payload = _json_body()
    engine = _engine_store().create(
        provider_name=str(payload.get("provider_name", "")),
        model_id=str(payload.get("model_id", "")),
        display_name=str(payload.get("display_name", "")),
        api_key=_optional_str(payload.get("api_key")),
        env_var=_optional_str(payload.get("env_var")),
        enabled=bool(payload.get("enabled", True)),
    )
    return jsonify(engine.to_dict())


@llm_engines.put("/llm-engines/<engine_id>")
def update_engine(engine_id: str) -> Response:
    payload = _json_body()
    engine = _engine_store().update(
        engine_id,
        provider_name=_optional_str(payload.get("provider_name")),
        model_id=_optional_str(payload.get("model_id")),
        display_name=_optional_str(payload.get("display_name")),
        enabled=payload.get("enabled") if "enabled" in payload else None,
        api_key=_optional_str(payload.get("api_key")),
        env_var=_optional_str(payload.get("env_var")),
    )
    return jsonify(engine.to_dict())


@llm_engines.delete("/llm-engines/<engine_id>")
def delete_engine(engine_id: str) -> Response:
    _engine_store().delete(engine_id)
    return jsonify({"deleted": True})


@llm_engines.post("/llm-engines/<engine_id>/test")
def test_engine(engine_id: str) -> Response:
    return jsonify(_engine_store().test(engine_id))


# --------------------------------------------------------------------------- #
# Runs
# --------------------------------------------------------------------------- #


@llm_engines.post("/videos/<video_id>/llm-runs")
def start_llm_run(video_id: str) -> Response:
    payload = _json_body()
    engine_id = payload.get("engine_id")
    if not engine_id:
        raise AnalyzerError("Choose an engine to run.")

    outlet = payload.get("outlet")
    if not isinstance(outlet, dict) or "x" not in outlet or "y" not in outlet:
        raise AnalyzerError(
            "Mark the outlet hole of the cup on the video frame before running an AI engine."
        )
    try:
        outlet_xy = (float(outlet["x"]), float(outlet["y"]))
    except (TypeError, ValueError) as exc:
        raise AnalyzerError("The marked outlet point is not valid.") from exc

    record = _video_store().get(video_id)
    engine = _engine_store().get(str(engine_id))
    job = _llm_run_service().submit(record, engine, outlet_xy=outlet_xy)
    return jsonify(job.to_dict())


@llm_engines.get("/llm-runs/<run_id>")
def llm_run_status(run_id: str) -> Response:
    return jsonify(_llm_run_service().get(run_id).to_dict())


@llm_engines.post("/llm-runs/<run_id>/cancel")
def cancel_llm_run(run_id: str) -> Response:
    return jsonify(_llm_run_service().cancel(run_id).to_dict())


@llm_engines.get("/llm-runs/<run_id>/evidence/<boundary>")
def llm_run_evidence_frame(run_id: str, boundary: str) -> Response:
    """The one decisive still frame for ``boundary`` ("start" or "end"),
    at the timestamp *this run itself* selected and grounded - see
    ``pipeline._nearest_grounded_evidence_ts``. Deliberately takes no
    timestamp from the request: unlike ``/videos/<id>/frame`` (a general
    frame-preview endpoint driven by whatever time the browser is
    scrubbed to), this endpoint always serves the server-stored audit
    timestamp for this specific run, so a client can never relabel an
    arbitrary frame as "the evidence" by supplying its own ``t`` - that
    was an explicit supervisor requirement (see
    diagnostics/llm_spike/DESIGN.md). 404s - never a fabricated image -
    when the run has no grounded evidence for this boundary at all
    (abstained, failed, or the deciding pass cited none).

    Falls back to the durable audit record (``llm_run_audit.py``) when the
    in-memory job is gone (server restart, or past its 6-hour retention) -
    so a past result's evidence images stay viewable exactly as long as
    its audit JSON does, not only while the job happens to still be in
    memory."""
    if boundary not in ("start", "end"):
        raise AnalyzerError("Unknown evidence boundary - expected 'start' or 'end'.")

    timestamp_s, video_id = _evidence_timestamp_and_video(run_id, boundary)
    if timestamp_s is None:
        raise NotFoundError("No grounded evidence image is available for this run.")

    record = _video_store().get(video_id)
    with VideoReader(record.path, record.info) as reader:
        frame = reader.frame_at(timestamp_s)
    response = Response(encode_jpeg(frame), mimetype="image/jpeg")
    response.headers["Cache-Control"] = "no-store"
    return response


def _evidence_timestamp_and_video(run_id: str, boundary: str) -> tuple[float | None, str]:
    key = "start_evidence_s" if boundary == "start" else "end_evidence_s"
    try:
        job = _llm_run_service().get(run_id)
    except NotFoundError:
        audit = _llm_run_service().get_audit(run_id)  # raises NotFoundError itself if missing
        derived = audit.get("derived") or {}
        return derived.get(key), audit["video_id"]
    timestamp_s = job.start_evidence_s if boundary == "start" else job.end_evidence_s
    return timestamp_s, job.video_id


@llm_engines.get("/llm-runs/<run_id>/audit")
def llm_run_audit(run_id: str) -> Response:
    """The full durable audit record for one run - client-safe by
    construction (see ``llm_run_audit.py``'s own module docstring: never a
    credential, a filesystem path, or an image byte). Serves as both the
    data source for the result card's "How the AI decided" section
    (fetched directly) and the "Download audit report (JSON)" button
    (the same URL, given a ``download`` attribute in the page) - the
    ``Content-Disposition`` header below only affects the latter; a plain
    ``fetch()`` ignores it. Readable after a page refresh or a server
    restart, since it comes from disk, not the in-memory job map."""
    audit = _llm_run_service().get_audit(run_id)
    response = jsonify(audit)
    response.headers["Content-Disposition"] = f'attachment; filename="llm-run-{run_id}-audit.json"'
    return response


@llm_engines.get("/llm-engines/<engine_id>/latest-audit")
def llm_engine_latest_audit(engine_id: str) -> Response:
    """The most recently finished run's audit record for ``engine_id`` -
    how the page finds "what this engine last did" after a refresh or
    restart, without already knowing a ``run_id``. 404s if this engine has
    never finished a run (never confused with an engine that ran but
    produced nothing - that case still has an audit record, just one
    whose ``final_verdict`` is ABSTAIN)."""
    return jsonify(_llm_run_service().get_latest_audit_for_engine(engine_id))


def _optional_str(value: Any) -> str | None:
    """``None`` stays ``None`` (field omitted / not being changed); an
    empty string is passed through as-is so a caller can explicitly clear
    an optional field rather than it being silently ignored."""
    if value is None:
        return None
    return str(value)
