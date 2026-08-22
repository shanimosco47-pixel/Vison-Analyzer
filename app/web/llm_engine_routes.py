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

from ..errors import AnalyzerError
from ..services.llm_engine_store import LLMEngineStore
from ..services.llm_run_service import LLMRunService

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

    record = _video_store().get(video_id)
    engine = _engine_store().get(str(engine_id))
    job = _llm_run_service().submit(record, engine)
    return jsonify(job.to_dict())


@llm_engines.get("/llm-runs/<run_id>")
def llm_run_status(run_id: str) -> Response:
    return jsonify(_llm_run_service().get(run_id).to_dict())


@llm_engines.post("/llm-runs/<run_id>/cancel")
def cancel_llm_run(run_id: str) -> Response:
    return jsonify(_llm_run_service().cancel(run_id).to_dict())


def _optional_str(value: Any) -> str | None:
    """``None`` stays ``None`` (field omitted / not being changed); an
    empty string is passed through as-is so a caller can explicitly clear
    an optional field rather than it being silently ignored."""
    if value is None:
        return None
    return str(value)
