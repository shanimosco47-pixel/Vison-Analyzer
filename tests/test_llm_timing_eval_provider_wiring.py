"""Offline tests for scripts/llm_timing_eval.py's ``--provider gemini`` and
``--provider openai`` wiring.

These verify routing only: that selecting a real provider name builds the
right ``TimingProvider`` with the right model ID, via whichever client
factory is injected. None of this needs the real ``google-genai``/``openai``
SDK installed or ``GEMINI_API_KEY``/``OPENAI_API_KEY`` set -
``_build_provider`` takes each client factory as a parameter specifically so
this is testable without either (see its own docstring in the script).
Real-provider behaviour (retries, parsing, grounding, ...) is already
covered by ``test_llm_timing_gemini_provider.py``,
``test_llm_timing_openai_provider.py``, and ``test_llm_timing_pipeline.py``
- this file only proves the CLI/harness plumbing routes to each correctly.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pytest

from app.analysis.llm_timing.gemini_provider import (
    DEFAULT_GEMINI_MODEL_ID,
    GeminiCallResult,
    GeminiTimingProvider,
)
from app.analysis.llm_timing.openai_provider import (
    DEFAULT_OPENAI_MODEL_ID,
    OpenAICallResult,
    OpenAITimingProvider,
)
from app.analysis.llm_timing.prompts import (
    PROMPT_END_CASCADE_COARSE_V1,
    PROMPT_END_CASCADE_REFINE_V1,
    PROMPT_END_COARSE_V2,
    PROMPT_START_REFINE_V1,
    PROMPT_V1,
)
from app.analysis.llm_timing.provider import (
    PermanentProviderError,
    ProviderRequest,
    TimedFrame,
)

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "llm_timing_eval.py"
_spec = importlib.util.spec_from_file_location("llm_timing_eval", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
llm_timing_eval = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = llm_timing_eval
_spec.loader.exec_module(llm_timing_eval)

_CONFIRMED_JSON = json.dumps(
    {
        "status": "confirmed",
        "start_s": 1.0,
        "end_s": 2.0,
        "start_uncertainty_s": 0.1,
        "end_uncertainty_s": 0.1,
        "confidence": 0.9,
        "reason_codes": [],
        "evidence_frame_timestamps_s": [1.0, 2.0],
    }
)

# ``response_text`` on both fake clients below is either a fixed string (the
# common case: most of these tests only care about model-ID routing, one
# call, any well-formed CONFIRMED payload will do) or a callable taking the
# actual ``parts`` sent for that call and returning the JSON text - needed
# for the two full-harness tests, where the coarse/fine/end-coarse/
# end-validate passes genuinely need different answers to reach a real
# CONFIRMED result under the two-stage end determination (see
# ``_truth_aware_response``).
_ResponseText = str | Callable[[list[dict]], str]

_FRAME_LABEL_RE = re.compile(r"\[frame at t=([\d.]+)s\]")
_CONTACT_SHEET_LABEL_RE = re.compile(r"\[CONTACT SHEET[^\]]*? at t=([\d.]+)s\]")


def _truth_aware_response(true_start_s: float, true_end_s: float) -> Callable[[list[dict]], str]:
    """Builds a ``response_text`` callable good enough to reach a real,
    grounded CONFIRMED result through the actual pipeline: it answers
    accurately for whichever pass (coarse/fine/end-coarse/end-validate) the
    request's own prompt text identifies, and - for the end-coarse call -
    only confirms if ``true_end_s`` actually falls within the submitted
    frames, abstaining (``no_break_found``) otherwise, the same shape a
    real model's answer takes."""

    def respond(parts: list[dict]) -> str:
        prompt_text = parts[0]["text"]
        # Frame labels only - parts[1:], never parts[0] (the prompt text
        # itself), which quotes the literal placeholder "[frame at t=...s]"
        # as part of its own instructions and would otherwise get matched
        # as a fake, non-numeric "frame timestamp".
        frame_timestamps_s = [
            float(match)
            for match in _FRAME_LABEL_RE.findall(" ".join(p.get("text", "") for p in parts[1:]))
        ]
        if prompt_text == PROMPT_V1:
            start_s, end_s, evidence = true_start_s, true_end_s, (true_start_s, true_end_s)
        elif prompt_text == PROMPT_START_REFINE_V1:
            start_s, end_s, evidence = true_start_s, true_start_s, (true_start_s,)
        elif prompt_text == PROMPT_END_COARSE_V2:
            if frame_timestamps_s and min(frame_timestamps_s) <= true_end_s <= max(
                frame_timestamps_s
            ):
                start_s, end_s, evidence = true_end_s, true_end_s, (true_end_s,)
            else:
                return json.dumps({"status": "abstain", "reason_codes": ["no_break_found"]})
        elif prompt_text in (PROMPT_END_CASCADE_COARSE_V1, PROMPT_END_CASCADE_REFINE_V1):
            # Perfect contact-sheet cascade stub: always confirms the
            # sheet's own centred candidate (these synthetic truths have a
            # single genuine, sustained break with no recovery to reject).
            # The composite image's individual panel timestamps are baked
            # into the image pixels, invisible to this text-only stub - only
            # the one anchor timestamp survives as a text caption - so later
            # checkpoints are computed from the cascade's own known, fixed
            # panel spacing (0.5s coarse / 0.25s refine) rather than read
            # back from the (here, empty) frame-label text.
            candidate_text = " ".join(p.get("text", "") for p in parts[1:])
            anchor_matches = _CONTACT_SHEET_LABEL_RE.findall(candidate_text)
            assert anchor_matches, "end-validate request missing a CONTACT SHEET frame label"
            candidate_ts = float(anchor_matches[0])
            step_s = 0.5 if prompt_text == PROMPT_END_CASCADE_COARSE_V1 else 0.25
            trend_checkpoints = (
                round(candidate_ts + step_s, 6),
                round(candidate_ts + 2 * step_s, 6),
            )
            start_s, end_s = candidate_ts, candidate_ts
            evidence = (candidate_ts, *trend_checkpoints)
        else:  # pragma: no cover - would mean a new pass was added and this helper wasn't updated
            raise AssertionError(f"unrecognized prompt text: {prompt_text[:80]!r}")
        is_cascade_pass = prompt_text in (
            PROMPT_END_CASCADE_COARSE_V1,
            PROMPT_END_CASCADE_REFINE_V1,
        )
        trend_checkpoints_out = trend_checkpoints if is_cascade_pass else ()
        return json.dumps(
            {
                "status": "confirmed",
                "start_s": start_s,
                "end_s": end_s,
                "start_uncertainty_s": 0.1,
                "end_uncertainty_s": 0.1,
                "confidence": 0.9,
                "reason_codes": [],
                "evidence_frame_timestamps_s": list(evidence),
                "trend_checkpoint_timestamps_s": list(trend_checkpoints_out),
            }
        )

    return respond


class _RecordingFakeClient:
    """Stands in for the real google-genai-backed client - records which
    model each call requested, no network or SDK involved."""

    def __init__(self, response_text: _ResponseText) -> None:
        self._response_text = response_text
        self.calls: list[str] = []

    def generate_content(self, *, model, parts, generation_config):
        self.calls.append(model)
        text = self._response_text(parts) if callable(self._response_text) else self._response_text
        return GeminiCallResult(text=text)


def _dummy_request() -> ProviderRequest:
    return ProviderRequest(
        prompt_version="test-v1",
        prompt_text="analyze this",
        frames=(TimedFrame(timestamp_s=0.0, image_bytes=b"\xff\xd8\xff\xe0fake"),),
        pass_name="coarse",
    )


def test_build_provider_gemini_routes_to_geminitimingprovider_with_default_model_id():
    client = _RecordingFakeClient(_CONFIRMED_JSON)
    provider = llm_timing_eval._build_provider(
        "gemini", {}, model_id=None, gemini_client_factory=lambda: client
    )
    assert isinstance(provider, GeminiTimingProvider)
    response = provider.analyze(_dummy_request())
    assert response.model_id == DEFAULT_GEMINI_MODEL_ID
    assert client.calls == [DEFAULT_GEMINI_MODEL_ID]


def test_build_provider_gemini_honors_an_explicit_model_id():
    client = _RecordingFakeClient(_CONFIRMED_JSON)
    provider = llm_timing_eval._build_provider(
        "gemini", {}, model_id="gemini-2.5-pro", gemini_client_factory=lambda: client
    )
    response = provider.analyze(_dummy_request())
    assert response.model_id == "gemini-2.5-pro"
    assert client.calls == ["gemini-2.5-pro"]


def test_build_provider_gemini_calls_the_injected_factory_lazily_not_the_real_one():
    # No GEMINI_API_KEY is set in this test environment and google-genai may
    # not even be installed - if this routed to the real
    # build_default_gemini_client it would raise before this test could
    # observe anything. It must not.
    calls = {"n": 0}

    def fake_factory():
        calls["n"] += 1
        return _RecordingFakeClient(_CONFIRMED_JSON)

    llm_timing_eval._build_provider("gemini", {}, model_id=None, gemini_client_factory=fake_factory)
    assert calls["n"] == 1


def test_build_provider_stub_perfect_is_unaffected():
    entry = {"true_start_s": 4.0, "true_end_s": 21.5}
    provider = llm_timing_eval._build_provider("stub-perfect", entry)
    response = provider.analyze(_dummy_request())
    payload = json.loads(response.raw_text)
    assert payload["start_s"] == 4.0
    assert payload["end_s"] == 21.5


def test_build_provider_rejects_unknown_provider_name():
    with pytest.raises(ValueError, match="gemini"):
        llm_timing_eval._build_provider("not-a-real-provider", {})


def test_evaluate_clip_routes_through_the_full_harness_offline_with_gemini(zahn_video):
    """End-to-end proof the wiring works through the actual harness path
    (_evaluate_clip -> run_llm_timing -> GeminiTimingProvider), entirely
    offline against the existing synthetic zahn_video fixture."""
    entry = {
        "clip_id": "offline-routing-check",
        "video_path": str(zahn_video.path),
        "outlet_x": zahn_video.truth["outlet_x"],
        "outlet_y": zahn_video.truth["outlet_y"],
        "true_start_s": zahn_video.truth["flow_start_s"],
        "true_end_s": zahn_video.truth["flow_end_s"],
    }
    client = _RecordingFakeClient(_truth_aware_response(entry["true_start_s"], entry["true_end_s"]))
    result = llm_timing_eval._evaluate_clip(
        entry,
        "gemini",
        llm_timing_eval.PipelineConfig(),
        model_id=None,
        gemini_client_factory=lambda: client,
    )
    assert result.coarse_model_id == DEFAULT_GEMINI_MODEL_ID
    assert result.fine_model_id == DEFAULT_GEMINI_MODEL_ID
    # coarse, fine (start), one whole-clip end-coarse call, one
    # trend-validation call - exactly four calls, not N scan windows.
    assert client.calls == [DEFAULT_GEMINI_MODEL_ID] * 4
    assert result.status == "confirmed"
    assert result.predicted_start_s == pytest.approx(entry["true_start_s"])
    assert result.predicted_end_s == pytest.approx(entry["true_end_s"])


# --------------------------------------------------------------------------- #
# --provider openai routing - mirrors the gemini routing tests above.
# --------------------------------------------------------------------------- #


class _RecordingFakeOpenAIClient:
    """Stands in for the real openai-SDK-backed client - records which model
    each call requested, no network or SDK involved."""

    def __init__(self, response_text: _ResponseText) -> None:
        self._response_text = response_text
        self.calls: list[str] = []

    def generate_content(self, *, model, parts, generation_config):
        self.calls.append(model)
        text = self._response_text(parts) if callable(self._response_text) else self._response_text
        return OpenAICallResult(text=text)


def test_build_provider_openai_routes_to_openaitimingprovider_with_default_model_id():
    client = _RecordingFakeOpenAIClient(_CONFIRMED_JSON)
    provider = llm_timing_eval._build_provider(
        "openai", {}, model_id=None, openai_client_factory=lambda: client
    )
    assert isinstance(provider, OpenAITimingProvider)
    response = provider.analyze(_dummy_request())
    assert response.model_id == DEFAULT_OPENAI_MODEL_ID
    assert client.calls == [DEFAULT_OPENAI_MODEL_ID]


def test_build_provider_openai_honors_an_explicit_model_id():
    client = _RecordingFakeOpenAIClient(_CONFIRMED_JSON)
    provider = llm_timing_eval._build_provider(
        "openai", {}, model_id="gpt-5", openai_client_factory=lambda: client
    )
    response = provider.analyze(_dummy_request())
    assert response.model_id == "gpt-5"
    assert client.calls == ["gpt-5"]


def test_build_provider_openai_calls_the_injected_factory_lazily_not_the_real_one():
    # No OPENAI_API_KEY is set in this test environment and openai may not
    # even be installed - if this routed to the real
    # build_default_openai_client it would raise before this test could
    # observe anything. It must not.
    calls = {"n": 0}

    def fake_factory():
        calls["n"] += 1
        return _RecordingFakeOpenAIClient(_CONFIRMED_JSON)

    llm_timing_eval._build_provider("openai", {}, model_id=None, openai_client_factory=fake_factory)
    assert calls["n"] == 1


def test_evaluate_clip_routes_through_the_full_harness_offline_with_openai(zahn_video):
    """End-to-end proof the wiring works through the actual harness path
    (_evaluate_clip -> run_llm_timing -> OpenAITimingProvider), entirely
    offline against the existing synthetic zahn_video fixture."""
    entry = {
        "clip_id": "offline-routing-check-openai",
        "video_path": str(zahn_video.path),
        "outlet_x": zahn_video.truth["outlet_x"],
        "outlet_y": zahn_video.truth["outlet_y"],
        "true_start_s": zahn_video.truth["flow_start_s"],
        "true_end_s": zahn_video.truth["flow_end_s"],
    }
    client = _RecordingFakeOpenAIClient(
        _truth_aware_response(entry["true_start_s"], entry["true_end_s"])
    )
    result = llm_timing_eval._evaluate_clip(
        entry,
        "openai",
        llm_timing_eval.PipelineConfig(),
        model_id=None,
        openai_client_factory=lambda: client,
    )
    assert result.coarse_model_id == DEFAULT_OPENAI_MODEL_ID
    assert result.fine_model_id == DEFAULT_OPENAI_MODEL_ID
    # coarse, fine (start), one whole-clip end-coarse call, one
    # trend-validation call - exactly four calls, not N scan windows.
    assert client.calls == [DEFAULT_OPENAI_MODEL_ID] * 4
    assert result.status == "confirmed"
    assert result.predicted_start_s == pytest.approx(entry["true_start_s"])
    assert result.predicted_end_s == pytest.approx(entry["true_end_s"])


# --------------------------------------------------------------------------- #
# ClipResult.raw_notes: the observability gap from the first real gate-1 run
# (a coarse-call provider_error abstained with no way to tell auth/quota/
# request-shape/model-availability failures apart) must be visible in the
# per-clip output, and whatever secret-shaped text a provider's own
# exception message might echo back must stay redacted getting there.
# --------------------------------------------------------------------------- #


class _RaisingFakeClient:
    """Simulates the real failure mode observed in the first gate-1 run: the
    provider call itself raises before any structured response comes back -
    e.g. a rejected/invalid API key, whose exception text a real SDK might
    echo straight back including the key itself."""

    def __init__(self, message: str) -> None:
        self._message = message

    def generate_content(self, *, model, parts, generation_config):
        raise PermanentProviderError(self._message)


def test_evaluate_clip_surfaces_raw_notes_on_a_provider_error(zahn_video):
    entry = {
        "clip_id": "provider-error-check",
        "video_path": str(zahn_video.path),
        "outlet_x": zahn_video.truth["outlet_x"],
        "outlet_y": zahn_video.truth["outlet_y"],
        "true_start_s": zahn_video.truth["flow_start_s"],
        "true_end_s": zahn_video.truth["flow_end_s"],
    }
    client = _RaisingFakeClient("401 invalid API key")
    result = llm_timing_eval._evaluate_clip(
        entry,
        "gemini",
        llm_timing_eval.PipelineConfig(),
        model_id=None,
        gemini_client_factory=lambda: client,
    )
    assert result.status == "abstain"
    assert "provider_error" in result.reason_codes
    # This is the whole point of the fix: previously ClipResult dropped this
    # entirely, leaving no way to distinguish auth/quota/schema/availability
    # failures from the reason code alone.
    assert "invalid API key" in result.raw_notes


def test_evaluate_clip_raw_notes_redacts_secret_shaped_text_from_a_provider_error(zahn_video):
    entry = {
        "clip_id": "secret-leak-check",
        "video_path": str(zahn_video.path),
        "outlet_x": zahn_video.truth["outlet_x"],
        "outlet_y": zahn_video.truth["outlet_y"],
        "true_start_s": zahn_video.truth["flow_start_s"],
        "true_end_s": zahn_video.truth["flow_end_s"],
    }
    fake_key = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"  # 32+ chars: secret-shaped
    client = _RaisingFakeClient(f"401 invalid API key: {fake_key}")
    result = llm_timing_eval._evaluate_clip(
        entry,
        "gemini",
        llm_timing_eval.PipelineConfig(),
        model_id=None,
        gemini_client_factory=lambda: client,
    )
    assert result.status == "abstain"
    assert result.raw_notes  # present, not dropped
    assert fake_key not in result.raw_notes
    assert "[redacted]" in result.raw_notes


def test_evaluate_clip_raw_notes_redacts_a_masked_credential_from_a_provider_error(zahn_video):
    """Reproduces the exact gap from the first live OpenAI gate-1 run: a 401
    invalid_api_key error whose text echoes a *masked* key
    ("prefix********suffix") rather than one long contiguous token. Neither
    visible fragment may reach ClipResult.raw_notes - which is exactly what
    both the JSON output (asdict(result)) and the console "Notes:" section
    in scripts/llm_timing_eval.py's main() print, so asserting on this one
    field covers both surfaces."""
    entry = {
        "clip_id": "masked-credential-leak-check",
        "video_path": str(zahn_video.path),
        "outlet_x": zahn_video.truth["outlet_x"],
        "outlet_y": zahn_video.truth["outlet_y"],
        "true_start_s": zahn_video.truth["flow_start_s"],
        "true_end_s": zahn_video.truth["flow_end_s"],
    }
    visible_prefix = "sk-proj-AbCd1234"
    visible_suffix = "WxYz9876"
    message = (
        f"Error code: 401 - invalid_api_key: Incorrect API key provided: "
        f"{visible_prefix}********{visible_suffix}."
    )
    client = _RaisingFakeClient(message)
    result = llm_timing_eval._evaluate_clip(
        entry,
        "openai",
        llm_timing_eval.PipelineConfig(),
        model_id=None,
        openai_client_factory=lambda: client,
    )
    assert result.status == "abstain"
    assert result.raw_notes
    assert visible_prefix not in result.raw_notes
    assert visible_suffix not in result.raw_notes
    assert "[redacted]" in result.raw_notes
    # Same field feeds --json output directly (asdict(result)) - proves the
    # JSON surface is safe too, not just the in-memory field.
    serialized = json.dumps(asdict(result))
    assert visible_prefix not in serialized
    assert visible_suffix not in serialized
