"""Tests for deterministic frame resizing and the request-size/frame-count
budget (Codex re-review, finding 2): no request should be allowed to exceed
an explicit byte/frame ceiling, sampling should thin gracefully down to a
documented precision floor before that, and an unsatisfiable budget must
abstain rather than silently send too few frames or an oversized request.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.analysis.llm_timing.pipeline import (
    PipelineConfig,
    _effective_tolerance_s,
    _estimated_request_bytes,
    _fit_frames_to_budget,
    _min_frames_for_window,
    _resize_for_encoding,
    run_llm_timing,
)
from app.analysis.llm_timing.provider import (
    ProviderRequest,
    RawProviderResponse,
    StubTimingProvider,
    TimedFrame,
    canned_json_response,
)
from app.analysis.llm_timing.schema import TimingStatus

PROMPT_VERSION = "test-prompt-v1"


def _frame(ts: float, size_bytes: int) -> TimedFrame:
    return TimedFrame(timestamp_s=ts, image_bytes=b"x" * size_bytes, media_type="image/jpeg")


# --------------------------------------------------------------------------- #
# _resize_for_encoding: deterministic, never upscales
# --------------------------------------------------------------------------- #


def test_resize_leaves_a_small_image_untouched():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    resized = _resize_for_encoding(image, max_dimension_px=768)
    assert resized.shape == image.shape


def test_resize_downscales_the_longer_side_to_the_cap():
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    resized = _resize_for_encoding(image, max_dimension_px=768)
    assert max(resized.shape[0], resized.shape[1]) == 768
    # aspect ratio preserved (within rounding)
    assert abs(resized.shape[1] / resized.shape[0] - 1920 / 1080) < 0.01


def test_resize_never_upscales_a_small_image():
    image = np.zeros((50, 50, 3), dtype=np.uint8)
    resized = _resize_for_encoding(image, max_dimension_px=768)
    assert resized.shape == image.shape


# --------------------------------------------------------------------------- #
# _estimated_request_bytes / _min_frames_for_window / _fit_frames_to_budget
# --------------------------------------------------------------------------- #


def test_estimated_request_bytes_accounts_for_base64_inflation_and_prompt():
    frames = [_frame(0.0, 3000), _frame(1.0, 3000)]
    estimate = _estimated_request_bytes(frames, "hello")
    # 6000 raw bytes * 4/3 + len("hello")
    assert estimate == int(6000 * 4 / 3) + 5


def test_min_frames_for_window_scales_with_span_and_tolerance():
    # A wider window needs more frames at the same tolerance.
    narrow = _min_frames_for_window(1.0, target_tolerance_s=0.75)
    wide = _min_frames_for_window(10.0, target_tolerance_s=0.75)
    assert wide > narrow
    # A tighter tolerance needs more frames for the same window.
    loose = _min_frames_for_window(3.0, target_tolerance_s=2.0)
    tight = _min_frames_for_window(3.0, target_tolerance_s=0.2)
    assert tight > loose


def test_fit_frames_to_budget_returns_unchanged_when_already_within_budget():
    frames = [_frame(float(i), 100) for i in range(10)]
    fitted = _fit_frames_to_budget(
        frames, prompt_text="p", max_request_bytes=10_000_000, min_frames=2
    )
    assert fitted == frames


def test_fit_frames_to_budget_thins_deterministically_to_fit():
    frames = [_frame(float(i), 100_000) for i in range(100)]  # 10MB raw -> ~13.3MB estimated
    fitted = _fit_frames_to_budget(
        frames, prompt_text="p", max_request_bytes=2_000_000, min_frames=5
    )
    assert fitted is not None
    assert len(fitted) < len(frames)
    assert len(fitted) >= 5
    assert _estimated_request_bytes(fitted, "p") <= 2_000_000
    # Deterministic: running it again on the same input gives the same result.
    fitted_again = _fit_frames_to_budget(
        frames, prompt_text="p", max_request_bytes=2_000_000, min_frames=5
    )
    assert [f.timestamp_s for f in fitted] == [f.timestamp_s for f in fitted_again]


def test_fit_frames_to_budget_returns_none_when_floor_still_does_not_fit():
    frames = [_frame(float(i), 10_000_000) for i in range(5)]  # each frame alone is huge
    fitted = _fit_frames_to_budget(
        frames, prompt_text="p", max_request_bytes=1_000_000, min_frames=5
    )
    assert fitted is None


def test_fit_frames_to_budget_never_thins_below_min_frames():
    frames = [_frame(float(i), 100_000) for i in range(50)]
    fitted = _fit_frames_to_budget(frames, prompt_text="p", max_request_bytes=1, min_frames=8)
    # Budget is impossible to hit at all, so this must abstain (None) rather
    # than return fewer than min_frames.
    assert fitted is None


# --------------------------------------------------------------------------- #
# _effective_tolerance_s: widens to match whatever density was achieved
# --------------------------------------------------------------------------- #


def test_effective_tolerance_uses_fallback_when_too_few_timestamps():
    assert _effective_tolerance_s([1.0], fallback_step_s=0.04) == pytest.approx(0.08)
    assert _effective_tolerance_s([], fallback_step_s=0.04) == pytest.approx(0.08)


def test_effective_tolerance_widens_to_the_largest_actual_gap():
    # Largest gap (3.0) dominates the 2x-fallback-step floor (0.08).
    timestamps = [0.0, 0.5, 3.5, 4.0]
    assert _effective_tolerance_s(timestamps, fallback_step_s=0.04) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# Integration: the pipeline actually thins under a tight budget, and
# actually abstains when even the floor cannot fit
# --------------------------------------------------------------------------- #


def _stub_matching_truth(truth: dict[str, float]):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=truth["flow_start_s"], end_s=truth["flow_end_s"], confidence=0.8
            )
        # Evidence near both boundaries so grounding still passes despite thinning.
        return canned_json_response(
            start_s=truth["flow_start_s"],
            end_s=truth["flow_end_s"],
            confidence=0.9,
            evidence_frame_timestamps_s=(truth["flow_start_s"], truth["flow_end_s"]),
        )

    return StubTimingProvider(respond)


def test_pipeline_thins_fine_frames_under_a_tight_byte_budget(zahn_video):
    provider = _stub_matching_truth(zahn_video.truth)
    # Real synthetic-video JPEGs here run several KB each. This budget
    # comfortably fits the coarse pass's 4-frame floor (~25KB) and the fine
    # pass's precision floor (~18 frames, ~115KB), but not native-fps dense
    # sampling of two ~3s windows (~150 frames, roughly 1MB) - forcing
    # thinning without being literally unsatisfiable.
    config = PipelineConfig(max_request_bytes=200_000)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        config=config,
    )
    assert len(provider.calls) == 2
    fine_call = provider.calls[-1]
    assert fine_call.pass_name == "fine"
    # Native-fps density over ~3s windows at 25fps would be dozens of
    # frames; the tight budget must have thinned it down substantially.
    assert len(fine_call.frames) < 30
    assert outcome.event is not None  # still confirms - just with fewer frames


def test_pipeline_abstains_with_request_too_large_when_fine_floor_cannot_fit(zahn_video):
    provider = _stub_matching_truth(zahn_video.truth)
    # Enough for the coarse pass's small floor, not enough for the fine
    # pass's larger (per-boundary-precision) floor.
    config = PipelineConfig(max_request_bytes=50_000)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        config=config,
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "request_too_large" in outcome.verdict.reason_codes
    assert len(provider.calls) == 1  # coarse succeeded; fine never got sent
    assert provider.calls[0].pass_name == "coarse"


def test_pipeline_abstains_with_request_too_large_when_even_coarse_floor_cannot_fit(zahn_video):
    provider = _stub_matching_truth(zahn_video.truth)
    config = PipelineConfig(max_request_bytes=1)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        config=config,
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "request_too_large" in outcome.verdict.reason_codes
    assert len(provider.calls) == 0  # never even reached the coarse call
