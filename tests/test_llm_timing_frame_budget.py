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
    _fit_fine_windows_to_budget,
    _fit_frames_to_budget,
    _GroundingRegion,
    _min_coarse_frames,
    _min_frames_for_window,
    _resize_for_encoding,
    _validate_grounding,
    run_llm_timing,
)
from app.analysis.llm_timing.provider import (
    ProviderRequest,
    RawProviderResponse,
    StubTimingProvider,
    TimedFrame,
    canned_json_response,
    spaced_trend_checkpoints,
)
from app.analysis.llm_timing.schema import TimingStatus, TimingVerdict

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
        if request.pass_name == "fine":
            # Start-only pass: reports the confirmed start as a degenerate
            # point, never the (discarded) end. Evidence near the claim so
            # grounding still passes despite thinning.
            return canned_json_response(
                start_s=truth["flow_start_s"],
                end_s=truth["flow_start_s"],
                confidence=0.9,
                evidence_frame_timestamps_s=(truth["flow_start_s"],),
            )
        if request.pass_name == "end_validate":
            # Perfect trend-validation stub: always confirms whichever
            # candidate the pipeline flagged.
            candidate_ts = next(f.timestamp_s for f in request.frames if f.is_candidate)
            checkpoints = spaced_trend_checkpoints(request.frames, candidate_ts)
            return canned_json_response(
                start_s=candidate_ts,
                end_s=candidate_ts,
                confidence=0.9,
                evidence_frame_timestamps_s=(candidate_ts,) + checkpoints,
                trend_checkpoint_timestamps_s=checkpoints,
            )
        assert request.pass_name == "end_coarse"
        window_times = [f.timestamp_s for f in request.frames]
        end_s = truth["flow_end_s"]
        if window_times and min(window_times) <= end_s <= max(window_times):
            return canned_json_response(
                start_s=end_s, end_s=end_s, confidence=0.9, evidence_frame_timestamps_s=(end_s,)
            )
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    return StubTimingProvider(respond)


def test_pipeline_thins_fine_frames_under_a_tight_byte_budget(zahn_video):
    provider = _stub_matching_truth(zahn_video.truth)
    # Real synthetic-video JPEGs here run several KB each. This budget
    # comfortably fits every pass's own precision floor (coarse ~19
    # frames/~163KB; the single start-only fine window ~9 frames; the
    # end-coarse sparse batch similarly small; the dense end-validate
    # window - now up to end_validation_max_span_s=10.0s wide, ~24 frames
    # at its own floor here - see PipelineConfig.end_validation_pre_s/
    # end_validation_post_s), but not native-fps dense sampling of the
    # ~3s fine window (~75 frames, several hundred KB) - forcing thinning
    # there without being literally unsatisfiable for any pass.
    config = PipelineConfig(max_request_bytes=300_000)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        config=config,
    )
    fine_call = provider.calls[1]
    assert fine_call.pass_name == "fine"
    # Native-fps density over a ~3s window at 25fps would be dozens of
    # frames; the tight budget must have thinned it down substantially.
    assert len(fine_call.frames) < 30
    assert outcome.event is not None  # still confirms - just with fewer frames


def test_pipeline_abstains_with_request_too_large_when_fine_floor_cannot_fit(zahn_video):
    provider = _stub_matching_truth(zahn_video.truth)
    # A tight target_tolerance_s makes each fine window's own floor (~61
    # frames, ~500KB) far larger than the coarse pass's floor (~19 frames
    # spread across the whole clip, ~163KB) - so a budget that comfortably
    # fits the coarse pass (200KB) still isn't enough for either fine
    # window's half of the same budget (100KB each).
    config = PipelineConfig(max_request_bytes=200_000, target_tolerance_s=0.1)
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


# --------------------------------------------------------------------------- #
# Per-window fine-pass coverage (Codex re-review round 3, finding 1): the
# start and end fine windows must be budgeted, thinned, and grounded
# independently - never merged into one list/one tolerance first.
# --------------------------------------------------------------------------- #


def test_min_coarse_frames_scales_with_duration_and_fine_margin():
    # A longer clip, or a tighter fine_margin_s, needs more coarse samples
    # to still guarantee one lands within fine_margin_s of the true
    # transition wherever it is.
    assert _min_coarse_frames(60.0, fine_margin_s=1.5) > _min_coarse_frames(10.0, fine_margin_s=1.5)
    assert _min_coarse_frames(60.0, fine_margin_s=0.5) > _min_coarse_frames(60.0, fine_margin_s=5.0)
    # Never below the old flat floor, even for a very short/generous case.
    assert _min_coarse_frames(1.0, fine_margin_s=10.0) >= 4


def test_fit_fine_windows_to_budget_gives_each_window_its_own_half_regardless_of_the_others_size():
    # The end window's frames are individually far larger (a "busier" JPEG)
    # than the start window's - under a merged single-list thinning pass,
    # this used to let one window's bulk crowd out the other's density. Each
    # must independently retain at least its own floor.
    start_frames = [_frame(float(i), 1_500) for i in range(40)]  # light
    end_frames = [_frame(20.0 + i, 7_000) for i in range(40)]  # heavy

    fitted_start, fitted_end = _fit_fine_windows_to_budget(
        start_frames,
        end_frames,
        prompt_text="p",
        max_request_bytes=200_000,
        start_min_frames=10,
        end_min_frames=10,
    )
    assert fitted_start is not None
    assert fitted_end is not None
    assert len(fitted_start) >= 10
    assert len(fitted_end) >= 10
    # Each window's own half of the budget is respected independently.
    assert _estimated_request_bytes(fitted_start, "p") <= 100_000
    assert _estimated_request_bytes(fitted_end, "p") <= 100_000


def test_fit_fine_windows_to_budget_reports_each_window_failure_independently():
    light = [_frame(float(i), 1_000) for i in range(20)]
    heavy = [_frame(20.0 + i, 500_000) for i in range(20)]  # can't fit even at the floor

    fitted_light, fitted_heavy = _fit_fine_windows_to_budget(
        light,
        heavy,
        prompt_text="p",
        max_request_bytes=200_000,
        start_min_frames=5,
        end_min_frames=5,
    )
    assert fitted_light is not None  # the light window still fits fine
    assert fitted_heavy is None  # the heavy window doesn't - reported separately


# --------------------------------------------------------------------------- #
# _validate_grounding / _GroundingRegion: evidence from the empty space
# between two disjoint windows - never actually sent to the provider - must
# never "ground" either boundary, and evidence legitimately drawn from one
# window must never ground the other window's claim.
# --------------------------------------------------------------------------- #


def _confirmed_verdict(*, start_s, end_s, evidence):
    return TimingVerdict(
        status=TimingStatus.CONFIRMED,
        start_s=start_s,
        end_s=end_s,
        start_uncertainty_s=0.1,
        end_uncertainty_s=0.1,
        confidence=0.9,
        reason_codes=(),
        evidence_frame_timestamps_s=evidence,
        model_id="stub-model",
        prompt_version="test-v1",
    )


def test_a_timestamp_never_actually_sent_cannot_ground_either_boundary_via_the_inter_window_gap():
    # A realistic pair of disjoint fine windows, far apart - exactly the
    # shape a real efflux clip produces (start ~4s, end ~21.5s).
    start_region = _GroundingRegion(
        bounds=(2.5, 5.5),
        submitted_timestamps_s=tuple(round(2.5 + i * 0.04, 6) for i in range(76)),
        time_tolerance_s=0.08,
    )
    end_region = _GroundingRegion(
        bounds=(20.0, 23.0),
        submitted_timestamps_s=tuple(round(20.0 + i * 0.04, 6) for i in range(76)),
        time_tolerance_s=0.08,
    )
    # 10.0 sits in the dead zone between the two windows - never sent to the
    # provider at all. Under the old merged-tolerance behaviour (tolerance
    # inflated to the ~14.5s inter-window gap), this single fabricated
    # timestamp would incorrectly "match" a submitted frame and "ground"
    # both start_s and end_s at once. It must not.
    verdict = _confirmed_verdict(start_s=4.0, end_s=21.5, evidence=(10.0,))
    result = _validate_grounding(
        verdict, start_region=start_region, end_region=end_region, max_uncertainty_s=0.5
    )
    assert result.status is TimingStatus.ABSTAIN
    assert "ungrounded_evidence" in result.reason_codes


def test_evidence_from_one_window_cannot_ground_the_other_windows_claim():
    start_region = _GroundingRegion(
        bounds=(2.5, 5.5),
        submitted_timestamps_s=tuple(round(2.5 + i * 0.04, 6) for i in range(76)),
        time_tolerance_s=0.08,
    )
    end_region = _GroundingRegion(
        bounds=(20.0, 23.0),
        submitted_timestamps_s=tuple(round(20.0 + i * 0.04, 6) for i in range(76)),
        time_tolerance_s=0.08,
    )
    # Evidence is real (genuinely submitted, in the end window) but there is
    # nothing anywhere near start_s - a verdict citing only end-window
    # evidence must not be treated as having grounded the start claim too.
    verdict = _confirmed_verdict(start_s=4.0, end_s=21.5, evidence=(21.48,))
    result = _validate_grounding(
        verdict, start_region=start_region, end_region=end_region, max_uncertainty_s=0.5
    )
    assert result.status is TimingStatus.ABSTAIN
    assert "evidence_far_from_claim" in result.reason_codes


def test_validate_grounding_confirms_when_each_boundary_has_its_own_nearby_evidence():
    start_region = _GroundingRegion(
        bounds=(2.5, 5.5),
        submitted_timestamps_s=tuple(round(2.5 + i * 0.04, 6) for i in range(76)),
        time_tolerance_s=0.08,
    )
    end_region = _GroundingRegion(
        bounds=(20.0, 23.0),
        submitted_timestamps_s=tuple(round(20.0 + i * 0.04, 6) for i in range(76)),
        time_tolerance_s=0.08,
    )
    verdict = _confirmed_verdict(start_s=4.0, end_s=21.5, evidence=(4.0, 21.5))
    result = _validate_grounding(
        verdict, start_region=start_region, end_region=end_region, max_uncertainty_s=0.5
    )
    assert result.status is TimingStatus.CONFIRMED


# --------------------------------------------------------------------------- #
# End-to-end: the same fabricated-inter-window-gap evidence, exercised
# through the real pipeline against the real (widely-separated) zahn_video
# windows, not just the unit-level helpers above.
# --------------------------------------------------------------------------- #


def test_pipeline_abstains_when_fine_evidence_is_nowhere_near_the_start_window(zahn_video):
    """The fine pass is start-only now, so the original "evidence in the
    gap between two merged fine windows" scenario (Codex re-review round 3,
    finding 1) can no longer occur - there is only ever one fine window.
    The underlying discipline it protected still matters though: a
    timestamp far outside the one window actually sent must not ground a
    claim, no matter how internally consistent the rest of the verdict
    looks."""

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=zahn_video.truth["flow_start_s"],
                end_s=zahn_video.truth["flow_end_s"],
                confidence=0.8,
            )
        # Nowhere near the start window (~[2.5, 5.5]) - never actually
        # extracted or sent.
        far_timestamp = zahn_video.truth["flow_end_s"]
        return canned_json_response(
            start_s=zahn_video.truth["flow_start_s"],
            end_s=zahn_video.truth["flow_start_s"],
            confidence=0.9,
            evidence_frame_timestamps_s=(far_timestamp,),
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "ungrounded_evidence" in outcome.verdict.reason_codes
    assert len(provider.calls) == 2  # both coarse and fine were sent
