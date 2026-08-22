"""Tests for the LLM timing spike (app/analysis/llm_timing).

This spike is not wired into the detector registry or the web layer (see the
package docstring and diagnostics/llm_spike/DESIGN.md), so these tests run
entirely offline against StubTimingProvider - no network, no API key. They
exist to prove the *scaffolding* is trustworthy: schema invariants hold,
every provider failure mode converges on ABSTAIN, and the coarse-to-fine
pipeline never fabricates an Event when it shouldn't.

They intentionally do NOT prove the spike measures real footage accurately -
that is evidence gate 1 in the design doc, and requires a real provider and
the two hand-verified real clips, neither of which exist in this repo.
"""

from __future__ import annotations

import pytest

from app.analysis.base_detector import EventStatus
from app.analysis.llm_timing.pipeline import PipelineConfig, run_llm_timing
from app.analysis.llm_timing.provider import (
    ProviderRequest,
    RawProviderResponse,
    StubTimingProvider,
    canned_json_response,
    parse_raw_response,
)
from app.analysis.llm_timing.schema import TimingStatus, TimingVerdict
from app.errors import ConfigurationError

PROMPT_VERSION = "test-prompt-v1"


def _candidate_timestamp(request: ProviderRequest) -> float:
    """The timestamp of the frame the pipeline marked as the trend-
    validation candidate (see ``TimedFrame.is_candidate``) - every
    "end_validate" request has exactly one."""
    for frame in request.frames:
        if frame.is_candidate:
            return frame.timestamp_s
    raise AssertionError("end_validate request missing a candidate-marked frame")


def _confirm_validation(
    request: ProviderRequest, *, confidence: float = 0.9
) -> RawProviderResponse:
    """A perfect trend-validation stub response: always validates whichever
    candidate the pipeline flagged. Good enough for tests whose synthetic
    fixture has a single genuine, sustained break with no recovery - the
    dedicated trend-validation regressions below script something more
    specific."""
    ts = _candidate_timestamp(request)
    return canned_json_response(
        start_s=ts, end_s=ts, confidence=confidence, evidence_frame_timestamps_s=(ts,)
    )


# --------------------------------------------------------------------------- #
# schema.TimingVerdict invariants
# --------------------------------------------------------------------------- #


def test_confirmed_verdict_requires_start_and_end():
    with pytest.raises(ConfigurationError):
        TimingVerdict(
            status=TimingStatus.CONFIRMED,
            start_s=None,
            end_s=5.0,
            start_uncertainty_s=0.1,
            end_uncertainty_s=0.1,
            confidence=0.9,
            reason_codes=(),
            evidence_frame_timestamps_s=(),
            model_id="m",
            prompt_version=PROMPT_VERSION,
        )


def test_confirmed_verdict_rejects_end_before_start():
    with pytest.raises(ConfigurationError):
        TimingVerdict(
            status=TimingStatus.CONFIRMED,
            start_s=5.0,
            end_s=1.0,
            start_uncertainty_s=0.1,
            end_uncertainty_s=0.1,
            confidence=0.9,
            reason_codes=(),
            evidence_frame_timestamps_s=(),
            model_id="m",
            prompt_version=PROMPT_VERSION,
        )


def test_abstain_verdict_requires_a_reason_code():
    with pytest.raises(ConfigurationError):
        TimingVerdict.abstain(reason_codes=(), model_id="m", prompt_version=PROMPT_VERSION)


def test_confidence_out_of_range_rejected():
    with pytest.raises(ConfigurationError):
        TimingVerdict(
            status=TimingStatus.ABSTAIN,
            start_s=None,
            end_s=None,
            start_uncertainty_s=0.0,
            end_uncertainty_s=0.0,
            confidence=1.5,
            reason_codes=("no_break_found",),
            evidence_frame_timestamps_s=(),
            model_id="m",
            prompt_version=PROMPT_VERSION,
        )


def test_duration_is_none_when_abstained():
    verdict = TimingVerdict.abstain(
        reason_codes=("no_break_found",), model_id="m", prompt_version=PROMPT_VERSION
    )
    assert verdict.duration_s is None


# --------------------------------------------------------------------------- #
# provider.parse_raw_response: every failure mode -> ABSTAIN, never a guess
# --------------------------------------------------------------------------- #


def test_parse_well_formed_confirmed_response():
    response = canned_json_response(start_s=4.0, end_s=20.6, confidence=0.9)
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.CONFIRMED
    assert verdict.start_s == 4.0
    assert verdict.end_s == 20.6
    assert verdict.duration_s == pytest.approx(16.6)


def test_parse_provider_error_abstains():
    response = RawProviderResponse(
        model_id="m", raw_text="", latency_s=0.0, error="timeout after 30s"
    )
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "provider_error" in verdict.reason_codes


def test_parse_non_json_abstains():
    response = RawProviderResponse(model_id="m", raw_text="not json at all", latency_s=0.0)
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "malformed_output" in verdict.reason_codes


def test_parse_missing_required_field_abstains():
    response = RawProviderResponse(
        model_id="m", raw_text='{"status": "confirmed", "confidence": 0.9}', latency_s=0.0
    )
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "malformed_output" in verdict.reason_codes


def test_parse_end_before_start_abstains_not_raises():
    response = RawProviderResponse(
        model_id="m",
        raw_text='{"status": "confirmed", "start_s": 10.0, "end_s": 2.0, "confidence": 0.9}',
        latency_s=0.0,
    )
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "invalid_invariant" in verdict.reason_codes


def test_parse_below_confidence_floor_abstains():
    response = canned_json_response(start_s=4.0, end_s=20.6, confidence=0.3)
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert "low_confidence" in verdict.reason_codes


def test_parse_model_requested_abstain_preserves_reason_codes():
    response = RawProviderResponse(
        model_id="m",
        raw_text='{"status": "abstain", "reason_codes": ["weak_contrast", "camera_motion"]}',
        latency_s=0.0,
    )
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN
    assert set(verdict.reason_codes) == {"weak_contrast", "camera_motion"}


# --------------------------------------------------------------------------- #
# parse_raw_response must be total: NaN/Infinity/non-numeric never raise,
# in either a "confirmed" or an "abstain" payload (Codex review, finding 2)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw_text",
    [
        '{"status": "confirmed", "start_s": 4.0, "end_s": 20.6, "confidence": NaN}',
        '{"status": "confirmed", "start_s": 4.0, "end_s": 20.6, "confidence": Infinity}',
        '{"status": "confirmed", "start_s": NaN, "end_s": 20.6, "confidence": 0.9}',
        '{"status": "confirmed", "start_s": 4.0, "end_s": Infinity, "confidence": 0.9}',
        '{"status": "confirmed", "start_s": "not-a-number", "end_s": 20.6, "confidence": 0.9}',
        '{"status": "confirmed", "start_s": 4.0, "end_s": 20.6, "confidence": "high"}',
        '{"status": "confirmed", "start_s": 4.0, "end_s": 20.6, "confidence": 0.9, '
        '"start_uncertainty_s": NaN}',
        '{"status": "confirmed", "start_s": 4.0, "end_s": 20.6, "confidence": 0.9, '
        '"reason_codes": "not-a-list"}',
    ],
)
def test_parse_never_raises_on_confirmed_payload_garbage(raw_text):
    response = RawProviderResponse(model_id="m", raw_text=raw_text, latency_s=0.0)
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN


@pytest.mark.parametrize(
    "raw_text",
    [
        '{"status": "abstain", "confidence": NaN}',
        '{"status": "abstain", "confidence": Infinity}',
        '{"status": "abstain", "confidence": "not-a-number"}',
        '{"status": "abstain", "reason_codes": ["ok"], "confidence": -Infinity}',
    ],
)
def test_parse_never_raises_on_abstain_payload_garbage(raw_text):
    response = RawProviderResponse(model_id="m", raw_text=raw_text, latency_s=0.0)
    verdict = parse_raw_response(response, prompt_version=PROMPT_VERSION, min_confidence=0.5)
    assert verdict.status is TimingStatus.ABSTAIN


# --------------------------------------------------------------------------- #
# pipeline.run_llm_timing: coarse-to-fine orchestration against a real video
# --------------------------------------------------------------------------- #


def _stub_matching_truth(
    truth: dict[str, float], *, fine_confidence: float = 0.9, end_scan_confidence: float = 0.9
):
    """A StubTimingProvider that answers accurately on every pass, including
    the chronological end-scan: it confirms the true break only for the
    scan window that actually contains it, and abstains (no_break_found)
    for every earlier window - the same shape a real model's answers take,
    so tests exercise the scan loop itself rather than a single fine call.
    """

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=truth["flow_start_s"], end_s=truth["flow_end_s"], confidence=0.8
            )
        if request.pass_name == "fine":
            # The fine pass is start-only now: it reports the confirmed
            # start as a degenerate point (start_s == end_s), never the
            # (now-irrelevant) end.
            return canned_json_response(
                start_s=truth["flow_start_s"],
                end_s=truth["flow_start_s"],
                confidence=fine_confidence,
                evidence_frame_timestamps_s=(truth["flow_start_s"],),
            )
        if request.pass_name == "end_validate":
            return _confirm_validation(request)
        assert request.pass_name == "end_scan"
        window_times = [f.timestamp_s for f in request.frames]
        end_s = truth["flow_end_s"]
        if window_times and min(window_times) <= end_s <= max(window_times):
            return canned_json_response(
                start_s=end_s,
                end_s=end_s,
                confidence=end_scan_confidence,
                evidence_frame_timestamps_s=(end_s,),
            )
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    return StubTimingProvider(respond)


def test_pipeline_confirms_event_matching_ground_truth(zahn_video):
    provider = _stub_matching_truth(zahn_video.truth)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.status is EventStatus.CONFIRMED
    assert outcome.event.start_s == pytest.approx(zahn_video.truth["flow_start_s"])
    assert outcome.event.end_s == pytest.approx(zahn_video.truth["flow_end_s"])
    # coarse, then fine (start), then one or more end-scan windows, each
    # CONFIRMED window immediately followed by a trend-validation call -
    # stopping once a candidate both occurs and validates.
    assert provider.calls[0].pass_name == "coarse"
    assert provider.calls[1].pass_name == "fine"
    assert len(provider.calls[1].frames) > len(provider.calls[0].frames)
    assert len(provider.calls) > 3  # at least one end-scan window + its validation
    assert all(c.pass_name in ("end_scan", "end_validate") for c in provider.calls[2:])
    assert provider.calls[-1].pass_name == "end_validate"
    assert provider.calls[-2].pass_name == "end_scan"
    # The winning end-scan window's frames actually contain the true break.
    winning_times = [f.timestamp_s for f in provider.calls[-2].frames]
    assert min(winning_times) <= zahn_video.truth["flow_end_s"] <= max(winning_times)
    assert outcome.end_scan_responses  # every scan call's diagnostics preserved
    assert outcome.end_validation_responses  # every validation call's diagnostics preserved
    assert outcome.end_scan_responses[-1].model_id == outcome.verdict.model_id


def test_pipeline_marks_low_but_passing_confidence_as_review(zahn_video):
    provider = _stub_matching_truth(zahn_video.truth, fine_confidence=0.6)
    config = PipelineConfig(min_confidence=0.5, review_confidence=0.75)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        config=config,
    )
    assert outcome.event is not None
    assert outcome.event.status is EventStatus.REVIEW


def test_pipeline_abstains_when_coarse_pass_abstains(zahn_video):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        return RawProviderResponse(
            model_id="m",
            raw_text='{"status": "abstain", "reason_codes": ["no_continuous_stream_found"]}',
            latency_s=0.0,
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
    assert outcome.fine_response is None  # never got that far
    assert len(provider.calls) == 1


def test_pipeline_abstains_when_fine_pass_abstains_after_confirmed_coarse(zahn_video):
    calls = {"n": 0}

    def respond(request: ProviderRequest) -> RawProviderResponse:
        calls["n"] += 1
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=zahn_video.truth["flow_start_s"],
                end_s=zahn_video.truth["flow_end_s"],
                confidence=0.8,
            )
        return RawProviderResponse(
            model_id="m",
            raw_text='{"status": "abstain", "reason_codes": ["ambiguous_evidence"]}',
            latency_s=0.0,
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
    assert outcome.fine_response is not None  # it did run, and then abstained
    assert calls["n"] == 2


def test_pipeline_abstains_on_malformed_provider_output_without_crashing(zahn_video):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        return RawProviderResponse(model_id="m", raw_text="<html>not json</html>", latency_s=0.0)

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "malformed_output" in outcome.verdict.reason_codes


def test_pipeline_abstains_when_coarse_window_is_implausibly_wide(zahn_video):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        # The coarse pass itself reports huge uncertainty on the start
        # boundary - too wide a window to fine-scan densely.
        return canned_json_response(
            start_s=zahn_video.truth["flow_start_s"],
            end_s=zahn_video.truth["flow_end_s"],
            start_uncertainty_s=15.0,
            confidence=0.8,
        )

    provider = StubTimingProvider(respond)
    # max_uncertainty_s raised so this test isolates the oversized-window
    # path from the separate uncertainty-cap path (see
    # test_pipeline_abstains_when_uncertainty_exceeds_cap).
    config = PipelineConfig(fine_max_span_s=6.0, max_uncertainty_s=20.0)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        config=config,
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "ambiguous_evidence" in outcome.verdict.reason_codes
    assert len(provider.calls) == 1  # never sent the (would-be enormous) fine batch


def test_pipeline_abstains_when_uncertainty_exceeds_cap(zahn_video):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        return canned_json_response(
            start_s=zahn_video.truth["flow_start_s"],
            end_s=zahn_video.truth["flow_end_s"],
            start_uncertainty_s=15.0,
            confidence=0.8,
        )

    provider = StubTimingProvider(respond)
    config = PipelineConfig(max_uncertainty_s=5.0)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
        config=config,
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "uncertainty_exceeds_cap" in outcome.verdict.reason_codes
    assert len(provider.calls) == 1  # rejected at the coarse pass, no fine call


# --------------------------------------------------------------------------- #
# _validate_grounding via run_llm_timing: a CONFIRMED verdict that parses
# cleanly but isn't tethered to the actual request must still abstain
# (Codex review, finding 1)
# --------------------------------------------------------------------------- #


def _stub_coarse_ok_fine_custom(truth, fine_response: RawProviderResponse):
    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=truth["flow_start_s"], end_s=truth["flow_end_s"], confidence=0.8
            )
        return fine_response

    return StubTimingProvider(respond)


def test_pipeline_abstains_when_fine_start_is_outside_its_window(zahn_video):
    # The fine window is only ~fine_margin_s wide around the coarse
    # estimate (default 1.5s); 0.0 is nowhere near a true start_s of 4.0.
    fine_response = canned_json_response(start_s=0.0, end_s=0.0, confidence=0.9)
    provider = _stub_coarse_ok_fine_custom(zahn_video.truth, fine_response)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "out_of_bounds" in outcome.verdict.reason_codes


def test_pipeline_abstains_when_evidence_is_empty(zahn_video):
    fine_response = canned_json_response(
        start_s=zahn_video.truth["flow_start_s"],
        end_s=zahn_video.truth["flow_start_s"],
        confidence=0.9,
        evidence_frame_timestamps_s=(),
    )
    provider = _stub_coarse_ok_fine_custom(zahn_video.truth, fine_response)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "ungrounded_evidence" in outcome.verdict.reason_codes


def test_pipeline_abstains_when_evidence_matches_no_submitted_frame(zahn_video):
    fine_response = canned_json_response(
        start_s=zahn_video.truth["flow_start_s"],
        end_s=zahn_video.truth["flow_start_s"],
        confidence=0.9,
        # A timestamp nowhere near anything the pipeline actually extracted
        # and sent - a fabricated citation.
        evidence_frame_timestamps_s=(9999.0,),
    )
    provider = _stub_coarse_ok_fine_custom(zahn_video.truth, fine_response)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "ungrounded_evidence" in outcome.verdict.reason_codes


def test_pipeline_abstains_when_evidence_is_grounded_but_far_from_the_claim(zahn_video):
    # start_lo is roughly flow_start_s - fine_margin_s = 4.0 - 1.5 = 2.5;
    # this is a real, submittable frame timestamp, but nowhere near the
    # claimed start_s (4.0).
    fine_response = canned_json_response(
        start_s=zahn_video.truth["flow_start_s"],
        end_s=zahn_video.truth["flow_start_s"],
        confidence=0.9,
        evidence_frame_timestamps_s=(2.5,),
    )
    provider = _stub_coarse_ok_fine_custom(zahn_video.truth, fine_response)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.ABSTAIN
    assert outcome.event is None
    assert "evidence_far_from_claim" in outcome.verdict.reason_codes


# --------------------------------------------------------------------------- #
# Chronological end-scan (supervisor-directed experiment, after four real
# gate-1 runs converged on one shared failure: a single wide end window let
# the model's own judgement drift to a late final-disappearance/thinning
# event instead of the first genuine break). The end is now searched for in
# bounded, overlapping windows after the confirmed start, stopping at the
# first grounded candidate.
# --------------------------------------------------------------------------- #


def test_pipeline_end_scan_stops_at_the_first_break_ignoring_a_later_resumed_flow_window(
    zahn_video,
):
    """A later window that would also confirm a plausible-looking break
    (simulating the stream resuming after the first break, then breaking
    again) must never even be asked about, once an earlier window has
    already produced a grounded candidate - the FIRST break wins, and the
    scan stops there."""
    end_scan_calls: list[ProviderRequest] = []
    first_break_s = 10.0
    resumed_break_s = 18.0  # only reachable if the scan wrongly kept going

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=first_break_s, confidence=0.9)
        if request.pass_name == "fine":
            # Start-only pass now: reports the confirmed start as a
            # degenerate point, never the (discarded) end.
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            return _confirm_validation(request)
        end_scan_calls.append(request)
        window_times = [f.timestamp_s for f in request.frames]
        for candidate in (first_break_s, resumed_break_s):
            if window_times and min(window_times) <= candidate <= max(window_times):
                return canned_json_response(
                    start_s=candidate,
                    end_s=candidate,
                    confidence=0.9,
                    evidence_frame_timestamps_s=(candidate,),
                )
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.end_s == pytest.approx(first_break_s)
    # Every end-scan window actually asked about stopped short of the later
    # "resumed flow" candidate - the scan never got there because it had
    # already stopped at the first grounded break.
    for call in end_scan_calls:
        window_times = [f.timestamp_s for f in call.frames]
        assert max(window_times) < resumed_break_s
    assert len(outcome.end_scan_responses) == len(end_scan_calls)
    # The winning end-scan window is immediately followed by its (accepted)
    # trend-validation call.
    assert end_scan_calls[-1] is provider.calls[-2]
    assert provider.calls[-1].pass_name == "end_validate"


def test_pipeline_end_scan_abstains_with_no_break_found_when_the_stream_never_breaks(zahn_video):
    """The stream stays continuous (or the model never sees a genuine
    break) all the way to the end of the clip - every scan window abstains,
    and the whole run converges on ABSTAIN, never a fabricated end."""

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=21.5, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
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
    assert "no_break_found" in outcome.verdict.reason_codes
    # Scanned all the way to the end of the clip, never fabricating a result.
    assert outcome.end_scan_responses
    assert all(c.pass_name == "end_scan" for c in provider.calls[2:])
    last_window_frames = [c for c in provider.calls if c.pass_name == "end_scan"][-1].frames
    assert max(f.timestamp_s for f in last_window_frames) == pytest.approx(
        zahn_video.duration_s, abs=0.1
    )


def test_pipeline_start_confirmation_is_independent_of_a_stale_coarse_end_estimate(zahn_video):
    """Reproduces the shape of a real gate-1 failure: the coarse pass's own
    end estimate is stale/wrong, close to the end of the clip - exactly the
    shape that, under the old combined start+end fine-window design, built
    an unusable fine end window (e.g. ~[25.500, 27.031]s) whose invalid
    end_s could abstain the *whole* run before the end-scan ever got to
    run, even though that end answer was never actually used downstream.
    The fine pass no longer builds or asks about an end window at all, so
    a bad coarse end estimate can't touch start confirmation, and the
    chronological end-scan still runs and finds the true first break
    (supervisor-directed fix, see diagnostics/llm_spike/DESIGN.md)."""
    stale_coarse_end_s = 25.9  # nowhere near the true end (21.5)

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=zahn_video.truth["flow_start_s"], end_s=stale_coarse_end_s, confidence=0.8
            )
        if request.pass_name == "fine":
            return canned_json_response(
                start_s=zahn_video.truth["flow_start_s"],
                end_s=zahn_video.truth["flow_start_s"],
                confidence=0.9,
            )
        if request.pass_name == "end_validate":
            return _confirm_validation(request)
        # end_scan
        window_times = [f.timestamp_s for f in request.frames]
        true_end_s = zahn_video.truth["flow_end_s"]
        if window_times and min(window_times) <= true_end_s <= max(window_times):
            return canned_json_response(
                start_s=true_end_s,
                end_s=true_end_s,
                confidence=0.9,
                evidence_frame_timestamps_s=(true_end_s,),
            )
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.start_s == pytest.approx(zahn_video.truth["flow_start_s"])
    assert outcome.event.end_s == pytest.approx(zahn_video.truth["flow_end_s"])
    # The fine request never covered anything near the coarse pass's (bad)
    # end estimate - it only ever spans the start window.
    fine_call = provider.calls[1]
    assert fine_call.pass_name == "fine"
    fine_times = [f.timestamp_s for f in fine_call.frames]
    assert max(fine_times) < 10.0


def test_pipeline_end_scan_never_considers_the_onset_transition_as_a_candidate_break(zahn_video):
    """Reproduces a real gate-1 failure: a third real rerun (commit
    ``894dfe6``) got a CONFIRMED start of ~4.033s (correct) but a
    false-CONFIRMED end of ~4.666s from the *first* end-scan window,
    ``[4.03, 7.03]`` - the model misread the stream's own onset transition
    (nothing visible -> stream visible), right next to the confirmed start,
    as if it were a break, with its own reason codes naming both
    ``stream_start_visible`` and ``first_break_in_window``.

    The fix is structural: the scan begins no earlier than ``start_hi``,
    the far edge of the grounded start-refinement window, so an onset
    timestamp is never even eligible to be submitted as a candidate end -
    a stub that is (unrealistically) willing to falsely confirm right next
    to the true start must never get the chance, and the later, true break
    must still be found."""
    false_onset_break_s = zahn_video.truth["flow_start_s"] + 0.633  # ~4.666s
    true_break_s = 15.0
    end_scan_calls: list[ProviderRequest] = []

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(
                start_s=zahn_video.truth["flow_start_s"], end_s=true_break_s, confidence=0.9
            )
        if request.pass_name == "fine":
            return canned_json_response(
                start_s=zahn_video.truth["flow_start_s"],
                end_s=zahn_video.truth["flow_start_s"],
                confidence=0.9,
            )
        if request.pass_name == "end_validate":
            return _confirm_validation(request)
        end_scan_calls.append(request)
        window_times = [f.timestamp_s for f in request.frames]
        for candidate in (false_onset_break_s, true_break_s):
            if window_times and min(window_times) <= candidate <= max(window_times):
                return canned_json_response(
                    start_s=candidate,
                    end_s=candidate,
                    confidence=0.9,
                    evidence_frame_timestamps_s=(candidate,),
                )
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.start_s == pytest.approx(zahn_video.truth["flow_start_s"])
    assert outcome.event.end_s == pytest.approx(true_break_s)
    # No end-scan window ever included the onset region - the false break
    # was never even a reachable candidate, not merely one that lost to a
    # later grounded confirmation.
    assert end_scan_calls
    for call in end_scan_calls:
        window_times = [f.timestamp_s for f in call.frames]
        assert min(window_times) > false_onset_break_s


# --------------------------------------------------------------------------- #
# Trend validation (supervisor-directed generalization of the onset fix
# above): a real break is the onset of a *sustained* shortening trend, not
# a single frame that happens to look shorter. Every end-scan candidate
# must pass a bounded follow-up trend check before the scan stops.
# --------------------------------------------------------------------------- #


def test_pipeline_rejects_a_candidate_that_fully_recovers_and_keeps_scanning(zahn_video):
    """A candidate break that looked plausible in isolation (one window)
    but fully recovers back toward the established baseline afterward - a
    visual/camera/contrast artifact, not a real break - must be rejected by
    trend validation, and the chronological scan must keep going and find
    the later, true break instead."""
    false_candidate_s = 10.0
    true_break_s = 16.0
    validation_calls: list[ProviderRequest] = []

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=true_break_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            validation_calls.append(request)
            candidate_ts = next(f.timestamp_s for f in request.frames if f.is_candidate)
            if candidate_ts == false_candidate_s:
                # Shortened briefly, then recovered back to the baseline
                # reach and stayed there - reject.
                return RawProviderResponse(
                    model_id="stub-model",
                    raw_text='{"status": "abstain", "reason_codes": ["trend_not_sustained"]}',
                    latency_s=0.01,
                )
            return canned_json_response(
                start_s=candidate_ts,
                end_s=candidate_ts,
                confidence=0.9,
                evidence_frame_timestamps_s=(candidate_ts,),
            )
        # end_scan
        window_times = [f.timestamp_s for f in request.frames]
        for candidate in (false_candidate_s, true_break_s):
            if window_times and min(window_times) <= candidate <= max(window_times):
                return canned_json_response(
                    start_s=candidate,
                    end_s=candidate,
                    confidence=0.9,
                    evidence_frame_timestamps_s=(candidate,),
                )
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.end_s == pytest.approx(true_break_s)
    assert outcome.event.end_s != pytest.approx(false_candidate_s)
    # The rejected candidate really was checked and rejected, not skipped
    # over by the scan.
    assert any(
        frame.timestamp_s == false_candidate_s
        for call in validation_calls
        for frame in call.frames
        if frame.is_candidate
    )
    assert len(outcome.end_validation_responses) >= 2


def test_pipeline_accepts_a_validated_candidate_reporting_its_own_t_not_a_later_point(zahn_video):
    """When a candidate is validated, the final result must report the
    candidate's OWN timestamp T - the first onset frame - never a later
    point from within the validation window, even if the validation call's
    own answer names a different (also-plausible) timestamp - e.g. a brief
    re-extension followed by continued, deeper shortening that a real model
    might cite as "where it's now clearly shorter"."""
    candidate_s = 10.0
    deeper_point_s = 11.5  # inside the validation horizon, later than candidate_s

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            # Validates the candidate, but (unrealistically, to prove the
            # pipeline does not just trust this call's own numbers) echoes
            # back a later, deeper timestamp instead of the candidate's own.
            return canned_json_response(
                start_s=deeper_point_s,
                end_s=deeper_point_s,
                confidence=0.9,
                evidence_frame_timestamps_s=(deeper_point_s,),
            )
        # end_scan
        window_times = [f.timestamp_s for f in request.frames]
        if window_times and min(window_times) <= candidate_s <= max(window_times):
            return canned_json_response(
                start_s=candidate_s,
                end_s=candidate_s,
                confidence=0.9,
                evidence_frame_timestamps_s=(candidate_s,),
            )
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.end_s == pytest.approx(candidate_s)
    assert outcome.event.end_s != pytest.approx(deeper_point_s)


def test_pipeline_accepts_a_candidate_with_sustained_per_second_shortening(zahn_video):
    """The straightforward positive case: a candidate break whose
    post-candidate frames show a clear, sustained shortening trend -
    evidence cited at the candidate itself plus later per-checkpoint
    frames - is accepted on the first candidate checked, no rejection or
    further scanning needed."""
    candidate_s = 12.0

    def respond(request: ProviderRequest) -> RawProviderResponse:
        if request.pass_name == "coarse":
            return canned_json_response(start_s=4.0, end_s=candidate_s, confidence=0.9)
        if request.pass_name == "fine":
            return canned_json_response(start_s=4.0, end_s=4.0, confidence=0.9)
        if request.pass_name == "end_validate":
            candidate_ts = next(f.timestamp_s for f in request.frames if f.is_candidate)
            later_frames = sorted(
                f.timestamp_s for f in request.frames if f.timestamp_s > candidate_ts
            )
            # Cite the candidate plus two later checkpoints - a stand-in for
            # "checked roughly per second through the validation horizon".
            checkpoints = (
                (candidate_ts, later_frames[len(later_frames) // 2], later_frames[-1])
                if later_frames
                else (candidate_ts,)
            )
            return canned_json_response(
                start_s=candidate_ts,
                end_s=candidate_ts,
                confidence=0.95,
                evidence_frame_timestamps_s=checkpoints,
            )
        # end_scan
        window_times = [f.timestamp_s for f in request.frames]
        if window_times and min(window_times) <= candidate_s <= max(window_times):
            return canned_json_response(
                start_s=candidate_s,
                end_s=candidate_s,
                confidence=0.9,
                evidence_frame_timestamps_s=(candidate_s,),
            )
        return RawProviderResponse(
            model_id="stub-model",
            raw_text='{"status": "abstain", "reason_codes": ["no_break_found"]}',
            latency_s=0.01,
        )

    provider = StubTimingProvider(respond)
    outcome = run_llm_timing(
        zahn_video.path,
        provider,
        prompt_version=PROMPT_VERSION,
        prompt_text="irrelevant for a stub",
    )
    assert outcome.verdict.status is TimingStatus.CONFIRMED
    assert outcome.event is not None
    assert outcome.event.end_s == pytest.approx(candidate_s)
    assert len(outcome.end_validation_responses) == 1  # validated immediately, no rejects


def test_pipeline_config_rejects_invalid_bounds():
    with pytest.raises(ConfigurationError):
        PipelineConfig(min_confidence=0.9, review_confidence=0.5).validate()
