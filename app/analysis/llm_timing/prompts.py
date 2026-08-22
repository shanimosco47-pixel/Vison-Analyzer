"""Versioned prompt templates for the LLM timing provider.

The supervisor's authorization requires model/prompt versions to be pinned
and logged. A prompt is part of the contract, not incidental text: changing
its wording changes what a given ``prompt_version`` means, so any edit here
must come with a new version string. Never mutate an existing template in
place.

``PROMPT_V1`` is the coarse-to-fine, criteria-first prompt this spike is
reproducing (refined by hand against two real clips before this spike
existed - see ``diagnostics/llm_spike/DESIGN.md``), rewritten to demand
strict JSON matching ``schema.TimingVerdict`` instead of a conversational
report, since a production caller cannot ask a human to re-copy the answer
into the right shape.

``PROMPT_END_SCAN_V1`` is used only by the chronological end-scan phase in
``pipeline.py`` (real gate-1 runs across four models converged on one
shared failure: the single wide fine-pass window let the model's own
end-of-stream judgement drift to a late final-disappearance/thinning event
rather than the first genuine break). It asks a narrower question - "does
the stream break for the first time within this specific short window" -
about one small window at a time, reusing ``schema.TimingVerdict``'s
existing shape by reporting the break as a degenerate point (``start_s ==
end_s``) rather than an interval.
"""

from __future__ import annotations

PROMPT_V1_ID = "zahn-efflux-v1"

PROMPT_V1 = """\
You are analyzing a single Zahn cup viscosity test as one self-contained
measurement. You are given a batch of frames extracted from the video,
each labelled with its exact timestamp in seconds - not the whole video
file. Frame selection (which timestamps) was done deterministically by the
calling pipeline, not by you; do not assume you are seeing every frame, and
rely only on the timestamps given.

This may be a coarse batch (frames spread sparsely across the whole clip,
to locate approximate windows) or a fine batch (every frame within a short
window, to pin down an exact transition) - the accompanying pass label
tells you which. If this is a coarse batch, report your best approximate
start_s/end_s and expect a follow-up fine batch to refine them; do not treat
a coarse-pass answer as final-precision.

MEASUREMENT RULE (use exactly this rule, do not invent your own):
- "start" = the first labelled frame in which liquid leaving the outlet
  forms a continuous, uniform stream (not a single drop, not a
  discontinuous dribble still connected to the fluid surface in the
  container).
- "end" = the first labelled frame in which that continuous stream shortens
  and no longer reaches the lower boundary it had been reaching. One frame
  of real shortening is enough; do not wait for confirmation from further
  frames. If the stream later becomes long again, that does not cancel the
  end you already found - stop at the first genuine shortening event.
- duration = end - start.

Do not rely on a general visual impression of the batch in place of
examining individual labelled frames and citing their timestamps as
evidence.

HAZARDS SPECIFIC TO THIS FOOTAGE:
- The cup, stream and background are often close to the same pale/dusty
  colour (industrial environment) - contrast can be very low. A drop in raw
  pixel brightness alone is not evidence of a break: look for continuity
  along the stream's own axis, not an absolute brightness threshold.
- The footage is handheld - there is camera shake. Do not confuse whole-frame
  motion (camera movement) with real motion of the stream itself.
- If the stream merely fades against a light background without a genuine
  geometric break, treat it as still continuous, not ended.
- Prefer evidence from any frame where the stream is visible against a
  darker background - the edge is unambiguous there.
- If no clear geometric edge is visible and the evidence is genuinely
  ambiguous, abstain rather than guess.

OUTPUT: reply with a single JSON object and nothing else (no prose before or
after it), matching exactly this shape:

{
  "status": "confirmed" | "abstain",
  "start_s": <float seconds, or null if abstaining>,
  "end_s": <float seconds, or null if abstaining>,
  "start_uncertainty_s": <float, your own +/- bound on start_s>,
  "end_uncertainty_s": <float, your own +/- bound on end_s>,
  "confidence": <float 0..1>,
  "reason_codes": [<short machine-readable strings, e.g. "weak_contrast",
                     "camera_motion", "no_break_found", "ambiguous_evidence",
                     "resumed_flow", "missing_outlet">],
  "evidence_frame_timestamps_s": [<the frame timestamps you actually
                                     inspected around each decision>],
  "raw_notes": "<short free-text explanation, for a human audit log only>"
}

If you are not confident, set "status" to "abstain", leave start_s/end_s
null, and explain why in reason_codes. Do not invent a confident-sounding
number when the evidence does not support one.
"""

PROMPT_END_SCAN_V1_ID = "zahn-efflux-end-scan-v1"

PROMPT_END_SCAN_V1 = """\
You are analyzing one short, consecutive window of frames from a Zahn cup
viscosity test video - not the whole clip. The stream's start has already
been confirmed by an earlier pass; you are being asked ONLY whether the
continuous stream breaks for the first time within THIS window. You will
be asked again about later windows if this one shows no break, so do not
try to judge anything beyond what these frames show.

Each frame is labelled with its exact timestamp in seconds. Frame selection
was done deterministically by the calling pipeline, not by you; rely only
on the timestamps given.

MEASUREMENT RULE (use exactly this rule, do not invent your own):
- A "break" is the first frame, within this window, in which the
  previously-continuous stream shortens and no longer reaches the lower
  boundary it had been reaching - a genuine geometric discontinuity, not a
  brightness/contrast change.
- If you find a break, report it. Do not keep looking further into the
  window for a "more complete" or later break - the FIRST one within this
  window is the answer.
- If the stream is continuous throughout every frame in this window (no
  break at all here), or a break already happened before this window and
  these frames only show the aftermath (drips, a shortened stream,
  reconnection, or nothing) rather than the transition itself, report
  "abstain" with reason_codes including "no_break_found" - do NOT report a
  break that is not the first genuine transition actually visible here.
- Later drips, the stream reappearing, further thinning, or the stream's
  final disappearance are NOT the answer unless one of them is itself the
  first genuine break visible in this specific window - report "abstain"
  for this window rather than report any of those as if it were the first
  break.

HAZARDS SPECIFIC TO THIS FOOTAGE:
- The cup, stream and background are often close to the same pale/dusty
  colour (industrial environment) - contrast can be very low. A drop in raw
  pixel brightness alone is not evidence of a break: look for continuity
  along the stream's own axis, not an absolute brightness threshold.
- The footage is handheld - there is camera shake. Do not confuse whole-frame
  motion (camera movement) with real motion of the stream itself.
- If the stream merely fades against a light background without a genuine
  geometric break, treat it as still continuous, not a break.
- If no clear geometric edge is visible and the evidence is genuinely
  ambiguous, abstain rather than guess.

OUTPUT: reply with a single JSON object and nothing else (no prose before or
after it), matching exactly this shape. This search reports a single moment
(a break), not an interval - set start_s and end_s to the SAME timestamp,
and start_uncertainty_s/end_uncertainty_s to the same value:

{
  "status": "confirmed" | "abstain",
  "start_s": <float seconds - the break's timestamp, identical to end_s;
              or null if abstaining>,
  "end_s": <float seconds - the break's timestamp, identical to start_s;
            or null if abstaining>,
  "start_uncertainty_s": <float, your own +/- bound on the timestamp>,
  "end_uncertainty_s": <float, the same +/- bound as start_uncertainty_s>,
  "confidence": <float 0..1>,
  "reason_codes": [<short machine-readable strings, e.g. "weak_contrast",
                     "camera_motion", "no_break_found", "ambiguous_evidence",
                     "resumed_flow", "missing_outlet">],
  "evidence_frame_timestamps_s": [<frame timestamps from THIS window you
                                     actually inspected around the break -
                                     immediately before/at/after it>],
  "raw_notes": "<short free-text explanation, for a human audit log only>"
}

If you are not confident, set "status" to "abstain", leave start_s/end_s
null, and explain why in reason_codes. Do not invent a confident-sounding
timestamp when the evidence does not support one.
"""

PROMPTS: dict[str, str] = {
    PROMPT_V1_ID: PROMPT_V1,
    PROMPT_END_SCAN_V1_ID: PROMPT_END_SCAN_V1,
}


def get_prompt(prompt_version: str) -> str:
    try:
        return PROMPTS[prompt_version]
    except KeyError as exc:
        raise KeyError(
            f"Unknown prompt_version {prompt_version!r}; known: {sorted(PROMPTS)}"
        ) from exc
