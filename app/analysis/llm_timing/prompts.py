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

PROMPTS: dict[str, str] = {PROMPT_V1_ID: PROMPT_V1}


def get_prompt(prompt_version: str) -> str:
    try:
        return PROMPTS[prompt_version]
    except KeyError as exc:
        raise KeyError(
            f"Unknown prompt_version {prompt_version!r}; known: {sorted(PROMPTS)}"
        ) from exc
