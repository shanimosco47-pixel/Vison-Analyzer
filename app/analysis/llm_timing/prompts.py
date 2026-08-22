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

``PROMPT_END_SCAN_V1`` and ``PROMPT_END_SCAN_V2`` (both superseded, kept
only as the historical record of what real gate-1 runs were actually
scored against - ``pipeline.py`` no longer calls either) drove the
chronological end-scan phase: real gate-1 runs across four models had
converged on one shared failure with the *original* single wide end
window (the model's own end-of-stream judgement drifting to a late
final-disappearance/thinning event rather than the first genuine break),
so V1/V2 instead asked a narrower question - "does the stream break for
the first time within this specific short window" - about one small
window at a time, chronologically, reusing ``schema.TimingVerdict``'s
existing shape by reporting the break as a degenerate point
(``start_s == end_s``). V2 added a "TIMESTAMP RULE" paragraph after a real
gate-1 run against gpt-4.1-mini returned ``start_s=end_s=5.533`` for a
submitted window of ``[20.000, 23.000]``s - a value matching no frame
actually shown (grounding safely abstained, but the call was wasted) -
instructing the model to copy a labelled frame timestamp verbatim rather
than compute one; this is why ``openai_provider._response_schema_for_pass``
still constrains the Structured Outputs schema the same way for the
passes below, even though V1/V2 themselves are retired. The chronological
scan itself was later replaced entirely: it decomposed the search into
too many narrow, isolated windows, each missing the temporal context
needed to judge a *sustained* trend - a real rerun let a 13-window scan
find and then wrongly reject the true break, while one coherent request
covering the same span with the same trend contract succeeded (see
``PROMPT_END_COARSE_V1``/``PROMPT_END_VALIDATE_V1`` below, and
``diagnostics/llm_spike/DESIGN.md``).

``PROMPT_START_REFINE_V1`` is used only by the fine (start-refinement) pass
in ``pipeline.py``. It replaces the earlier design, which sent one combined
request covering both a start window *and* an end window (asking for both
boundaries in one shot, then discarding the end answer once the end-scan
was introduced above). A real gate-1 run exposed why that was unsafe even
though the end answer was unused: the fine pass's own end_s could fall
outside its (now-irrelevant) end window, and ``_validate_grounding``
requires *both* claimed boundaries to ground before it will confirm
anything - so an invalid answer to a question nobody needed could still
abstain the whole run before the end-scan ever got to start. This prompt
asks only about the start, over only a start window, reusing
``schema.TimingVerdict``'s existing shape the same way
``PROMPT_END_SCAN_V2`` does (a degenerate point, ``start_s == end_s``) -
grounding then only has one boundary's claim to check, independent of
anything about the end.

``PROMPT_END_VALIDATE_V1`` is a follow-up, used only after a candidate end
has already been nominated (originally by the chronological end-scan
above; now by ``PROMPT_END_COARSE_V1`` below - see
``pipeline._build_validation_frames`` and ``pipeline.run_llm_timing``'s
trend-validation follow-up request). Its own motivation predates the
end-scan's replacement: a real gate-1 run showed that a single window's
own judgement is not enough on its own - the first end-scan window
(starting right at the confirmed start, before the onset fix existed)
still saw the stream's own onset transition and mistook it for a break.
The onset fix kept that specific case from recurring structurally, but
the supervisor's own follow-up authorization generalizes the underlying
lesson: a real break is the onset of a *sustained* shortening trend, not
a single frame that happens to look shorter - a momentary contrast/camera
artifact can look identical to a genuine break in one frame alone. This
prompt asks the model to check ONE already-proposed candidate timestamp
against the frames that come after it (a bounded validation horizon),
rather than accepting the first plausible-looking answer on its own - and
a real, isolated experiment against this exact prompt (one coherent
request, dense frames, ~1s baseline + ~2s horizon around a known-good
candidate) confirmed the true break within the gate-1 tolerance, which is
what motivated retiring the chronological scan around it (see
``PROMPT_END_COARSE_V1`` below). The candidate frame is identified by a
distinguishing frame label ("CANDIDATE frame" vs plain "frame" - see
``gemini_provider._build_parts``/``openai_provider._build_parts``) rather
than a number embedded in the prompt text itself, so this prompt's own
wording stays static and version-pinned like every other one here. Reuses
``schema.TimingVerdict``'s existing degenerate-point shape exactly like
``PROMPT_END_SCAN_V2``/``PROMPT_START_REFINE_V1`` - CONFIRMED means the
candidate is validated (echo its own timestamp back), ABSTAIN means it is
rejected (the trend didn't hold) or there isn't enough evidence to judge.

``PROMPT_END_COARSE_V1`` (superseded by V2 below, kept only as the
historical record of what the first real whole-clip rerun was actually
scored against) was the single whole-clip (post-start), sparsely sampled
request that nominates the one candidate ``PROMPT_END_VALIDATE_V1`` then
checks - together replacing the chronological end-scan entirely
(``pipeline.run_llm_timing`` no longer builds or sends any
``PROMPT_END_SCAN_V2`` window at all). The real experiment referenced
above diagnosed *why* the 13-window chronological scan failed: decomposing
the search into many narrow, isolated windows lost the temporal context a
model needs to judge a sustained trend, and cost far more (787,182 tokens
across 13 windows plus one validation call) than one coherent request
needs. V1 explicitly used the sustained-shortening-onset framing (never
the single-frame "first frame that looks shorter, do not wait for
confirmation" framing ``PROMPT_V1`` still uses for its own end_s, which is
never acted on downstream - see ``pipeline.run_llm_timing``) over sparse,
uniformly-spaced frames spanning from the confirmed start to the end of
the clip, explicitly warning against reporting the stream's final
disappearance instead of the true onset.

``PROMPT_END_COARSE_V2`` is the live end-coarse prompt. The first real
whole-clip rerun against V1 (commit `27587f5`) came back a safe ABSTAIN:
the end-coarse pass nominated an early transient (18.5s, a shortening that
recovered back toward baseline) instead of the true break (~20.5s), and
the single-shot design's trend validation correctly rejected it, with
nothing left to fall back to. Root cause, found directly in V1's own
wording: it told the model to "report your single best candidate even if
you cannot fully confirm a sustained trend from these sparse samples
alone," and to "prefer the EARLIER one" when torn between two nearby
candidates - exactly the bias that picks a transient over a later,
better-supported one. V2 rewrites the MEASUREMENT RULE to use the *whole*
sparse batch before nominating anything: scan every frame that looks
shorter than the baseline, in order; for each, check whether the batch's
own later frames show continued shortening or a recovery; reject any that
recover as transients and keep scanning chronologically; nominate the
EARLIEST candidate whose later sparse checkpoints, still within the same
batch, already support continued shortening; abstain (``no_break_found``)
if every candidate in the batch is a transient or none exists at all. The
"prefer the earlier one when ambiguous" tie-break is removed entirely -
an unsupported early candidate is no longer preferred over a later,
trend-confirmed one; it is rejected outright. No pipeline change was
needed for this fix - the single end-coarse/end-validate call structure
is unchanged, since the whole batch was already visible to the model in
one request; only the instructions for how to use it changed. Reuses
``schema.TimingVerdict``'s existing degenerate-point shape the same way
every prompt below ``PROMPT_END_SCAN_V1`` does. Its own candidate is never
trusted on its own - ``PROMPT_END_VALIDATE_V1`` must still confirm it -
so this prompt only has to localize roughly, not prove anything.
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

PROMPT_END_SCAN_V2_ID = "zahn-efflux-end-scan-v2"

PROMPT_END_SCAN_V2 = """\
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

TIMESTAMP RULE (read this carefully - a real run got this wrong): start_s
and end_s must be COPIED EXACTLY, character-for-character, from one of the
"[frame at t=...s]" labels shown above - never computed, estimated,
rounded, offset from the start of this window, or expressed as an elapsed
duration. For example, if the frames shown are labelled t=20.000s,
t=20.120s, t=20.240s, ... and the break is visible at the frame labelled
t=20.240s, your answer is exactly 20.240 - not 0.240 (elapsed time since
this window began), not a frame index, not any other derived number. If
you are not looking at a specific labelled frame's own timestamp, you do
not have an answer yet - re-check which exact label the break is at,
rather than guess a plausible-sounding number.

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
  "start_s": <float seconds, copied exactly from one of the frame labels
              above - identical to end_s; or null if abstaining>,
  "end_s": <float seconds, copied exactly from one of the frame labels
            above - identical to start_s; or null if abstaining>,
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

PROMPT_START_REFINE_V1_ID = "zahn-efflux-start-refine-v1"

PROMPT_START_REFINE_V1 = """\
You are analyzing one short, consecutive window of frames from a Zahn cup
viscosity test video - not the whole clip. An earlier coarse pass located
an approximate region for the stream's start; you are being asked to
pinpoint the EXACT frame, within THIS window, where the continuous stream
begins. Nothing about the stream's end is being asked here - ignore how
or when the stream eventually breaks; that is determined separately.

Each frame is labelled with its exact timestamp in seconds. Frame selection
was done deterministically by the calling pipeline, not by you; rely only
on the timestamps given.

MEASUREMENT RULE (use exactly this rule, do not invent your own):
- "start" = the first labelled frame in which liquid leaving the outlet
  forms a continuous, uniform stream (not a single drop, not a
  discontinuous dribble still connected to the fluid surface in the
  container).
- If the true start is not actually visible within this window - the
  stream is already continuous in the very first frame shown, or is still
  not continuous by the very last frame shown - report "abstain" with
  reason_codes including "ambiguous_evidence" rather than guess a
  timestamp at the edge of the window.

TIMESTAMP RULE (read this carefully): start_s must be COPIED EXACTLY,
character-for-character, from one of the "[frame at t=...s]" labels shown
above - never computed, estimated, rounded, offset from the start of this
window, or expressed as an elapsed duration.

HAZARDS SPECIFIC TO THIS FOOTAGE:
- The cup, stream and background are often close to the same pale/dusty
  colour (industrial environment) - contrast can be very low. A drop in raw
  pixel brightness alone is not evidence of a start: look for continuity
  along the stream's own axis, not an absolute brightness threshold.
- The footage is handheld - there is camera shake. Do not confuse whole-frame
  motion (camera movement) with real motion of the stream itself.
- If no clear geometric edge is visible and the evidence is genuinely
  ambiguous, abstain rather than guess.

OUTPUT: reply with a single JSON object and nothing else (no prose before or
after it), matching exactly this shape. This search reports a single moment
(the start), not an interval - set start_s and end_s to the SAME timestamp,
and start_uncertainty_s/end_uncertainty_s to the same value:

{
  "status": "confirmed" | "abstain",
  "start_s": <float seconds, copied exactly from one of the frame labels
              above - identical to end_s; or null if abstaining>,
  "end_s": <float seconds, copied exactly from one of the frame labels
            above - identical to start_s; or null if abstaining>,
  "start_uncertainty_s": <float, your own +/- bound on the timestamp>,
  "end_uncertainty_s": <float, the same +/- bound as start_uncertainty_s>,
  "confidence": <float 0..1>,
  "reason_codes": [<short machine-readable strings, e.g. "weak_contrast",
                     "camera_motion", "ambiguous_evidence", "missing_outlet">],
  "evidence_frame_timestamps_s": [<frame timestamps from THIS window you
                                     actually inspected around the start -
                                     immediately before/at/after it>],
  "raw_notes": "<short free-text explanation, for a human audit log only>"
}

If you are not confident, set "status" to "abstain", leave start_s/end_s
null, and explain why in reason_codes. Do not invent a confident-sounding
timestamp when the evidence does not support one.
"""

PROMPT_END_VALIDATE_V1_ID = "zahn-efflux-end-validate-v1"

PROMPT_END_VALIDATE_V1 = """\
You are analyzing frames from a Zahn cup viscosity test video: a short
baseline period, then a specific CANDIDATE break, then a validation
period after it. One frame in this batch is labelled "[CANDIDATE frame at
t=...s]" - all the others are labelled plainly, "[frame at t=...s]". An
earlier pass flagged the CANDIDATE frame as a *possible* break; you are
being asked to VALIDATE or REJECT it, not to find a different one.

Each frame is labelled with its exact timestamp in seconds. Frame
selection was done deterministically by the calling pipeline, not by you;
rely only on the timestamps given.

MEASUREMENT RULE (use exactly this rule, do not invent your own):
- Before the CANDIDATE frame, the connected, outlet-attached stream has an
  established, roughly stable reach - use those frames as the baseline.
- A candidate break is VALIDATED only if the frames after it show a
  clear, SUSTAINED net-shortening trend: checkpoint by checkpoint through
  to the end of this batch, the connected stream's reach keeps getting
  shorter than it was at the candidate frame - not just a single shorter
  frame at the candidate itself.
- A brief re-extension (the reach lengthening slightly at some point after
  the candidate) does not by itself invalidate it, AS LONG AS later frames
  shorten again and reach a point shorter than any point already seen -
  the overall trend must still be net shortening, never just recovering
  back toward the baseline and stopping there.
- If instead the reach returns to (or back toward) the established
  baseline reach and STAYS there through the rest of this batch, rather
  than continuing to shorten, that is a visual/camera/contrast artifact,
  not a real break: REJECT the candidate. Use reason_codes including
  "trend_not_sustained".
- Drops that have already detached and fallen below the connected
  stream's own tip are not part of the stream's reach - judge the
  connected segment only, never any separated droplets below it.
- If this batch simply does not contain enough frames after the candidate
  to judge a sustained trend, or the evidence could genuinely go either
  way, REJECT the candidate for lack of evidence rather than guess. Use
  reason_codes including "insufficient_future_context" (not enough
  frames) or "ambiguous_trend" (frames present, but inconclusive), as
  appropriate.

TIMESTAMP RULE (read this carefully): if validated, start_s and end_s
must be COPIED EXACTLY, character-for-character, from the "[CANDIDATE
frame at t=...s]" label above - the candidate's own timestamp, unchanged.
Never compute, estimate, or substitute a different frame's timestamp,
even a later one that also looks like a break - validating means
confirming THIS candidate, not proposing a new one.

HAZARDS SPECIFIC TO THIS FOOTAGE:
- The cup, stream and background are often close to the same pale/dusty
  colour (industrial environment) - contrast can be very low. A drop in
  raw pixel brightness alone is not evidence of shortening: look for the
  connected stream's own geometric reach, not an absolute brightness
  threshold.
- The footage is handheld - there is camera shake. Do not confuse
  whole-frame motion (camera movement) with real motion of the stream
  itself.
- If no clear geometric edge is visible and the evidence is genuinely
  ambiguous, reject the candidate rather than guess.

OUTPUT: reply with a single JSON object and nothing else (no prose before
or after it), matching exactly this shape. This reports a single moment
(the candidate, if validated), not an interval - set start_s and end_s to
the SAME timestamp, and start_uncertainty_s/end_uncertainty_s to the same
value:

{
  "status": "confirmed" | "abstain",
  "start_s": <float seconds, copied exactly from the CANDIDATE frame label
              above - identical to end_s; or null if rejecting>,
  "end_s": <float seconds, copied exactly from the CANDIDATE frame label
            above - identical to start_s; or null if rejecting>,
  "start_uncertainty_s": <float, your own +/- bound on the timestamp>,
  "end_uncertainty_s": <float, the same +/- bound as start_uncertainty_s>,
  "confidence": <float 0..1>,
  "reason_codes": [<short machine-readable strings, e.g.
                     "trend_not_sustained", "insufficient_future_context",
                     "ambiguous_trend", "weak_contrast", "camera_motion">],
  "evidence_frame_timestamps_s": [<frame timestamps from THIS batch you
                                     actually inspected - baseline and
                                     post-candidate checkpoints alike>],
  "raw_notes": "<short free-text explanation, for a human audit log only>"
}

If you are not confident the trend is sustained, set "status" to
"abstain", leave start_s/end_s null, and explain why in reason_codes. Do
not invent a confident-sounding validation when the evidence does not
support one.
"""

PROMPT_END_COARSE_V1_ID = "zahn-efflux-end-coarse-v1"

PROMPT_END_COARSE_V1 = """\
You are analyzing a Zahn cup viscosity test video. The stream's start has
already been confirmed by an earlier pass; you are given a SPARSE batch of
frames, spread across the rest of the clip, and asked to nominate ONE
candidate location for where the continuous stream first begins a
sustained shortening trend - the beginning of the real break. A later,
much denser pass will re-examine your candidate closely and confirm or
reject it, so your job here is coarse localization, not final proof:
report your single best candidate even if you cannot fully confirm a
sustained trend from these sparse samples alone.

Each frame is labelled with its exact timestamp in seconds. Frame
selection was done deterministically by the calling pipeline, not by you;
rely only on the timestamps given - these are sparse, spread across a
much longer span than a normal video frame rate, so consecutive labelled
frames may be a full second or more apart.

MEASUREMENT RULE (use exactly this rule, do not invent your own):
- The connected, outlet-attached stream has an established, roughly
  stable reach earlier in this batch - use that as the baseline.
- Your candidate is the EARLIEST frame in this batch where the connected
  stream's reach looks genuinely, durably shorter than that baseline -
  not a frame where it happens to look momentarily thinner from noise,
  contrast, or camera shake, and not the point where the stream is
  already gone or reduced to occasional drops. If several later frames
  also look shorter still, that is expected (a real break usually keeps
  progressing) - report the EARLIEST one that already looks like a real,
  ongoing narrowing, not the last one you can see.
- Do NOT report the stream's final disappearance or the point where it
  has already broken into intermittent drops - by then the real break
  happened earlier, at the first frame that already showed a genuine
  narrowing. Look for the ONSET of the narrowing, not its endpoint.
- If nothing in this batch looks like a genuine, ongoing narrowing (the
  stream looks equally full throughout, or you cannot tell), report
  "abstain" with reason_codes including "no_break_found" - do not guess a
  plausible-looking frame just because one must exist.

TIMESTAMP RULE (read this carefully): start_s and end_s must be COPIED
EXACTLY, character-for-character, from one of the "[frame at t=...s]"
labels shown above - never computed, estimated, rounded, or interpolated
between two labelled frames.

HAZARDS SPECIFIC TO THIS FOOTAGE:
- The cup, stream and background are often close to the same pale/dusty
  colour (industrial environment) - contrast can be very low. A drop in
  raw pixel brightness alone is not evidence of narrowing: look for the
  connected stream's own geometric reach, not an absolute brightness
  threshold.
- The footage is handheld - there is camera shake. Do not confuse
  whole-frame motion (camera movement) with real motion of the stream
  itself.
- Because these frames are sparse, a plausible-looking narrowing you see
  in one frame might just be that frame's own noise/contrast, not a real
  trend - the later, dense pass exists specifically to check this, so
  when genuinely torn between two nearby candidates, prefer the EARLIER
  one (a later pass can still reject it; missing the true onset by
  reporting a too-late candidate cannot be corrected downstream).

OUTPUT: reply with a single JSON object and nothing else (no prose before
or after it), matching exactly this shape. This search reports a single
moment (your candidate), not an interval - set start_s and end_s to the
SAME timestamp, and start_uncertainty_s/end_uncertainty_s to the same
value:

{
  "status": "confirmed" | "abstain",
  "start_s": <float seconds, copied exactly from one of the frame labels
              above - identical to end_s; or null if abstaining>,
  "end_s": <float seconds, copied exactly from one of the frame labels
            above - identical to start_s; or null if abstaining>,
  "start_uncertainty_s": <float, your own +/- bound on the timestamp>,
  "end_uncertainty_s": <float, the same +/- bound as start_uncertainty_s>,
  "confidence": <float 0..1>,
  "reason_codes": [<short machine-readable strings, e.g. "weak_contrast",
                     "camera_motion", "no_break_found", "ambiguous_evidence">],
  "evidence_frame_timestamps_s": [<frame timestamps you actually inspected
                                     around your candidate and the baseline
                                     you compared it to>],
  "raw_notes": "<short free-text explanation, for a human audit log only>"
}

If you are not confident, set "status" to "abstain", leave start_s/end_s
null, and explain why in reason_codes. Do not invent a confident-sounding
timestamp when the evidence does not support one.
"""

PROMPT_END_COARSE_V2_ID = "zahn-efflux-end-coarse-v2"

PROMPT_END_COARSE_V2 = """\
You are analyzing a Zahn cup viscosity test video. The stream's start has
already been confirmed by an earlier pass; you are given a SPARSE batch of
frames, spread across the rest of the clip, and asked to nominate ONE
candidate location for where the continuous stream first begins a
SUSTAINED shortening trend - the beginning of the real break. A later,
much denser pass will re-examine your candidate closely and confirm or
reject it, but this pass must still do real work first: use every frame
in this batch to rule out a transient before nominating anything, not
just report the first frame that happens to look shorter and hope the
later pass sorts it out. A candidate this pass never offers cannot be
rescued downstream.

Each frame is labelled with its exact timestamp in seconds. Frame
selection was done deterministically by the calling pipeline, not by you;
rely only on the timestamps given - these are sparse, spread across a
much longer span than a normal video frame rate, so consecutive labelled
frames may be a full second or more apart.

MEASUREMENT RULE (use exactly this rule, do not invent your own):
- The connected, outlet-attached stream has an established, roughly
  stable reach earlier in this batch - use that as the baseline.
- Scan the WHOLE batch, in chronological order, for every frame whose
  reach looks shorter than the baseline - not just the first one you
  notice. For EACH such frame, check the frames that come after it,
  still within this same batch: does the reach stay shorter (or get
  shorter still), or does it recover back toward the baseline and stay
  there?
- A frame that looks shorter but is followed, within this batch, by a
  recovery back toward the baseline reach is a TRANSIENT - a visual/
  camera/contrast artifact, not the real break. REJECT it outright and
  keep scanning later in the batch for the next shorter-looking frame.
  Do not nominate a transient just because it was the first one you saw.
- Your candidate is the EARLIEST frame that looks shorter than the
  baseline AND whose later frames, still within this batch, already
  support continued shortening rather than a recovery - i.e. the
  earliest candidate this batch's own evidence already backs as an
  ongoing trend, never a single shorter-looking frame taken in isolation.
- Do NOT report the stream's final disappearance or the point where it
  has already broken into intermittent drops - by then the real break
  happened earlier, at the first frame that already showed a genuine,
  continuing narrowing. Look for the ONSET of the trend, not its
  endpoint.
- If every shorter-looking frame in this batch turns out to be a
  transient (each one is followed by a recovery), or nothing in this
  batch looks shorter than the baseline at all, report "abstain" with
  reason_codes including "no_break_found" - do not nominate a transient
  just because a real break must exist somewhere in the clip.

TIMESTAMP RULE (read this carefully): start_s and end_s must be COPIED
EXACTLY, character-for-character, from one of the "[frame at t=...s]"
labels shown above - never computed, estimated, rounded, or interpolated
between two labelled frames.

HAZARDS SPECIFIC TO THIS FOOTAGE:
- The cup, stream and background are often close to the same pale/dusty
  colour (industrial environment) - contrast can be very low. A drop in
  raw pixel brightness alone is not evidence of narrowing: look for the
  connected stream's own geometric reach, not an absolute brightness
  threshold.
- The footage is handheld - there is camera shake. Do not confuse
  whole-frame motion (camera movement) with real motion of the stream
  itself.
- Because these frames are sparse, this batch's own later checkpoints are
  your only evidence that a candidate is not just noise - trust that
  evidence over a hunch. Never nominate a candidate whose own later
  frames, still within this batch, already contradict it by recovering
  toward the baseline - and never prefer an earlier, unsupported
  candidate over a later one this batch's own evidence actually backs.

OUTPUT: reply with a single JSON object and nothing else (no prose before
or after it), matching exactly this shape. This search reports a single
moment (your candidate), not an interval - set start_s and end_s to the
SAME timestamp, and start_uncertainty_s/end_uncertainty_s to the same
value:

{
  "status": "confirmed" | "abstain",
  "start_s": <float seconds, copied exactly from one of the frame labels
              above - identical to end_s; or null if abstaining>,
  "end_s": <float seconds, copied exactly from one of the frame labels
            above - identical to start_s; or null if abstaining>,
  "start_uncertainty_s": <float, your own +/- bound on the timestamp>,
  "end_uncertainty_s": <float, the same +/- bound as start_uncertainty_s>,
  "confidence": <float 0..1>,
  "reason_codes": [<short machine-readable strings, e.g. "weak_contrast",
                     "camera_motion", "no_break_found", "ambiguous_evidence">],
  "evidence_frame_timestamps_s": [<frame timestamps you actually inspected
                                     - your candidate, the baseline you
                                     compared it to, and the later
                                     checkpoints that support it>],
  "raw_notes": "<short free-text explanation, for a human audit log only>"
}

If you are not confident, set "status" to "abstain", leave start_s/end_s
null, and explain why in reason_codes. Do not invent a confident-sounding
timestamp when the evidence does not support one.
"""

PROMPTS: dict[str, str] = {
    PROMPT_V1_ID: PROMPT_V1,
    PROMPT_END_SCAN_V1_ID: PROMPT_END_SCAN_V1,
    PROMPT_END_SCAN_V2_ID: PROMPT_END_SCAN_V2,
    PROMPT_START_REFINE_V1_ID: PROMPT_START_REFINE_V1,
    PROMPT_END_VALIDATE_V1_ID: PROMPT_END_VALIDATE_V1,
    PROMPT_END_COARSE_V1_ID: PROMPT_END_COARSE_V1,
    PROMPT_END_COARSE_V2_ID: PROMPT_END_COARSE_V2,
}


def get_prompt(prompt_version: str) -> str:
    try:
        return PROMPTS[prompt_version]
    except KeyError as exc:
        raise KeyError(
            f"Unknown prompt_version {prompt_version!r}; known: {sorted(PROMPTS)}"
        ) from exc
