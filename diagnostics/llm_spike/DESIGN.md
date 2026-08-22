# LLM-assisted efflux timing: feasibility spike

Status: **design + evaluation harness only. Not wired into the detector
registry, the web layer, or any user-facing workflow.** This document exists
to record the design, the evidence gates it must clear, and exactly what is
and isn't implemented yet, so a reviewer doesn't have to read the diff to
find that out.

## 1. Why this branch exists

PR #4 (`claude/zahn-cup-timing-diagnosis-hm5tv4`) spent five structurally
different classical computer-vision mechanisms trying to autonomously track
a Zahn cup's outlet through hand-held video (Lucas-Kanade optical flow +
RANSAC, background-motion compensation, edge/contour silhouette matching,
fixed-geometry multi-band consensus templates) and never got past ~2-2.5%
trusted-frame coverage on the one real clip available. The supervisor's
final instruction on that PR was to stop implementation - PR #4 stays draft,
unmerged, and untouched - because the remaining choices were product-level,
not technical.

Outside that PR, as a side experiment, a general-purpose multimodal model
(ChatGPT, using its own code-execution tool against the real video file) was
given a carefully-specified prompt - explicit start/end criteria, a
coarse-to-fine scanning strategy, and warnings about the footage's specific
hazards (low contrast, camera shake, fading vs. breaking) - and asked to
measure two real clips' efflux time blind. Both answers were independently
verified by hand afterwards to within about 0.1-0.4s of the true value -
noticeably better than any classical mechanism PR #4 tried, and better than
a blind attempt with a different model (Gemini) that used no code execution
and was off by 3.4s.

The supervisor reviewed that finding and authorized this spike (full text in
the PR #4 comment thread) with an explicit scope: **a bounded feasibility
investigation, not a production rollout.** This document and this branch are
that investigation.

## 2. What was authorized, and what wasn't

Authorized:

- Investigate whether a production-callable LLM API (not an interactive
  chat/Code Interpreter session) can measure efflux timing accurately and
  honestly enough to be worth productionising.
- Build this on a clean branch from `main` - not on top of PR #4's failed
  tracking machinery.
- Network dependency and per-analysis cost, for the production-floor videos
  in question, which are not considered sensitive.

Not authorized (yet):

- Wiring a real provider into the main user workflow.
- Any production-default recommendation, before the gate-3 blinded
  evaluation (§5) has actually been run.
- Touching PR #4. It remains the historical record of the classical-CV
  attempt and stays draft/unmerged.

## 3. Architecture

```
                 ┌─────────────────────────────────────────────┐
                 │              run_llm_timing()                │
                 │            (pipeline.py)                     │
                 │                                               │
  video file ──▶ │  1. probe duration/fps (VideoReader)          │
                 │  2. deterministic coarse frame extraction     │
                 │     (every coarse_step_s, whole clip)         │
                 │  3. provider.analyze(coarse batch) ──────────┼──▶ TimingProvider
                 │  4. parse_raw_response -> TimingVerdict        │   (schema.py / provider.py)
                 │     not CONFIRMED?  -> abstain, stop            │
                 │  5. size a fine window per boundary,            │
                 │     widened by the coarse verdict's OWN          │
                 │     uncertainty; too wide -> abstain, stop         │
                 │  6. deterministic fine frame extraction            │
                 │     (every native frame, both windows)               │
                 │  7. provider.analyze(fine batch) ─────────────────────▶ TimingProvider
                 │  8. parse_raw_response -> TimingVerdict                  │
                 │     not CONFIRMED? -> abstain, stop                       │
                 │  9. CONFIRMED -> base_detector.Event                       │
                 │     (status CONFIRMED or REVIEW by confidence)               │
                 └─────────────────────────────────────────────────────────────┘
```

Key design decisions and why:

- **Frame extraction is the pipeline's job, not the provider's.** Every
  frame a provider sees was chosen deterministically by `pipeline.py` via
  `VideoReader.frame_at()` and JPEG-encoded before the call. This is
  reproducible/auditable regardless of vendor, and doesn't depend on a
  vendor's video-ingestion capability (sampling rate, whether it even
  accepts raw video files) being consistent, documented, or even known - a
  real gap this spike's own predecessor experiment ran into with Gemini,
  which silently subsampled video at roughly 1fps with no way to change it.
- **The provider boundary (`TimingProvider.analyze`) is a plain function
  from a frame batch to raw text.** No vendor SDK, no network call,
  anywhere above that seam. `StubTimingProvider` is the only implementation
  that exists in this spike; a real one is a few dozen lines behind the same
  interface (§6).
- **Conversational text is never authoritative.** A provider's raw response
  is parsed and validated (`provider.parse_raw_response`) into a
  `TimingVerdict` (`schema.py`) before anything downstream touches it. Every
  failure mode - a provider-level error, invalid JSON, a missing field, a
  value that fails `TimingVerdict`'s own invariants, confidence below the
  configured floor - converges on the same `ABSTAIN` verdict. Nothing
  fabricates a start/end time to paper over a gap; see `never confidently
  wrong` in the module docstrings, and PR #4's `test_..._never_confidently_
  wrong` tests for the same discipline applied to the classical tracker.
- **Abstain propagates to "no event", not to a lower-confidence guess.** An
  `ABSTAIN` verdict at either pass produces `PipelineOutcome.event = None`.
  A caller is expected to treat that exactly like "the detector found
  nothing" and fall back to the existing assisted/manual workflow - this
  spike does not (and should not) replace that fallback, only supplement it.
- **The fine window is sized from the coarse pass's own reported
  uncertainty**, not a fixed margin, and each boundary (start, end) is
  checked independently against `fine_max_span_s`. A wide window means "the
  coarse pass wasn't sure", which is a legitimate reason to abstain rather
  than fine-scan a huge, low-value frame batch - it is emphatically not
  triggered by start and end simply being far apart in time, which is
  normal for a real efflux measurement (order tens of seconds).

## 4. The structured output contract

Every provider response must parse into this JSON shape (see `prompts.py`'s
`PROMPT_V1` for the exact instruction, and `schema.TimingVerdict` for the
validated Python form):

```json
{
  "status": "confirmed" | "abstain",
  "start_s": <float | null>,
  "end_s": <float | null>,
  "start_uncertainty_s": <float>,
  "end_uncertainty_s": <float>,
  "confidence": <float 0..1>,
  "reason_codes": [<string>, ...],
  "evidence_frame_timestamps_s": [<float>, ...],
  "raw_notes": "<string, audit only, never read programmatically>"
}
```

`reason_codes` is deliberately not a closed enum enforced at parse time
(see `schema.PIPELINE_REASON_CODES` / `MODEL_REASON_CODES` for the
documented vocabulary) - an unrecognised code from the model is kept as
data, not treated as a parse failure, so a new failure mode is visible
rather than silently swallowed.

## 5. Evidence gates (from the supervisor's authorization, verbatim scope)

None of these have been run yet - see §7.

1. **Blind reproduction.** Both already hand-verified real clips, run
   through this pipeline with a real provider, blind (the pipeline must not
   receive the true start/end - only `scripts/llm_timing_eval.py`'s manifest
   does, for scoring afterwards). Each must land within ±0.75s on both
   start and end, with no human correction after seeing the answer.
2. **Adversarial regressions.** New synthetic or real fixtures for: white/
   low-contrast background, gradual fading vs. a true break, flow that
   breaks and resumes, a missing/occluded outlet, and a provider that errors
   or returns malformed output - each must abstain cleanly, not guess.
3. **Blinded evaluation set.** At least 15-20 varied real clips, manually
   labelled, run blind. Report the full error distribution, worst case,
   abstention rate, false-confident rate, latency, and cost per analysis.
   Two clips succeeding (the side experiment that motivated this spike) is
   sufficient to justify continuing the spike - it is explicitly **not**
   sufficient to recommend a production default.

`scripts/llm_timing_eval.py` computes exactly the metrics gate 1 and gate 3
ask for (per-clip start/end/duration error, abstention, false-confident,
latency) from a JSON manifest of labelled clips - see its docstring for the
format. It runs today only against `--provider stub-perfect`, a harness
self-test that answers with the manifest's own ground truth; the report it
prints says so loudly so its numbers can't be mistaken for gate evidence.

## 6. What's implemented vs. deferred

| Piece | Status |
|---|---|
| `schema.py` - strict `TimingVerdict` contract, all invariants | Done |
| `provider.py` - `TimingProvider` seam, `parse_raw_response`, `StubTimingProvider` | Done |
| `prompts.py` - versioned `PROMPT_V1` (pinned, JSON-only, criteria-first) | Done |
| `pipeline.py` - coarse-to-fine orchestration, abstain propagation | Done |
| `scripts/llm_timing_eval.py` - gate 1/3 metrics harness | Done (stub-only) |
| Unit/integration tests against `StubTimingProvider` + synthetic fixtures | Done (19 tests, `tests/test_llm_timing_pipeline.py`) |
| A real vendor provider (OpenAI / Anthropic / Google, behind `TimingProvider`) | **Not started - needs an API key, explicitly deferred** |
| The two real hand-verified clips as committed fixtures | **Not available - real footage isn't committed to this repo; see `tests/conftest.py`'s own docstring on why** |
| Adversarial regression fixtures (gate 2) | Not started |
| Blinded 15-20 clip evaluation set (gate 3) | Not started - blocked on both of the above |
| Cost/token accounting in `RawProviderResponse` | Fields exist (`prompt_tokens`, `completion_tokens`) but nothing populates them yet - depends on a real provider |
| Prompt A/B: raw-video-with-vendor-code-execution vs. our own deterministic frame batching | Open design question - see §8 |

## 7. Immediate next step (needs supervisor input first)

Per the authorization, this branch stops here for review before a real
provider is wired in. The next concrete steps, in order, once reviewed:

1. Supply an API key (as an environment variable, never committed - see
   `config.example.env` for this repo's existing convention) for one
   vendor to start with.
2. Implement one concrete `TimingProvider` behind the existing interface -
   no change to `pipeline.py`, `schema.py`, or the tests should be needed.
3. Obtain the two real clips (or equivalents) as local files to run gate 1
   for real, not blind-in-name-only against a harness self-test.
4. Build gate 2's adversarial fixtures (most can reuse `tests/conftest.py`'s
   existing synthetic-video generation patterns).
5. Only after 1-4: assemble and run the gate-3 blinded set.

## 8. Open design question worth flagging now

The side experiment that motivated this spike gave the model the *entire
raw video file* and let it use its own code-execution tool
(`cv2.VideoCapture` inside a sandboxed Python session) to do its own
coarse-to-fine frame extraction internally. This spike instead does that
extraction ourselves and sends discrete timestamped frames (§3) - more
auditable, reproducible, and provider-agnostic (works with any vision API,
not just ones offering code-execution-with-file-access), but it has not
been verified empirically to reproduce the same accuracy the raw-video
approach demonstrated. That comparison is worth running explicitly once a
real provider exists, rather than assumed either way.

## 9. Operational considerations accepted by the user, documented anyway

- **Network dependency**: every analysis requires a live call to an external
  API. No offline fallback beyond the existing manual/assisted workflow.
- **Per-analysis cost**: real money per video, scaling with frame count and
  model choice. Not yet measured - `RawProviderResponse` has the fields to
  track it once a real provider is wired in.
- **Data leaves the premises**: production-floor footage is sent to a
  third-party API. The user has stated these specific videos are not
  considered sensitive; this should not be treated as a blanket policy for
  all future footage without asking again.
- **Latency**: a full coarse+fine analysis is at least two sequential API
  round-trips, plausibly several seconds to tens of seconds depending on
  vendor and frame count - unmeasured until a real provider exists.
