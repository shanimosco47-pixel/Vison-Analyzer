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
- Designing (not implementing) a multi-engine architecture: one uploaded
  video submitted to several independently-configured engines, with
  results shown separately - see §10. Included as a design/backend
  requirement for this checkpoint; the interaction is not wired into a UI.

Not authorized (yet):

- Wiring a real provider into the main user workflow.
- Any production-default recommendation, before the gate-3 blinded
  evaluation (§5) has actually been run.
- Touching PR #4. It remains the historical record of the classical-CV
  attempt and stays draft/unmerged.
- Building the Settings UI or any other web-layer wiring for multi-engine
  configuration (§10) - the backend contract exists; the interaction
  surface does not yet.

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
| `engine_config.py` - `EngineConfig`, failure-isolated `run_llm_timing_for_engines` | Done (backend only - see §10) |
| `redaction.py` - centralized, bounded sanitization for all provider/model free text | Done - see §11 |
| `pipeline._validate_grounding` - anchors a parsed verdict to the actual video/request | Done - see §11 |
| `pricing.py` + `PipelineOutcome` cost/usage reporting | Done - pricing table empty until a real model is priced, see §11 |
| `gemini_provider.py` - `GeminiTimingProvider`, the first real adapter | Done, unit-tested with an injected fake client - **not called anywhere; no live request exists in this spike** |
| A real vendor provider actually wired to a live key | **Not started - needs an API key, explicitly deferred; `build_default_gemini_client` exists but nothing calls it** |
| The two real hand-verified clips as committed fixtures | **Not available - real footage isn't committed to this repo; see `tests/conftest.py`'s own docstring on why** |
| Adversarial regression fixtures (gate 2) | Not started |
| Blinded 15-20 clip evaluation set (gate 3) | Not started - blocked on both of the above |
| Populated pricing entries for real models | Not started - `PRICING_TABLE` is empty; add an entry once a model is actually being called |
| Prompt A/B: raw-video-with-vendor-code-execution vs. our own deterministic frame batching | Open design question - see §8 |
| Settings UI (engine entries, `+` control, credential fields, connection validation) | **Not started - explicitly deferred until reviewed, see §10** |

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

## 10. Multi-engine architecture (approved, not yet wired into the UI)

Second authorization (PR #4 comment thread): the eventual workflow should
let a user submit one uploaded video to **multiple independently-configured
engines** and see every engine's result separately - not a single "the
answer is X" number. This section is the interaction/configuration contract
requested for this checkpoint. **Nothing below is implemented in the web
layer.** The backend pieces (`engine_config.py`) exist and are tested;
Settings UI, HTTP routes, and any storage for engine configuration do not.

### 10.1 Interaction contract (for the future Settings UI)

- A Settings control opens engine configuration.
- **One engine entry shown by default.** A clear **`+`** control adds
  another entry; each entry can be individually removed.
- Each entry exposes:
  - **Provider / model selection** (which `TimingProvider` implementation,
    which model).
  - **Enabled** toggle - a disabled entry stays configured but is skipped at
    run time (see `EngineRunStatus.SKIPPED`), not deleted.
  - **Credential configuration**, entered and displayed **masked** - the UI
    never shows a credential's actual value once set, only that a reference
    is configured.
  - **Connection validation** - an explicit action to check the configured
    credential/model actually works, before it's relied on for a real run.
  - **Remove**.
- Submitting a video runs it against every *enabled* entry. Results are
  presented **per engine, side by side** - including status
  (confirmed/abstain/error), confidence, uncertainty, latency, and estimated
  cost per engine - specifically so engines disagreeing with each other is
  visible to the user, not hidden.
- **No averaging, no automatic winner.** If two engines disagree, both
  answers are shown; nothing here picks or blends them.
- **One engine failing (bad credential, unreachable, malformed output, an
  unexpected exception) must never prevent the other enabled engines from
  producing their own result.**

### 10.2 Backend contract (`app/analysis/llm_timing/engine_config.py`)

- `EngineConfig` - `engine_id`, `provider_name`, `model_id`,
  `credential_ref`, `enabled`, `display_name`. Validated eagerly
  (`__post_init__`): no empty required fields, and `credential_ref` is
  rejected outright if it looks like an actual key rather than a reference
  (`_looks_like_a_raw_secret` - a heuristic safety net, not a substitute for
  secret scanning elsewhere).
- `run_llm_timing_for_engines(video_path, engines, provider_factory, ...)` -
  runs every enabled engine through `run_llm_timing` independently, each
  inside its own try/except (the isolation boundary the interaction
  contract above requires), and returns `list[EngineOutcome]` in input
  order. Disabled entries become `EngineRunStatus.SKIPPED` without ever
  calling `provider_factory` (so no connection is attempted for a
  configured-but-off engine). A crash becomes `EngineRunStatus.ENGINE_ERROR`
  with a redacted message, never an exception that takes down the other
  engines' results.
- `provider_factory: Callable[[EngineConfig], TimingProvider]` is supplied
  by the caller, not this module - `engine_config.py` stays vendor-free,
  matching the "provider adapters behind the narrow interface" requirement.
  A real factory (not implemented yet) is where `credential_ref` actually
  gets resolved - reading an environment variable or a secret-store entry -
  server-side, at call time.

### 10.3 Secret handling policy

- **Server-side only.** `EngineConfig.credential_ref` is a reference (an
  environment variable name or secret-store key), never a credential value.
  The dataclass has nowhere to put an actual secret.
- **Never returned to or stored by the browser.** `EngineConfig.to_dict()`
  only ever contains the reference, which is safe to log or send to a
  client - there is no code path in this module that touches a resolved
  secret value at all.
- **Never committed, logged, or included in diagnostics.** Enforced in two
  places: `EngineConfig.__post_init__` rejects a `credential_ref` that looks
  like a raw key, and `run_llm_timing_for_engines` redacts anything
  credential-shaped out of an `ENGINE_ERROR`'s message
  (`_redact` / `_RAW_SECRET_SCAN_PATTERNS`) before it's stored on the
  returned `EngineOutcome`.
- Both guards are **heuristic, defense-in-depth** - they catch the easy
  mistake of pasting a real key where a reference belongs, and scrub a
  vendor SDK's exception text if it happens to echo one back. They are not
  a substitute for the real provider implementation raising clean errors in
  the first place, or for secret-scanning elsewhere in the toolchain.

### 10.4 Tests

`tests/test_llm_timing_engines.py` (7 tests): `EngineConfig` validation
(empty fields, raw-secret-shaped `credential_ref` rejected, a plausible env
var reference accepted); a disabled engine is skipped and never reaches
`provider_factory`; one engine erroring does not block sibling engines, and
the failing engine's error is reported without stopping the batch; a
`RuntimeError` containing a secret-shaped substring is redacted before
being stored; two engines confirming genuinely different, disagreeing
answers both survive untouched in the output - proving nothing here
merges or picks a winner.

## 11. Codex review round (commit `76d0ddb`) - findings addressed

A review left directly on PR #5 found five real gaps in the design above.
All five are fixed on this branch; this section is the record of what
changed and why, so the fix is traceable back to the finding that demanded
it.

**1. A verdict wasn't anchored to the actual video/request.** Everything in
§3/§4 validated a verdict's own internal shape - it had no idea whether
`start_s` was inside the video, or whether `evidence_frame_timestamps_s`
corresponded to anything actually sent. A model could return a finite,
internally-consistent, *fabricated* answer and it would parse as CONFIRMED.
Fixed with `pipeline._validate_grounding`, run on both the coarse and fine
verdicts before either is trusted: `start_s`/`end_s` must fall inside the
window whose frames were actually submitted for that pass (the whole video
for the coarse pass, the sized fine window for the fine pass); every
`evidence_frame_timestamps_s` entry must be within a documented, pass-scaled
tolerance of a frame that was actually sent (`ungrounded_evidence`
otherwise); at least one evidence timestamp must sit near each claimed
boundary (`evidence_far_from_claim` otherwise); and a new
`PipelineConfig.max_uncertainty_s` caps how large `start_uncertainty_s`/
`end_uncertainty_s` may be on a CONFIRMED verdict (`uncertainty_exceeds_cap`
otherwise). Every failure converges on ABSTAIN, same discipline as
`parse_raw_response`. 9 new tests in `tests/test_llm_timing_pipeline.py`.

**2. Malformed ABSTAIN payloads could raise instead of abstaining.**
`parse_raw_response`'s "abstain" branch built a `TimingVerdict` outside the
try/except that guarded the "confirmed" branch, so a non-numeric or
NaN/Infinity `confidence` in an abstain payload could raise instead of
producing a clean abstain. Fixed by routing both branches
(`_parse_confirmed`/`_parse_abstain`) through the same try/except, and by
adding explicit `math.isfinite()` checks to every numeric field in
`TimingVerdict.__post_init__` (`start_s`, `end_s`, both uncertainties,
confidence) - a naive range/comparison check silently passes NaN, since any
comparison against NaN is `False`. 12 new parametrized regression tests
covering NaN/Infinity/non-numeric in every field, on both payload shapes.

**3. Secret/error redaction had a bypass.** The per-engine redaction in
`engine_config.py` only ever ran on caught Python exceptions - a provider's
own reported `error` string, and a model's own `raw_notes` field, reached
`TimingVerdict.raw_notes` (and from there `to_dict()`, logs, diagnostics)
unredacted. Fixed by extracting the redaction (and a length bound - "bounded
sanitization", per the finding) into `redaction.py`, and routing every
untrusted free-text value through `sanitize_untrusted_text()` before it
reaches a constructed object: `provider.py`'s `_abstain`/`_parse_confirmed`/
`_parse_abstain`, and `engine_config.py`'s exception-message path (now
importing the shared function instead of keeping its own copy).

**4. `credential_ref` was exposed as a client-safe field.** `EngineConfig.to_dict()`
returned the raw reference (an env var name / secret-store key) - server
configuration a client has no legitimate need to see, and a future
free-text-editable field would let a browser pick which server secret gets
resolved. Fixed: `to_dict()` now returns `credential_configured: bool`
instead of `credential_ref`. §10.1/§10.3 above are written to match: a
future "add credential" UI action is write-only into a server-side secret
store, and the reference it produces is never echoed back to the client.

**5. The evaluation contract lacked cost/token evidence.** Gate 3 needs
per-analysis cost, and nothing surfaced it. Added `pricing.py`
(`PRICING_TABLE_VERSION`, an explicitly empty-until-populated
`PRICING_TABLE`, `estimate_cost_usd` - returns `None`, never a silent `0.0`,
for an unpriced model or missing usage) and extended `PipelineOutcome` with
`total_retries`, `total_latency_s`, `total_tokens`, and
`estimated_cost_usd()`, all surfaced in `to_dict()`. `scripts/llm_timing_eval.py`'s
`ClipResult`/`_summarize` now report per-clip and aggregate model IDs,
provider latency, retries, token usage, and cost - `None`/omitted rather
than fabricated wherever a real provider hasn't reported it yet.

**Then, one real provider adapter.** `gemini_provider.py` adds
`GeminiTimingProvider`, the first concrete `TimingProvider` implementation,
pinned to `gemini-2.5-flash-lite` by default (`DEFAULT_GEMINI_MODEL_ID`,
overridable, never implicit) with a small bounded retry loop. It is built
around an injected `GeminiClient` protocol specifically so it is fully
unit-testable (10 tests in `tests/test_llm_timing_gemini_provider.py`) with
a fake client - no network, no SDK dependency anywhere in the test suite.
The one function that imports the real `google.generativeai` SDK
(`build_default_gemini_client`) does so lazily inside its own body and
reads the API key only from a named environment variable - and **nothing in
this spike calls it**. No live request exists anywhere on this branch.

**Gates re-run after all of the above**: full Python suite (`pytest -q`) -
green; the pre-existing Node suite (`node --test tests_js/*.test.js`) -
green, unaffected (nothing in this round touched the web layer); `ruff
check .` - clean; `ruff format --check .` - clean; `mypy app` - clean
except one finding in `app/web/routes.py` confirmed present on `main`
before this branch existed, unrelated to any of this work.

Still stopped here, per the standing instruction: no live gate, no
production/UI wiring, without further review.
