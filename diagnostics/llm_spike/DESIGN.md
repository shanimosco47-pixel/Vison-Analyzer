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
| `pricing.py` + `PipelineOutcome` cost/usage reporting | Done - `gemini-2.5-flash-lite` priced (unofficial-source caveat), see §12 |
| `gemini_provider.py` - `GeminiTimingProvider`, built on the GA `google-genai` SDK | Done, unit-tested with an injected fake client - **not called anywhere; no live request exists in this spike** |
| Deterministic frame resize + request-size/frame-count budget | Done - see §12 |
| Retry policy (transient-only, bounded backoff, correct retry accounting) | Done - see §12 |
| A real vendor provider actually wired to a live key | **Not started - needs an API key, explicitly deferred; `build_default_gemini_client` exists but nothing calls it** |
| The two real hand-verified clips as committed fixtures | **Not available - real footage isn't committed to this repo; see `tests/conftest.py`'s own docstring on why** |
| Adversarial regression fixtures (gate 2) | Not started |
| Blinded 15-20 clip evaluation set (gate 3) | Not started - blocked on both of the above |
| Official-source verification of the Gemini pricing entry | Not started - see §12's sourcing caveat |
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

## 12. Codex re-review round (commit `2cdcff6`) - three live-gate blockers addressed

A second review pass, after §11's fixes, found the design was sound but not
yet safe to point at a live key. Three findings, all fixed on this branch.

**1. Deprecated SDK.** `build_default_gemini_client` imported
`google.generativeai`, which Google has deprecated in favour of the GA
`google-genai` package. Rewritten against `from google import genai` /
`google.genai.types` / `client.models.generate_content(...)`, translating
this module's vendor-neutral `parts: list[dict]` shape into
`types.Part.from_text`/`types.Part.from_bytes` only inside the real client
wrapper - `GeminiTimingProvider` and its tests are unaffected, since they
only ever see the generic dict shape. `requirements-llm-spike.txt` (new,
not part of `requirements.txt`) documents the dependency and why it's
optional - nothing in the test suite imports it, and `build_default_gemini_client`
still isn't called anywhere.

**2. No request-size/frame-count budget.** The fine pass could send every
native-fps full-resolution JPEG across both boundary windows, base64
included - on a 1080p/30fps clip, comfortably over Gemini's ~20MB inline
limit. Fixed with three pieces, all in `pipeline.py`:
- `_resize_for_encoding`: every frame is downscaled (never upscaled) to
  `PipelineConfig.max_frame_dimension_px` (default 768px longer side)
  before encoding - deterministic, provider-agnostic.
- `_min_frames_for_window`: derives a documented sampling floor from
  `PipelineConfig.target_tolerance_s` (default 0.75s, matching gate 1) - at
  least one sample per half-tolerance, so the fine pass never samples
  sparser than the gate can tolerate.
- `_fit_frames_to_budget`: deterministically thins an already-extracted
  frame list until the estimated serialized size
  (`_estimated_request_bytes`, base64-inflated) fits
  `PipelineConfig.max_request_bytes` (default 18MB, under Gemini's limit
  with margin) - never below the floor above. If even the floor doesn't
  fit, the pass aborts with `request_too_large` **before any provider
  call is made** - never a silently-thinned request below the gate's
  precision floor, and never an oversized one sent anyway.
- Grounding's tolerance (`_effective_tolerance_s`) now derives from the
  actual gaps between whichever frames were actually selected after
  thinning, not the nominal pre-thinning step - so a legitimately-cited
  frame isn't rejected purely because thinning coarsened the achieved
  density.

Applied to both passes (the coarse pass gets a simpler fixed floor of 4
frames - it only has to locate an approximate region, not hit gate-level
precision). 16 new tests (`tests/test_llm_timing_frame_budget.py`): resize
bounds, budget estimation, deterministic thinning (including a
determinism check - same input, same output), the floor-unsatisfiable
case, and two pipeline-level integration tests proving real thinning and a
real `request_too_large` abstain both actually fire.

**3. Two uncertainty/retry problems:**
- `PipelineConfig.max_uncertainty_s` defaulted to 5.0s against a ±0.75s
  acceptance target - a verdict admitting multi-second endpoint
  uncertainty could still emit a plain CONFIRMED `Event`. Lowered to
  **0.5s**, documented as itself uncalibrated until gate 3's blinded set
  says otherwise (§ docstring on the field).
- `RawProviderResponse.retries` counted total attempts, not retries after
  the initial one (`max_retries=0` on an immediate failure reported
  `retries=1`). Fixed in `GeminiTimingProvider.analyze`'s retry loop -
  `retries=0` now means exactly what it says. Additionally: retries are
  now only attempted for `provider.TransientProviderError` (or a bare
  `TimeoutError`/`ConnectionError`) - `provider.PermanentProviderError`
  and any exception the adapter can't classify are treated as
  non-retryable by default (retrying an auth failure or a request Gemini
  already rejected as too large wastes time and money for no chance of a
  different outcome), with exponential backoff
  (`backoff_base_s * 2**attempt`, capped at `backoff_max_s`) via an
  injectable `sleep_fn` so tests never actually sleep. `build_default_gemini_client`'s
  real wrapper classifies `google.genai.errors.ClientError` (4xx) as
  permanent and `ServerError` (5xx) as transient. 9 new/rewritten tests in
  `tests/test_llm_timing_gemini_provider.py`.

**Gemini pricing entry.** `PRICING_TABLE["gemini-2.5-flash-lite"]` is now
populated ($0.10/1M input, $0.40/1M output tokens) instead of empty.
**Sourcing caveat, stated plainly**: the official page
(`ai.google.dev/gemini-api/docs/pricing`) was unreachable from this
sandbox (network egress blocked); the figure is corroborated across
several independent third-party pricing aggregators as of August 2026, not
read directly from Google, and is commented as such in `pricing.py` -
re-verify against the official source before it's relied on for any real
billing decision. **Separately, and worth the supervisor's attention
regardless of pricing**: multiple of those same third-party sources
appeared to report a retirement date for `gemini-2.5-flash-lite`. **This
claim was wrong** - see §13, finding 3, for the correction after the
supervisor read the official page directly.

**Gates re-run after all of the above**: `pytest -q` - green (full suite,
80 tests total across this branch's spike test files); `node --test
tests_js/*.test.js` - green, unaffected; `ruff check .` - clean; `ruff
format --check .` - clean; `mypy app` - clean except the same
pre-existing, unrelated `app/web/routes.py` finding noted in §11.

Still stopped here: no live gate, no production/UI wiring, without further
review.

## 13. Codex re-review round (commit `70f0b7c`) - per-window coverage, 429 retry, pricing correction

A third review pass, after §12's fixes. Three findings, all fixed on this
branch.

**1. Fine-pass frames were merged across the two boundary windows before
thinning and before computing grounding tolerance - the biggest of the
three findings.** `run_llm_timing`'s fine pass previously built one
combined, sorted set of native-fps timestamps spanning *both* the start
window and the end window, thinned that single list against the shared
byte budget, and computed one shared `_effective_tolerance_s` across all
of it. For a real clip the two windows are typically many seconds apart
(the zahn_video fixture's are ~14.5s apart: start ~[2.5s, 5.5s], end
~[20.0s, 23.0s]) - `_effective_tolerance_s` widens to the *largest gap* in
its input, so that huge inter-window gap, not either window's own achieved
sampling density, dominated the tolerance used for every grounding check.
Two concrete consequences, both demonstrated by new tests: a timestamp
that was *never actually sent to the provider at all* (sitting in the dead
zone between the two windows) could "match a submitted frame" purely
because it fell within the inflated tolerance of a real frame at the edge
of one window, and evidence legitimately drawn from one window could
"ground" the *other* window's claim.

Fixed by treating the two windows as fully independent from extraction
through grounding, never merged:
- `_dense_timestamps(lo, hi, step_s)`: native-fps timestamps for one
  window only (the old inline set-comprehension, extracted so it can be
  called once per window).
- `_fit_fine_windows_to_budget(start_frames, end_frames, ...)`: thins each
  window against its own half of `PipelineConfig.max_request_bytes`,
  independently - a window with larger or more numerous frames (a busier
  scene, a higher-detail boundary) can no longer crowd out the other
  window's density during thinning. Either half failing to fit its own
  floor is reported separately in the resulting `request_too_large`
  abstain's `raw_notes` (which window, how many frames, what floor).
- `_GroundingRegion` (new frozen dataclass: `bounds`,
  `submitted_timestamps_s`, `time_tolerance_s`) and a generalized
  `_validate_grounding(verdict, *, start_region, end_region,
  max_uncertainty_s)`: each boundary's "does this evidence correspond to a
  frame actually sent" and "is there evidence near this boundary's claim"
  checks now use *only that boundary's own* submitted frames and its own
  `_effective_tolerance_s` (computed from that window's frames alone,
  never the other window's). Evidence must match a region *and* be near
  that region's own claim to ground it - evidence from the end window
  cannot ground the start claim even if it happens to be numerically
  closer to `start_s` than to `end_s`. The coarse pass (one shared frame
  batch, not yet split into windows) passes the same region for both
  `start_region` and `end_region` - its behaviour is unchanged.
- `_min_coarse_frames(duration_s, fine_margin_s)` replaces the previous
  flat coarse floor of 4 frames. The fine window built around the coarse
  estimate only covers +/- `fine_margin_s` around it, so if the coarse
  pass's own sampling (after budget-driven thinning) has gaps wider than
  `fine_margin_s`, the true transition could fall in a gap the coarse pass
  never saw closely enough to estimate, and the resulting fine window
  could miss it entirely. The new floor
  (`max(4, ceil(duration_s / fine_margin_s) + 1)`) guarantees some coarse
  sample lands within `fine_margin_s` of the true transition wherever it
  is, at the cost of needing more bytes for a long clip's coarse pass -
  reflected in `_oversized_abstain`'s per-pass detail message if the
  budget can't support it.

7 new tests in `tests/test_llm_timing_frame_budget.py`: `_min_coarse_frames`
scaling with duration/`fine_margin_s`; `_fit_fine_windows_to_budget`
retaining each window's own floor independently of the other window's
frame size, and reporting each window's failure separately; three
`_validate_grounding`/`_GroundingRegion` unit tests reproducing the exact
fabricated-evidence-in-the-gap and evidence-from-the-wrong-window failure
modes described above (and confirming a genuinely well-grounded verdict
still confirms); and one full pipeline-level integration test
(`test_pipeline_abstains_when_fine_evidence_only_exists_in_the_gap_between_windows`)
reproducing the same fabricated-evidence scenario end-to-end against the
real, widely-separated zahn_video fine windows.

Existing `tests/test_llm_timing_frame_budget.py` byte-budget integration
tests were recalibrated against the new coarse floor: the 26s zahn_video
fixture's coarse floor is now ~19 frames (~163KB) rather than the old flat
4 frames (~25KB), which made the previous "coarse fits, fine doesn't"
budget value (50KB) no longer reachable - that scenario now needs a
tighter `target_tolerance_s` to keep each fine window's own floor above
the coarse floor's byte requirement (see the updated test's comment for
the exact figures probed against the real fixture).

**2. Gemini `ClientError` 408/429 were treated as unconditionally
permanent (never retried).** A 429 (rate limited) or 408 (timeout) is
exactly the shape of failure a bounded retry is *for* - as a `ClientError`
(4xx), the previous adapter code retried nothing in that family, wasting
otherwise-recoverable calls. Fixed:
- `TransientProviderError` gained an optional `retry_after_s` field - a
  raiser can supply a server-suggested delay (e.g. from a 429's
  `Retry-After` header) that overrides `GeminiTimingProvider`'s own
  computed exponential backoff for the next attempt.
- `_classify_client_error(exc)` (new, standalone function in
  `gemini_provider.py`): 408/429 -> `TransientProviderError` (carrying
  `retry_after_s` when extractable), every other 4xx -> unchanged
  `PermanentProviderError`. Extracted as a standalone function
  specifically so the classification *policy* is unit-testable with a
  fake exception object, independent of whether the real `google-genai`
  SDK is installed.
- `_client_error_status_code`/`_client_error_retry_after_s`: best-effort
  attribute extraction (`code`/`status_code`; `retry_after` or a
  `response.headers["Retry-After"]`), each explicitly marked
  **unverified against live SDK documentation** in its own docstring -
  this sandbox still cannot reach `ai.google.dev` to confirm the exact
  attribute names the real SDK's exception objects use. Falls back to
  `None` (never guesses a number) when nothing matches, which
  `_classify_client_error` and the retry loop both treat safely (no
  status code -> permanent; no retry-after -> the existing exponential
  backoff).

10 new tests in `tests/test_llm_timing_gemini_provider.py`: 429/408
classify transient, 400/401/403/413/422/500 (and an unrecognised bare
exception) classify permanent, status-code extraction from either
attribute name, `retry_after_s` extraction from a direct attribute and
from a response-header mapping, and an adapter-level integration test
confirming `GeminiTimingProvider.analyze` actually sleeps the
server-suggested delay (not its own backoff guess) when a raised
`TransientProviderError` carries one.

**3. Pricing comment stated a false model-retirement claim as fact.**
§12 flagged - correctly, as an unverified claim needing confirmation -
that several third-party pricing aggregators appeared to report Google
retiring `gemini-2.5-flash-lite` on 2026-10-16, and surfaced this as an
operational risk worth the supervisor's attention. The supervisor's
review, reading the official pages directly
(`ai.google.dev/gemini-api/docs/pricing`,
`ai.google.dev/gemini-api/docs/deprecations`) - something this sandbox
still cannot do (network egress to `ai.google.dev` remains blocked, tried
again this round) - reported that claim is **wrong**: no shutdown or
retirement date is listed for this model on the official deprecations
page, likely confusion with a different preview/model on the aggregator
side. `pricing.py`'s sourcing comment has been rewritten to remove the
false claim entirely and instead: confirm the $0.10/$0.40 per-1M-token
figures against the official pricing page (per the same review), state
plainly that this module still can't fetch that page itself and is
relying on the reviewer's direct reading of it, and note there is no
listed retirement date - while still flagging that this is worth a
periodic recheck by whoever *can* reach the official pages, not a
permanently-settled fact either way. This module continuing to be unable
to verify its own sourcing claims independently is itself worth carrying
forward as a standing operational note, not just a one-time correction.

**Gates re-run after all of the above**: `pytest -q` - green (full suite);
`node --test tests_js/*.test.js` - green, unaffected; `ruff check .` -
clean; `ruff format --check .` - clean; `mypy app` - clean except the same
pre-existing, unrelated `app/web/routes.py` finding noted in §11 and §12.

Still stopped here: no live gate, no production/UI wiring, without further
review. Per the supervisor's round-3 instruction: report the new commit
and exact tests, then await explicit authorization before the controlled
real-clip/API-key gate-1 run.

## 14. First real gate-1 runs (four models) and a chronological end-scan experiment

Gate 1 was authorized and run for real, locally, by the operator - a real
clip, a real API key, never in this sandbox. Results, reported back via PR
comment (no credentials, clip, or reference values ever posted to GitHub):

- **`gemini-2.5-flash-lite`** (round-3 default): coarse call rejected with
  a 401 `invalid_api_key` on the first attempt - a masked-credential
  redaction gap (`prefix********suffix` slipping past both existing
  patterns) was found and fixed (`11f4a8f`, §"masked credential" fix,
  committed between this section and §13) before any accuracy result was
  produced.
- **`gemini-3.5-flash-lite`**: completed, but **false-confident** - end
  error +5.1s (215,671 tokens).
- **`gemini-3.5-flash`**: safe **abstain** (`ambiguous_evidence`,
  `no_break_found`) - the fine pass's end window, centered on the coarse
  pass's own (wrong) estimate, left the true break in a gap it never saw
  (244.3s latency, 232,427 tokens).
- **`gpt-4o-mini`** (after the OpenAI adapter, §"Add OpenAI adapter" commit
  `69799e2`): coarse completed, but the fine request exceeded the model's
  128K context window.
- **`gpt-4.1-mini`**: safe **abstain** (`out_of_bounds`) - the coarse pass
  selected a late end estimate (~26s), so the fine end window was
  `[25.500, 27.031]` and structurally could not contain the true first
  break (~20.5s). 109,303 tokens, ~40.9s latency.

**Diagnosis, shared across all four models**: the coarse pass's own
end-of-stream judgement gravitates toward final disappearance/late
thinning near the end of the clip, not the first genuine break of the
continuous stream - this repro'd the exact ambiguity `PROMPT_V1`'s
MEASUREMENT RULE already tries (and fails) to rule out by wording alone
("stop at the first genuine shortening event"). Once the coarse estimate
lands late, the existing architecture's single narrow fine window (coarse
estimate ± `fine_margin_s`) cannot recover regardless of model quality -
the true break is simply never in the frames sent.

**Experiment authorized and implemented** (this round): start detection is
completely unchanged (the existing coarse pass + combined start/end fine
call, byte budgets, dual-region grounding - none of that code was touched).
Its `end_s` is discarded. The end is instead searched for **chronologically**,
in bounded, overlapping windows (`_end_scan_windows` in `pipeline.py`) moving
forward from the confirmed `start_s` to the end of the clip:

- Each window reuses the exact frame-density/byte-budget machinery the fine
  pass already used (`_dense_timestamps`, `_extract_frames`,
  `_fit_frames_to_budget`, `_min_frames_for_window`).
- Each window's provider call uses a new prompt, `PROMPT_END_SCAN_V1`,
  asking a narrower question ("does the stream break for the first time
  within *this* window") and reusing `schema.TimingVerdict`'s existing
  shape unchanged by reporting a break as a degenerate point
  (`start_s == end_s == the break's timestamp`) rather than an interval -
  no schema change was needed.
- Each window is graded with the *same* `_validate_grounding`/
  `_GroundingRegion` machinery the coarse pass already uses for a single
  shared region (`start_region == end_region == this window`), so evidence
  for a candidate can only come from frames immediately before/at/after it,
  within that window's own achieved density - not the wide, model-judged
  window the previous approach relied on.
- The scan **stops at the first grounded CONFIRMED window** - a later
  window (e.g. one that would show the stream resuming, thinning further,
  or finally disappearing) is never even requested once an earlier one has
  been confirmed. If every window from the confirmed start to the end of
  the clip abstains, the whole run abstains with `no_break_found` - never a
  fabricated end.
- `PipelineOutcome` gained `end_scan_responses: tuple[RawProviderResponse, ...]`
  (every scan-window call, including the winner) so every pass's
  cost/latency/retries stays individually inspectable - `total_retries`,
  `total_latency_s`, `total_tokens`, and `estimated_cost_usd()` all fold
  scan calls in, and `to_dict()`/the harness's console+JSON output report
  `end_scan_window_count` alongside the existing coarse/fine fields.
- `scripts/llm_timing_eval.py`'s `--provider stub-perfect` self-test was
  updated to be end-scan-aware (branching on `pass_name` the same way the
  real adapters' test doubles now do) - its old "answer every pass
  identically" shape would never ground a scan window (a window's bounds
  check requires both the echoed `start_s` and `end_s` to fall inside that
  one window, which two boundaries seconds apart never both do), so the
  self-test would have silently started reporting ABSTAIN on every run.

Two new synthetic regressions in `test_llm_timing_pipeline.py`, both
against the existing `zahn_video` fixture with scripted per-window
`StubTimingProvider` responses (no real clip needed): a stream that breaks,
then would show a plausible-looking "resumed flow" break further along -
confirms the *first* break and proves the later window is never even
requested; and a stream that never breaks - scans to the end of the clip
and abstains `no_break_found`. Every existing pipeline/engine/pricing/eval-
wiring test whose stub answered identically regardless of `pass_name` was
updated to be end-scan-aware in the same way (`_stub_matching_truth`,
`_perfect_provider`, `_confirms_at`, `_respond_confirming_every_pass`) -
tests that abort before the fine pass confirms (grounding failures,
oversized-budget aborts, coarse-level rejections) were unaffected, since
the end-scan phase is never reached in those cases.

**Gates**: `pytest -q` - green (451 tests total across the full repo);
`ruff check .` / `ruff format --check .` - clean; `mypy app` - clean except
the same pre-existing, unrelated `app/web/routes.py` finding noted in §11;
`node --test tests_js/*.test.js` - green, unaffected. `--provider
stub-perfect` re-smoke-tested end-to-end against the synthetic fixture
after the fix above.

This is explicitly an **experiment**, not a production-default
recommendation - gate 1 has not yet been rerun against a real clip with
this change (no real clip or API key exists in this sandbox; the operator
reruns it locally with `--provider openai --model-id gpt-4.1-mini` first,
per the authorization). Still stopped here: no UI, no additional
providers, no Gemini-side tuning, no unrelated cleanup. PR #5 stays draft.

## 15. Two rerun blockers found and fixed: an OpenAI timestamp contract gap, and a discarded-but-still-blocking fine end_s

Two more real reruns against `gpt-4.1-mini`, each exposing one more
concrete gap before any accuracy evidence could be produced:

**Rerun 1** (commit `d52ac27`): the chronological scan reached the
*correct* `[20.000, 23.000]s` window (true break ~20.5s) - the end-scan
architecture worked - but the model returned `start_s=end_s=5.533`, a
value matching no frame actually shown. Grounding safely rejected it
(`out_of_bounds`), but the call was wasted (40.7s latency, 134,152
tokens). Fixed two ways, both scoped to the end-scan phase only (commit
`e835d0a`): `openai_provider._response_schema_for_pass` constrains the
OpenAI Structured Outputs schema so `start_s`/`end_s` for an `end_scan`
call can only be one of the timestamps actually submitted for that window
(`anyOf: [{type: number, enum: [...]}, {type: null}]` - `strict: True`
Structured Outputs validates this before the response is ever returned,
so the failure becomes structurally impossible to receive back); and
`PROMPT_END_SCAN_V2` (a new prompt version - never mutate one in place)
adds an explicit "TIMESTAMP RULE" instructing the model to copy a shown
label verbatim, as defense-in-depth for a provider (Gemini) without the
same schema guarantee.

**Rerun 2** (commit `e835d0a`): a *different* blocker, before the
end-scan even got to run. Safe `ABSTAIN`/`out_of_bounds` - the combined
start+end fine pass returned `end_s=5.933` for its own stale end window
(`[25.500, 27.031]`, built from the coarse pass's own wrong end estimate).
That `end_s` was never actually used downstream (the end-scan phase
already ignored it, per §14) - but `_validate_grounding` requires *both*
claimed boundaries to ground before confirming anything, so an invalid
answer to a question nobody needed could still abstain the whole run
before the end-scan ever started.

Fixed by removing the end window from the fine pass entirely, not just
ignoring its answer: the fine pass now sends **only** the start window
(the preferred fix per the supervisor's authorization - "a bounded
start-only fine request/contract") via a new prompt, `PROMPT_START_REFINE_V1`,
asking a single question ("pinpoint the exact start frame in this
window") and reusing `schema.TimingVerdict`'s shape the same way
`PROMPT_END_SCAN_V2` does (`start_s == end_s`, a degenerate point).
Grounding is a single shared region (`start_region == end_region ==` the
one start window), the same pattern the coarse pass and each end-scan
window already use - no changes to `_validate_grounding`/
`_GroundingRegion` themselves. The now-unused dual-window budget-splitting
path (`_fit_fine_windows_to_budget`) was left in place rather than
removed, to stay inside the requested scope - it is fully dead code as of
this round, worth removing in a future pass if the supervisor wants that.
The start window now also gets the *full* per-request byte budget
(previously halved to share with the discarded end window), a strict
improvement in achievable start-window density, not a tradeoff.

Every existing stub whose "fine" response answered with a distinct
`(start_s, end_s)` pair was updated to the new degenerate-point contract;
one test whose exact scenario (evidence in the gap *between two merged
fine windows*) became structurally impossible under the new one-window
design was rewritten to cover the same underlying discipline (fabricated
evidence must never ground a claim) against the new shape instead of
deleted. One new regression
(`test_pipeline_start_confirmation_is_independent_of_a_stale_coarse_end_estimate`)
reproduces the real failure's shape directly: the coarse pass reports a
wildly wrong end estimate (mimicking the stale `[25.500, 27.031]s`
window), and the test proves the fine request never even attempts to
cover anything near it, start still confirms, and the end-scan still
finds the true first break independent of the bad coarse signal.

**Gates**: `pytest -q` - green (457 tests total across the full repo);
`ruff check .` / `ruff format --check .` - clean; `mypy app` - clean
except the same pre-existing, unrelated `app/web/routes.py` finding noted
in §11.

Still an experiment, not a production-default recommendation. No UI, no
additional providers, no unrelated cleanup beyond what these two fixes
directly required. PR #5 stays draft - ready for the next rerun.

## 16. Third rerun blocker: the end-scan's own first window could see the onset transition and misread it as a break

A third real rerun against `gpt-4.1-mini` (base commit `894dfe6`) got past
both prior blockers - `start_s` CONFIRMED at ~4.033s (correct, +0.133s) -
but then falsely CONFIRMED `end_s` at ~4.666s from the *first* end-scan
window, `[4.03, 7.03]s` (true end ~20.5s, error -15.834s). That window's
own `reason_codes` named both `stream_start_visible` and
`first_break_in_window`: the scan's first window began at the locked
start itself, so it still showed the onset transition (nothing visible ->
stream visible), and the model misread that transition as the break it
was asked to find.

Fixed structurally, not by prompt wording alone (commit `594889f`): the
end-scan now begins no earlier than `start_hi` - the far edge of the
grounded start-refinement window - rather than at `locked_start_s`
itself. Concretely, `pipeline.run_llm_timing` computes
`scan_from_s = min(max(start_hi, locked_start_s), duration_s)` and passes
that (not `locked_start_s`) to `_end_scan_windows`. Because grounding
already guarantees `start_lo <= locked_start_s <= start_hi`, this can
only ever push the scan's starting point later, never earlier - a
continuous-stream baseline has already been established by the time any
end-scan window's frames are extracted, so no onset/pre-flow frame is
ever eligible to be submitted as a candidate end timestamp at all. This
is a request-construction change, not a grounding-relaxation: the enum
constraint and every existing `_validate_grounding` check are unchanged.

One new regression
(`test_pipeline_end_scan_never_considers_the_onset_transition_as_a_candidate_break`)
reproduces the real failure's shape: a stub willing to falsely confirm a
break at a timestamp ~0.63s after the true start (mirroring the real
run's ~4.666s) is proven unreachable - no end-scan window's submitted
frames ever include that timestamp - while the true, later break is still
found and confirmed correctly.

**Gates**: `pytest -q` - green (458 tests total across the full repo);
`ruff check .` / `ruff format --check .` - clean; `mypy app` - clean
except the same pre-existing, unrelated `app/web/routes.py` finding noted
in §11; `node --test tests_js/*.test.js` - unaffected (Python-only
change), green.

Still an experiment. No UI, no additional providers, no unrelated
cleanup. PR #5 stays draft - ready for the next rerun.

## 17. Trend validation: a break is a sustained shortening trend, not a single frame

The onset fix above (§16) closed one specific failure - the scan's own
first window seeing the onset transition - but the supervisor's own
follow-up authorization generalizes the lesson it exposed: a single
window's local judgement is not enough on its own for *any* candidate,
anywhere in the scan, because a momentary contrast/camera artifact can
look exactly like a genuine break in one frame, wherever it happens to
occur. The fix: a chronological end-scan window CONFIRMING is now only a
*candidate*; before the scan stops, every candidate must also pass a
bounded trend-validation follow-up request (commit `22e1605`).

**Contract** (verbatim from the authorization): before a candidate `T`,
at least ~1s of frames must show an already-established, continuous,
outlet-connected stream at a roughly stable reach (the baseline); `T`
itself is the first clear shortening away from that baseline; frames
after `T` (a ~2s validation horizon) are *mandatory* evidence, not
optional - the connected reach must show sustained net shortening,
checkpoint by checkpoint, through to the end of the horizon; a brief
re-extension is tolerated only if later frames shorten again past the
deepest point already seen; shortening that instead recovers back to the
baseline and stays there is a visual/camera/contrast artifact and must be
rejected; drops already detached from the connected stream's tip don't
count as stream length; insufficient future context or a genuinely
ambiguous trend must abstain rather than guess; and when validated, the
pipeline must report `T` itself - the first onset frame - never a later
point the validation call's own answer might name.

**Implementation** (`pipeline.py`, all in `run_llm_timing`'s end-scan
loop plus the new `_build_validation_frames` helper):

- Every CONFIRMED end-scan window becomes `candidate_ts =
  window_verdict.end_s`, not an immediate `end_candidate`. A validation
  window `[max(locked_start_s, candidate_ts - end_validation_baseline_s),
  min(duration_s, candidate_ts + end_validation_horizon_s)]` (new
  `PipelineConfig` fields, default 1.0s/2.0s per the authorization's own
  floors) is built, densely sampled the same way every other pass is, with
  the candidate's own frame timestamp always included (unioned into the
  dense timestamp set before extraction) and always surviving
  budget-fitting even if thinning would otherwise have dropped it
  (`_build_validation_frames` re-inserts it via `_merge_frames_sorted` if
  needed) - a validation request with no labelled candidate frame at all
  would be useless, not just imprecise.
- The candidate frame is marked `TimedFrame.is_candidate=True`, a new
  field threaded through unchanged for every other frame/pass. Both
  `gemini_provider._build_parts` and `openai_provider._build_parts` label
  it `"[CANDIDATE frame at t=...s]"` instead of the plain `"[frame at
  t=...s]"` every other frame gets - this is how the model is told *which*
  submitted frame is the proposed break, without putting a numeric value
  into the (version-pinned, otherwise-static) prompt text itself, keeping
  every prompt's own wording exactly reproducible from its
  `prompt_version` alone, same discipline as every prompt before it.
- The request is sent via a new prompt, `PROMPT_END_VALIDATE_V1`, with
  `pass_name="end_validate"`, reusing `schema.TimingVerdict`'s existing
  degenerate-point shape exactly like `PROMPT_END_SCAN_V2`/
  `PROMPT_START_REFINE_V1` (CONFIRMED means validated - echo the
  candidate's own timestamp back; ABSTAIN means rejected or unable to
  judge). Grounded with the same single-region `_validate_grounding` every
  other pass uses - no changes to that function or `_GroundingRegion`.
  `openai_provider._response_schema_for_pass` extends its existing
  enum-of-submitted-timestamps constraint (originally added for
  `"end_scan"`, see §14) to also cover `"end_validate"`, for the same
  reason: this pass must also echo one submitted timestamp verbatim, and
  the schema makes fabricating a different one structurally impossible for
  OpenAI's Structured Outputs to return, not just something grounding
  detects after paying for the call.
- On a validated candidate, `end_candidate` is set to the *original*
  end-scan `window_verdict` - never to the validation call's own
  verdict - so the final `start_s`/`end_s` always reports `T`, the
  candidate's own timestamp, regardless of what the validation call itself
  answered (per the authorization's point 7). The validation verdict's
  confidence/reason_codes/evidence still fold into the final result for
  audit purposes (`min()` of all three passes' confidence, union of all
  three passes' reason_codes/evidence).
- On a rejected or unvalidatable candidate (trend didn't hold,
  insufficient future context, ambiguous, malformed, ungrounded, or the
  validation request itself couldn't fit the byte budget), the pipeline
  does **not** abstain the whole run - it keeps scanning chronologically
  forward for a later candidate, exactly mirroring how a single window's
  own abstain was already handled before this round. Only exhausting every
  scan window without a validated candidate converges on the existing
  `no_break_found` ABSTAIN.
- `PipelineOutcome` gained `end_validation_responses` (a new tuple field,
  parallel to `end_scan_responses`, default `()` for backward
  compatibility), folded into `total_retries`/`total_latency_s`/
  `total_tokens`/`estimated_cost_usd()`/`to_dict()` exactly like every
  other pass - kept as its own field (not merged into
  `end_scan_responses`) so `end_scan_window_count` keeps meaning "how many
  scan windows were tried", not conflated with how many candidates were
  checked for a sustained trend; `to_dict()` gained the parallel
  `end_validation_call_count`/`end_validation_total_retries`/
  `end_validation_total_latency_s` keys.

**Tests**: every existing stub that scripted an "end_scan" pass with no
"end_validate" branch would otherwise hit its own `assert
request.pass_name == "end_scan"` or fall through to unrelated logic
(mass test-double breakage, the same shape as every architectural round
before this one) - fixed across `test_llm_timing_pipeline.py`,
`test_llm_timing_frame_budget.py`, `test_llm_timing_pricing.py`,
`test_llm_timing_engines.py`, and `test_llm_timing_eval_provider_wiring.py`
by adding a perfect trend-validation branch (always confirms whichever
candidate the pipeline flagged) to each. Three new regressions in
`test_llm_timing_pipeline.py` cover exactly the three scenarios named in
the authorization:
`test_pipeline_rejects_a_candidate_that_fully_recovers_and_keeps_scanning`
(transient shortening + full recovery - rejected, scan continues and
finds the later true break),
`test_pipeline_accepts_a_validated_candidate_reporting_its_own_t_not_a_later_point`
(the validation call itself names a different, later "deeper" timestamp -
proves the final result still reports the original candidate `T`, not
that later point),
`test_pipeline_accepts_a_candidate_with_sustained_per_second_shortening`
(the straightforward positive case, evidence cited at per-checkpoint
timestamps). Two more targeted provider-level tests
(`test_a_candidate_marked_frame_gets_a_distinguishing_label`, one per
provider) prove the CANDIDATE frame label, and one more
(`test_response_schema_for_end_validate_constrains_start_and_end_to_submitted_timestamps`)
proves the extended OpenAI schema constraint.

**Gates**: `pytest -q` - green (464 tests total across the full repo);
`ruff check .` / `ruff format --check .` - clean; `mypy app` - clean
except the same pre-existing, unrelated `app/web/routes.py` finding noted
in §11; `node --test tests_js/*.test.js` - 44/44, unaffected (Python-only
change).

Still an experiment. No UI, no additional providers, no unrelated
cleanup beyond what this round's authorization directly required. PR #5
stays draft - ready for the next rerun.

## 18. Replacing the chronological end-scan with a two-stage strategy

A real-clip rerun against `gpt-4.1-mini` on commit `22e1605` (the trend-
validation round) came back a safe `no_break_found` ABSTAIN: 13 scan
windows, one candidate, rejected by trend validation. 426.578s runtime,
326.641s provider latency, 787,182 tokens. The supervisor then ran a
focused, isolated experiment to diagnose why: one request, 26 timestamped
frames at 0.2s spacing spanning the known 18.0-23.0s region, the same
trend-validation contract (baseline before, mandatory future checkpoints,
sustained-shortening definition, structured audit). Result: CONFIRMED
`first_break_s=20.2` against a manual truth of ~20.5s (error -0.3s,
inside the ±0.75s gate-1 tolerance), in 22.047s and 15,943+1,135 tokens,
zero retries.

Diagnosis: decomposing the end search into 13 narrow, isolated
chronological windows lost the temporal context a model needs to judge a
*sustained* trend, and the fragmentation cost two orders of magnitude
more (787,182 tokens/~7 minutes) than one coherent request needed
(~17,000 tokens/~22s). One coherent request with the same trend contract
correctly confirmed the true break; 13 isolated ones did not.

**Replacement** (commit `66b2996`): the chronological end-scan (`PROMPT_END_SCAN_V1`/
`PROMPT_END_SCAN_V2`, `_end_scan_windows`) is retired - `pipeline.py` no
longer builds or sends any `end_scan` request at all - replaced by a
bounded two-stage strategy, targeting 2-3 total provider calls for the
whole end determination instead of N scan windows:

1. **End-coarse** (new `PROMPT_END_COARSE_V1`, `pass_name="end_coarse"`):
   ONE request, sparse frames uniformly sampled at `coarse_step_s` across
   the whole post-start clip (`[start_hi, duration_s]` - the same onset-
   safety floor the chronological scan already established, preserved
   unchanged: an onset/pre-flow frame is still never eligible as a
   candidate). Explicitly uses the sustained-shortening-onset framing
   ("report the EARLIEST frame that already looks like a real, ongoing
   narrowing, not the stream's final disappearance") rather than the
   original `PROMPT_V1` end definition ("first frame that looks shorter,
   do not wait for confirmation") that caused the very first late-
   disappearance drift failure four gate-1 runs converged on (§14). Its
   own candidate is coarse and untrusted - a later pass must still
   validate it.
2. **End-validate** (existing `PROMPT_END_VALIDATE_V1`, unchanged - this
   is the exact contract the isolated experiment above validated): ONE
   dense, bounded follow-up request around the end-coarse candidate
   (`end_validation_baseline_s`/`end_validation_horizon_s`, unchanged
   defaults 1.0s/2.0s), confirming or rejecting the sustained-shortening
   trend.

Unlike the old design, a rejected or unvalidatable candidate now falls
straight through to ABSTAIN (the assisted/manual fallback) - single-shot,
not a search for another candidate, per the authorization. `end_candidate`
is still always the end-coarse pass's own verdict, never overwritten by
the validation call's own answer (the "report T, not a later point" rule
from §17 carries over unchanged).

**`PipelineOutcome` reshaped**: `end_scan_responses`/
`end_validation_responses` (tuples, sized for an unbounded scan) become
`end_coarse_response`/`end_validation_response` (single `Optional`
fields, matching the existing `fine_response` pattern) - honest about the
new architecture's fixed shape, and nothing outside `pipeline.py` read
the old tuple-shaped fields or their `to_dict()` keys (confirmed by
inspection - `scripts/llm_timing_eval.py` only reads `coarse_response`/
`fine_response`/the aggregate `total_*` properties), so this is a clean
rename, not a breaking change to anything that depends on it.
`_end_scan_windows` and `_fit_fine_windows_to_budget` are now both fully
dead code, deliberately left in place rather than removed (same choice as
§15) - flagged here again for the supervisor to decide on a cleanup pass.

**Tests**: the full `_stub_matching_truth`-style mass update this spike's
architectural rounds always require, across `test_llm_timing_pipeline.py`,
`test_llm_timing_frame_budget.py`, `test_llm_timing_pricing.py`,
`test_llm_timing_engines.py`, `test_llm_timing_eval_provider_wiring.py` -
every stub scripting `pass_name == "end_scan"` now scripts
`"end_coarse"` instead. One test
(`test_pipeline_end_scan_stops_at_the_first_break_ignoring_a_later_resumed_flow_window`)
tested a mechanism (multi-window scan-stop ordering) that no longer
exists structurally under a single whole-clip request, and was retired
rather than force-repurposed - retained only implicitly via the new
`test_pipeline_confirms_event_matching_ground_truth`'s exact-call-count
assertion (`["coarse", "fine", "end_coarse", "end_validate"]`). Two tests
were rewritten for the new single-shot-abstain behavior: what was
`test_pipeline_rejects_a_candidate_that_fully_recovers_and_keeps_scanning`
is now `test_pipeline_abstains_when_the_only_candidate_fails_trend_validation`
(rejected candidate -> immediate ABSTAIN, never a second attempt), and
what was `test_pipeline_end_scan_abstains_with_no_break_found_when_the_stream_never_breaks`
is now `test_pipeline_abstains_when_end_coarse_finds_no_candidate` (one
abstaining end-coarse call, never reaching validation). The onset-safety
regression (§17) and the stale-coarse-end-estimate regression (§15)
carried over with only their pass-name check updated - the underlying
disciplines are unchanged.

**Gates**: `pytest -q` - green (463 tests total across the full repo, one
fewer than before this round due to the retired test above);
`ruff check .` / `ruff format --check .` - clean; `mypy app` - clean
except the same pre-existing, unrelated `app/web/routes.py:261` finding
noted since §11; `node --test tests_js/*.test.js` - 44/44, unaffected
(Python-only change).

Still an experiment. No UI, no additional providers, no unrelated
cleanup beyond what this round's authorization directly required. Model
configurability and `gpt-4.1-mini` preserved for the next rerun, per the
authorization. PR #5 stays draft - ready for one real whole-clip rerun.

## 19. Codex review of §18: OpenAI schema gap for "end_coarse", and future-context enforced structurally

A Codex review of commit `66b2996` (before any live rerun) caught two
gaps in the two-stage replacement, both fixed here (commit `27587f5`):

**OpenAI enum constraint missed the new pass.**
`openai_provider._response_schema_for_pass` still special-cased only
`"end_scan"` and `"end_validate"` - the new `"end_coarse"` request (the
whole-clip candidate-nomination call) carried the exact same
timestamp-fabrication risk the enum constraint exists to close (§14/§17),
but was never added to the check when it replaced the chronological scan.
Fixed by adding `"end_coarse"` to the constrained-pass tuple; grounding
would have safely caught a fabricated candidate anyway, but the whole
point of the schema-level fix was to make the failure structurally
impossible rather than merely detected after paying for the call - this
closes that gap for the pass that actually runs now. New regression:
`test_response_schema_for_end_coarse_constrains_start_and_end_to_submitted_timestamps`.

**The "~2s future context" rule was prompt-only, not structural.**
`validation_hi` was computed as `min(duration_s, candidate_ts +
end_validation_horizon_s)` - a candidate close enough to the end of the
clip silently got a *shorter* validation horizon than the authorized
minimum, relying entirely on the model itself noticing and self-abstaining
(`PROMPT_END_VALIDATE_V1`'s own "insufficient_future_context" instruction)
rather than any code-level guarantee. Fixed by checking `candidate_ts +
cfg.end_validation_horizon_s > duration_s` *before* building the
validation request at all: a candidate that can't receive the full
horizon now abstains immediately (`insufficient_future_context`), and the
validation call is never sent. Once past that check, `validation_hi` is
no longer clamped by `duration_s` at all (the check already guarantees it
fits) - every validation request that does get sent now receives the
authorized horizon in full, never a silently-shortened one. New
regression:
`test_pipeline_abstains_when_the_candidate_has_no_room_for_the_full_validation_horizon`,
proving both the immediate ABSTAIN and that the validation request is
never sent.

**Gates**: `pytest -q` - green (465 tests total across the full repo);
`ruff check .` / `ruff format --check .` - clean; `mypy app` - clean
except the same pre-existing, unrelated `app/web/routes.py:261` finding
noted since §11; `node --test tests_js/*.test.js` - 44/44, unaffected.

No other changes. PR #5 stays draft - ready for one real whole-clip
rerun.

## 20. PROMPT_END_COARSE_V2: eliminating transients before nominating a candidate

The first real whole-clip rerun (commit `27587f5`) came back a safe but
unhelpful ABSTAIN: 83.297s harness / 61.423s provider, 162,404 tokens,
zero retries. `end_coarse` nominated `18.500s`; dense validation correctly
rejected it because the reach fluctuated and returned near baseline. The
true break is ~20.5s.

Root cause, found directly in `PROMPT_END_COARSE_V1`'s own wording: it
told the model to "report your single best candidate even if you cannot
fully confirm [a sustained trend]," and, when torn between two nearby
candidates, to "prefer the EARLIER one." In a single-shot design where a
rejected candidate converges straight on ABSTAIN (§18/§19), those two
instructions together are exactly the bias that selects an early
transient over a later, better-supported break - and make the true
candidate unreachable, since there is no second attempt.

**Fix** (commit `86c083d`, prompt-only - "no pipeline expansion or extra
calls" per the authorization): `PROMPT_END_COARSE_V2` rewrites the
MEASUREMENT RULE to use the whole sparse batch before nominating
anything, instead of stopping at the first shorter-looking frame. Scan
every frame in the batch, chronologically, that looks shorter than the
established baseline; for each, check whether the batch's own later
frames show continued shortening or a recovery; reject any that recover
as transients and keep scanning; nominate the earliest candidate whose
later checkpoints, still within the same batch, already support continued
shortening; abstain (`no_break_found`) if every candidate in the batch is
a transient or none exists. The "prefer the earlier one when ambiguous"
tie-break is removed entirely - an unsupported early candidate is now
rejected outright, never preferred over a later, trend-confirmed one.
`pipeline.py` was updated to import and send `PROMPT_END_COARSE_V2`
instead of V1 - a one-line rename at each of its ~10 use sites, no logic
change, since the whole sparse batch was already visible to the model in
one request; only the instructions for how to use it changed.
`PROMPT_END_COARSE_V1` is preserved unchanged as the historical record of
what the `27587f5` rerun was actually scored against, per this file's own
"never mutate a prompt in place" rule.

**Tests**: this is the first fix in the whole spike where the change is
purely in prompt *wording*, not code - no pipeline/provider logic needed
to change, so pipeline-level regressions alone can't pin it down (a stub
can't "reason" the way a real model does). New file
`tests/test_llm_timing_prompts.py` asserts directly on the prompt text:
the exact root-cause phrasing ("report your single best candidate even
if you cannot fully...", "prefer the EARLIER...") is confirmed present,
unchanged, in V1 and absent from V2; V2 is confirmed to instruct
rejecting a transient and continuing chronologically, requiring later-
checkpoint support, and still abstaining when nothing qualifies; both
versions stay independently registered under their own IDs. One new
pipeline regression
(`test_pipeline_reaches_the_true_break_in_one_call_when_end_coarse_avoids_the_transient`)
reproduces the real failure's shape (18.5s transient, 20.5s true break)
and proves two things structurally: `run_llm_timing` is actually wired to
`PROMPT_END_COARSE_V2_ID` now, and a model that reports the true,
trend-confirmed candidate (as V2 asks it to) still reaches CONFIRMED in
exactly one end-coarse call and one validation call - the plumbing itself
needed nothing further.

**Gates**: `pytest -q` - green (473 tests total across the full repo,
seven more than before this round: six new prompt-content assertions plus
one new pipeline regression); `ruff check .` / `ruff format --check .` -
clean; `mypy app` - clean except the same pre-existing, unrelated
`app/web/routes.py:261` finding noted since §11; `node --test
tests_js/*.test.js` - 44/44, unaffected.

Model configurability and `gpt-4.1-mini` preserved. No UI, no unrelated
cleanup. PR #5 stays draft - ready for one real whole-clip rerun.

## 21. Trend validation may refine the onset, not only confirm/reject it

A Codex review ran the real clip against commit `86c083d`
(`PROMPT_END_COARSE_V2`). Result: still safe ABSTAIN, not gate
acceptance - the coarse nomination improved from the rejected 18.5s
transient to 21.5s, but dense validation rejected 21.5s itself as
ambiguous (reach fluctuated/re-extended near baseline; weak contrast;
camera motion). Truth is ~20.5s ±0.2. 76.985s harness / 55.062s provider,
162,627 tokens, zero retries.

The gap: `PROMPT_END_VALIDATE_V1`'s contract could only CONFIRM the exact
sparse candidate or REJECT it outright - never report a different,
better-supported timestamp from evidence it was already looking at. At
the default 1.0s baseline margin, the dense window sent for a 21.5s
candidate already reaches back to 20.5s - the true break - but V1's own
TIMESTAMP RULE explicitly forbade reporting anything but the CANDIDATE
frame's own label. This is the same class of gap the isolated experiment
(§17/§18) already proved works when the dense pass is trusted with its
own evidence: one coherent request, given the right contract, found
20.2s (0.3s from truth) unprompted.

**Fix** (commit `b42bb9a`, no pipeline expansion): `PROMPT_END_VALIDATE_V2`
reframes the CANDIDATE frame as a reference point, not a fixed answer.
The model must find the EARLIEST frame anywhere in the window - baseline
before the candidate, evidence after it - where a SUSTAINED shortening
trend is confirmed by later checkpoints still within the same batch,
using the same trend-vs-transient discipline V1 already had (brief
re-extension tolerated only if later frames shorten past the prior
deepest point; a return-to-baseline-and-stay is a rejected transient).
It may report the candidate's own timestamp, an earlier one the
baseline supports, or occasionally a later one - whichever the batch's
own evidence actually confirms. The TIMESTAMP RULE now permits copying
any submitted frame's label, not just the CANDIDATE's.

No pipeline expansion was needed for the *request* shape -
`openai_provider._response_schema_for_pass` already enum-constrained
`"end_validate"` to every timestamp *submitted* in the window (not just
the candidate's own), so the schema-level guarantee needed no change;
only V1's own prompt wording was preventing the model from using
evidence already in front of it. Two things did change in `pipeline.py`:

1. `run_llm_timing` now reports the *validation* pass's own `end_s` as
   the final answer, not the end-coarse candidate's - the "report T, not
   a later point" rule from §17 is deliberately superseded here (it
   existed to stop the validation call from unilaterally drifting past
   its own candidate; V2 is allowed to refine within its own grounded
   window instead, and `_validate_grounding`'s existing bounds/evidence
   checks are the safety net, unchanged).
2. The "at least ~2s future context" structural check (§19) is now
   re-verified against whatever timestamp validation actually reports,
   not just the original sparse candidate: `validation_verdict.end_s +
   end_validation_horizon_s > validation_hi` converges on ABSTAIN
   (`insufficient_future_context`) rather than trusting an onset the
   model could not have actually confirmed from what it was shown. Since
   `validation_hi = candidate_ts + end_validation_horizon_s` is fixed by
   the pre-flight check already run for `candidate_ts`, this makes
   backward refinement (toward the baseline) always safe by construction
   - the window only grows wider from any earlier point - while forward
   refinement (past the sparse candidate) is structurally always short on
   trailing evidence and reliably abstains. (§22 below replaced this
   pre-flight-check-plus-point-anchored-window design entirely - see
   there for what superseded it.)

The end-coarse candidate itself (`candidate_ts`) is preserved in
`Event.details["end_coarse_candidate_s"]`, distinct from the final
(possibly refined) `end_s`, for audit.

**Tests**: `tests/test_llm_timing_pipeline.py` gained three regressions
matching the four scenarios requested (the fourth - transients still
abstain - was already covered by the existing
`test_pipeline_abstains_when_the_only_candidate_fails_trend_validation`,
unaffected by this round):
`test_pipeline_accepts_a_validation_refined_onset_earlier_than_the_sparse_candidate`
(backward refinement, using the real rerun's own numbers - 21.5s sparse,
20.5s refined - CONFIRMED at 20.5s, not 21.5s),
`test_pipeline_abstains_when_a_refined_onset_falls_outside_the_validation_window`
(a refined timestamp outside the submitted window's bounds is rejected,
proving grounding's existing bounds check still applies under the new
freedom), and
`test_pipeline_abstains_when_the_refined_onset_has_no_room_for_its_own_future_horizon`
(forward refinement past the sparse candidate abstains
`insufficient_future_context`, proven to happen only after the
validation call actually ran). The old
`test_pipeline_accepts_a_validated_candidate_reporting_its_own_t_not_a_later_point`
tested the now-deliberately-reversed rule and was replaced by these
three, not left alongside them.

**Gates**: `pytest -q` - green (475 tests total across the full repo);
`ruff check .` / `ruff format --check .` - clean; `mypy app` - clean
except the same pre-existing, unrelated `app/web/routes.py:261` finding
noted since §11; `node --test tests_js/*.test.js` - 44/44, unaffected.
No real-clip rerun was run from this sandbox (no API key or real clip
exists here, as every prior round has noted) - the operator runs that
next, per the standing workflow.

## 22. The point-anchored validation window itself was the bug: replaced with a wide coherent window

A Codex review ran the real clip against commit `b42bb9a` (§21's
region-refinement change). Result: still safe ABSTAIN - refined onset
16.533s, validation window `[?, 18.50]`s, truth ~20.5s. 78.516s harness /
56.734s provider, 163,330 tokens, zero retries. Reason given: `16.533 +
2.0s` exceeded the submitted horizon by 0.033s.

Reconstructing the run's own numbers from the pipeline's arithmetic
(`validation_hi = candidate_ts + end_validation_horizon_s`, so
`candidate_ts = 18.50 - 2.0 = 16.50s`): end-coarse's sparse candidate had
moved to **16.5s**, now ~4 seconds *before* the true break, rather than
after it as in the two prior rounds (18.5s, 21.5s). Under §21's
point-anchored window (`candidate_ts - 1.0s` to `candidate_ts + 2.0s`,
i.e. `[15.5, 18.5]` for this run), the true break at 20.5s was not merely
missed - it was structurally outside the request entirely. The 0.033s
future-context shortfall that actually produced the ABSTAIN was
incidental: even a validation response that had cleanly CONFIRMED inside
`[15.5, 18.5]` would have been wrong by roughly 4 seconds, the exact
"confidently wrong" outcome this whole design exists to prevent. The
supervisor's explicit instruction was **not** to relax the future-context
comparison - doing so would have converted this safe abstain into a false
positive nearly 4s early - and to diagnose and redesign instead of
tweaking wording again.

Three consecutive real reruns (18.5s → 21.5s → 16.5s candidates, against
a truth near 20.5s) established that the sparse end-coarse pass's own
error can swing several seconds in *either* direction. A window sized
only from a tight baseline/horizon around one possibly-wrong candidate
cannot be fixed by choosing a better baseline/horizon - any fixed small
window anchored to a single point has some coarse error large enough to
put the truth outside it.

**Fix** (commit `c529554`): the dense validation window is now sized as one
coherent region wide enough to contain the observed coarse error range,
not point-anchored:

- `PipelineConfig.end_validation_baseline_s`/`end_validation_horizon_s`
  (1.0s/2.0s, used only to size the request) are replaced by four fields:
  `end_validation_pre_s` (4.0s default, before the candidate),
  `end_validation_post_s` (6.0s default, after the candidate - sizing
  only), `end_validation_max_span_s` (10.0s hard cap on
  `pre_s + post_s`, enforced in `PipelineConfig.validate()` - this stays
  one coherent request, never an open-ended or multi-candidate search),
  and `end_validation_min_future_s` (2.0s default - the *structural*
  future-evidence requirement, now fully decoupled from `post_s`'s
  sizing role).
- `run_llm_timing` computes `validation_lo = max(locked_start_s,
  candidate_ts - end_validation_pre_s)` and `validation_hi =
  min(duration_s, candidate_ts + end_validation_post_s)` - clamped by the
  clip's own bounds on both sides, never past either edge.
- The old *pre-flight* check (`candidate_ts + end_validation_horizon_s >
  duration_s` → abstain before ever sending the request) is gone
  entirely: a candidate near the clip's end now still gets a window and
  a validation request, clamped at `duration_s`, exactly per the
  supervisor's "if clipping at the video end still leaves >=2s after a
  refined T, it is eligible; otherwise abstain" instruction. Eligibility
  is decided once, post-hoc, against whichever timestamp validation
  actually reports: `validation_verdict.end_s + end_validation_min_future_s
  > validation_hi` converges on ABSTAIN (`insufficient_future_context`)
  - the same check §21 introduced, just re-parameterized on the new
    fields and (now) the request's own actual, possibly clip-clamped
    upper bound rather than a value guaranteed by a pre-flight check that
    no longer exists.
- `PROMPT_END_VALIDATE_V2` needed **no wording change and no new
  version**: its MEASUREMENT RULE already scans the *whole* submitted
  batch chronologically for the earliest sustained onset without hard-
  coding any specific window size - the fix is entirely in how wide a
  window `pipeline.py` builds before asking, not in what the prompt
  tells the model to do with it. This is the same lesson as §20 in
  reverse: there, the bug was in the prompt and the pipeline needed no
  change; here, the bug is in the pipeline's window sizing and the
  prompt needs no change.

**Diagnosability for future failures**: the supervisor also asked that
the eval artifact retain enough detail to diagnose the *next* real
failure without another paid rerun. `PipelineOutcome` gained
`pass_verdicts: dict[str, TimingVerdict]` - each pass's own
post-grounding verdict (candidate/refined timestamp, evidence, reason
codes, raw_notes, model/prompt IDs), populated incrementally so a run
that abstains partway through still records every pass that ran before
the abstain. `scripts/llm_timing_eval.py`'s `ClipResult` now carries the
same dict (via each verdict's existing `to_dict()`) into the `--json`
artifact. Nothing new here carries image bytes or credentials -
`TimingVerdict.raw_notes` was already sanitized/bounded (see
`redaction.py`) before this. `Event.details` also gained
`end_validation_window_s: [validation_lo, validation_hi]` - the actual
window sent - alongside the existing `end_coarse_candidate_s`, so a
CONFIRMED result's own audit trail shows both the original sparse
nomination and the window it was validated against, distinctly.

**Tests**: the five scenarios requested -
`test_pipeline_clamps_the_validation_window_to_locked_start_s_for_an_early_candidate`
(early candidate, window clamped at the confirmed start, not
`candidate - pre_s`),
`test_pipeline_clamps_the_validation_window_to_the_clip_end_and_still_confirms`
and
`test_pipeline_sends_validation_then_abstains_when_clip_end_clamping_leaves_no_future_room`
(late candidate / clip-end clamping, both the succeeds-with-enough-room
and the aborts-for-lack-of-room cases, proving the pre-flight abort is
gone and eligibility is checked post-hoc against the actual clamped
window),
`test_pipeline_abstains_when_the_refined_onset_has_no_room_for_its_own_future_horizon`
(updated numbers - the refined-T future-horizon rule, re-verified under
the new wider window), byte-budget abstention
(`test_pipeline_abstains_with_request_too_large_when_the_wide_validation_window_cannot_fit`),
and the real-failure reproduction
`test_pipeline_reaches_a_break_the_old_narrow_validation_window_could_not_reach`
(16.5s sparse candidate, 20.5s true break - exactly this run's own
numbers - CONFIRMED at 20.5s, proving the old `[15.5, 18.5]` window could
never have reached it while the new `[12.5, 22.5]` window does). Two new
`PipelineConfig.validate()` regressions cover the new invariants
(`end_validation_pre_s + post_s` over the hard cap;
`end_validation_min_future_s` over `end_validation_post_s`).

**Gates**: `pytest -q` - green (481 tests total across the full repo, up
from 475 - six new/replaced regressions net this round); `ruff check .` /
`ruff format --check .` - clean; `mypy app` - clean except the same
pre-existing, unrelated `app/web/routes.py:261` finding noted since §11;
`node --test tests_js/*.test.js` - 44/44, unaffected. No real-clip rerun
was run from this sandbox (no API key or real clip exists here, as every
prior round has noted) - the operator runs that next, per the standing
workflow.

No pipeline expansion, no UI, no unrelated cleanup. PR #5 stays draft -
ready for one real whole-clip rerun.

## 23. App integration: an Experimental LLM analysis section

A Codex real-clip result against commit `c529554` (gpt-4.1-mini): start
4.033s (+0.133s error), end 19.566s (-0.934s vs manual 20.5s), duration
error -1.067s - a usable CONFIRMED result, but missing the strict ±0.75s
gate by 0.184s. Explicit direction: stop spending rounds on algorithm/
prompt corrections and move to hands-on experimentation instead, with the
result kept clearly marked Experimental / review required. This section
adds a bounded, usable slice of that: a UI section in the existing upload
workflow that runs the same uploaded clip through one or more
independently-configured engines and shows every result side by side,
with the API key entirely server-side and never round-tripped to the
browser. Not merged, not marked ready - the PR stays draft.

### Architecture

Four new backend modules, kept separate from the classical-detector code
path so nothing about it changes:

- `app/services/secret_store.py` - the only module that ever touches a
  real API key's *value*. Backed by the third-party `keyring` package
  (optional dependency, lazily imported - see `requirements-llm-spike.txt`),
  which auto-selects Windows Credential Manager (backed by DPAPI) on
  Windows, Keychain on macOS, Secret Service on Linux - satisfying the
  "OS-protected secret mechanism on Windows" requirement without this
  application needing to know which OS it's running on. A machine with no
  usable backend at all degrades to "can't save a key here" rather than
  falling back to something less protected; the operator uses the
  environment-variable path instead for that case.
- `app/services/llm_engine_store.py` - CRUD for configured engines,
  persisting only non-secret metadata (provider, model, display name,
  enabled) to one JSON file under `AppConfig.data_dir`. A `credential_ref`
  is one of two shapes: `"secret:<engine_id>"` (a saved key, resolved via
  `SecretStore`) or `"env:<VAR_NAME>"` (the documented development
  fallback - an operator-managed environment variable; nothing is stored
  server-side for this path beyond the variable's own name). Building a
  real provider still goes through the existing, unmodified
  `build_default_openai_client`/`build_default_gemini_client` (both read a
  credential from a *named* environment variable only, a deliberate
  pre-existing choice - never a literal through a function argument that
  could end up in a traceback or repr); a resolved secret is materialized
  into a per-engine, unique process environment variable
  (`VISION_ANALYZER_LLM_ENGINE_SECRET_<ENGINE_ID>`) immediately before
  constructing the client, so two engines' credentials can never collide
  even when resolved concurrently by independent runs.
- `app/services/llm_run_service.py` - one background job per engine run,
  mirroring `AnalysisService`'s existing shape (staged progress,
  cooperative cancellation, retention) but calling
  `pipeline.run_llm_timing` directly rather than through
  `engine_config.run_llm_timing_for_engines` - that function's per-engine
  `try/except` would misreport a user-initiated cancellation as an engine
  crash, which this module keeps distinct. Frames-only by construction:
  `run_llm_timing` only ever extracts individual JPEG frames from the
  video and sends those in each `ProviderRequest` - the original video
  file is never opened by, or passed to, a provider at all (regression:
  `test_a_frames_only_provider_request_never_carries_the_video_file`).
- `app/web/llm_engine_routes.py` - the REST surface
  (`GET/POST /api/llm-engines`, `PUT/DELETE /api/llm-engines/<id>`,
  `POST /api/llm-engines/<id>/test`,
  `POST /api/videos/<id>/llm-runs`, `GET/POST /api/llm-runs/<id>[/cancel]`),
  registered from `routes.create_app` alongside the existing blueprint
  (shares its app-wide `AnalyzerError` -> JSON-message handling
  unchanged). `EngineConfig.to_dict()` already excluded `credential_ref`
  (pre-existing, §10) - this is the first place anything actually calls
  it over HTTP.

One small, additive change to the existing pipeline:
`pipeline.run_llm_timing` gained an optional `on_stage: Callable[[str],
None]` parameter, called with the pass name ("coarse"/"fine"/
"end_coarse"/"end_validate") immediately before that pass's provider call
is sent - and never after, and never for a pass a run doesn't reach. Two
uses, both optional and additive: staged progress for the run service to
report, and cooperative cancellation - the service's own callback checks
a cancel flag and raises `RunCancelled`, which propagates straight out of
`run_llm_timing` (no new exception type in the pipeline module itself; it
has no opinion on what "cancelled" means to its caller).

### UI

A new "Experimental: LLM analysis" section in `index.html`, positioned
right after the upload step (it only needs an uploaded video, not a
chosen classical mode) and gated the same way the existing steps are
(`enableStep`). Per engine: a card with Run/Cancel/Test/Edit/Delete,
staged progress text + a settled progress bar (the server reports named
stages, not a fraction), and a result panel that always carries an
"Experimental - review required" badge alongside the confirmed/abstained/
error state, start/end/duration, confidence/uncertainty, and reason
notes. "+ Add engine" opens a form: display name, provider (OpenAI or
Google Gemini - not a hard-coded single model, any model ID the account
has access to), and a credential fieldset offering either "save an API
key" (OS-protected storage) or "advanced: use an environment variable
already set on this machine". The existing manual-correction/classical-
analysis workflow is completely unchanged - this section is additive,
never a replacement path.

`app/web/static/app.js` gained the DOM-wiring for all of this plus four
pure, unit-tested functions (`llmEngineSubtitle`, `llmResultBadge`,
`llmRunStatusLine`, `validateLLMEngineForm`) - the same "pure logic
exported via `module.exports`, DOM wiring exercised by the Flask test
client instead" split the existing Zahn-review code already uses.
Dynamic content (engine display names, run messages) is built via
`document.createElement`/`.textContent` throughout, never
`innerHTML`-with-interpolation, so a display name a user typed can never
be interpreted as markup.

### Security properties, verified by tests, not just asserted here

- **Write-only secrets, all the way through**: a submitted API key is
  handed to `LLMEngineStore.create`/`update`, saved via `SecretStore`, and
  never appears in any HTTP response, the persisted `llm_engines.json`
  file, or a log line (`tests/test_web_llm_engines.py::test_create_with_an_api_key_saves_it_write_only`,
  `test_the_saved_api_key_never_appears_in_the_persisted_metadata_file`;
  `tests/test_llm_engine_store.py::test_the_persisted_file_never_contains_the_raw_api_key`).
- **No browser storage of any kind is used for a key** - the form field is
  a plain `type="password"` input whose value is sent once, in the POST
  body, over the loopback-only connection this application already runs
  on (§9), and never written back into the DOM or `localStorage`/
  `sessionStorage`.
- **A machine with no OS secret backend fails safely, not silently**:
  `SecretStore.save` raises `SecretStoreUnavailable` rather than falling
  back to something unprotected; the route surfaces this as a clear 400,
  and this is exactly the path exercised in this sandbox, since `keyring`
  has no real OS backend to select here
  (`test_create_with_an_api_key_when_no_secret_backend_exists_fails_safely`).
- **Frames-only**: see above.
- **Safe error handling**: a missing/unresolvable credential, an
  unsupported provider, or an unexpected crash inside a run all converge
  on the run's `status: "failed"` with a message safe to show
  (`AnalyzerError.user_message` or a fixed generic sentence - the
  technical detail goes to the server log only, matching this
  application's existing discipline elsewhere) - never a raw exception,
  never the credential's value
  (`tests/test_llm_run_service.py::test_a_missing_credential_fails_the_run_with_a_safe_message`,
  `tests/test_web_llm_engines.py::test_run_with_a_missing_credential_fails_with_a_safe_message`).
- **Provider timeout/error -> safe UI state**: unchanged pre-existing
  behaviour (§10) - `OpenAITimingProvider.analyze`/`GeminiTimingProvider.analyze`
  never let a client exception escape; it becomes an `error`-carrying
  `RawProviderResponse`, which flows through the normal ABSTAIN path
  (`provider_error` reason code) rather than crashing the run.
- **Cancellation**: honoured cooperatively between pipeline passes, proven
  by asserting the *next* pass's request never reaches the provider after
  a mid-run cancel
  (`test_cancel_mid_run_stops_before_the_next_pass_completes`), and
  immediately for a still-queued run
  (`test_cancel_before_the_run_starts_marks_it_cancelled_immediately`).

### What's deliberately out of scope this round

- The "Test" button resolves the credential and constructs a real
  provider client - it does **not** make a live API call (that would cost
  real time/money on every click); `ok: true` means "this key/model is
  configured and the client builds", not "a request round-tripped". Said
  explicitly in the button's own result message.
- No cross-engine rate limiting or request queuing beyond the existing
  `MAX_CONCURRENT_RUNS = 3` worker pool.
- No true fractional progress bar - the server reports named stages
  (coarse/fine/end_coarse/end_validate), not a percentage, so the bar
  settles at a fixed "in progress" fill rather than implying false
  precision.
- No aggregation/averaging/winner-picking across engines - unchanged from
  the existing `EngineConfig`/`run_llm_timing_for_engines` design (§10):
  every engine's result is shown independently, disagreement is the
  point.

### Manual test procedure

1. `pip install -r requirements-llm-spike.txt` (needs `openai`,
   `google-genai`, and/or `keyring` depending on what's being tested).
2. `python -m app.main` (or however the app is normally started locally).
3. Upload a video. The "Experimental: LLM analysis" section becomes
   usable immediately (no need to pick a classical mode first).
4. Click "+ Add engine": choose OpenAI or Gemini, enter any model ID,
   either paste a real API key (saved via the OS-protected store on a
   supporting machine) or switch to "use an environment variable" and
   name one already exported in the shell the server was started from.
   Save.
5. Click "Test" - confirms the credential resolves and the client
   constructs, without spending a real request.
6. Click "Run" - staged progress text updates every ~1.2s; "Cancel"
   appears while queued/running and stops the run before its next pass.
7. On completion: a result card with the Experimental/review-required
   badge, confirmed/abstained state, start/end/duration/confidence/
   uncertainty if confirmed, and reason notes if any.
8. Add a second engine (different provider or model) and run it against
   the same clip - both result cards persist side by side, independently.
9. Delete an engine - confirm it disappears from the list and a
   subsequent "Test"/"Run" against its old ID 404s.
10. Confirm the classical analysis workflow (upload -> mode -> configure
    -> analyse -> results) still works exactly as before, untouched by
    any of the above.

### Gates

`pytest -q` - 545 passed (up from 481; 64 new tests: 12 for
`secret_store.py`, 24 for `llm_engine_store.py`, 7 for
`llm_run_service.py`, 19 for the web routes, 2 for `pipeline.py`'s new
`on_stage` parameter). `ruff check .` / `ruff format --check .` - clean.
`mypy app` - clean except the same pre-existing, unrelated
`app/web/routes.py` "Unused type: ignore" finding noted since §11 (line
number shifted by the new imports/registrations, not a new issue - fixed
once a variable-naming collision of my own between the OpenAI and Gemini
client branches in `llm_engine_store.build_provider`, a real mypy catch).
`node --test tests_js/*.test.js` - 69/69 (44 existing + 25 new, covering
provider/model-selection validation, write-only-secret form behaviour,
and the result/status-line helpers behind the comparison UI).

No real vendor call was made from this sandbox (no API key here, as
every prior round has noted) - manual verification against a real key is
the operator's next step. PR #5 stays draft, not marked ready.

## 24. Codex UI review of §23: secrets in os.environ, no request timeout, overclaimed cancellation

A Codex review of commit `58479b5` (the app-integration slice) found the
integration direction sound (focused suite: 116 Python + 25 JS, passing)
but flagged three user-test blockers, not algorithm polish:

**1. Saved secrets were copied into `os.environ` and never removed.**
`LLMEngineStore._materialize_credential` wrote
`VISION_ANALYZER_LLM_ENGINE_SECRET_<ID>` into the process environment
before every `build_provider`/`test` call and never cleared it - a saved
API key was readable by the whole process for the server's entire
lifetime, the opposite of the OS-secret-store boundary the key was saved
to protect in the first place.

**Fix**: `build_default_openai_client`/`build_default_gemini_client` each
gained an `api_key: str | None = None` keyword argument. When given, it
is passed straight to the SDK client constructor; the original
`api_key_env_var` positional argument is untouched and still works
exactly as before for the `"env:<VAR_NAME>"` (operator-managed) path.
`LLMEngineStore.build_provider`/`_resolve_credential` (renamed from
`_materialize_credential`) now returns `(api_key, api_key_env_var)` -
exactly one non-`None` - and never touches `os.environ` itself; the
`_env_var_for_secret` helper and the per-engine env-var scheme it
implemented are gone entirely. Regressions: `test_llm_provider_client_factories.py`
snapshots `os.environ` before/after a `build_default_*_client(api_key=...)`
call and asserts it is byte-for-byte unchanged (using a fake SDK module
injected into `sys.modules`, since neither real SDK is installed here);
`test_llm_engine_store.py::test_build_provider_passes_a_saved_secret_directly_never_via_os_environ`
does the same at the `LLMEngineStore` layer and additionally asserts the
resolved key string appears in no environment variable's value.

**2. No bounded per-vendor request timeout.** Neither real SDK client was
constructed with an explicit timeout, so a stalled request could hang
indefinitely - the staged-progress UI would keep showing "in progress"
for a request that was never coming back.

**Fix**: both factories gained `timeout_s: float = 120.0`
(`DEFAULT_REQUEST_TIMEOUT_S`), passed to the SDK client constructor
(`OpenAI(..., timeout=timeout_s)`; `genai.Client(..., http_options=types.HttpOptions(timeout=timeout_s*1000))`
- milliseconds, UNVERIFIED against live documentation like this module's
other SDK-shape notes, network access to ai.google.dev being blocked
here). A timeout already converged on the existing safe ABSTAIN/failed
path with zero further change: OpenAI's `APITimeoutError` is a subclass
of `APIConnectionError`, already classified `TransientProviderError`; the
Gemini wrapper gained a best-effort, class-name-based
`_looks_like_a_timeout` check (`"timeout" in type(exc).__name__.lower()`
- not an `isinstance` check against a confirmed SDK exception type, same
caveat) so the failure message names the configured bound explicitly
("did not complete within 7s") rather than leaving the SDK's raw text;
anything that doesn't match this heuristic re-raises completely
unchanged, so `GeminiTimingProvider.analyze`'s own pre-existing catch-all
still handles it safely regardless. Regressions inject a fake SDK client
whose call raises a timeout-shaped exception and assert the resulting
`TransientProviderError`'s message names the configured `timeout_s`.

**3. Cancel was only checked between passes but presented as immediate.**
`on_stage` was already correctly cooperative (checked before each of the
four pipeline passes, never mid-request - see §23), but the "Cancel
requested" UI gave no indication of that: a click looked like it should
stop an in-flight vendor request, which it structurally cannot (neither
SDK's request object is exposed to `LLMRunService` for that).

**Fix**: `LLMRunService.cancel` now distinguishes three cases explicitly -
a still-queued job stops immediately with nothing ever sent ("Cancelled
before it started", unchanged); a running job gets an honest "Cancel
requested - stopping after the current provider request finishes (it
cannot be interrupted mid-request)" message, status staying `"running"`
until the next `on_stage` check actually raises and the worker's own
`except RunCancelled` sets the final "Run cancelled" state - the
in-flight-request caveat is stated outright rather than implied away.
`app/web/static/app.js`'s `cancelLLMRun` disables the Cancel button and
relabels it "Cancelling..." immediately on click, before the network
round-trip, so repeated clicks don't suggest a faster stop is possible;
`runLLMEngine` resets that state when a fresh run starts. Neither SDK's
own request-cancellation was wired in this round (would need investigating
each SDK's own cancellation support cleanly, out of scope per the
review's own "otherwise document the bounded behavior" fallback) - the
bounded, honest cooperative semantics are what's shipped and documented,
not faked as immediate. Regression:
`test_llm_run_service.py::test_cancel_of_a_running_job_reports_an_honest_stopping_message`
cancels from inside an in-flight stub response and asserts the message
right after `cancel()` returns (before the run has actually finished).

### Manual test procedure (updated)

Same as §23's, with two additions to verify at steps 4-6:

- After saving an API key (step 4), inspect the server process's
  environment (e.g. `cat /proc/<pid>/environ` on Linux, or add a
  temporary log line) and confirm no `VISION_ANALYZER_LLM_ENGINE_SECRET_*`
  variable - or the key's own value under any name - appears there,
  before or after clicking Run.
- Click Cancel while a run is genuinely mid-request (a real, slow vendor
  call): the message should read "Cancel requested - stopping after the
  current provider request finishes..."), the button should grey out
  immediately, and the run should still take until the in-flight request
  returns before actually stopping - confirming the UI's own claim about
  itself rather than a faster fake stop.

### Gates

`pytest -q` - 557 passed (up from 545; 12 new: 9 in
`test_llm_provider_client_factories.py` covering both factories'
`api_key`/`timeout_s` behavior plus the `_looks_like_a_timeout` helper,
2 replacing/extending the old os.environ-materialization test in
`test_llm_engine_store.py`, 1 for the honest cancel message in
`test_llm_run_service.py`). `ruff check .` / `ruff format --check .` -
clean. `mypy app` - clean except the same pre-existing, unrelated
`app/web/routes.py` finding noted since §11 (a real mypy catch of my own
fixed along the way: `build_provider`'s `api_key_env_var` needed an
explicit `assert ... is not None` to narrow past `_resolve_credential`'s
`tuple[str | None, str | None]` return type). `node --test
tests_js/*.test.js` - 69/69, unaffected (this round's cancel-button
disabling is DOM wiring, exercised by the manual procedure above, not by
the pure-function JS tests, matching how the rest of this file's DOM code
is tested).

No real vendor call was made from this sandbox (still no SDK installed,
no API key here). PR #5 stays draft, not marked ready.

## 25. Frame auditability, evidence images, a model selector, and a browser regression gate

A supervisor change request, quoted here for the product principle it's
built to satisfy: "for every AI timing result, I must be able to answer,
from the website itself, *what exactly did the AI see, and which two
submitted frames support the reported start and end?*" Four requirements,
none of which change the classical Zahn detector, the LLM prompt/pipeline
strategy, or the timing defaults themselves (explicitly out of scope -
the ±1.5s fine-refine margin, the coarse/end-scan windows, etc. are
unchanged; this round only makes the *existing* windows auditable).

### 1. What was actually sent to the AI

`PipelineOutcome` gained `pass_frames: dict[str, tuple[float, ...]]` -
one entry per pass that actually ran ("coarse"/"fine"/"end_coarse"/
"end_validate"; a pass that never ran is simply absent, not an empty
tuple, so the UI's "Not run" state is driven by key absence rather than
an ambiguous zero-frame range). Populated in `run_llm_timing` at the
exact point each pass's `ProviderRequest.frames` is built - after
`_fit_frames_to_budget`'s thinning, never before - so a regression
(`test_pipeline_thinned_pass_frames_match_the_frames_actually_sent`)
proves that when a tight byte budget forces the fine pass down to fewer
than 30 frames, `pass_frames["fine"]` is exactly that thinned set, not
the denser pre-thinning plan. The frontend's new "Frames sent to AI"
`<details>` section (`buildLLMFramesSentSection` in `app.js`) renders one
row per pass with a first/last/count summary
(`llmPassFrameSummary`) and a nested expandable list of every individual
submitted timestamp (`llmFormatFrameTimestamps`, all to 0.1s) - so a
sparse, thinned request can't be misread as a continuous range. A static
explanation of the fine pass's default ±1.5s margin sits alongside the
*actual* computed range for the current run, satisfying requirement #3
without touching the margin itself.

### 2. The two decisive evidence images

A new pure selector, `pipeline._nearest_grounded_evidence_ts(evidence_ts,
submitted_ts, boundary_ts)`, picks the grounded verdict's own cited
evidence timestamp nearest the reported boundary, then snaps to the
nearest entry in that pass's *actual submitted* timestamps - so the
result is always an exact frame that pass really sent, never an
interpolated or reconstructed one. Returns `None` (never fabricates) when
either input is empty. `run_llm_timing`'s final CONFIRMED block computes
`start_evidence_s`/`end_evidence_s` this way and stores them in
`Event.details`; ABSTAIN produces no `Event` at all, so there is no
per-boundary special-casing needed to keep ABSTAIN from presenting a
fabricated image - it's structural. `LLMRunJob` exposes both as
properties read straight from `event_dict` (single source of truth, no
separate field to drift out of sync) and a new endpoint,
`GET /api/llm-runs/<run_id>/evidence/<start|end>`, serves the frame at
that server-stored timestamp - deliberately taking **no** timestamp from
the request, so a client can never relabel an arbitrary frame as "the
evidence" by supplying its own `t`
(`test_evidence_image_endpoint_uses_the_server_stored_timestamp_not_a_client_supplied_one`
proves a spoofed `?t=` query param is ignored). 404s, never a fabricated
image, when a run has no grounded evidence for that boundary
(`test_evidence_endpoint_404s_rather_than_fabricating_for_an_abstained_run`).
The frontend renders both images with a floating timestamp-overlay label
(`llmEvidenceLabel`) or an explicit "No grounded evidence image
available" message when `None`.

### 3. A server-side model allowlist, not free text

`llm_engine_store.py` gained `SUPPORTED_MODELS: dict[str, tuple[str,
...]]` - the single source of truth for which model IDs are selectable
per provider, deliberately curated from this repository's own real-call
history in this file (§14 onward) rather than "whatever the vendor
happens to offer": `openai: ("gpt-4.1-mini",)` (the model behind every
real gate-1 rerun from §15 onward) and `gemini: ("gemini-3.5-flash-lite",
"gemini-3.5-flash")` (both completed real, schema-conformant calls in
§14's four-model round, even though their timing accuracy was poor - a
separate question from whether this code has verified they accept the
required image + structured-output request shape). Each provider
module's own no-`model_id`-given fallback (`gpt-5-mini`,
`gemini-2.5-flash-lite`) is deliberately *not* on the list - neither has a
real-call record here, and `LLMEngineStore` always passes an explicit
`model_id`, so those defaults are never reached through this UI anyway.
`_validated_model_id` enforces the allowlist independently inside both
`create()` and `update()` - not just in the new `GET
/api/llm-model-options` endpoint the frontend's `<select>` populates from
- so DOM or request tampering can't select an unsupported model either
way. `update()`'s edit semantics: the existing model is preserved only if
it's still valid for the (possibly just-changed) provider; otherwise
`_validated_model_id` raises, and the frontend shows a disabled
placeholder option plus a visible warning rather than silently falling
back to some other model
(`test_update_switching_provider_requires_a_model_valid_for_the_new_provider`).

### 4. A browser regression gate with a deterministic stub provider

`tests_js/playwright/` is new: `e2e_server.py` builds the real
`create_app()` against a throwaway data directory and monkeypatches
`LLMEngineStore.build_provider` at the class level to a
`StubTimingProvider` (the same stubbing pattern
`tests/test_llm_run_service.py` already uses) - so this suite needs no
API key, makes no network call, and costs nothing, while still exercising
the real pipeline, real frame extraction, and real HTTP layer end to end
(a real `werkzeug` server on an ephemeral port, not `app.test_client()` -
Playwright needs an actual socket). `llm_flow.test.js` drives a headless
Chromium through the complete flow end to end: upload a synthetic video,
add an engine through the Provider/Model dropdowns, run it, verify the
result card's Start/End/Duration/Confidence/Uncertainty, verify the
Frames-sent-to-AI section names all four passes with non-"Not run"
frame-range summaries and an expandable full timestamp list, verify both
evidence images load with a correctly formatted `t = X.Xs` overlay label,
exercise Test/Edit/Delete, and confirm a page reload preserves the
configured engine (re-uploading afterward, since `#step-llm` -
like every other step in this wizard - goes back to its
`pointer-events: none` "disabled" styling on reload until a video is
re-selected; that's pre-existing, uniform behavior across all five steps,
not something this round introduced or was asked to change). A final
assertion confirms zero console errors and zero HTTP responses ≥400
occurred anywhere during the run. Two real bugs surfaced and were fixed
because of this suite, not despite it:

- `buildLLMFramesSentSection` appended each pass's `<dt>`/`<dd>` directly
  as siblings into the `.details-grid` container instead of wrapping them
  in a `<div>` the way the existing `appendDetail` helper does - since
  `.details-grid` is a CSS grid over its *direct* children, this silently
  misaligned every pass's label from its own value (visible in the first
  screenshot taken; invisible to any DOM-content assertion that doesn't
  look at layout). Fixed by wrapping each pair in its own `div`, matching
  the established convention.
- The two `page.waitForSelector("#llm-engine-form.hidden")` calls timed
  out once exercised end to end, because Playwright's default `state:
  "visible"` can never be satisfied by an element matching a selector
  that requires the CSS class making it `display: none` - a genuine
  contradiction in the original selector, not a flaky wait. Replaced with
  `waitForSelector("#llm-engine-form", { state: "hidden" })`.

Run with `NODE_PATH=/opt/node22/lib/node_modules node --test
tests_js/playwright/llm_flow.test.js` (Playwright is preinstalled
globally in this sandbox, not as a repo dependency - see the file's own
header comment); intentionally **not** part of `node --test
tests_js/*.test.js` since it needs a live server and a browser, not just
Node. 13/13 passed (1 wrapper + 12 subtests), ~3.5s.

### Screenshots inspected

Two screenshots were captured by the suite (desktop 1280px and narrow
375px viewports, both `fullPage`) and visually inspected as part of this
round, not just asserted on programmatically: the result card's badges,
detail grid, Frames-sent-to-AI section (all four passes correctly
aligned after the grid fix above), and both evidence images with their
timestamp overlays all remain legible at both widths - the narrow layout
reflows the frame-summary grid to a single column rather than
overflowing or truncating. Not committed to the repository (generated
artifacts, regenerable by re-running the suite).

### Files changed

- `app/analysis/llm_timing/pipeline.py` - `pass_frames` field +
  threading through every `PipelineOutcome`/`_abstain_outcome` call site;
  `_nearest_grounded_evidence_ts`; `start_evidence_s`/`end_evidence_s` in
  the CONFIRMED `Event.details`.
- `app/services/llm_engine_store.py` - `SUPPORTED_MODELS`,
  `_validated_model_id`, wired into `create()`/`update()`.
- `app/services/llm_run_service.py` - `LLMRunJob.start_evidence_s`/
  `.end_evidence_s` properties, exposed in `to_dict()`.
- `app/web/llm_engine_routes.py` - `GET /api/llm-model-options`,
  `GET /api/llm-runs/<id>/evidence/<boundary>`.
- `app/web/templates/index.html` - Model `<input>` replaced with a
  `<select>` plus a warning-hint `<span>`.
- `app/web/static/app.js` - model-dropdown population/preservation logic,
  the Frames-sent-to-AI and evidence-image builders, three new pure
  helpers (`llmPassFrameSummary`, `llmFormatFrameTimestamps`,
  `llmEvidenceLabel`).
- `app/web/static/styles.css` - `.llm-frames-sent`, `.llm-frame-
  timestamps`, `.llm-evidence-*`.
- `tests/test_llm_timing_pipeline.py`, `tests/test_llm_engine_store.py`,
  `tests/test_web_llm_engines.py` - new coverage per requirement above,
  plus mechanical `gpt-5-mini`/`gemini-2.5-flash` -> `gpt-4.1-mini`/
  `gemini-3.5-flash` renames forced by the new allowlist.
- `tests_js/llm_engines.test.js` - unit tests for the three new pure
  helpers.
- `tests_js/playwright/e2e_server.py`, `tests_js/playwright/
  llm_flow.test.js` - new browser regression suite (see above).

### Gates

`pytest -q` - 578 passed (up from 557; +21: 9 in
`test_llm_timing_pipeline.py` for `pass_frames`/evidence-timestamp
selection, 6 in `test_llm_engine_store.py` for the model allowlist, 6 in
`test_web_llm_engines.py` for the evidence endpoint and
`/api/llm-model-options`). `ruff check .` - clean (one line-length fix in
`pipeline.py` along the way). `ruff format --check .` - clean (one file
reformatted, `test_web_llm_engines.py`). `mypy app` - clean except the
same pre-existing, unrelated `app/web/routes.py:273` finding noted since
§11/§24 (confirmed unrelated to this round: reproduces identically with
this round's changes stashed out). `node --test tests_js/*.test.js` -
82/82 (up from 69; +13 for the three new pure helpers). New browser
suite: 13/13 (see above). At least one deterministic stub-provider run
covers the complete flow end to end; no real-provider smoke test was
possible from this sandbox (no vendor SDK installed, no API key here) -
the same limitation noted in every prior round of this file.

No known limitation remains for requirements #1, #2, or #4. Requirement
#3's UI half is the static explanation text plus the actual per-run
range shown inside the Frames-sent-to-AI section - no numeric default
changed. Per the request: **not merged**, PR #5 stays draft.

## 26. Automatic, durable AI decision audit trail

A second supervisor change request against §25's own work, prompted by a
real operator complaint after a hands-on run: an end-coarse pass
nominated an early candidate, the dense validation window built around it
was consequently too narrow to see the clip's real continuation, and
there was no way to understand *why* without DevTools or manual JSON
extraction. Explicit closing test: "for every AI timing result, I must be
able to answer, from the website itself, what exactly did the AI see, and
which two submitted frames support the reported start and end" - this
round extends that to *every* outcome, not only a CONFIRMED one, and
makes the answer durable across a page refresh and a server restart, not
just visible while a browser tab happens to still be polling. No change
to the classical detector, the LLM prompts, or any pipeline/timing
default - explicitly out of scope, same as §25.

### Two new derived-decision fields, threaded through every outcome

`PipelineOutcome` gained `derived: dict[str, ...]`
(`pipeline.py`) - `locked_start_s`, `start_evidence_s`,
`end_coarse_candidate_s`, `end_validation_window_s`, `end_evidence_s` -
populated the moment each is actually decided inside `run_llm_timing`,
*not* only inside the final CONFIRMED block (which already had this data
in `Event.details`, untouched). The validation window in particular is
recorded the instant `validation_lo`/`validation_hi` are computed, before
the validation call is even sent - so a run that abstains because that
window was too narrow, or because the byte budget refused it outright,
still shows exactly what window the pipeline chose and why. Threaded
through all nine `PipelineOutcome`/`_abstain_outcome` return sites, the
same incremental pattern §25's `pass_frames` already established - and in
threading it, one *pre-existing* gap from that round was closed along the
way: the "fine window implausibly wide" abstain path
(`ambiguous_evidence`) had never been passing `pass_verdicts`/`pass_frames`
through at all, silently dropping the coarse pass's own already-computed
detail from that one outcome; it now does, matching every other abstain
path (`tests/test_llm_timing_pipeline.py::test_pipeline_derived_decisions_survive_a_rejected_end_candidate`
reproduces the operator's exact numbers - candidate 5.5s, window
[1.5, 11.5]s - and proves both `derived` and `pass_frames`/`pass_verdicts`
are populated even though the run overall abstains).

### The durable audit record

New `app/services/llm_run_audit.py`:

- `build_audit_record(...)` - a pure function assembling one JSON-safe
  dict from a run's identity, an optional `PipelineOutcome` (`None` for a
  run that never reached the pipeline at all, e.g. a credential failure -
  the record is still built, with empty `passes`/`derived` and whatever
  `error` says, per "persist safe partial information for ABSTAIN,
  failure and cancellation too"), and nothing else - it is never given a
  `VideoRecord`'s real filesystem path, an `EngineConfig.credential_ref`,
  or a `TimedFrame`'s image bytes, so none of those can appear in its
  output by construction, not by a redaction pass trying to catch them
  afterward. Every free-text field (`raw_notes`, the top-level `error`) is
  still routed through `sanitize_untrusted_text` again at build time as
  defense in depth, even though both sources already sanitize their own
  text.
- `LLMRunAuditStore` - one JSON file per run under
  `data_dir/llm_run_audits/<run_id>.json`, written atomically
  (write-to-temp, then rename - same pattern `llm_engine_store.py` already
  uses). `get(run_id)` validates `run_id` against a strict
  `[A-Za-z0-9_-]{1,128}` pattern before it ever becomes a path component -
  "safe run IDs only" is an explicit requirement, and this is the one
  place a run_id arrives from a URL path parameter.
  `latest_for_engine(engine_id)` is a plain directory scan (not a
  maintained index file, which could itself drift out of sync with what's
  actually on disk) - fine at this application's scale.
  `purge_before(cutoff_epoch_s)` deletes everything older than the
  cutoff, and is called from `LLMRunService.purge_expired()` with the
  *same* `RUN_RETENTION_S` cutoff already applied to the in-memory job
  map - bounded cleanup tied to the existing retention policy, per the
  requirement, not a new one.

`LLMRunService` now takes an `AppConfig` (previously just the engine
store) and builds/saves an audit record on every terminal outcome -
complete, failed, *and* cancelled, including a run cancelled before it
ever started (which previously skipped the worker's `try/finally`
entirely and would have produced no audit at all; fixed by an explicit
save on that early-return path too). One real ordering bug was caught
and fixed while wiring this up, by the new integration tests themselves
racing against it: the first version set `job.status` to its terminal
value *before* the `finally` block that saved the audit, so a poller
could see `status: "complete"` and immediately request the audit before
it existed on disk. Fixed by introducing a `terminal_status` local that
every branch sets instead of `job.status` directly, saving the audit
against that value inside `finally`, and only then publishing it onto
`job.status` - so by the time any caller can observe a terminal status,
the durable audit is already there.

### HTTP surface

Three additions to `llm_engine_routes.py`:

- `GET /api/llm-runs/<run_id>/audit` - the full record, `Content-Disposition:
  attachment` set (a plain `fetch()` ignores that header; only a
  browser-navigated/`<a download>` click respects it - so this one URL
  serves both the page's own rendering *and* the "Download audit report
  (JSON)" button, never two endpoints that could drift apart).
- `GET /api/llm-engines/<engine_id>/latest-audit` - how the page finds
  "what this engine last did" after a refresh without already knowing a
  `run_id`.
- The existing evidence-image endpoint (§25) now falls back to the
  durable audit's own `derived` timestamps when the in-memory job is gone
  (restart, or past the 6-hour retention) - previously it 404'd the
  instant the job left memory even though the image was still derivable
  from disk; evidence images now stay viewable exactly as long as the
  audit JSON does.

### UI: "How the AI decided" and the download button

`app.js` gained `buildHowAIDecidedSection(audit)` - one block per pass, in
run order, in plain language (`llmPassPlainLanguage`), each pass's own
`raw_notes` appended verbatim, plus (only under "End coarse") a connector
sentence naming the exact candidate-to-window derivation
(`llmEndCoarseToValidateExplanation`) - the specific link the requirement
calls out by name. A pass that never ran reads "This pass did not run.",
never silently omitted, so an ABSTAIN's chain is exactly as inspectable
as a CONFIRMED one. `buildAuditDownloadLink(runId)` is a plain
`<a download>` pointed at the audit endpoint above.

This meant switching the existing Frames-sent-to-AI and evidence-image
sections (§25) from reading `job.outcome`/`job.start_evidence_s` to
reading the durable `audit.passes`/`audit.derived` instead - not a
cosmetic change: it is what makes the *entire* result card, not just the
new section, render identically whether it just finished live or is
being restored on page load from `GET /api/llm-engines/<id>/latest-audit`
(`loadLatestAuditForEngine`, called for every configured engine right
after `loadLLMEngines()` on every page load). A live run's card still
gets this data one fetch sooner than before (`fetchLLMAuditAndRender`,
called once a poll sees a terminal status) rather than reading it
straight off the poll response - the extra round trip is invisible in
practice given the ordering fix above, and buys a single source of truth
for both cases instead of two rendering paths that could disagree.

### Redaction, verified not asserted

Beyond `build_audit_record`'s structural guarantee (§ above), tests in
`tests/test_llm_run_audit.py` and `tests/test_web_llm_engines.py` prove,
over the actual HTTP response body, that a secret-shaped string injected
into a model's own `raw_notes` is redacted (`[redacted]`), that the
configured engine's env-var name and the word "credential" never appear,
and that the real uploaded video's filesystem path never appears - not by
inspecting the code path, but by grepping the actual serialized JSON a
client would receive.

### Files changed

- `app/analysis/llm_timing/pipeline.py` - `PipelineOutcome.derived` +
  threading; one pre-existing threading gap fixed (see above).
- `app/services/llm_run_audit.py` - new: `build_audit_record`,
  `LLMRunAuditStore`.
- `app/services/llm_run_service.py` - takes `AppConfig`; builds/persists
  the audit on every terminal outcome (including cancel-before-start);
  the `terminal_status` ordering fix; `get_audit`/
  `get_latest_audit_for_engine`; audit cleanup wired into
  `purge_expired()`.
- `app/web/routes.py` - passes `config` into `LLMRunService`.
- `app/web/llm_engine_routes.py` - `GET /api/llm-runs/<id>/audit`,
  `GET /api/llm-engines/<id>/latest-audit`; the evidence endpoint's
  durable-audit fallback.
- `app/web/static/app.js` - `buildHowAIDecidedSection`,
  `buildAuditDownloadLink`, `llmPassPlainLanguage`,
  `llmEndCoarseToValidateExplanation`; Frames-sent-to-AI/evidence sections
  switched to read from the durable audit; `loadLatestAuditForEngine`/
  `fetchLLMAuditAndRender`.
- `app/web/static/styles.css` - `.llm-how-decided*`, `.llm-audit-actions`.
- `tests/test_llm_timing_pipeline.py`,
  `tests/test_llm_run_audit.py` (new),
  `tests/test_web_llm_engines.py`,
  `tests/test_llm_run_service.py` - new coverage per requirement #7
  below, plus the `app_config` fixture both LLMRunService test files now
  share (needed once its constructor takes an `AppConfig`).
- `tests_js/llm_engines.test.js` - unit tests for the two new pure
  helpers.
- `tests_js/playwright/e2e_server.py` - gained a `scenario` argument
  (`confirmed` default, `wrong_candidate` new) so the same harness can
  reproduce the operator's exact wrong-result shape deterministically.
- `tests_js/playwright/llm_flow.test.js` - a second top-level browser
  test for the wrong-candidate scenario; the first suite's "no console
  errors" check now excludes the one expected, harmless 404 from
  `loadLatestAuditForEngine` probing an engine that has never run
  (matched by URL via Playwright's `msg.location()`, not by the browser's
  generic message text, so an unrelated real console error still fails
  the check).

### Regression coverage (requirement #7, one item each)

- **Automatic persistence**: `test_a_completed_run_is_automatically_persisted_and_retrievable`.
- **Reload after a new app/service instance**: `test_audit_is_readable_from_a_brand_new_service_instance`
  and `test_audit_store_survives_a_fresh_instance_same_directory` - a
  second `LLMRunService`/`LLMRunAuditStore` pointed at the same
  `data_dir` reads what the first wrote; the in-memory job map does not
  carry over (proven 404ing), the durable audit does.
- **Exact post-thinning timestamps**: `test_build_audit_record_exact_post_thinning_timestamps_match_pass_frames`.
- **Per-pass raw_notes/reason/evidence**: `test_build_audit_record_for_a_confirmed_outcome_has_every_pass`.
- **Candidate -> window derivation**: `test_build_audit_record_for_a_rejected_candidate_still_shows_the_chain`
  (backend) and the whole second Playwright suite (browser, see below).
- **Safe partial audits**: `test_build_audit_record_for_a_run_that_never_reached_the_pipeline`,
  `test_a_missing_credential_failure_still_persists_a_safe_partial_audit`,
  `test_a_run_cancelled_before_it_starts_still_persists_a_safe_partial_audit`.
- **Secret/path/image-byte redaction**: `test_build_audit_record_redacts_secret_shaped_text_in_raw_notes_and_error`,
  `test_build_audit_record_never_carries_image_bytes_credentials_or_paths`,
  `test_audit_record_over_the_wire_never_carries_credentials_or_the_video_path`.
- **UI rendering**: JS unit tests for both new pure helpers; the browser
  suite's "How the AI decided explains the candidate -> window -> abstain
  chain" step reads the rendered DOM text, not just checks the section
  exists.
- **JSON download**: the browser suite's "Download audit report (JSON)"
  step actually clicks the link, captures the real download via
  Playwright's `download` event, and parses the saved file to confirm it
  matches the on-page chain.
- **Bounded cleanup**: `test_audit_store_purge_before_deletes_old_records_and_keeps_new_ones`,
  `test_purge_expired_also_removes_the_durable_audit`.
- **The wrong-candidate shape itself, in the browser**: the second
  Playwright suite reproduces the operator's exact numbers end to end
  (candidate 5.5s, window [1.5, 11.5]s, clip continuing to 26s) via a new
  `wrong_candidate` scenario in `e2e_server.py`, and asserts the rendered
  "End coarse" and "End validate" paragraphs name the candidate, the
  window bounds, and the abstain reason - not merely that a section with
  that heading is present.

### Gates

`pytest` - 604 passed (up from 578; +26: 2 new in
`test_llm_timing_pipeline.py` for `derived` and the wrong-candidate
chain, 18 in the new `test_llm_run_audit.py`, 6 new in
`test_web_llm_engines.py`'s `TestLLMRunAudit`, the rest mechanical
fixture-sharing changes, not new behavior). `ruff check .` /
`ruff format --check .` - clean (one file reformatted along the way).
`mypy app` - clean except the same pre-existing, unrelated
`app/web/routes.py:273` finding noted since §11 (still present with this
round's changes stashed out). `node --test tests_js/*.test.js` - 96/96
(up from 82; +14 for the two new pure helpers). Browser suite - two
top-level tests, 19 subtests total, all passing: the original confirmed-
flow suite (now also proving the durable-audit-driven rendering matches
the old job-driven one) plus the new wrong-candidate suite. Screenshots
at 1280px and 375px were captured and visually inspected for both
scenarios - the "How the AI decided" section stays legible and correctly
collapsed/expandable at both widths; a dedicated screenshot of the
expanded wrong-candidate explanation was also inspected directly (not
committed - regenerable by re-running the suite).

No real-provider smoke test was possible from this sandbox (no vendor
SDK installed, no API key here) - the same limitation noted in every
prior round. No known limitation remains for any of the seven numbered
requirements in the request. Per the request: **not merged**, PR #5
stays draft.
