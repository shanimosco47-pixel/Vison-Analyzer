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

**Fix** (commit TBD, prompt-only - "no pipeline expansion or extra
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
