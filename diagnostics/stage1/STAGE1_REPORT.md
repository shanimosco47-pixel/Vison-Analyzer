# Stage 1 — outlet tracking for the Zahn cup detector

Implementation evidence and limitations, per the supervisor's approval
comment on PR #4 (2026-08-20). **This is not ready to merge as a final
review** — it is the evidence package for Codex review, on a still-draft PR.

**Update, same PR:** the first Codex review round asked for five specific
changes (§6). All five were implemented, tested, and re-validated (§7).
A second review round of that revision found three remaining blocking
gaps plus two minor issues (§9); all were addressed and re-validated
(§10). A third review round found one remaining blocking gap in that
fix's own semantics (§12); it was addressed and re-validated (§13). A
fourth review round found a state-timing bug in *that* fix (§14); it is
addressed and re-validated (§15). A fifth, independent verification round
found and closed two reproducibility gaps (§16). A real hand-held clip was
then supplied for the first time and **fails the real-footage acceptance
criterion** at the current head; §17 is the diagnosis and a proposed
design. §18's four-point implementation of that design was re-validated
only on synthetic fixtures (§19–§20), since the real clip was not
available in this environment. A supervisor review then ran that exact
head against the real clip directly and found two further, concrete
blockers (§21): the frontend never actually sent `outlet_reference_s`,
and the reference-frame search could falsely lock onto the wrong part of
the frame. Both are addressed and re-validated (§22–§23), including a
runtime bug the new fixtures used to prove it caught in themselves before
this round was done (§21.4). A supervisor review then ran that head
(`07ec4ee`) against the real clip directly: the frontend fix and the t=0
false-lock fix both held, but the underlying translucent/low-texture
tracking failure (anticipated and flagged out of Stage 1's reach at
§18.3/§20) reproduced on the gate clip itself, plus a suspected frame
double-count. §24 is the root-cause analysis and design proposal
requested in response, plus the double-count's confirmed root cause -
**no implementation in this round**, per instruction. Sections 1–5 are
the original Stage 1
submission, left as-is as the historical record of the first round;
§6–§8 cover the first round's fixes; §9–§10 cover the second; §12–§13
cover the third; §14–§15 cover the fourth.

Product decision in force (final, per the approval comment): single Zahn
mode, outlet tracking on by default, `zahn_track_outlet` as a config-only
rollback lever, no user-visible mode switch. Real-footage validation: no
clip supplied, so validated against deterministic synthetic evidence only;
`zahn_max_endpoint_uncertainty_s = 0.5` shipped as the initial default,
documented as synthetic-only. Stage 2 and Stage 3 stay deferred, per Stage
0's recommendation and this approval.

---

## 1. What was built

`app/analysis/outlet_tracker.py` (new) — `OutletTracker`: Lucas-Kanade over
a feature set on the cup body, forward-backward validated, similarity
transform fit with `estimateAffinePartial2D` (RANSAC), carrying the outlet
as a point under that transform. States `tracked` / `predicted` / `lost`
exactly as specified; reacquisition after `lost` is a template match against
a reference patch anchored on the *strongest* detected feature (not the
outlet, and not a plain average of all features — see §4.2), independently
verified against `track_reacquire_min_correlation` before tracking resumes.

`app/analysis/zahn_detector.py` — `ZahnCupDetector.run()` decodes one fixed,
unscaled capture window per frame (guard region + `track_search_margin_px`
on every side) when tracking is on; the tracker estimates drift within it,
and the guard/ROI sub-crop used for scoring is re-cut from that same decoded
frame at the tracked position before being resized and blurred to the scale
scoring has always run at — no second decode. `FlowStateMachine` gained a
`trusted` parameter: `lost`/`predicted` frames reset persistence exactly as
disturbed frames already did, and additionally open a *gap* that is checked
when flow-end is confirmed — if the gap sitting between the last confirmed
liquid and the next trusted evidence is wider than
`zahn_max_endpoint_uncertainty_s`, `end_confirmed` is forced `False`,
`end_uncertain` is set, and the reported bounds are `[last_activity_s,
first_trusted_timestamp_after_the_gap]` — the earliest-supported timestamp,
with explicit uncertainty, never a falsely precise number.

`app/config.py` — eight new `ZahnConfig` fields (`zahn_track_outlet`,
`track_search_margin_px`, `track_max_frame_displacement_px`,
`track_min_inliers`, `track_min_features`, `track_max_bridge_s`,
`track_reacquire_min_correlation`, `zahn_max_endpoint_uncertainty_s`), each
independently justified and validated in `ZahnConfig.validate()`.
`_zahn_config_from_params` now delegates to the shared `apply_overrides`
instead of reimplementing it — the old hand-rolled version coerced booleans
wrong (`bool("false") == True`), a real bug for the new `zahn_track_outlet`
field coming from a web form.

Diagnostics: tracking-state transitions (timestamp, state, reacquired,
`roi`/`guard_roi`/`search_roi` in source coordinates) are recorded live
during the run — geometry only, not pixels, capped at
`MAX_DIAGNOSTIC_TRANSITIONS` — and exposed via
`DetectorResult.diagnostics["tracking_transitions"]` for `--save-diagnostics`
to render.

## 2. Test results

```
python -m pytest          324 passed        (Stage 0 baseline: 305 passed, 0 failed)
python -m ruff check .    All checks passed
python -m ruff format --check .   46 files already formatted
python -m mypy app        Success: no issues found in 29 source files (was 28)
node --test tests_js/*.test.js   44 pass, 0 fail
```

No pre-existing failure was touched. Every test that passed at the Stage 0
baseline still passes, unmodified, with tracking on by default — including
the existing static-camera Zahn integration tests
(`tests/test_pipeline_integration.py::TestZahnDetector`), which now exercise
the tracked code path and still assert the same tolerances they always did.
Two runs of the tracking-specific suites back to back produced identical
results (RANSAC's internal randomness does not make this flaky at these
tolerances).

New tests, 19 total:

* `tests/test_outlet_tracker.py` (12) — tracker states in isolation, no
  video decoding: init failure on a textureless patch, translation tracked
  to sub-3px error, combined camera+cup drift, an implausible single-frame
  jump rejected outright, a covered cup bridging `predicted` then `lost`,
  predicted extrapolation direction, reacquisition at the true new position,
  reacquisition *refused* against a differently-textured decoy (the
  "verified, not assumed" requirement), and a gap with no valid candidate
  anywhere in frame staying `lost` rather than guessing.
* `tests/test_zahn_tracking_integration.py` (7), over generated video with
  both the camera and the cup drifting and known ground truth:
  * the fixed-ROI path reproduces the original bug (error `> 0.5s`, always
    too long);
  * tracking (default) measures the same clip within the ±0.5 s synthetic
    tolerance;
  * flow start likewise within tolerance;
  * confidence for the hand-held clip is never higher than for the existing
    static `zahn_video` fixture;
  * **the true-break-during-a-gap case**: outlet swung out of frame across
    both the true stream break and the true end — asserts
    `end_confirmed is False`, `end_uncertain is True`, and that the reported
    bounds actually bracket both the true break and the true end (not just
    that a number exists);
  * the same gap clip under the fixed-ROI path, showing `frames_untracked ==
    0` — the fixed path has no notion of the gap at all, which is exactly
    the "falsely precise" risk tracking exists to close;
  * a runtime/memory bound (elapsed time under the clip's own duration, RSS
    growth bounded) so a regression here would fail loudly.
* `tests/conftest.py` — two new session-scoped fixtures
  (`handheld_zahn_video`, `handheld_gap_zahn_video`) built from
  `tests/_synthetic_handheld.py`, a trimmed adaptation of
  `diagnostics/stage0/make_handheld_clip.py`'s world/camera/hand-offset
  model, generating ground truth that is *drawn*, matching Stage 0's method.

## 3. Runtime and memory, before and after

Measured on the `handheld_zahn_video` fixture (480×360, 25 FPS, 22 s, both
camera and cup drifting, white-on-white):

| | efflux measured | error vs. 14.50 s true | status / confidence | frames | throughput | peak RSS growth |
| --- | --- | --- | --- | --- | --- | --- |
| **before** (`zahn_track_outlet=False`) | 16.28 s | **+1.78 s** | confirmed / 0.98 | 497 | 776 fps | 3.6 MB |
| **after** (tracking, default) | 14.28 s | **−0.22 s** | confirmed / 0.89 | 447 | 448 fps | 2.6 MB |

Tracking costs roughly 40% throughput on this clip (decoding one wider,
unscaled capture window per frame instead of a small pre-scaled crop) — still
~20× faster than real time on ordinary CPU, and memory is unaffected (still
one frame's worth of small crops held at a time; nothing scales with video
length). Confidence is lower under tracking (0.89 vs. the fixed path's
falsely-confident 0.98) even before any frame goes untracked, purely from
the fixed path's frozen background staying more "self-consistent" with
itself — which is precisely the mechanism Stage 0 identified as producing
false confidence, not genuine agreement with the truth.

## 4. Limitations — said explicitly, not left to be discovered

**4.1 — The real-footage acceptance criterion remains unverified.** No hand-held
clip was supplied (the approval explicitly authorised proceeding without
one). `zahn_max_endpoint_uncertainty_s = 0.5` and the ±0.5 s synthetic
tolerance are both calibrated only against generated footage. If real
footage later shows this needs adjustment, that is expected, not a defect
in this stage — see `diagnostics/stage0/STAGE0_REPORT.md` §7 for what
would be needed to close this out.

**4.2 — The reacquisition reference patch's anchor mattered more than expected.**
Initial implementation centred the patch on the outlet itself; end-to-end
testing on the gap fixture (§2) showed this produced spurious reacquisitions
against smooth wall background during genuine occlusion (a low-variance
patch correlates unreliably against other low-variance regions under
normalised cross-correlation) — the tracker flickered `lost`/`tracked`
through the whole occlusion window instead of staying `lost`. Fixed by
anchoring the patch on the single strongest detected feature
(`cv2.goodFeaturesToTrack`'s first, highest-response point — typically the
rim or handle) instead, which resolved it cleanly (§2's reacquisition tests,
plus the gap integration test's bounds now correctly bracketing the true
break). Documented here because it is exactly the kind of failure mode that
looks fine in isolation and only shows up under a real gap, which is why
the gap integration test exists rather than only the unit-level ones.

**4.3 — Deferred, per the Stage 0 recommendation and this approval:**
Stage 2 (continuity/contrast-anchor tracing) and Stage 3 (global
camera-motion compensation, distinct from cup tracking). Stage 0 measured
contrast as *not* the driver of the reported field bug in white-on-white
footage; a high-contrast, heavy-motion combination that would need Stage 2
is out of this stage's scope. Confirmed still true here:
`test_hand_held_confidence_is_not_higher_than_a_steady_run` and the
regression test both pass without it. The existing scene-wide disturbance
mechanism (`disturbance_area_ratio`) is unchanged and is a different concern
from cup tracking, as Stage 0 noted.

**4.4 — Start-time uncertainty is not bounded the same way end-time is.**
By construction, `flow_start_s` can only be reported *later* than the true
start when a gap coincides with it (persistence requires an unbroken
trusted run) — never earlier, never falsely precise. That asymmetry with
the true risk (a falsely-precise, too-early number) means start needed no
uncertainty-bound machinery to stay honest; only the end side, where a
falsely early confirmation was the actual failure mode, does. Noted here as
a deliberately scoped-down decision, not an oversight.

**4.5 — The capture window is fixed for the whole run, not re-centred.**
`track_search_margin_px` (default 80 px) bounds total drift the tracker can
follow across the *entire* recording, not per-segment. A recording where the
operator pans far enough, for long enough, to exceed that margin will show
increasing `lost` frames rather than the tracker re-anchoring further out.
This is the simpler of two designs considered in Stage 0's report; the
alternative (periodically re-seeking to a wider window centred on the last
known position) is more robust to very large sustained pans but adds real
complexity for a case not evidenced in the (synthetic, so far) test corpus.
Operators with footage that pans further than the default should raise
`track_search_margin_px` — documented in the README's settings table.

**4.6 — Disturbance status is not exposed as `clear`/`unavailable`.** The
performance/diagnostics section of the original brief asked for this, but it
depends on Stage 3's camera-motion-confidence estimate, which is deferred
per §4.3. The existing boolean `disturbed` diagnostic is unchanged and
accurate for what it measures (scene-wide pixel change); it does not yet
carry a distinct "compensation confidence too low to say" state.

## 5. What to look for in review

* `app/analysis/outlet_tracker.py` — the tracked → predicted → lost state
  machine, and the reacquisition patch anchor (§4.2).
* `app/analysis/zahn_detector.py::FlowStateMachine` — the `trusted` gating
  and the gap-uncertainty bookkeeping (`_gap_open`, `_gap_after_activity`),
  and specifically that it does **not** apply to plain scene-wide
  `disturbed` frames (a first draft did, and broke an existing pre-approved
  test — `test_disturbance_does_not_count_toward_the_end_persistence`;
  fixed by scoping the gap logic to `not trusted` only).
* `app/analysis/zahn_detector.py::ZahnCupDetector.run()` — `wide_capture` vs.
  the mutable `use_tracking`: the decode shape is fixed once the sampling
  plan starts, so the scoring branch must key off the fixed flag, not the
  one that can flip mid-run if tracker initialisation fails.

## 6. Codex review round — five required changes, addressed

### 6.1 — Feature selection is now genuinely cup-bounded

**Finding:** production feature selection was not actually cup-bounded — a
strongly textured, independently-moving background could out-compete the
cup for the similarity fit or the reacquisition anchor.

**Fix:** `outlet_tracker._feature_mask` / `_detect_features` now take
`outlet_xy`, `half_width_px`, and `height_above_px` and mask *both* axes to
a cup-sized box near/above the outlet, not just a vertical exclusion band
below it. `OutletTracker.__init__` requires `cup_half_width_px` /
`cup_height_above_px`; a new `_detect_cup_features()` wraps every detection
site (initial detection, re-detection when tracked points thin out,
post-reacquisition re-detection) so no code path can fall back to
unbounded, frame-wide detection. `ZahnCupDetector.run()` computes the bound
from the guard region itself (`cup_half_width_px = guard_roi.width // 2`,
`cup_height_above_px = guard_roi.height`) and passes it through.

**Evidence:** new regression
`tests/test_outlet_tracker.py::TestTracking::test_a_strongly_textured_independently_moving_background_is_ignored`
places a strongly textured block well outside the cup box, drifting in the
opposite direction from the cup so the two never overlap in-frame, and
asserts the tracked offset stays within tolerance of the cup's own motion.
Verified as a genuine regression test (not a tautology) by monkeypatching
the bound back to frame-wide and confirming it fails before the fix and
passes after.

**Regression discovered and fixed along the way:** tightening the bound
exposed that the pre-existing `zahn_video` fixture's cup — a bare
rectangle — has only 4 geometric corners, two of which sit inside
`_FEATURE_EXCLUSION_BELOW_OUTLET_PX` right at the outlet, leaving exactly 2
trackable features regardless of how the box was sized (below
`track_min_features = 4`). This broke 6 tests in
`test_pipeline_integration.py` and `test_web.py` that reuse that fixture
with tracking on. Root-caused with a diagnostic sweep of bound sizes
against the fixture (feature count stuck at 2 throughout). Fixed by giving
the fixture's cup real texture above the outlet — a rim ellipse and a
handle rectangle, both well clear of the exclusion zone — rather than
loosening the production bound or exclusion logic; confirmed via the same
sweep script that the corrected fixture now yields 20–55 features. The
enrichment does not touch the stream or timing pixels those tests assert
on.

### 6.2 — Tracker init failure now fails loudly, never silently falls back

**Finding:** when the tracker could not even initialise (e.g. no texture
above the outlet), the old code fell back to the fixed-ROI path — silently
using exactly the known-bad geometry Stage 0 diagnosed, and reporting it as
a normal (potentially confirmed) result.

**Fix:** `TrackerInitError` raised during first-frame construction now
returns `_tracking_unavailable_result()` — a `FAILED` result, zero events,
`frames_analysed = 0`, and an explicit reason in both `summary["reasons"]`
and `warnings`. Only an *explicit* `zahn_track_outlet=False` may use the
fixed-ROI path; tracking-on-by-default (the product decision) now means
tracking either works or the run fails honestly, never a silent downgrade.

**Evidence:** new `tests/test_zahn_tracking_integration.py::TestTrackingInitFailure`,
built on a new session-scoped `blank_video_path` fixture (a perfectly flat
clip — no texture above any point): `test_no_texture_above_outlet_fails_loudly_when_tracking_is_on`
asserts `status == "failed"`, no events, zero frames analysed, and a
tracking-specific warning; `test_the_same_clip_with_tracking_explicitly_off_does_not_fail`
confirms the same clip with `zahn_track_outlet=False` proceeds through the
ordinary fixed-ROI "no stream found" outcome instead of being blocked
before a single frame decodes — the one sanctioned rollback still works.

### 6.3 — Start-side uncertainty, symmetric to the end side

**Finding:** a long tracking gap around flow *start* was still falsely
precise — "the start can only be late" is not actually safe, because a
late-but-precise start reports a duration shorter than true, with full
confidence.

**Fix:** `FlowStateMachine` now opens a gap timer (`_gap_start_ts`) on any
untrusted sample, not only after flow has started. If flow is confirmed to
start immediately after such a gap closes, `_update_waiting()` sets
`measurement.start_uncertain` and `start_uncertainty_bounds` — the same
shape as the existing end-side machinery. `_status_for()` and
`score_confidence()` both treat `start_uncertain` exactly like
`end_uncertain`: forced `REVIEW`, confidence capped at 0.5, and an
explanatory reason naming the gap span.

**Evidence:** new fixture `handheld_gap_at_start_zahn_video` (occlusion
2.6–4.2 s, straddling the true 3.0 s start) and new
`TestBreakDuringATrackingGapAtStart` (2 tests):
`test_start_is_unconfirmed_with_bounds_not_falsely_precise` asserts
`start_uncertain`, bounds that bracket the true start, `REVIEW` status, and
confidence ≤ 0.5; `test_the_event_is_preserved_for_review_without_a_precise_duration`
checks the `Event` object itself carries no precise `efflux_seconds` (see
§6.4).

### 6.4 — Uncertain endpoints no longer emit a precise `efflux_seconds` or `Event`

**Finding:** even with `end_uncertain` / (now) `start_uncertain` set, the
result still emitted an exact, authoritative `efflux_seconds` and an
`Event` with a precise duration — the uncertainty flag existed but nothing
downstream actually honoured it.

**Fix:** `_build_result()` now suppresses `efflux_seconds` whenever
`measurement.start_uncertain or measurement.end_uncertain` — treated the
same as the existing `FAILED`-status suppression — and instead computes
`efflux_seconds_bounds` via a new `_efflux_bounds()` helper: the widest
plausible range from combining whichever endpoint(s) are uncertain with
the other's exact value (earliest-start/latest-end for the upper bound,
latest-start/earliest-end for the lower). The `Event` creation condition
no longer requires a non-`None` `efflux`, so a candidate is still preserved
for review — its `details["efflux_seconds"]` is `None` and
`details["efflux_seconds_bounds"]` carries the range instead, alongside the
existing `start_uncertain`/`end_uncertain` flags. `warnings` text now
distinguishes start-only, end-only, and both-uncertain phrasing rather than
one generic message.

**Evidence:** `summary["efflux_seconds"] is None` and
`summary["efflux_seconds_bounds"]` bracketing the true efflux time are now
asserted in both `TestBreakDuringATrackingGap` (end-side) and
`TestBreakDuringATrackingGapAtStart` (start-side); the new
`test_the_event_is_preserved_for_review_without_a_precise_duration` reads
`result.events[0].details` directly rather than only the summary, so a
regression that fixed the summary but left the `Event` payload precise
would still be caught.

### 6.5 — Bounded/streamed per-frame diagnostics

**Finding:** the state-transition JSON (capped at
`MAX_DIAGNOSTIC_TRANSITIONS`) does not show the tracked outlet position,
search window, scoring ROI, or state for the frames actually analysed —
below the requested contract of being able to inspect a run without
loading the full video into memory.

**Fix:** `ZahnCupDetector.run()` now opens (via `_open_tracking_frames_log()`)
a JSONL file when a `diagnostics_dir` param is present, and
`_write_tracking_frame_log()` appends one record per analysed frame —
`timestamp_s`, `state`, `trusted`, `reacquired`, `search_roi`, `roi`,
`guard_roi` — written and flushed to disk as the run progresses, so memory
use does not grow with run length. `run()` was split into `run()` (owns
the file's lifecycle, guaranteeing it closes via `finally` even on an early
return or exception) and `_run_loop()` (the per-frame loop, unchanged in
behaviour). `diagnostics_dir` is not a user-facing parameter: it is
injected by `AnalysisService._run_job()` — `<config.diagnostics_dir>/<job_id>`
— only when the server operator's `save_diagnostics` config flag is on,
independent from the existing `save_diagnostics` *request* param that
separately gates the lightweight, bounded, always-in-memory transitions
summary. The resulting path is surfaced via both
`DetectorResult.diagnostics["tracking_frames_log"]` and
`job.diagnostics_paths` (→ `diagnostics_files` in the job API response).

**Evidence:** new `TestTrackingFramesLog` (2 tests):
`test_a_frame_evidence_file_is_streamed_when_a_diagnostics_dir_is_given`
asserts the file exists, has exactly one JSON line per analysed frame with
the documented keys, and monotonic timestamps;
`test_no_file_is_written_without_a_diagnostics_dir` confirms the default
(no `diagnostics_dir` param) writes nothing to disk. Overhead measured on
`handheld_zahn_video` (447 frames): 0.881 s mean without the log vs.
0.893 s mean with it, over 3 trials each — about 1% — see §7.

## 7. Re-validation after the fixes

```
python -m pytest          331 passed        (pre-round: 324 passed, 0 failed)
python -m ruff check .    All checks passed
python -m ruff format --check .   all files already formatted
python -m mypy app        Success: no issues found in 29 source files
node --test tests_js/*.test.js   44 pass, 0 fail
```

7 new tests this round (§6.1–§6.5); no pre-existing test was modified to
pass except where §4's assertions were *strengthened* (adding
`efflux_seconds is None`/bounds checks to the existing gap tests, per
§6.4) — nothing was loosened.

**Runtime, before this round's fixes vs. after**, measured on
`handheld_zahn_video` (480×360, 25 FPS, 22 s, both camera and cup
drifting), comparing the pre-review commit (`0c8f5c4`, checked out into a
scratch worktree) against the current tree, 3 trials each, tracking on
(default), no `diagnostics_dir`:

| | mean elapsed | trials |
| --- | --- | --- |
| before (`0c8f5c4`) | 0.911 s | 0.923 / 0.890 / 0.921 |
| after (this revision, no diagnostics log) | 0.881 s | 0.925 / 0.866 / 0.851 |
| after, with `diagnostics_dir` set (streamed log on) | 0.893 s | 0.923 / 0.868 / 0.888 |

The five fixes — tighter feature bounding, the init-failure early return,
the extra gap bookkeeping for start uncertainty, and the bounds
computation — cost nothing measurable (within run-to-run noise); streaming
447 per-frame JSON records to disk adds roughly 1%. Peak memory remains
bounded by construction: the frames log is written and flushed one line at
a time, never accumulated.

## 8. What to look for in this round's review

* `app/analysis/outlet_tracker.py::_feature_mask` / `_detect_cup_features`
  (§6.1) — both axes now bounded, and every detection call site routed
  through the same bounded method.
* `app/analysis/zahn_detector.py::ZahnCupDetector.run()` /`_run_loop()`
  (§6.2, §6.5) — `TrackerInitError` handling returns before any frame is
  scored, and the `run()`/`_run_loop()` split exists solely so the frames
  log's `finally`-close is guaranteed regardless of how the loop exits.
* `app/analysis/zahn_detector.py::FlowStateMachine` (§6.3) — the
  `_pending_start_gap` bookkeeping mirrors the existing end-side gap logic
  deliberately; a change to one without the other is worth flagging.
* `app/analysis/zahn_detector.py::ZahnCupDetector._build_result` /
  `_efflux_bounds` (§6.4) — the suppression condition
  (`start_uncertain or end_uncertain`) and that it applies to the `Event`
  payload, not only `summary`.
* `app/services/analysis_service.py::AnalysisService._run_job` (§6.5) —
  `diagnostics_dir` is only ever injected server-side, gated on the
  operator config flag, never accepted as a request param.

## 9. Second Codex review round — three blocking gaps, addressed

Codex re-reviewed `c0e4cc8` (§6–§8's revision): the five original findings
were substantially addressed, but re-review surfaced three remaining
blocking gaps, plus two minor issues. All five are fixed below.

### 9.1 — An uncertain measurement still produced an Event with a precise `duration_s`

**Finding:** §6.4's fix set `details["efflux_seconds"]` to `null` on the
`Event`, but `Event.duration_s` is a property computed fresh as
`end_s - start_s`, independent of `details` — and it is what
`Event.to_dict()` serializes, what `EventLog.rows()` (and therefore the
CSV export) reads, and what `summarise()` sums into
`total_active_seconds`. The uncertainty flag existed on the Event but
nothing in the shared Event/EventLog contract actually respected it, so a
precise duration was still reachable through every one of those paths.

**Fix, exactly as Codex's own suggested direction:** rather than change the
shared `Event` contract (used by every detector) to support a genuinely
unknown/bounded duration, `_build_result()` now emits **no `Event` at all**
when `measurement.start_uncertain or measurement.end_uncertain`. The
bounded candidate remains fully visible in `summary`
(`flow_start_s`/`flow_end_s`/`efflux_seconds_bounds`/`start_uncertain`/
`end_uncertain`) — nothing about being able to inspect an uncertain
measurement was lost — it simply never becomes an `Event` a downstream
consumer (UI, API, CSV, event count) could mistake for a confirmed
occurrence.

**Evidence:** both `TestBreakDuringATrackingGap` (end-uncertain) and
`TestBreakDuringATrackingGapAtStart` (start-uncertain) in
`tests/test_zahn_tracking_integration.py` now include
`test_no_event_is_emitted_for_an_uncertain_measurement`, using a shared
`_assert_no_precise_duration_leaks()` helper that checks, at every layer
Codex named: `result.events == []`, `DetectorResult.to_dict()["events"] ==
[]`, `build_event_log(...).rows() == []`, the CSV export has no data row,
and `summarise(...)["total_active_seconds"] == 0`.

### 9.2 — `diagnostics_dir` was a client-controlled filesystem write path

**Finding:** `POST /analyses` forwards its `params` dict essentially
unchanged; `AnalysisService.submit` stored it on the job; and
`ZahnCupDetector.configure()` read `params["diagnostics_dir"]` straight
through — so `_open_tracking_frames_log()` would `mkdir`/write wherever a
caller named, even with `AppConfig.save_diagnostics` off. A caller could
choose any writable server path.

**Fix:** `analysis_service.py` gained `RESERVED_PARAM_KEYS = {"diagnostics_dir"}`
and `_sanitize_params()`, applied in `submit()` **before** the detector is
even constructed for planning and **before** the value is stored on
`job.params` — a request can no longer get this key queued at all, let
alone acted on. `_run_job()` was also hardened to assign (not merely
`setdefault`) `diagnostics_dir` from `AppConfig.diagnostics_dir` only when
`self.config.save_diagnostics` is on, popping any stray key first — belt
and suspenders on top of the earlier sanitization.

**Evidence:** two new regression tests in `tests/test_web.py`:
`test_a_submitted_diagnostics_dir_is_stripped_and_cannot_write_anywhere`
posts a `diagnostics_dir` pointed at an attacker-chosen `tmp_path`
directory against a default (`save_diagnostics=False`) app, and asserts
both that the stored job's params never contain the key and that the
directory is never created; `test_diagnostics_dir_stays_server_owned_even_with_diagnostics_enabled`
does the same against a `save_diagnostics=True` app and additionally
confirms the legitimate frames log is written under the server's own
`AppConfig.diagnostics_dir / job_id` instead.

### 9.3 — A long tracking gap in the *middle* of flow was forgotten once liquid returned

**Finding:** `FlowStateMachine._update_flowing` cleared `_gap_after_activity`
the instant fresh trusted liquid arrived. A gap wider than
`zahn_max_endpoint_uncertainty_s` occurring mid-flow — liquid before,
liquid after, then a later, ordinary end — left no trace by the time that
end was confirmed, so the result came back `CONFIRMED` even though a span
long enough to lose confidence in had gone unobserved. This conflicts with
the acceptance criterion that a measurement be invalid/unconfirmed when
tracking is lost longer than the defined duration, wherever in the run
that happens.

**Fix:** `FlowStateMachine` gained `_unresolved_flow_gap`, set once (the
first qualifying gap only, never overwritten) whenever a gap closes while
flowing and exceeds `zahn_max_endpoint_uncertainty_s` — and, unlike
`_gap_after_activity`, deliberately **never cleared** by subsequent trusted
liquid. When an end is eventually confirmed, both signals are combined:
if either the immediately-adjacent gap or any earlier unresolved one
qualifies, `end_confirmed` is forced `False`, `end_uncertain` is set, and
`end_uncertainty_bounds` becomes the widest span covering whichever
qualifying gap(s) were seen.

**Evidence:** new `TestUntrackedGapsDuringFlow` in
`tests/test_zahn_state_machine.py` (fast, unit-level — no video decoding,
`FlowStateMachine.update(..., trusted=False)` fed directly): a >0.5 s
untracked gap in the middle of an otherwise clean run (liquid before,
liquid after, a later normal end) now reports `end_confirmed=False`,
`end_uncertain=True`, with bounds bracketing the mid-flow gap even though
the reported `end_s` itself is unchanged; a companion test confirms a
short (<0.5 s) mid-flow gap does *not* taint the result, isolating the
behaviour to gap width, not position.

### 9.4 — Minor: stale `TrackerInitError` docstring

**Finding:** the docstring still instructed callers to "fall back to the
fixed-ROI path" — exactly the behaviour §6.2 removed.

**Fix:** rewritten to state the actual (and only sanctioned) contract:
report failure with no confirmed measurement while `zahn_track_outlet=True`;
the fixed-ROI path is available only via an explicit
`zahn_track_outlet=False` set before tracking was ever attempted, never as
a reaction to this exception.

### 9.5 — Minor: the frames log wrote each line but did not flush as claimed

**Finding:** `_open_tracking_frames_log()`'s docstring said lines are
"written and flushed... as the run progresses," but `_write_tracking_frame_log()`
only called `.write()` — Python's own buffering could hold lines back for
an arbitrary, unbounded time, so the file was not actually inspectable
mid-run the way the report claimed.

**Fix:** added an explicit `frames_log.flush()` after every write, so the
claim is now literally true rather than merely aspirational.

## 10. Re-validation after the second round's fixes

```
python -m pytest          336 passed        (previous: 331 passed, 0 failed)
python -m ruff check .    All checks passed
python -m ruff format --check .   all files already formatted
python -m mypy app        Success: no issues found in 29 source files
node --test tests_js/*.test.js   44 pass, 0 fail
```

5 new tests this round: 2 in `test_zahn_state_machine.py` (§9.3), 1 in
`test_zahn_tracking_integration.py` (§9.1 — `TestBreakDuringATrackingGap`
gained a no-Event assertion; `TestBreakDuringATrackingGapAtStart`'s
existing Event test was rewritten in place rather than added to), 2 in
`test_web.py` (§9.2).

**Runtime**, same `handheld_zahn_video` fixture, 3 trials each, after this
round's fixes (the new per-frame `flush()` call from §9.5 is the only
change with any plausible runtime cost):

| | mean elapsed | trials |
| --- | --- | --- |
| no `diagnostics_dir` | 0.880 s | 0.940 / 0.861 / 0.838 |
| with `diagnostics_dir` (now flushed per line) | 0.897 s | 0.898 / 0.872 / 0.921 |

Consistent with the first round's measurements (§7) — no regression;
flushing every line costs roughly 2% over the already-negligible
streaming overhead, still comfortably within run-to-run noise.

## 11. What to look for in this round's review

* `app/analysis/zahn_detector.py::ZahnCupDetector._build_result` (§9.1) —
  the Event-creation condition now includes `not uncertain`; `summary`
  still carries the full bounded candidate.
* `app/services/analysis_service.py::_sanitize_params` /
  `AnalysisService.submit` / `_run_job` (§9.2) — sanitization happens
  before the job is even constructed, not only before the detector runs.
* `app/analysis/zahn_detector.py::FlowStateMachine._unresolved_flow_gap`
  (§9.3) — set once, on the first qualifying gap, and specifically **not**
  cleared by `_update_flowing`'s liquid-resumes branch (contrast with
  `_gap_after_activity` just above it, which still is).

## 12. Third Codex review round — one remaining blocking gap, addressed

Codex re-reviewed `82dda38` (§9–§11's revision): the Event/CSV/summary
duration leak, the server-owned diagnostics path, and the stale
documentation/flush issues were all confirmed sound. One blocker remained
in the *semantics* of §9.3's own fix.

### 12.1 — The mid-flow gap's own span was still (wrongly) reported as `end_uncertainty_bounds`

**Finding:** §9.3 added `_unresolved_flow_gap` to stop a mid-flow gap from
being forgotten, but the fix folded that gap's own `(gap_start, gap_end)`
timestamps into `end_uncertainty_bounds` alongside the genuinely-adjacent
case. That is backwards: trusted liquid *was* seen again after the gap
closed - that is precisely what makes it "mid-flow" rather than
"adjacent-to-the-end" - which proves the true end is **not** "somewhere in
that gap". Reporting the gap's span as the end's bound therefore fabricated
a range that could actively **exclude** the real, later end and duration.
The new unit test from §9.3 only checked that the bounds bracketed the
gap itself, which codified the wrong semantics rather than catching the
bug.

**Fix:** `FlowMeasurement` gained a dedicated `end_gap_unresolved: bool`
field, separate from `end_uncertain`/`end_uncertainty_bounds`.
`_unresolved_flow_gap` changed from carrying a `(start, end)` span to a
plain `bool` - there is no honest span to carry. When it fires,
`end_confirmed` is forced `False` and `end_uncertain`/`end_gap_unresolved`
are both set `True` (still suppressing the Event and the precise
`efflux_seconds`, exactly as for any other uncertain measurement), but
`end_uncertainty_bounds` is left untouched by it - populated only by the
existing, still-correct adjacent-gap logic (`_gap_after_activity`, which
is itself only non-`None` when *no* trusted liquid was seen between it
closing and the end being confirmed - i.e. it was never actually
susceptible to this bug). `_efflux_bounds()` now returns `None` outright
(not a degenerate range collapsed onto the raw measurement) when
`end_gap_unresolved` is set and no independently-valid
`end_uncertainty_bounds` exists - "null when the evidence only establishes
invalidity," per the review's own suggested contract. The `score_confidence`
reason and the `_build_result` review warning were both rewritten for this
case specifically: they no longer say the true end "could have occurred
earlier" (contradicted by the evidence - liquid *was* seen afterward);
instead they say plainly that continuity through the gap could not be
verified.

A mid-flow gap can still coexist with a *separate*, genuinely adjacent gap
right before the end (two different gaps in the same run) - that case is
unaffected: the adjacent gap's own bounds are still honestly computed and
reported, `end_gap_unresolved` from the earlier one is layered on top
purely to keep the measurement unconfirmed, without touching those bounds.

**Evidence:**
* `tests/test_zahn_state_machine.py::TestUntrackedGapsDuringFlow` rewritten:
  `test_a_long_gap_in_the_middle_of_flow_still_poisons_a_later_clean_end`
  now asserts `end_uncertainty_bounds is None` (previously asserted the
  fabricated bound); a new
  `test_a_mid_flow_gap_plus_a_separate_adjacent_end_gap_keeps_the_adjacent_bounds`
  covers the two-gaps-in-one-run case, confirming the adjacent gap's
  bounds survive untouched.
* New `tests/test_zahn_tracking_integration.py::TestUnresolvedGapInTheMiddleOfFlow`,
  over a new synthetic fixture `handheld_gap_mid_flow_zahn_video`
  (occlusion 8.0–9.6s, entirely inside the continuous-stream window,
  nowhere near the true start or break/end - liquid trustedly visible
  immediately before and after, flow then runs normally to an ordinary
  later end): `test_the_measurement_is_unconfirmed_without_a_fabricated_bound`
  asserts `end_confirmed is False`, `end_gap_unresolved is True`,
  `end_uncertainty_bounds is None`, `efflux_seconds_bounds is None`, and
  that the reported `flow_end_s` itself stays close to the true end (its
  own timing was never actually in question);
  `test_no_event_is_emitted_and_the_warning_does_not_claim_an_earlier_end`
  reuses the §9.1 `_assert_no_precise_duration_leaks` helper and asserts
  the literal phrase "could have occurred earlier" is absent from the
  warning text.

## 13. Re-validation after the third round's fix

```
python -m pytest          339 passed        (previous: 336 passed, 0 failed)
python -m ruff check .    All checks passed
python -m ruff format --check .   all files already formatted
python -m mypy app        Success: no issues found in 29 source files
node --test tests_js/*.test.js   44 pass, 0 fail
```

3 new tests this round: 1 in `test_zahn_state_machine.py` (the two-gaps
case), 2 in `test_zahn_tracking_integration.py` (the new mid-flow fixture).
One existing unit test's assertions were corrected in place (§12.1) rather
than added to, since it had codified the wrong semantics.

Runtime, same `handheld_zahn_video` fixture, 3 trials, no `diagnostics_dir`:
0.838 / 0.865 / 0.875 s, mean 0.859 s - consistent with every prior
round's measurements (§7, §10); this round changed only how bounds are
computed at end-confirmation, not what gets decoded or how often.

## 14. Fourth Codex review round — a state-timing bug in the third round's own fix

Codex re-reviewed `34e0391` (§12–§13's revision): the bounds fix was
confirmed directionally correct, but flagged one remaining state-semantics
bug in exactly the code that fix touched.

### 14.1 — `end_gap_unresolved` was promoted at gap-close time, before knowing what followed

**Finding:** `update()`'s gap-close handling set `_unresolved_flow_gap =
True` the moment a qualifying gap closed while flowing - *before* the
first trusted sample after it had even been classified as liquid or
absence. That meant the *ordinary* adjacent-to-the-end case (a gap closes,
then only trusted absence follows, confirming the end - liquid never
resumes) was also being labelled `end_gap_unresolved`, even though nothing
demonstrated resumption. Such a measurement then got the mid-flow warning
("liquid was seen again afterward"), which is simply false for that case -
exactly the wording §12.1 introduced specifically to avoid saying when
liquid genuinely never returned.

A second, related gap: when a *real* mid-flow gap and a separate,
genuinely adjacent end gap coexisted in the same run, the `if/elif` chains
in both `score_confidence` and `_build_result`'s warnings reported only
whichever check ran first, silently dropping the other's genuine reason.

**Fix:**
* The promotion of `_unresolved_flow_gap` moved out of `update()`'s
  gap-close block entirely, into `_update_flowing`'s liquid-present
  branch - the one point that actually demonstrates resumption. A gap
  closing now only ever populates `_gap_after_activity` (a *candidate*);
  whether it becomes an honest adjacent bound (absence follows, never
  promoted) or `end_gap_unresolved` (liquid resumes, promoted right there
  before `_gap_after_activity` is cleared) is decided by what happens
  next, never by the gap closing alone.
* `score_confidence`'s `not end_confirmed` block and `_build_result`'s
  `REVIEW`-status warnings block were both changed from `if/elif` chains
  to independent `if`s (with a `has_adjacent_end_bound` guard requiring
  genuine bounds, not just `end_uncertain`), so a mid-flow reason and an
  adjacent-gap reason are both reported when both genuinely apply, instead
  of one masking the other.

**Evidence:**
* `tests/test_zahn_state_machine.py::TestUntrackedGapsDuringFlow::test_a_gap_immediately_adjacent_to_the_end_is_not_mislabelled_mid_flow` -
  the case the bug actually manifested in: one qualifying gap, trusted
  absence only afterward, asserts `end_gap_unresolved is False`.
* `tests/test_zahn_tracking_integration.py::TestBreakDuringATrackingGap::test_this_adjacent_gap_is_not_mislabelled_as_a_resumed_mid_flow_gap` -
  the same case over real synthetic video (the existing
  `handheld_gap_zahn_video` fixture, occlusion spanning both the true
  break and the true end - liquid never returns), asserting
  `summary["end_gap_unresolved"] is False` and that the warning text
  contains "could have occurred earlier" but neither "seen again" nor
  "resumed".
* `tests/test_zahn_state_machine.py::TestConfidence::test_a_mid_flow_gap_and_a_separate_adjacent_gap_both_get_a_reason` -
  two distinct gaps in one run (one genuinely mid-flow, one genuinely
  adjacent), asserting `score_confidence`'s `reasons` contains *both* the
  mid-flow phrase and the adjacent-gap phrase, not just one.
* The existing two-gap test from §12
  (`test_a_mid_flow_gap_plus_a_separate_adjacent_end_gap_keeps_the_adjacent_bounds`)
  still passes unchanged - it happened not to distinguish the buggy and
  fixed promotion timing (both set `end_gap_unresolved=True` there, since
  a genuine mid-flow gap was already present), which is exactly why the
  dedicated adjacent-only test above was needed to actually catch this.

## 15. Re-validation after the fourth round's fix

```
python -m pytest          342 passed        (previous: 339 passed, 0 failed)
python -m ruff check .    All checks passed
python -m ruff format --check .   all files already formatted
python -m mypy app        Success: no issues found in 29 source files
node --test tests_js/*.test.js   44 pass, 0 fail
```

3 new tests this round, all described above. Runtime, same
`handheld_zahn_video` fixture, 3 trials, no `diagnostics_dir`: 0.808 /
0.824 / 0.900 s, mean 0.844 s - consistent with every prior round; this
was purely a state-timing and reason-composition fix, no change to what
gets decoded.

## 16. Independent verification round — Windows reproducibility and evidence accuracy

After acceptance at `7181500`, the supervisor independently verified from a
fresh Python 3.12 virtualenv built from `requirements-dev.txt`, including on
Windows, and reopened acceptance with three findings. None touch the
tracking/timing logic itself; all three are about reproducibility and
evidence accuracy.

### 16.1 — `tests/test_zahn_tracking_integration.py` could not be collected on Windows

**Finding:** the module imported the POSIX-only `resource` module at module
scope (used by exactly one test, for peak-RSS measurement), so importing it
at all raised `ModuleNotFoundError` on Windows before pytest could collect
any test in the file - not just the one test that actually needs it.

**Fix:** the import is now wrapped in `try`/`except ImportError`, falling
back to `resource = None`. `test_runtime_and_memory_are_bounded` (the one
test that reads `resource.getrusage`) is now decorated
`@pytest.mark.skipif(resource is None, ...)`, so it - and only it - is
skipped on Windows; every other test in the module, including the rest of
`TestPerformance`, collects and runs normally. A new
`test_runtime_is_bounded_without_the_platform_specific_rss_check` keeps the
portable half of the check (elapsed time under the clip's own duration)
running unconditionally on every platform, so Windows still gets a runtime
sanity check even without the RSS measurement.

**Verified** (no Windows machine available here) by simulating the failure
mode directly: monkeypatching `builtins.__import__` to raise `ImportError`
for `resource` and re-importing the module fresh confirms it now imports
cleanly with `resource is None`, and running `TestPerformance` under that
same simulated condition collects both tests, skips exactly the RSS one,
and passes the portable one.

### 16.2 — A clean install could resolve a NumPy version mypy cannot check under this project's Python target

**Finding:** `numpy>=1.26,<3` let a clean install resolve NumPy 2.5.2,
whose own `.pyi` stubs (`numpy/__init__.pyi`) use PEP 695 `type` statement
syntax - syntax mypy only accepts under `python_version >= "3.12"`. This
project's mypy config targets `python_version = "3.10"` (the project's own
minimum supported Python), so `mypy app` fails while merely *parsing*
NumPy's stub file, with no code of this project's own involved at all. This
is a known, currently-unresolved mypy limitation
([python/mypy#21178](https://github.com/python/mypy/issues/21178)): syntax
gating by `python_version` is applied even inside third-party stub files.

**Fix:** `numpy` is now bounded `>=1.26,<2.5` in both `pyproject.toml` and
`requirements.txt`, with a comment explaining why. Verified directly in
this environment: the installed NumPy 2.4.6 (the newest release satisfying
the new bound) contains no `type` statements anywhere in its stub tree
(`grep` came back empty), and `mypy app` passes cleanly against it. The
exact 2.5.2 reproduction could not be re-run here (this environment's
package index does not offer a version past 2.4.6), so the fix is verified
by confirming the newest *allowed* version is clean, not by reproducing the
failing version - noted explicitly rather than claimed as directly
reproduced.

### 16.3 — The report's `ruff format --check .` claims were not accurate at full-repo scope

**Finding:** every round's evidence block in this report claims
`ruff format --check .` (or equivalent) passed, but that was only ever
actually run against the files touched in that round - five pre-existing
`diagnostics/stage0/*.py` scratch scripts (committed before Stage 1, never
touched by any of this stage's changes) would be reformatted, so the
repo-wide claim was false as written.

**Fix:** formatted those five files (`ablate.py`, `make_handheld_clip.py`,
`save_frames.py`, `track_probe.py`, `widen_roi.py`) - a pure, deterministic
whitespace/line-wrap operation with no behavioural change, confirmed by
`git diff` showing only reformatting. One additional line
`ruff format` could not auto-wrap (an f-string past the 100-column limit in
`save_frames.py`) was split by hand. This makes every prior round's
`ruff format --check .` claim retroactively true at full-repo scope, rather
than requiring every evidence line in six sections of report history to be
rewritten to a narrower, previously-undocumented scope.

### 16.4 — Re-validation

```
python -m pytest          343 passed        (previous: 342 passed, 0 failed)
python -m ruff check .    All checks passed
python -m ruff format --check .   55 files already formatted (repo-wide, no exclusions)
python -m mypy app        Success: no issues found in 29 source files
node --test tests_js/*.test.js   44 pass, 0 fail
```

1 new test this round (§16.1's portable runtime check). Windows collection
and the skip behaviour were verified by simulation, per §16.1, since no
Windows machine is available in this environment - **exact clean-environment
Windows pass/skip counts, and confirmation of the NumPy bound against the
originally-reported 2.5.2, remain to be run by the supervisor** on their
own Windows/clean-venv setup, as neither is reproducible from here.

## 17. First real-footage validation — reopened, diagnosis and proposed design (not yet implemented)

A real hand-held clip (`20260820_184144.mp4`, 1080×1920 portrait, 811
frames, 30.0028 fps, 27.03 s) was supplied for the first time and run
against `c1d091f`. It fails: `FAILED` status, no start/end/duration,
806–811 of 811 (or 671–676 of 676) frames untracked depending on params.
Manual frame review puts the true flow window at roughly 3.8–20.6 s
(efflux ≈ 16.6 s ± 0.2 s review uncertainty). Two blockers were named. This
section is the diagnosis and a proposed design for both, **stopped here
before any implementation**, per the instruction not to attempt a broad
rewrite without review. No production or test code changed in this round -
only this report.

### 17.1 — Blocker 1: the frame-picker's timestamp never reaches the backend

**Confirmed by reading the code.** `loadFrameFromPreview()`
(`app/web/static/app.js:371`) stores the picker frame's timestamp in
`state.frameTime`, but `collectParams()` (`app/web/static/app.js:492`)
never reads it - only `state.outlet`/`state.roi` are sent.
`ZahnCupDetector.configure()` (`app/analysis/zahn_detector.py:749`) takes
those coordinates as `self.outlet_xy` with no associated timestamp, and
`run()` constructs the `OutletTracker` on the *first* decoded sample -
i.e. at `analysis_start_s` (default `0.0`), never at whatever frame the
user actually looked at. A mark made on a later preview frame is silently
reinterpreted as a mark on frame 0 (or whatever `analysis_start_s` is).

**Synthetically confirmed, with a second, deeper mechanism found in the
process.** A portrait (1080×1920), resolution-scaled synthetic clip (see
`build_handheld_clip` in `tests/_synthetic_handheld.py`, `camera_drift_px`/
`hand_drift_px`/`tremor_px` scaled ×2.25 to represent the same real-world
motion at higher pixel density; ground truth `flow_start_s=3.9`) was run
three ways:

| scenario | `outlet` marked at | `analysis_start_s` | result |
| --- | --- | --- | --- |
| A (baseline) | t=0 (matches default) | 0 (default) | confirmed, 0.98 confidence, `flow_start_s=3.9` - correct |
| B (the reported bug) | t=4.5 s | 0 (unchanged - the bug) | confirmed, 0.98 confidence, `flow_start_s=3.9` - correct *in this run*, because the t=0→t=4.5 drift in this synthetic clip (≈26 px) was small enough for the tracker to self-correct from a slightly wrong anchor. **This does not mean the bug is harmless** - see the real clip's reported 800+/811 untracked frames, which implies drift far larger than 26 px; the synthetic clip's drift magnitude was a rough estimate, not measured from the real footage, so scenario B under-reproduces the real severity. It reproduces the *mechanism*, not the *magnitude*. |
| C (the user's attempted workaround) | t=4.5 s | 4.5 (matched) | **`status=review`, `flow_start_s=20.0`, `flow_end_s=26.97`, `efflux_seconds=6.97`** - wrong by roughly 15 s against the true 3.9 s start, with `confidence=0.5` and reason "video ended while liquid was still visible." Silently, badly wrong - not just imprecise. |

Scenario C is the important one: it reproduces the report's "starting there
can discard or contaminate the true start" concretely, and running it down
found the actual mechanism, distinct from the tracking/geometry question
in §17.2 below - **`StreamActivityScorer`'s background model**
(`app/analysis/zahn_detector.py:216-224`):

```python
if self._background is None:
    self._background = current.copy()
    return ScoreSample(value=0.0, extras={"outlet_score": 0.0, "noise_sigma": 0.0})
```

The very first frame the scorer ever sees becomes the "empty cup"
baseline, unconditionally - there is a real safeguard against the stream
ever being absorbed into the background afterward (line 268: "the
background is only refreshed while nothing is happening, so the stream
can never dissolve into the reference image"), but that safeguard begins
only from the *second* frame. If `analysis_start_s` lands after flow has
already started - exactly what happens when a user sets it to the first
frame where the outlet becomes markable, because the outlet was not
markable any earlier - the first frame already shows the stream, so the
stream itself becomes "background," and the scorer stops seeing it as
liquid for as long as it stays in the same place. Scenario C's continuous
stream (3.9-20.0 s) sits still relative to the tracked crop and gets
absorbed; the intermittent drops after 20.0 s move to a new position each
frame and stand out against the (now stream-shaped) background, which is
why detection only picks up around t≈20.0 - matching the observed
`flow_start_s=20.0` exactly.

**This is the load-bearing finding for the design below:** fixing only the
outlet-coordinate/reference-frame mismatch is not sufficient. Any design
that lets `analysis_start_s` (i.e. where scoring/background-model decoding
begins) land after flow has already started will keep producing this
failure mode, independent of whether the tracker itself is correctly
anchored.

### 17.2 — Blocker 2: tracking not robust on this real cup/reframing

Even with `analysis_start_s=4.5` and matching coordinates, the supervisor
reports the tracker goes untracked almost immediately (671/676 untracked,
4 liquid frames) - a different failure from §17.1's background-model
problem, since that scenario already has the coordinates and start time
consistent with each other.

**What was checked and ruled out:** the ROI/guard/cup-feature-search-box
geometry (`default_roi_width_px=60`, `guard_margin_px=40`,
`track_search_margin_px=80` - all fixed absolute pixels, independent of
`video.width`/`video.height`) was a plausible suspect, since the real clip
is 1080×1920 versus this project's ~480×360 synthetic fixtures. Testing a
resolution-scaled synthetic clip at the real clip's exact resolution with
proportionally scaled camera/hand drift (scenario A above) tracked
cleanly - 0 untracked frames, 0.98 confidence. **The fixed-pixel geometry
constants are not, by themselves, sufficient to reproduce a tracking
failure at this resolution and this drift magnitude.** That does not clear
them entirely (the real clip's actual drift magnitude and cup pixel size
are unknown here - "large early reframing" could still exceed what was
tested), but it means resolution alone is not the likely root cause.

**What could not be checked here:** the real clip's outlet region is
described as "low-texture translucent cup" - material behaviour
(translucency, refraction as liquid moves behind/through the cup wall,
low local contrast) that this project's synthetic generator does not
attempt to model (its cup is a flat, opaque intensity block plus a rim and
handle, added in an earlier round specifically to *avoid* too few
trackable corners - see the Codex-review §6.1 fixture fix). Whether
`cv2.goodFeaturesToTrack`'s corner detection genuinely cannot find enough
stable corners on real translucent material, or whether frame-to-frame LK
correspondence is unstable under real refraction/reflection changes, or
whether `track_min_features=12`/`track_min_inliers=6`/
`track_reacquire_min_correlation=0.6` are simply mistuned for this
material, cannot be determined without either the real clip or a
synthetic fixture that actually models low-texture/translucent rendering
- neither of which exists yet.

**Recommended next step, not yet done:** the streamed per-frame
diagnostics built this session (`diagnostics_dir`, §6.5/§9.5) exist
precisely for this - re-running the real clip with diagnostics enabled
would show, frame by frame, the tracked outlet position, search window,
scoring ROI, state and reacquisition decisions, which would show *where
and why* tracking is failing (feature count trending to zero? RANSAC
inlier count collapsing? reacquisition never clearing
`track_reacquire_min_correlation`?) rather than guessing from aggregate
counts. That diagnosis has to happen against the real clip or an
accurately modelled synthetic one; it was not attempted here since the
clip is not available in this environment and a low-texture/translucent
synthetic model does not exist yet to build one from.

### 17.3 — Proposed design for §17.1 (recommended: correctable within Stage 1)

The key insight from §17.1's investigation: the fix does **not** need
bidirectional/backward tracking or a second decode pass. It can reuse the
gap-uncertainty machinery already built and tested this stage (§6.3, §9.3,
§12.1, §14.1) almost unchanged:

1. **Wire the timestamp through.** `loadFrameFromPreview()` already has
   the picker frame's timestamp; `collectParams()` needs to send it
   alongside `outlet`/`roi` as a new field (e.g. `outlet_reference_s`).
   `ZahnCupDetector.configure()` reads it, defaulting to `analysis_start_s`
   when absent (preserves today's behaviour for direct API callers that
   don't supply it - no breaking change to the params contract).
2. **Do not move where scoring/decoding starts.** The sampling plan keeps
   spanning `[analysis_start_s, analysis_end_s]` exactly as today - this
   is what keeps `StreamActivityScorer`'s background model calibrating on
   a genuinely quiet frame (§17.1's finding) rather than moving the
   corruption to a different starting point.
3. **Delay tracker construction until the reference frame, not the first
   sample.** `_run_loop`'s `if first_frame:` block currently constructs
   `OutletTracker` unconditionally on the very first decoded sample. It
   would instead wait until `sample.timestamp_s >= outlet_reference_s`,
   constructing the tracker there with the user's (now correctly matched)
   coordinates. Samples before that point have no tracker yet - fed to
   `machine.update(..., trusted=False)` with a zero-value sample, exactly
   like an ordinary untracked/`lost` span today. `TrackerInitError` at the
   reference frame is handled exactly as it is now (§6.2/§9.2): loud
   `FAILED`, no confirmed measurement, only `zahn_track_outlet=False` may
   fall back.
4. **Let the existing uncertainty machinery do the rest.** If the true
   flow start falls before the reference frame - exactly this clip's
   situation - it now sits inside a leading "gap" the same shape as any
   other untracked span. `start_uncertain`/`start_uncertainty_bounds`
   (§6.3) already exist to handle precisely this: report a bounded,
   honestly-uncertain start (`[analysis_start_s, outlet_reference_s]` at
   worst) rather than either a wrong precise number (§17.1's scenario C)
   or a silent failure (§17.1's scenario B pattern on the real clip's true
   drift magnitude).

This is a bounded, comprehensible change confined to: one frontend field,
one new detector param with a backward-compatible default, and a
restructure of *when* (not *whether*) `_run_loop` constructs the tracker.
It does not touch `OutletTracker` itself, the state machine's core logic,
or introduce a second decode pass. **Recommendation: implementable within
Stage 1's existing scope**, pending review of this design.

### 17.4 — Scope recommendation

* **§17.1 (frame-picker/reference-frame semantics): Stage 1-correctable.**
  Proposed design above; requires review before implementation, not a
  scope decision.
* **§17.2 (real low-texture/translucent-cup tracking robustness): scope
  currently undetermined.** Could not be reproduced or root-caused from
  here without either the real clip or an accurately-modelled synthetic
  low-texture/translucent fixture. Depending on what the recommended
  per-frame diagnostics run turns up, this may be a parameter-tuning fix
  within Stage 1, or it may be exactly the "continuity/contrast-anchor
  tracing" and motion-robustness work Stage 0 already scoped out to Stage
  2/3 (§4.3 of this report, and Stage 0's own recommendation). **No claim
  either way is made here** - this needs either the real clip run through
  the streamed diagnostics, or an explicit decision to treat it as a
  Stage 2/3 question without further synthetic diagnosis.

**Not implementing either fix in this round**, per instruction: stopping
for review of this diagnosis and the §17.3 design before any code change.
PR #4 stays draft, unmerged; the real-footage criterion is **not** claimed
to pass.

## 18. Design review of §17.3 — four required changes, addressed

§17.3's proposed design ("treat everything before the reference frame as
one honestly-uncertain gap") was reviewed against real streamed
diagnostics run on the actual clip and **rejected as insufficient**: it
converts a wrong answer into an unmeasurable one rather than recovering
the true, measurable start, and does not touch the geometry or
distractor-lock failure modes at all. Four required changes were given.
All four are addressed below; PR #4 stays draft, unmerged; Stages 2 and 3
are not started.

### 18.1 — Requirement 1: recover the true start, don't just flag it unmeasurable

**Design change from §17.3.** Rather than leaving every pre-reference
frame untracked and letting the existing gap-uncertainty machinery report
a wide, honest "don't know," the tracker is now constructed *before* the
sampling loop starts, pre-armed against a frame read directly at
`outlet_reference_s` — then run forward from `analysis_start_s` in a
normal, already-armed `TRACKED`/`PREDICTED`/`LOST` state, exactly as if
the user had marked the outlet on the very first frame. This recovers a
real, trackable start instead of only bounding an unknown one.

Implementation, `app/analysis/zahn_detector.py`:

* `configure()` now parses `outlet_reference_s` from `self.params`,
  clamped to `[start_s, end_s]`, defaulting to `self.start_s` (today's
  behaviour, unchanged, for callers that don't supply it — no breaking
  change to the params contract).
* `run()`: when `outlet_reference_s > start_s`, it reads that one frame
  out-of-band via `VideoReader.frame_at()` (a seek, not part of the
  forward-streaming sample loop), converts it to grayscale, and
  constructs `OutletTracker` against it immediately — before the first
  sample is streamed — instead of waiting for `_run_loop`'s
  `if first_frame:` block to construct it on whatever frame happens to be
  first. `TrackerInitError` at this point is handled exactly as before
  (§6.2/§9.2): loud `FAILED`, no confirmed measurement.
* New `OutletTracker.__init__` parameter `start_in_search: bool = False`.
  When the tracker is constructed this way (pre-armed from a reference
  frame, forward-run starting earlier), it starts in `LOST` with no
  confident observation rather than assuming the reference frame's
  position is already correct at `analysis_start_s` — the existing
  reacquisition search (unchanged, the same machinery used for ordinary
  mid-run reacquisition) is what finds and locks onto the real outlet
  position as frames stream in from the true start. This is the load-
  bearing piece: it is why the true start is *recovered*, not merely
  *un-penalised*.

**Verified**, `tests/test_zahn_tracking_integration.py::TestOutletReferenceFrame`
(3 tests, new fixture `portrait_reference_frame_zahn_video`, portrait
1080×1920, resolution-scaled drift, true `flow_start_s=3.9`, outlet marked
at `t=4.5`):

| test | what it checks | result |
| --- | --- | --- |
| `test_the_bug_marking_a_later_frame_is_applied_to_frame_zero` | without `outlet_reference_s`, reproduces the pre-fix failure (`flow_start_s` wrong by more than the synthetic tolerance, or `None`) | passes — confirms the bug still exists *without* the fix, so the next test is a real regression check, not a tautology |
| `test_outlet_reference_s_recovers_the_true_start` | with `outlet_reference_s=4.5`, `analysis_start_s` left at its default (0) | passes — `status=confirmed`, `flow_start_s` and `efflux_seconds` both within ±0.5s of ground truth |
| `test_a_reference_frame_matching_analysis_start_s_is_unaffected` | `outlet_reference_s == analysis_start_s` (today's usage pattern) | passes — unchanged behaviour, no regression for existing callers |

### 18.2 — Requirement 2: fix the fixed scoring geometry for portrait outlet motion

**Root cause, confirmed by reading the code.** `build_roi_from_click()`
computed the scoring ROI's height as
`min(configured_height, video.height - y)` — leaving zero reserved
headroom below the ROI for the tracker's own search/guard margins
(`guard_margin_px`, `track_search_margin_px`). On a portrait frame where
the outlet starts well down the frame and drifts further down under
camera/hand motion (this clip's actual geometry), the guard region has
nowhere left to expand into and clips against the frame's bottom edge —
independent of whether the tracker itself is correctly following the cup.
This is a distinct failure mode from §17.2's tracking-robustness question.

**Fix**, `app/analysis/zahn_detector.py::build_roi_from_click()`: the
height computation now reserves
`downward_headroom = config.guard_margin_px // 2 + config.track_search_margin_px`
pixels below the ROI before clamping to the frame edge, so the guard
geometry has the same drift budget on a tall portrait frame as it already
had on this project's landscape fixtures. The reservation is expressed in
terms of the tracker's own configured margins, not a new fixed pixel
constant, so it scales with configuration rather than reproducing the
same class of bug at a different fixed size.

**Verified**, `tests/test_zahn_tracking_integration.py::TestPortraitGuardGeometry`
(1 test, same portrait fixture as §18.1): asserts `frames_untracked /
frames_analysed < 0.1` and `status == "confirmed"` — i.e. the overwhelming
majority of frames track cleanly, not the near-total failure that
edge-clipping guard geometry produces. Full existing landscape fixture
suite re-run unchanged (no regression at the resolution this geometry was
originally tuned against).

### 18.3 — Requirement 3: prevent locking onto background through the translucent cup

**What was built.** A continuous patch-correlation trust gate,
`OutletTracker._patch_correlation_at()`, checked on every accepted
RANSAC/inlier/displacement transform (not only at reacquisition time, as
`track_reacquire_min_correlation` already was): the anchor feature's
implied position under the fitted transform is checked against the
original reference patch via `cv2.matchTemplate` (`TM_CCOEFF_NORMED`), and
the transform is only accepted as `tracked` if that correlation clears a
new, separately-tunable threshold, `ZahnConfig.track_min_patch_correlation`
(default `0.45`, deliberately looser than `track_reacquire_min_correlation`'s
`0.6` — a long correct run is expected to drift further from one fixed
reference snapshot than a fresh reacquisition search, so the same bar
would false-reject good tracking). This closes off the case where RANSAC
inlier count and pixel-displacement bounds alone are satisfied by a
smoothly, self-consistently moving *distractor* rather than the cup
itself — exactly the "why transforms are accepted" gap named in the
requirement.

An initial implementation of the check assumed the reference patch was
symmetric around its anchor; `_extract_patch` in fact clips asymmetrically
near frame/crop edges, which made the check permanently fail
(`correlation == -1.0` even at zero displacement) and broke every Zahn
test. Fixed by recording the patch's true, possibly-asymmetric extraction
offset once at `__init__` and reusing that exact offset for every
subsequent correlation check, rather than re-deriving symmetric bounds
from the candidate center each time.

**What this does and does not solve — evidenced, not asserted.** Two new
synthetic scenarios (`tests/_synthetic_handheld.py`,
`build_translucent_cup_clip` plus an inline near-distractor generator in
`tests/conftest.py`) were built to separate two distinct sub-cases:

* **Distractor drifts near the cup mid-run, after correct initialisation**
  (the anchor feature was genuine cup texture when tracking began; a
  strong, stationary distractor patch is elsewhere in frame and the
  camera/hand drift brings the tracked region close to it later) — **this
  is the case the correlation check fixes.** Pre-fix, RANSAC/inlier/
  displacement checks alone accept the distractor's own smooth motion and
  report `status=confirmed` with high confidence on a wrong number.
  Post-fix, verified by
  `TestTrackingRefusesAStrongNearbyDistractor::test_the_result_is_not_confidently_wrong`:
  the run either reports a genuinely correct measurement, or stays
  honestly uncertain (`status != confirmed` with `confidence <= 0.5`) —
  never both `status=confirmed` and wrong. This is a real safety property:
  the tracker cannot be confidently, silently wrong about this failure
  mode any more.
* **Distractor already overlaps the cup at initialisation** — the true
  worst case for a low-texture, translucent cup, where the corner
  detector's strongest-response features may themselves be background
  visible *through* the cup wall rather than the cup's own (weak) edges.
  Run against `build_translucent_cup_clip`'s worst-case construction
  (distractor overlapping the cup from `t=0`): the tracker locks onto the
  distractor's structure from the very first frame — `confidence=0.98`, 0
  untracked frames, a fully confident, fully wrong result. The
  correlation check cannot catch this because it is circular in this
  case: the reference patch itself was extracted from the wrong
  (background/distractor) structure at `__init__` time, so every
  subsequent frame correctly, consistently matches *that* — the check
  verifies self-consistency, not correctness against the true cup. No
  patch-similarity-based check evaluated *after* anchor selection can fix
  a wrong anchor selected *at* initialisation; distinguishing "cup" from
  "background visible through cup" at that moment requires information
  the current architecture doesn't have — most plausibly an independent
  estimate of camera-only motion (the frame content that should move with
  the *background*, not the cup) to identify which candidate features move
  with the camera and which move with the handheld object. That is Stage
  3's deferred camera-motion-compensation scope, not a Stage-1-bounded fix.
  **This fixture is intentionally not wired into a passing assertion** —
  it demonstrates a known, evidenced Stage 1 limit, not a bug left
  unfixed by oversight. It is kept as test infrastructure for when Stage 3
  is authorized.

**Conclusion for requirement 3:** the requested "add a regression whose
tracked outlet follows the cup, with diagnostics proving scale-relative
outlet error, trusted-frame coverage, and why transforms are accepted" is
delivered for the tractable sub-case (mid-run drift toward a nearby
distractor) — real fix, real regression test, real evidence of "why
transforms are accepted" (the correlation gate itself, plus the trust
fraction it produces). For the untractable sub-case (distractor already
overlapping at initialisation), the same rigor is applied in the other
direction: precise evidence for *why* it cannot be solved within Stage 1's
architecture, rather than either a false claim of success or an unexamined
guess.

### 18.4 — Requirement 4: validate against the supplied evidence target

**Not performed against the real clip in this environment** —
`20260820_184144.mp4` is not accessible from this session (as in every
prior round; see §17's framing). The closest available evidence is the
synthetic fixture built for §18.1 (portrait, resolution-scaled to this
clip's actual 1080×1920 geometry, reference frame at `t=4.5` after true
flow has already started at `t=3.9`): with all three fixes in place, that
scenario reports `flow_start_s` and `efflux_seconds` within the synthetic
±0.5s tolerance of ground truth (§18.1's second test). This is **not** a
substitute for running the actual supplied clip and comparing against the
manual/visual ≈16.6s efflux target — it demonstrates the *mechanism* is
correctly wired (reference-frame decoupling, search-from-start
reacquisition, portrait guard geometry) against a fixture built to match
the clip's known geometry, not that the real clip now passes. **The real
±0.75s acceptance criterion against the actual clip is not claimed to be
met here** — that validation needs to be run by whoever has the clip.

## 19. Re-validation after this round's fixes

```
python -m pytest          348 passed        (pre-round: 343 passed, 0 failed)
python -m ruff check .    All checks passed
python -m ruff format --check .   51 files already formatted
python -m mypy app        3 pre-existing "unused type: ignore" errors,
                           identical on d8c3fdf before this round's changes
                           (opencv-stub/mypy-version drift, unrelated to
                           this round — confirmed by diffing mypy output
                           against a worktree checked out at d8c3fdf)
node --test tests_js/*.test.js   44 pass, 0 fail
```

5 new tests this round (§18.1's 3, §18.2's 1, §18.3's 1); no existing test
was weakened.

**Runtime**, same methodology as §7/§10, 3 trials each, `handheld_zahn_video`-
equivalent landscape fixture (480×360), comparing `d8c3fdf` (pre-round, in
a scratch worktree) against this round's tree:

| | mean elapsed | trials |
| --- | --- | --- |
| before (`d8c3fdf`) | 1.335 s | 1.309 / 1.371 / 1.326 |
| after (this round) | 1.375 s | 1.514 / 1.345 / 1.266 |

The difference (~3%) is within this environment's run-to-run noise (the
"after" trials alone span 1.266–1.514s) — the new per-frame
`_patch_correlation_at` check, the pre-read reference frame, and the
search-from-start reacquisition path do not add a measurable cost beyond
that noise floor. (Absolute numbers in this environment run higher than
earlier rounds' ~0.9s figures reported in §7/§10; that appears to be
sandbox-level performance variance unrelated to this round's changes,
since the `d8c3fdf` baseline measured *in this same environment* is
likewise ~1.3s, not ~0.9s — the before/after comparison within one
environment is what's load-bearing here, not the absolute figure against
older rounds.) The new portrait (1080×1920) fixture, run for the first
time this round since it didn't exist before, measures 5.672s mean — six
times the pixel count of the landscape fixture, consistent with roughly
linear scaling in frame area rather than a regression.

## 20. Status and what remains

* **Requirement 1** (reference-frame timeline recovery): done, verified
  against a portrait fixture matching the real clip's geometry.
* **Requirement 2** (portrait scoring-geometry fix): done, verified.
* **Requirement 3** (translucent-cup distractor lock): partially done —
  a real, verified safety-property fix for the tractable sub-case
  (mid-run drift toward a nearby distractor); the untractable sub-case
  (distractor already overlapping at cup initialisation) is precisely
  evidenced as out of Stage 1's reach and requires Stage 3's deferred
  camera-motion-compensation work.
* **Requirement 4** (validation against the real clip's ~16.6s target):
  **not performed** — no access to the real clip from this environment.
  Synthetic-only evidence is reported in §18.4/§18.1, explicitly caveated
  as not a substitute for running the actual file.

PR #4 stays **draft and unmerged**. Stages 2 and 3 are **not started**.
Stopping here for review, per instruction.

## 21. Real-clip validation found two further blockers — addressed

A supervisor comment ran `eabe29a` (§18–§20's head) against the real clip
(`20260820_184144.mp4`) with the UI-equivalent click `(610,1025)` and
`outlet_reference_s=4.5`. Result: `status=failed`, no start/end/efflux,
761/811 frames untracked, only 46 liquid frames. Two concrete blockers:

1. `app/web/static/app.js::collectParams()` never sent `state.frameTime`
   as `outlet_reference_s` — `git grep` found no frontend occurrence at
   all. Every outlet mark was silently applied as if made on frame zero,
   regardless of what §18.1's backend wiring supported.
2. The reference-frame search ran chronologically from `analysis_start_s`
   and falsely reacquired at t=0, moving the ROI to `(593,1619)` — nowhere
   near the true outlet. The fixed capture/search window, pinned to the
   original click for the whole run, also could not follow the cup once it
   drifted outside it later in the clip.

### 21.1 Frontend wiring

`collectParams()` never read `state.frameTime` at all (§18's backend
support for `outlet_reference_s` had no caller). Fixed by sending it
whenever an outlet click (not a drawn region) is submitted:

```js
params.outlet = currentState.outlet;
params.outlet_reference_s = currentState.frameTime;
```

Refactored to take `currentState`/`getValue` explicitly (both default to
the real page's `state`/`el(...).value`) so this is directly assertable
outside a browser, matching this repo's existing pattern for
DOM-independent unit tests (`tests_js/review.test.js`'s
`module.exports` guard). New `tests_js/analysis-params.test.js` asserts
the actual submitted payload: `outlet_reference_s` present and correct
for a click (including the `frameTime: 0` case a falsy-value bug would
drop), absent for a drawn region or a non-Zahn mode.

### 21.2 Anchor-centric bidirectional tracking, replacing the blind
     chronological search

§18.1's fix pre-armed a tracker at the reference frame but then searched
for it *chronologically forward from `analysis_start_s`* by template match
alone — a cold search with no motion continuity between consecutive
frames. That is exactly what a supervisor review against real footage
found could falsely "reacquire" on an unrelated patch of the scene right
at frame zero.

Replaced with genuine optical-flow continuity in both directions from a
single trusted anchor (`ZahnCupDetector._process_pre_reference_span`):
decode every frame from `analysis_start_s` to `outlet_reference_s` once,
forward (an ordinary, cheap sequential read); track *backward* through
that buffer, from the reference frame to the first one, with the same
Lucas-Kanade tracker used going forward — optical flow does not care
which way time runs, only that consecutive frames are close in content.
Score every one of those frames in the true, forward chronological order
`StreamActivityScorer`'s background model and `FlowStateMachine` require,
using the backward pass's per-frame positions. The reference frame is the
only frame this run ever treats as trustworthy without verification;
every other frame's position is reached from it by tracking, never by
guessing where a stored patch might match. `OutletTracker`'s now-unused
`start_in_search` parameter (last round's pre-arm mechanism) was removed
rather than left as dead code.

Bounded in memory by `outlet_reference_s - analysis_start_s` — the gap
between where a real clip's outlet first becomes markable and where
analysis should start, inherently small in the workflow this exists for,
not the length of the whole recording. The search window does not
recentre during this backward span (see §21.3) — a deliberately scoped
limit, since the span is inherently short and this fix already resolves
the specific false-reacquisition failure mode found on real footage.

### 21.3 Search window recentring on the last credible estimate

The capture/search window was fixed for the whole run once built
(`_build_capture_roi()`, called once in `run()`): a guard-sized box padded
by `track_search_margin_px` (80px default), centred on the original
click and never moved. Once the true outlet drifted further than that
from where it was clicked — exactly what the supervisor's diagnostic
showed ("the real cup later moves outside the current x=460..760 search
window") — the pixels it drifted to were simply never decoded. No
tracker or reacquisition search, however good, can find something that
was never in the crop.

`ZahnCupDetector._run_loop` now drives one or more decode *segments*
(`_run_segment`), almost always just one. A segment ends early and
requests a *recentre* when the tracked outlet is lost on a frame right
after having genuinely been tracked or predicted (a fresh loss, not a
repeat of one already given up on) — bounded by `MAX_TRACKING_RECENTRES`
(8) so a pathological run cannot recentre without limit. `_build_capture_roi`
now accepts an optional centre (source coordinates), defaulting to the
guard region's own centre — reproducing the original click-anchored
window exactly when omitted.

A recentre candidate is *verified*, not assumed: before accepting it, the
still-existing tracker's own stored reference patch must correlate with
the candidate position above `track_reacquire_min_correlation` — the same
bar a cold reacquisition search already holds a candidate to
(`OutletTracker.reference_patch_correlation`, a new public entry point
onto the existing `_patch_correlation_at` machinery). A failed
verification touches neither the tracker nor the window: the still-`lost`
tracker keeps trying its own verified in-window reacquisition every frame
from there, exactly as it did before this capability existed.

### 21.4 Three bugs this design caught in its own test suite before reaching review

None of these reached a passing state undetected — each was found by a
dedicated regression test failing, not by later inspection. Documented
because catching them here, rather than on the next real clip, is the
point of writing the tests first.

* **Recentring on any fresh loss, not just a loss near the window's
  edge.** The first version only recentred when the loss coincided with
  the tracked position sitting near the current window's edge. A new
  synthetic fixture with a steady, one-directional outlet pan (see below)
  showed the *dominant* real failure mode is a `predicted` bridge timing
  out (`track_max_bridge_s`) while still comfortably in-bounds — the
  extrapolated position never gets near the edge before the bridge simply
  runs out. Dropped the edge-proximity condition entirely: any fresh loss
  now triggers a recentre attempt (still verified — see below — and still
  budget-bounded).
* **A verification gate was needed, and its absence broke the occlusion
  tests.** Recentring on every fresh loss, unconditionally accepting a
  freshly-constructed tracker at the extrapolated position, immediately
  regressed four existing tests
  (`TestBreakDuringATrackingGap*`/`TestUnresolvedGapInTheMiddleOfFlow`):
  during a genuine, prolonged occlusion (the outlet swung out of frame
  entirely), the extrapolated "last credible estimate" was still treated
  as a confident re-anchor, "recovering" tracking on plain background
  noise mid-gap and corrupting the endpoint-uncertainty machinery those
  tests exist to check. Fixed by §21.3's correlation-verification gate.
  The *first* attempt at that fix still cold-started a brand-new tracker
  as the fallback when verification failed — which has no correlation
  check of its own, reopening the same hole one level deeper. Fixed by
  making a failed verification a genuine no-op (§21.3's final paragraph).
* **A recentred window sized around the wrong point.** `_build_capture_roi`
  centres its window on the *guard region's* own centre; the first
  recentre implementation passed the raw *outlet* position instead. The
  guard region is not symmetric around the outlet (it reaches much
  further down, for the stream, than up, for the cup body), so this
  under-sized/mis-positioned the recentred window - on one synthetic
  fixture with a perfectly *static* cup and zero drift, a single early,
  spurious recentre event left the window too short to ever contain the
  guard region again, and every subsequent frame silently failed the
  guard-cut (`frames_untracked` = 100%, despite the tracker itself
  reporting `tracked` throughout - state and trustedness are logged from
  different sources here, which is precisely why the discrepancy was
  visible in the diagnostics rather than just a wrong final number).
  Fixed by shifting the candidate outlet position by the guard region's
  fixed offset from the original click before building the window, so it
  is always sized around where the guard region would actually be.

New synthetic regression coverage (`TestSearchWindowRecentres`,
`tests/conftest.py`'s `wide_pan_zahn_video`): a *steady, one-directional*
outlet pan — not the existing fixtures' oscillating hand/camera drift,
whose sinusoidal reversals stress velocity extrapolation in a way a real
single reframe usually does not — of 380px, well beyond the ~150px a
fixed window bounds. A flat (untextured) background, deliberately unlike
most other fixtures: with the camera held still and only the cup panning,
a textured background's spatial pattern sliding past the tracked window
as it follows the cup reads as spurious activity to
`StreamActivityScorer`'s own background model - a real, separate
scoring-side limitation worth its own coverage another time, not
conflated here with the one thing this fixture exists to test. The test
asserts recentring actually engaged (`diagnostics["recentre_events"]`
non-empty, bounded by `MAX_TRACKING_RECENTRES`), that untracked frames
stay a small minority (a fixed window would have gone `lost` for the rest
of the run once first exceeded), and that the reported result is either
accurate and confirmed or honestly capped in confidence — never
confidently wrong.

## 22. Re-validation after this round's fixes

```
python -m pytest          349 passed        (pre-round: 348 passed, 0 failed)
python -m ruff check .    All checks passed
python -m ruff format --check .   55 files already formatted
python -m mypy app        Success: no issues found in 29 source files
                           (this environment's mypy run is clean this
                           round - the 3 "unused type: ignore" errors
                           noted last round as pre-existing/unrelated did
                           not reappear; consistent with that finding,
                           since they were already attributed to
                           environment/stub-version drift rather than any
                           code in this repo)
node --test tests_js/*.test.js   51 pass, 0 fail (pre-round: 44 pass)
```

1 new Python test this round (§21.4's `TestSearchWindowRecentres`; the
edge-proximity/verification/guard-centring bugs it and the pre-existing
occlusion tests caught were fixed in place, not worked around); 4 new
JavaScript tests (§21.1).

**Runtime**: `TestPerformance` (`tests/test_zahn_tracking_integration.py`)
already asserts elapsed time stays comfortably under the clip's own
duration and peak RSS growth stays bounded, and passed unchanged this
round on the same `handheld_zahn_video` fixture used for that check in
every prior round - the added machinery (segment/recentre bookkeeping,
the backward-tracking pass) does not change that on a clip whose drift
never actually triggers a recentre, which is the common case this
property needs to hold for. A clip that *does* recentre pays one extra
out-of-band frame read (`_read_capture_frame_gray`) per recentre attempt,
bounded by `MAX_TRACKING_RECENTRES` (8) - a fixed, small cost independent
of clip length, not re-measured separately here since it is bounded by
construction rather than by empirical variance the way the rest of this
report's runtime figures are.

## 23. Status and what remains

* **Requirement 1** (reference-frame timeline recovery): done, and now
  additionally hardened against the specific false-reacquisition failure
  a real clip exposed (§21.2), with the frontend wiring that made it
  reachable at all now actually in place (§21.1).
* **Requirement 2** (portrait scoring-geometry fix): done, unchanged this
  round.
* **Requirement 3** (translucent-cup distractor lock): unchanged this
  round - still partially done, per §20's status, with the untractable
  sub-case still requiring Stage 3.
* **Requirement 4** (validation against the real clip's ~16.6s target):
  **still not performed** — no access to the real clip from this
  environment. This round's fixes were built and validated entirely
  against the supervisor's *description* of the real clip's failure
  (the exact click, `outlet_reference_s`, and the reported diagnostic
  counts) and new synthetic fixtures constructed to reproduce that class
  of failure, never against the file itself. That remains the one
  criterion this report cannot honestly claim to have met.

PR #4 stays **draft and unmerged**. Stages 2 and 3 are **not started**.
Stopping here for review, per instruction.

## 24. Root cause of the real-clip tracking failure, and a design proposal

**No code in this section is implemented.** Per instruction, this is
analysis and proposal only, stopped here for review before any further
work. `07ec4ee` is unchanged.

### 24.1 What the supervisor's run on `07ec4ee` showed

Exact head, real clip, click `(610,1025)`, `outlet_reference_s=4.5`,
default analysis start:

* `status=review`, no precise efflux.
* Candidate start `5.133s`, end `26.231s`, bounds `21.098–21.698s` -
  against a visual reference of ~3.9s start, ~20.5s end, ~16.6s duration.
* 772/812 frames untrusted (95%); trusted runs mostly `4.466–6.033s`, then
  brief re-trusted spans at `25.498–25.664s` (0.166s) and
  `26.164–26.231s` (0.067s).
* The position estimate sits around x≈599–663 the whole time; the real
  cup later moves to x≈390. "Recentering therefore follows the wrong
  structure, not the cup."

### 24.2 Root cause

This is the failure §18.3 and §20 already named and flagged as out of
Stage 1's reach - the real clip reproduces it, it does not introduce a
new one. Restating precisely, against this run's own numbers:

The tracker locks on cleanly right at the reference frame (the
`4.466–6.033s` trusted run - about 1.5s either side of the 4.5s anchor,
where the anchor's own reference patch still matches by construction) and
then goes `lost` for the next ~15s as the cup keeps drifting. Every
reacquisition and every recentre attempt after that is verified against
the *same* stored reference patch
(`OutletTracker._reference_patch`/`reference_patch_correlation`,
§21.3) - built once, from whichever feature `goodFeaturesToTrack` ranked
strongest inside the cup-sized box at the reference frame
(`OutletTracker.__init__`'s anchor-selection comment). On an opaque,
well-textured cup that is genuinely the cup's own rim or handle. On a
translucent, low-texture one - this clip, evidently - it can just as
easily be structure visible *through* the cup: background, not cup.

If it is background, the verification bar this stage relies on
(`track_reacquire_min_correlation`/`track_min_patch_correlation`, both
correlating a *candidate* against that *same* stored patch) cannot catch
it, for a structural reason, not a tuning one: **the check can only ever
ask "does this still look like what I started with," and if what it
started with was already background, then more of that same background -
wherever it later appears - answers yes.** That is exactly consistent
with the estimate holding at x≈599–663 (near the original click) rather
than following the cup to x≈390: if the locked structure is static
background (or moves only with camera pan, not with the cup's own
independent hand-drift), the gap to the true, further-drifting cup
position only grows over time, which is what "recentering follows the
wrong structure" describes. The two brief re-trusted spans right at the
end (`25.498–25.664s`, `26.164–26.231s` - 0.166s and 0.067s, a handful of
frames each) read as one-off spurious correlations clearing the threshold
momentarily, not genuine re-locks - short enough that a persistence
requirement (§24.3.1) would very likely have suppressed them, though that
cannot be confirmed without the clip.

This is a real limitation of Stage 1's verification design, not a bug in
this round's specific implementation of it: §18.3 already found the
equivalent case (a distractor overlapping the cup at initialisation) and
called it unsolvable at this layer for the identical reason - "the
correlation check is circular in that case, since the reference patch
itself is extracted from the wrong structure." This run is the first
evidence that the identical circularity also defeats the *reacquisition
and recentre* paths on real footage, not only the initialisation case the
synthetic fixture targeted.

### 24.3 What can and cannot be done inside Stage 1

**24.3.1 Two bounded, low-risk mitigations that reduce - not eliminate -
the failure rate.** Neither addresses the root cause above; both are
worth doing regardless, because they attack the parts of this run's
failure that *are* just tuning/robustness gaps, not the circular-
verification problem itself:

* **Persistence-gate reacquisition and recentre.** Both currently accept
  on a single frame's correlation clearing the threshold
  (`OutletTracker._match_reference_patch`,
  `ZahnCupDetector._run_loop`'s recentre branch). Requiring the *next*
  frame to independently reconfirm before flipping to `tracked` would
  very likely have suppressed this run's two brief false locks (0.166s
  and 0.067s - too short for two independent frames to both spuriously
  agree) without weakening a genuine reacquisition, which by construction
  stays correlated for longer than one frame.
* **Bias anchor selection toward where a rim genuinely tends to survive
  translucency**, rather than "whichever corner `goodFeaturesToTrack`
  ranks strongest anywhere in the cup-sized box." `_draw_translucent_cup`'s
  own docstring already identifies the rim as "the one feature a real
  translucent cup usually keeps a visible edge on." Restricting the
  anchor search to an arc near the top of the box (where a rim would be),
  rather than the whole box, would reduce the chance the strongest corner
  found is background showing through the body lower down - a real,
  Stage-1-scoped algorithmic change, but a shape-prior tuned to *this*
  clip's evident cup geometry, unvalidated against any other real
  footage, and it does nothing for a cup whose rim itself is also
  low-contrast.

**24.3.2 Why the actual failure needs a scope change, not a tuning
pass.** §24.2's argument is structural: any verification built on
self-correlation to a single stored reference cannot distinguish "the
cup, moved" from "more of the same background the reference was already
built from." Closing that gap needs a signal that is *not* derived from
the reference patch at all - independent evidence of which motion in the
frame is the cup's and which is the camera's, which is exactly Stage 3's
already-deferred camera-motion-compensation scope (§9/§18.3 both name
it). Nothing inside Stage 1's current architecture supplies that signal;
the scorer's own per-pixel disturbance measure was considered and
rejected for this - a genuinely static-relative-to-camera cup rim and a
genuinely static background patch produce the same low frame-to-frame
difference, so it doesn't discriminate the two cases either.

**24.3.3 A Stage-1-scoped alternative that sidesteps the problem instead
of solving it: more than one operator-marked anchor.** This round's
`_process_pre_reference_span` already tracks bidirectionally between a
single trusted anchor and both ends of the clip. The same machinery
generalises directly to *multiple* anchors along the timeline (e.g. the
operator marks the outlet a second time near where the stream is
expected to end, not just once near the start): each unsupervised
tracking segment then only has to survive the distance between two
human-verified points, not the whole clip unaided, bounding this failure
mode's blast radius without requiring Stage 3's camera-motion work at
all. This is a product/UX decision (asking for more than one click), not
an algorithmic one - flagged here as the "explicit scope change" this
review asked for if full in-algorithm resolution isn't possible, rather
than implemented without that authorisation.

### 24.4 The suspected frame double-count: confirmed, root-caused, not fixed

Reproduced synthetically (not on the real clip - no access to it), and
it is a genuine off-by-one, distinct from §24.2-§24.3's tracking-design
question:

```
fps=30.0  reference_s=3.333  ->  reference frame index 99
forward_start_s = reference_timestamp + plan.effective_interval_s
                 = 3.300000 + 0.033333 = 3.333333
frame_index_at(3.333333, fps=30.0) = 99   # same frame again, not 100
```

`run()` computes the forward pass's starting timestamp
(`forward_start_s`) by *adding* one frame interval to the reference
frame's own timestamp in floating point, then re-deriving a frame index
from that sum via `frame_index_at`'s `floor(timestamp * fps)`. The round
trip (`index / fps`, `+ 1/fps`, `* fps`) is not guaranteed to land back
on exactly `index + 1` in floating point; when it lands a hair under, the
forward pass's first `iter_samples()` call re-decodes and re-scores the
reference frame a second time - consistent with 812 analysed rows on an
811-frame clip. Confirmed reproducible with `app.video.sampling`'s own
functions directly (fps=30.0, reference_s=3.333 above; also fps=29.97,
reference_s=4.166666) - not every fps/reference_s combination triggers
it, which is why it didn't show up in this round's own re-validation
fixtures (whose reference timestamps happened not to land on an affected
combination).

**Proposed fix** (not applied): carry the reference frame's own integer
index (`FrameSample.index`, already available on `reference_sample` in
`_process_pre_reference_span`) out of that method instead of only its
timestamp, and compute the forward pass's start via
`frames_to_seconds(reference_index + 1, fps)` - integer arithmetic on the
index, no floating-point round trip through a sum. Holding this fix for
the next authorised round, per "stop implementation."

## 25. Status

Requirements 1-2 stand as in §23. Requirement 3 (background-through-cup
rejection): the real clip confirms the sub-case §18.3 already flagged
unsolvable at this layer is not a synthetic-only concern - it is the
dominant failure mode on the one real clip tested against. §24.3
proposes what is and is not fixable inside Stage 1's current scope, and a
scope-preserving alternative (multiple anchors) that neither has been
implemented nor validated. Requirement 4 (the ±0.75s real-footage
criterion): **not met** - `status=review`, no confirmed efflux, per
§24.1's numbers.

PR #4 stays **draft and unmerged**. Stages 2 and 3 are **not started**.
No implementation this round; stopped here for review, per instruction.

## 26. The two-anchor design, implemented per authorisation

Supervisor decision (PR comment, following §25): proceed with §24.3.3's
Stage-1-scoped two-anchor design, with a specific product/tracking
contract, evidence requirements, and explicit non-goals (no camera-motion
compensation, no Stage 2/3). This section is that implementation.

### 26.1 Product and frontend

One Zahn mode, tracking always on - unchanged. The operator now marks the
outlet on **two** frames: an early one (clearly visible) and a late one
(near the expected stream end). Each mark is stored as source coordinates
plus its own frame's exact timestamp - `state.outletEarly` /
`state.outletLate`, each `{x, y, t}` - replacing the old single
`state.outlet`. The frame picker gained an explicit early/late radio
toggle (auto-advancing to "late" once "early" is marked, never silently
overwriting an already-set anchor), both anchors are drawn with distinct
labels, and `resetOutletAnchorsState` clears both on video upload / mode
change / "clear region" - no stale anchor ever survives past a reset.
`collectParams()` sends `outlet`/`outlet_reference_s` and
`outlet_end`/`outlet_end_reference_s` independently, only when each anchor
is actually set. `tests_js/analysis-params.test.js` covers both-set,
either-alone, replace, and reset-then-collect; `node --test tests_js/*.test.js`:
58/58 pass.

### 26.2 Backend contract

`ZahnCupDetector.configure()`: while `zahn_track_outlet` (default) is on,
`outlet` alone is rejected outright - both anchors are required together.
Each anchor's timestamp is validated independently (must parse, must fall
inside `[start_s, end_s]`), and the pair is validated together (late must
be strictly after early - equal or inverted timestamps are rejected the
same way). `zahn_track_outlet=False` is untouched: the single-click,
fixed-ROI rollback path has no reason to gain a new UI requirement an
operator who explicitly opted out of tracking has no reason to satisfy.

Tracking is now three chronological segments, not two:

* **Segment A** (`_process_pre_reference_span`, unchanged from §21.2) -
  only runs if `outlet_reference_s > start_s`, bidirectional between the
  early anchor and the analysis start.
* **Segment B** (`_process_between_anchors_span`, new) - between the two
  anchors. Both are unconditionally trusted; every interior frame is
  independently tracked forward-from-early and backward-from-late over
  the same buffered span, and `_reconcile_between_anchors` only marks a
  frame trusted when **both** directions report `TRACKED` (not
  `PREDICTED`, not `LOST`) **and** their positions agree within
  `track_reconciliation_max_disagreement_px` (new config field, 30px
  default - deliberately distinct from the single-frame
  `track_max_frame_displacement_px`, since this bounds disagreement
  between two independently-accumulated tracks over a potentially long
  span, not one frame's own motion). Disagreement or loss is never
  silently interpolated across - it is untrusted, exactly like any other
  gap, and feeds the same `FlowStateMachine` uncertainty logic segments A
  and C already used. This is the requirement stated most explicitly in
  the authorisation: "do not accept a template match or interpolated
  position as trusted merely because it lies between anchors."
* **Segment C** (`_run_loop`, unchanged) - from the late anchor to the end
  of the analysed range, using the existing recentring/reacquisition
  machinery.

The frame-boundary double-count from §24.4 is fixed as proposed there:
`_run_two_anchor`'s Segment C start is now `frames_to_seconds(late_index +
1, fps)` via `frame_index_at`, integer arithmetic on the frame index
rather than a floating-point `timestamp + interval` round trip re-derived
through `floor(timestamp * fps)`. An 811-frame clip can no longer produce
812 analysed rows.

### 26.3 The two bounded safeguards, and one regression they did not cause

Both §24.3.1 mitigations are implemented as bounded safeguards, per the
authorisation ("with regressions proving brief false correlations do not
become trusted"):

* **Two-frame reacquisition/recentre persistence.**
  `OutletTracker._attempt_reacquisition` now requires two consecutive
  frames' correlation checks to independently agree (within
  `track_max_frame_displacement_px`) before confirming - a single
  spurious hit sets `_pending_reacquisition` and stays `lost`; only a
  second, consistent hit promotes to `tracked`. `TestReacquisitionPersistence`
  (3 new tests in `test_outlet_tracker.py`) proves a single matching frame
  never confirms, on its own or followed by nothing, and that two
  *inconsistent* hits restart the count rather than confirming.
* **Rim-biased anchor selection.** `OutletTracker.__init__` picks the
  reacquisition/correlation reference feature from the top-of-box third
  of `goodFeaturesToTrack`'s ranked corners, not unconditionally the
  single strongest wherever it sits - a shape prior for translucent cups,
  where the rim survives translucency better than the body (§24.3.1).

Re-running the full synthetic suite after wiring both safeguards into the
new three-segment `run()` initially showed 5 of the (then) 25 integration
tests failing - most strikingly `handheld_zahn_video` (the original field-
report regression fixture, an ordinary hand-held, *opaque* cup with no
translucency at all): 41% of frames untracked, recovered start off by
0.8s, status `review` instead of `confirmed`. This looked at first like it
could be the two-anchor contract's own new conservatism working as
designed (independent bidirectional tracking over a long, unaided span
*should* sometimes disagree) - but the same design explicitly asked for
"regressions proving" the safeguards themselves are not the cause, so
before accepting that explanation this round isolated it directly rather
than assuming it:

* **A/B on the persistence gate**: monkeypatched `_attempt_reacquisition`
  back to single-frame confirmation and re-ran `handheld_zahn_video`
  standalone. Untracked frames barely moved (184 → 177 of 447) - the
  persistence gate was not the cause.
* **A/B on rim bias**: reverted the anchor-selection tier from "top third"
  to the original `points[0]` (single strongest corner anywhere) and
  re-ran the same fixture. Untracked frames dropped from 184 to **11**
  (2.5%), status flipped back to `confirmed`, recovered start matched
  exactly (3.0s). Sweeping the tier size (top-third → top-1/6 → top-10 →
  top-5 → top-3) showed a sharp threshold: top-3 matches the unbiased
  baseline exactly; anything wider degrades badly. The mechanism: the
  anchor point does not only seed reacquisition - `_attempt_tracking`'s
  *continuous* per-frame patch-correlation gate (§21.3) verifies every
  accepted frame against the same reference patch, so a weaker corner
  promoted purely for sitting higher in the box was costing ordinary,
  opaque-cup tracking precision on essentially every frame, not just
  robustness during reacquisition.

**Fix applied**: the rim-bias tier is now the top 3 corners by response
(not the top third), still biased toward the highest of those when a
genuine choice exists, but no longer able to pick a corner meaningfully
weaker than the strongest available. Re-running the full suite after this
one-line change: all 5 originally-failing tests pass, including
`TestSearchWindowRecentres` (fixed separately, §26.4) and
`TestBreakDuringATrackingGapAtStart` (the `efflux_seconds_bounds is None`
failure resolved as a side effect - it was downstream of the same
reconciliation disagreement, not a separate bug). `translucent_cup_near_
distractor_zahn_video` - the fixture the rim bias exists for - stayed
green throughout every A/B step above; nothing in this repository's
synthetic suite currently depends on the wider tier.

### 26.4 `wide_pan_zahn_video`'s late-anchor placement

This fixture's default `late_reference_s` (`duration - 1.0` = 15.0s of a
16s clip) placed nearly the entire 380px pan inside segment B, which by
design never recentres - leaving segment C only 1s, too little of the pan
left for `TestSearchWindowRecentres` to observe a `recentre_events` entry.
Fixed by moving this fixture's `late_s` to 4.0s specifically (unlike
every other fixture's "near the end" default) so most of the pan happens
*after* the late anchor, in segment C, where the recentring machinery
under test can actually engage.

### 26.5 New evidence required by the authorisation

* **Validation** (`TestTwoAnchorValidation`, 6 new tests): missing late
  anchor, late anchor outside the frame, a reference timestamp outside
  the analysed range, a late anchor before the early one, duplicate
  (equal) anchor timestamps - all rejected with `InvalidROIError`, raised
  before any frame is decoded. A dedicated positive case confirms
  duplicate *coordinates* with distinct, validly-ordered timestamps are
  accepted (a cup that genuinely has not moved is not an error).
* **False correlation between anchors**
  (`test_a_false_correlation_between_anchors_is_never_marked_trusted`,
  new, on `translucent_cup_near_distractor_zahn_video`): reads the
  per-frame tracking log directly rather than only the summary, and
  asserts every frame marked `trusted` is closer to the true, per-frame
  cup position than to the distractor - the frame-level form of "a false
  lock in one direction cannot promote to trusted just because it lies
  between the anchors," not just "the final answer happens not to be
  wrong."
* **Frame-boundary fix**: covered indirectly by every two-anchor
  integration test now exercising `_run_two_anchor`'s Segment C start
  through real fixtures at varied fps/reference-timestamp combinations,
  none of which reproduce the double-count.
* **Runtime/memory** (`handheld_zahn_video`, 22s/25fps/447 analysed
  frames, standalone measurement): 1.34s wall time end-to-end (~16.4x
  real time, comparable to §22's prior 776→448fps single-anchor
  measurement), process RSS 82.7MB before the run → 150.2MB peak during
  it (a ~68MB delta attributable to this run, unchanged order of
  magnitude from before this round - the three-segment design still
  buffers only capture-crop-sized frames, bounded by the anchor-to-anchor
  gap, not the whole clip).
* **Diagnostic state counts**, same fixture and run: 436/447 frames
  `tracked` and `trusted`, 11 `lost` (all untrusted, none `predicted`), 0
  reacquisitions needed - a clean run end to end, matching the `confirmed`
  status and exact-start-time result now that the rim-bias regression is
  fixed.
* **Existing regressions**: full suite, 359 tests (was 324 before this
  session), all pass - `pytest`, `ruff check`, `ruff format --check`,
  `mypy` (29 source files) all clean, `node --test tests_js/*.test.js`
  58/58. `TestTrackingRefusesAStrongNearbyDistractor`,
  `TestBreakDuringATrackingGap*`, `TestUnresolvedGapInTheMiddleOfFlow`,
  the event-contract (`_assert_no_precise_duration_leaks`) and security
  (`diagnostics_dir` stripping) tests all still pass unmodified in
  substance (only updated to supply the now-mandatory second anchor where
  they exercise the outlet-click path at all).

### 26.6 Status

Requirements 1-3 (product contract, tracking contract, both bounded
safeguards) are implemented as specified, with the rim-bias regression
found and fixed before it could reach review, not after. Requirement 4
(the ±0.75s real-footage criterion) is still **not independently
verified against the real clip** - per the authorisation, that claim is
the supervisor's own local run against this exact head to make, not
something asserted here.

PR #4 stays **draft and unmerged**. Stages 2 and 3 are **not started**.
