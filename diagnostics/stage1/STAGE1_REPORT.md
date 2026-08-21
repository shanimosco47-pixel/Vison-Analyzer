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
design, stopped here for review before any implementation, per instruction.
Sections 1–5 are the original Stage 1
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
