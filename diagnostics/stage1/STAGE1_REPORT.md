# Stage 1 — outlet tracking for the Zahn cup detector

Implementation evidence and limitations, per the supervisor's approval
comment on PR #4 (2026-08-20). **This is not ready to merge as a final
review** — it is the evidence package for Codex review, on a still-draft PR.

**Update, same PR:** the first Codex review round asked for five specific
changes (§6). All five are implemented, tested, and re-validated (§7) in
this revision. Sections 1–5 below are the original Stage 1 submission,
left as-is as the historical record of what that round reviewed; §6–§8
cover what changed since.

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
