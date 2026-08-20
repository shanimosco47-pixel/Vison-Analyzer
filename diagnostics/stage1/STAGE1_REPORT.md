# Stage 1 — outlet tracking for the Zahn cup detector

Implementation evidence and limitations, per the supervisor's approval
comment on PR #4 (2026-08-20). **This is not ready to merge as a final
review** — it is the evidence package for Codex review, on a still-draft PR.

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
