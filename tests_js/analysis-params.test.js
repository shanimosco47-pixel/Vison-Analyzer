/*
 * Unit tests for collectParams (app/web/static/app.js): the payload sent to
 * POST /api/analyses.
 *
 * A supervisor review running the real analysis pipeline against actual
 * hand-held footage found that outlet_reference_s - the timestamp the
 * outlet was actually clicked on - was never sent by the browser at all:
 * the click handler set state.frameTime, but collectParams() never read it.
 * A second review round, after that fix, authorized a two-anchor Zahn
 * design (one early, one late outlet mark, each with its own timestamp) to
 * bound how far unsupervised tracking ever has to run without a
 * human-verified checkpoint. These tests assert directly on the submitted
 * payload for both anchors so neither regression can reoccur silently.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const { collectParams, resetOutletAnchorsState } = require("../app/web/static/app.js");

// collectParams() reads UI-only tuning fields (persistence, sensitivity,
// ...) through a getValue(id) callback so these tests don't need a DOM;
// the value itself is irrelevant to what's under test here.
const irrelevantFieldValue = (_id) => 0;

function zahnState(overrides = {}) {
  return {
    mode: "zahn_cup",
    outletEarly: null,
    outletLate: null,
    roi: null,
    ...overrides,
  };
}

test("collectParams - Zahn cup, both anchors marked", async (t) => {
  await t.test("sends both (x, y, t) anchors under their own param names", () => {
    const state = zahnState({
      outletEarly: { x: 610, y: 1025, t: 4.5 },
      outletLate: { x: 390, y: 980, t: 21.2 },
    });
    const params = collectParams(state, irrelevantFieldValue);
    assert.deepEqual(params.outlet, { x: 610, y: 1025 });
    assert.equal(params.outlet_reference_s, 4.5);
    assert.deepEqual(params.outlet_end, { x: 390, y: 980 });
    assert.equal(params.outlet_end_reference_s, 21.2);
  });

  await t.test("sends a t=0 anchor timestamp, not falsily dropped", () => {
    const state = zahnState({
      outletEarly: { x: 10, y: 20, t: 0 },
      outletLate: { x: 50, y: 60, t: 12 },
    });
    const params = collectParams(state, irrelevantFieldValue);
    // frameTime/t: 0 is exactly the case a `anchor.t || ...` fallback would
    // break - must be present and exactly 0, not treated as "no timestamp".
    assert.equal("outlet_reference_s" in params, true);
    assert.equal(params.outlet_reference_s, 0);
  });
});

test("collectParams - Zahn cup, an anchor not yet marked", async (t) => {
  await t.test("omits the late anchor's params entirely when only the early one is marked", () => {
    const state = zahnState({ outletEarly: { x: 610, y: 1025, t: 4.5 }, outletLate: null });
    const params = collectParams(state, irrelevantFieldValue);
    assert.deepEqual(params.outlet, { x: 610, y: 1025 });
    assert.equal(params.outlet_reference_s, 4.5);
    assert.equal("outlet_end" in params, false);
    assert.equal("outlet_end_reference_s" in params, false);
  });

  await t.test("omits the early anchor's params entirely when only the late one is marked", () => {
    const state = zahnState({ outletEarly: null, outletLate: { x: 390, y: 980, t: 21.2 } });
    const params = collectParams(state, irrelevantFieldValue);
    assert.equal("outlet" in params, false);
    assert.equal("outlet_reference_s" in params, false);
    assert.deepEqual(params.outlet_end, { x: 390, y: 980 });
    assert.equal(params.outlet_end_reference_s, 21.2);
  });

  await t.test("sends neither anchor's params when nothing is marked", () => {
    const state = zahnState();
    const params = collectParams(state, irrelevantFieldValue);
    assert.equal("outlet" in params, false);
    assert.equal("outlet_end" in params, false);
  });
});

test("collectParams - a replaced anchor overwrites the old one, not both", () => {
  // The UI overwrites state.outletEarly/outletLate in place when the
  // operator re-marks an anchor (see onPointerUp) - collectParams only
  // ever sees the latest value, so this is really asserting there is no
  // stale-anchor accumulation for it to (wrongly) send.
  const state = zahnState({
    outletEarly: { x: 20, y: 20, t: 1.0 },
    outletLate: { x: 90, y: 90, t: 9.0 },
  });
  const first = collectParams(state, irrelevantFieldValue);
  state.outletEarly = { x: 25, y: 25, t: 1.5 };
  const second = collectParams(state, irrelevantFieldValue);
  assert.notDeepEqual(second.outlet, first.outlet);
  assert.deepEqual(second.outlet, { x: 25, y: 25 });
  assert.equal(second.outlet_reference_s, 1.5);
  assert.deepEqual(second.outlet_end, first.outlet_end); // untouched
});

test("resetOutletAnchorsState", async (t) => {
  await t.test("clears both anchors, any drawn region, and the marking target", () => {
    const state = zahnState({
      outletEarly: { x: 610, y: 1025, t: 4.5 },
      outletLate: { x: 390, y: 980, t: 21.2 },
      roi: { x: 1, y: 1, width: 2, height: 2 },
      markingAnchor: "late",
    });
    resetOutletAnchorsState(state);
    assert.equal(state.outletEarly, null);
    assert.equal(state.outletLate, null);
    assert.equal(state.roi, null);
    assert.equal(state.markingAnchor, "early");
  });

  await t.test("a reset state sends neither anchor - simulating a video/mode swap", () => {
    // Regression for "do not silently reuse stale anchors after a
    // video/mode reset": collectParams on a freshly-reset state must not
    // resurrect the anchors that were cleared.
    const state = zahnState({
      outletEarly: { x: 610, y: 1025, t: 4.5 },
      outletLate: { x: 390, y: 980, t: 21.2 },
    });
    resetOutletAnchorsState(state);
    const params = collectParams(state, irrelevantFieldValue);
    assert.equal("outlet" in params, false);
    assert.equal("outlet_end" in params, false);
  });
});

test("collectParams - non-Zahn modes", async (t) => {
  await t.test("never sends any outlet anchor params outside Zahn mode", () => {
    const state = {
      mode: "activity_scan",
      outletEarly: { x: 1, y: 2, t: 3 },
      outletLate: { x: 4, y: 5, t: 6 },
      roi: null,
    };
    const params = collectParams(state, irrelevantFieldValue);
    assert.equal("outlet" in params, false);
    assert.equal("outlet_reference_s" in params, false);
    assert.equal("outlet_end" in params, false);
    assert.equal("outlet_end_reference_s" in params, false);
  });

  await t.test("still sends a drawn region for non-Zahn modes", () => {
    const state = {
      mode: "activity_scan",
      outletEarly: null,
      outletLate: null,
      roi: { x: 5, y: 5, width: 40, height: 80 },
    };
    const params = collectParams(state, irrelevantFieldValue);
    assert.deepEqual(params.roi, state.roi);
  });
});
