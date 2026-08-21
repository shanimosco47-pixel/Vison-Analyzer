/*
 * Unit tests for collectParams (app/web/static/app.js): the payload sent to
 * POST /api/analyses.
 *
 * A supervisor review running the real analysis pipeline against actual
 * hand-held footage found that outlet_reference_s - the timestamp the
 * outlet was actually clicked on, wired through the backend since the
 * previous Stage 1 round - was never sent by the browser at all: the click
 * handler set state.frameTime, but collectParams() never read it. Every
 * outlet mark was silently treated as if it had been made on frame zero.
 * These tests assert directly on the submitted payload so that regression
 * cannot reoccur silently again.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const { collectParams } = require("../app/web/static/app.js");

// collectParams() reads UI-only tuning fields (persistence, sensitivity,
// ...) through a getValue(id) callback so these tests don't need a DOM;
// the value itself is irrelevant to what's under test here.
const irrelevantFieldValue = (_id) => 0;

test("collectParams - Zahn cup, outlet click", async (t) => {
  await t.test("sends outlet_reference_s alongside the marked outlet", () => {
    const state = {
      mode: "zahn_cup",
      outlet: { x: 610, y: 1025 },
      roi: null,
      frameTime: 4.5,
    };
    const params = collectParams(state, irrelevantFieldValue);
    assert.deepEqual(params.outlet, { x: 610, y: 1025 });
    assert.equal(params.outlet_reference_s, 4.5);
  });

  await t.test("sends outlet_reference_s of 0 when the click was on frame zero", () => {
    const state = { mode: "zahn_cup", outlet: { x: 10, y: 20 }, roi: null, frameTime: 0 };
    const params = collectParams(state, irrelevantFieldValue);
    // Must be present and exactly 0, not falsily dropped - `frameTime: 0`
    // is exactly the case a `state.frameTime || ...` fallback would break.
    assert.equal("outlet_reference_s" in params, true);
    assert.equal(params.outlet_reference_s, 0);
  });
});

test("collectParams - Zahn cup, drawn region", async (t) => {
  await t.test("omits outlet and outlet_reference_s when a region was drawn instead", () => {
    const state = {
      mode: "zahn_cup",
      outlet: null,
      roi: { x: 5, y: 5, width: 40, height: 80 },
      frameTime: 4.5,
    };
    const params = collectParams(state, irrelevantFieldValue);
    assert.deepEqual(params.roi, state.roi);
    assert.equal("outlet" in params, false);
    assert.equal("outlet_reference_s" in params, false);
  });
});

test("collectParams - non-Zahn modes", async (t) => {
  await t.test("never sends outlet_reference_s outside Zahn mode", () => {
    const state = {
      mode: "activity_scan",
      outlet: null,
      roi: null,
      frameTime: 4.5,
    };
    const params = collectParams(state, irrelevantFieldValue);
    assert.equal("outlet_reference_s" in params, false);
    assert.equal("outlet" in params, false);
  });
});
