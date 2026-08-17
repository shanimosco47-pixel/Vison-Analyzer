/*
 * Unit tests for the pure logic behind the Zahn review panel
 * (app/web/static/app.js). No browser, no DOM, no build step - these run
 * with Node's built-in test runner:
 *
 *   node --test tests_js/
 *
 * Only the functions exported through the `module.exports` guard at the
 * bottom of app.js are under test here; everything DOM-facing is covered
 * instead by the Chromium browser check (see README, "Testing").
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  formatTimeTenths,
  clampSeekTime,
  stepSeekTime,
  boundaryJumpTime,
  isValidRoi,
  computeReviewCanvasSize,
  timelinePercent,
  shouldShowZahnReview,
} = require("../app/web/static/app.js");

test("formatTimeTenths", async (t) => {
  await t.test("formats seconds under a minute with zero-padded minutes", () => {
    assert.equal(formatTimeTenths(16.3), "00:16.3");
  });

  await t.test("formats a value at exactly zero", () => {
    assert.equal(formatTimeTenths(0), "00:00.0");
  });

  await t.test("pads a single-digit seconds value", () => {
    assert.equal(formatTimeTenths(65.2), "01:05.2");
  });

  await t.test("includes an hour segment past 3600s, without padding the hour", () => {
    assert.equal(formatTimeTenths(3661.4), "1:01:01.4");
  });

  await t.test("clamps a negative input to zero rather than showing a sign", () => {
    assert.equal(formatTimeTenths(-2.5), "00:00.0");
  });

  await t.test("returns a placeholder for non-finite or missing input", () => {
    for (const value of [NaN, Infinity, -Infinity, null, undefined, "16.3"]) {
      assert.equal(formatTimeTenths(value), "--:--.-");
    }
  });
});

test("clampSeekTime", async (t) => {
  await t.test("passes through a value already inside the range", () => {
    assert.equal(clampSeekTime(5, 10), 5);
  });

  await t.test("clamps to zero", () => {
    assert.equal(clampSeekTime(-3, 10), 0);
  });

  await t.test("clamps to the duration", () => {
    assert.equal(clampSeekTime(12, 10), 10);
  });

  await t.test("clamps only below zero when duration is unknown", () => {
    assert.equal(clampSeekTime(-1, undefined), 0);
    assert.equal(clampSeekTime(1e9, undefined), 1e9);
    assert.equal(clampSeekTime(1e9, NaN), 1e9);
    assert.equal(clampSeekTime(1e9, 0), 1e9);
  });

  await t.test("treats a non-finite or missing time as zero before clamping", () => {
    assert.equal(clampSeekTime(NaN, 10), 0);
    assert.equal(clampSeekTime(undefined, 10), 0);
  });
});

test("stepSeekTime", async (t) => {
  await t.test("steps forward and clamps at the duration", () => {
    assert.equal(stepSeekTime(9.95, 0.1, 10), 10);
  });

  await t.test("steps backward and clamps at zero", () => {
    assert.equal(stepSeekTime(0.05, -0.1, 10), 0);
  });

  await t.test("treats a falsy current time as zero", () => {
    assert.equal(stepSeekTime(0, 0.1, 10), 0.1);
  });
});

test("boundaryJumpTime", async (t) => {
  await t.test("jumps back by the default one-second context", () => {
    assert.equal(boundaryJumpTime(4.0, 30), 3.0);
  });

  await t.test("clamps at zero for a boundary near the start", () => {
    assert.equal(boundaryJumpTime(0.3, 30), 0);
  });

  await t.test("accepts a custom context window", () => {
    assert.equal(boundaryJumpTime(10, 30, 2.5), 7.5);
  });

  await t.test("returns null for an undetected boundary", () => {
    assert.equal(boundaryJumpTime(null, 30), null);
    assert.equal(boundaryJumpTime(undefined, 30), null);
  });
});

test("isValidRoi", async (t) => {
  await t.test("accepts a well-formed rectangle with no video bounds supplied", () => {
    assert.equal(isValidRoi({ x: 10, y: 20, width: 30, height: 40 }), true);
  });

  await t.test("accepts a rectangle that fits inside the given video bounds", () => {
    assert.equal(isValidRoi({ x: 10, y: 20, width: 30, height: 40 }, 640, 480), true);
  });

  await t.test("rejects a missing ROI", () => {
    assert.equal(isValidRoi(null), false);
    assert.equal(isValidRoi(undefined), false);
  });

  await t.test("rejects a zero or negative width/height", () => {
    assert.equal(isValidRoi({ x: 0, y: 0, width: 0, height: 40 }), false);
    assert.equal(isValidRoi({ x: 0, y: 0, width: 30, height: -1 }), false);
  });

  await t.test("rejects a non-numeric field", () => {
    assert.equal(isValidRoi({ x: "0", y: 0, width: 30, height: 40 }), false);
  });

  await t.test("rejects a rectangle that runs past the video bounds", () => {
    assert.equal(isValidRoi({ x: 600, y: 20, width: 100, height: 40 }, 640, 480), false);
    assert.equal(isValidRoi({ x: 10, y: 460, width: 30, height: 40 }, 640, 480), false);
  });

  await t.test("rejects negative coordinates against known video bounds", () => {
    assert.equal(isValidRoi({ x: -5, y: 20, width: 30, height: 40 }, 640, 480), false);
  });
});

test("computeReviewCanvasSize", async (t) => {
  await t.test("scales the longer side up to the target, preserving aspect ratio", () => {
    assert.deepEqual(computeReviewCanvasSize(100, 50, 400), { width: 400, height: 200 });
  });

  await t.test("scales a tall ROI by its height when height is the longer side", () => {
    assert.deepEqual(computeReviewCanvasSize(40, 200, 400), { width: 80, height: 400 });
  });

  await t.test("returns null for a degenerate ROI", () => {
    assert.equal(computeReviewCanvasSize(0, 50, 400), null);
    assert.equal(computeReviewCanvasSize(50, 0, 400), null);
    assert.equal(computeReviewCanvasSize(50, 50, 0), null);
  });
});

test("timelinePercent", async (t) => {
  await t.test("computes the proportional position", () => {
    assert.equal(timelinePercent(5, 20), 25);
  });

  await t.test("clamps to 100 for a time past the (possibly stale) duration", () => {
    assert.equal(timelinePercent(25, 20), 100);
  });

  await t.test("returns null when the time or duration is unusable", () => {
    assert.equal(timelinePercent(null, 20), null);
    assert.equal(timelinePercent(5, 0), null);
    assert.equal(timelinePercent(5, NaN), null);
  });
});

test("shouldShowZahnReview", async (t) => {
  await t.test("shows for a Zahn cup summary", () => {
    assert.equal(shouldShowZahnReview({ mode: "zahn_cup" }), true);
  });

  await t.test("hides for any other mode or a missing summary", () => {
    assert.equal(shouldShowZahnReview({ mode: "robot_activity" }), false);
    assert.equal(shouldShowZahnReview(null), false);
    assert.equal(shouldShowZahnReview(undefined), false);
  });
});
