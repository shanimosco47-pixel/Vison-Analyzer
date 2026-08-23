/*
 * Unit tests for the pure logic behind the Experimental LLM analysis
 * section (app/web/static/app.js). No browser, no DOM, no build step:
 *
 *   node --test tests_js/*.test.js
 *
 * Only the functions exported through the `module.exports` guard at the
 * bottom of app.js are under test here - the DOM-wiring functions
 * (rendering engine cards, polling a run, wiring the add/edit form) are
 * exercised instead by the backend's Flask test-client tests
 * (tests/test_web_llm_engines.py) and by manual browser testing.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");

const {
  llmEngineSubtitle,
  llmResultBadge,
  llmRunStatusLine,
  validateLLMEngineForm,
  llmPassFrameSummary,
  llmFormatFrameTimestamps,
  llmEvidenceLabel,
  llmPassPlainLanguage,
  llmEndCoarseToValidateExplanation,
} = require("../app/web/static/app.js");

test("llmEngineSubtitle", async (t) => {
  await t.test("joins provider and model with a slash", () => {
    assert.equal(
      llmEngineSubtitle({ provider_name: "openai", model_id: "gpt-5-mini" }),
      "openai / gpt-5-mini"
    );
  });

  await t.test("returns an empty string for a missing engine", () => {
    assert.equal(llmEngineSubtitle(null), "");
    assert.equal(llmEngineSubtitle(undefined), "");
  });
});

test("llmResultBadge", async (t) => {
  await t.test("confirmed status gets the ok badge", () => {
    assert.deepEqual(llmResultBadge("confirmed"), { text: "Confirmed", cls: "ok" });
  });

  await t.test("abstain status gets the review badge, not a failure colour", () => {
    const badge = llmResultBadge("abstain");
    assert.equal(badge.cls, "review");
    assert.match(badge.text, /abstain/i);
  });

  await t.test("an unrecognised or missing status still returns something displayable", () => {
    const badge = llmResultBadge(null);
    assert.equal(typeof badge.text, "string");
    assert.ok(badge.text.length > 0);
    assert.equal(badge.cls, "review");
  });
});

test("llmRunStatusLine", async (t) => {
  await t.test("a queued job reads simply", () => {
    assert.equal(llmRunStatusLine({ status: "queued" }), "Queued");
  });

  await t.test("a running job shows its stage message and elapsed time", () => {
    const line = llmRunStatusLine({
      status: "running",
      message: "Confirming the candidate break",
      elapsed_s: 12.34,
    });
    assert.equal(line, "Confirming the candidate break (12.3s elapsed)");
  });

  await t.test("a running job with no message yet still shows something", () => {
    const line = llmRunStatusLine({ status: "running", elapsed_s: 0 });
    assert.match(line, /Running/);
    assert.match(line, /0\.0s elapsed/);
  });

  await t.test("a complete job prefers its own completion message", () => {
    assert.equal(
      llmRunStatusLine({ status: "complete", message: "Confirmed: 16.50s (experimental - review required)" }),
      "Confirmed: 16.50s (experimental - review required)"
    );
  });

  await t.test("a failed job shows the error, not a generic word", () => {
    assert.equal(
      llmRunStatusLine({ status: "failed", error: "This engine's saved API key could not be found." }),
      "This engine's saved API key could not be found."
    );
  });

  await t.test("a cancelled job reads simply, regardless of any stale message", () => {
    assert.equal(llmRunStatusLine({ status: "cancelled", message: "Confirming..." }), "Cancelled");
  });

  await t.test("a null job is handled without throwing", () => {
    assert.equal(llmRunStatusLine(null), "");
  });
});

test("validateLLMEngineForm - provider and model selection", async (t) => {
  await t.test("a fully valid api_key form has no errors", () => {
    const errors = validateLLMEngineForm({
      providerName: "openai",
      modelId: "gpt-5-mini",
      credentialMode: "api_key",
      apiKey: "sk-abc123",
      envVar: "",
    });
    assert.deepEqual(errors, []);
  });

  await t.test("a fully valid env_var form has no errors", () => {
    const errors = validateLLMEngineForm({
      providerName: "gemini",
      modelId: "gemini-2.5-flash",
      credentialMode: "env_var",
      apiKey: "",
      envVar: "GEMINI_API_KEY",
    });
    assert.deepEqual(errors, []);
  });

  await t.test("missing provider is rejected", () => {
    const errors = validateLLMEngineForm({
      providerName: "",
      modelId: "gpt-5-mini",
      credentialMode: "api_key",
      apiKey: "sk-abc",
    });
    assert.ok(errors.some((e) => /provider/i.test(e)));
  });

  await t.test("missing or blank model ID is rejected", () => {
    const errors = validateLLMEngineForm({
      providerName: "openai",
      modelId: "   ",
      credentialMode: "api_key",
      apiKey: "sk-abc",
    });
    assert.ok(errors.some((e) => /model/i.test(e)));
  });
});

test("validateLLMEngineForm - write-only secret behaviour", async (t) => {
  await t.test("api_key mode with a blank key is rejected", () => {
    const errors = validateLLMEngineForm({
      providerName: "openai",
      modelId: "gpt-5-mini",
      credentialMode: "api_key",
      apiKey: "   ",
      envVar: "",
    });
    assert.ok(errors.some((e) => /api key/i.test(e)));
  });

  await t.test("env_var mode with a blank name is rejected", () => {
    const errors = validateLLMEngineForm({
      providerName: "openai",
      modelId: "gpt-5-mini",
      credentialMode: "env_var",
      apiKey: "",
      envVar: "",
    });
    assert.ok(errors.some((e) => /environment variable/i.test(e)));
  });

  await t.test('"unchanged" mode (editing without touching the credential) needs neither field', () => {
    const errors = validateLLMEngineForm({
      providerName: "openai",
      modelId: "gpt-5-mini",
      credentialMode: "unchanged",
      apiKey: "",
      envVar: "",
    });
    assert.deepEqual(errors, []);
  });

  await t.test("an unrecognised credential mode is rejected rather than silently accepted", () => {
    const errors = validateLLMEngineForm({
      providerName: "openai",
      modelId: "gpt-5-mini",
      credentialMode: "",
      apiKey: "",
      envVar: "",
    });
    assert.ok(errors.length > 0);
  });
});

test("llmPassFrameSummary", async (t) => {
  await t.test("a pass that never ran (undefined) reads 'Not run'", () => {
    assert.equal(llmPassFrameSummary(undefined), "Not run");
  });

  await t.test("an empty list also reads 'Not run', not a zero-frame range", () => {
    assert.equal(llmPassFrameSummary([]), "Not run");
  });

  await t.test("summarizes an unsorted list as first-to-last with a frame count", () => {
    assert.equal(llmPassFrameSummary([4.02, 1.0, 2.56]), "1.0s to 4.0s - 3 frames");
  });

  await t.test("uses the singular 'frame' for exactly one submitted timestamp", () => {
    assert.equal(llmPassFrameSummary([7.3]), "7.3s to 7.3s - 1 frame");
  });
});

test("llmFormatFrameTimestamps", async (t) => {
  await t.test("sorts and formats every timestamp to 0.1s, comma-joined", () => {
    assert.equal(llmFormatFrameTimestamps([4.02, 1.0, 2.56]), "1.0s, 2.6s, 4.0s");
  });

  await t.test("an undefined or empty list formats as an empty string", () => {
    assert.equal(llmFormatFrameTimestamps(undefined), "");
    assert.equal(llmFormatFrameTimestamps([]), "");
  });

  await t.test("does not mutate the input array", () => {
    const timestamps = [3.0, 1.0, 2.0];
    llmFormatFrameTimestamps(timestamps);
    assert.deepEqual(timestamps, [3.0, 1.0, 2.0]);
  });
});

test("llmEvidenceLabel", async (t) => {
  await t.test("a numeric timestamp reads 't = X.Xs'", () => {
    assert.equal(llmEvidenceLabel(12.34), "t = 12.3s");
  });

  await t.test("null (no grounded evidence) reads the explicit no-evidence message", () => {
    assert.equal(llmEvidenceLabel(null), "No grounded evidence image available");
  });

  await t.test("undefined and non-finite values also read the no-evidence message", () => {
    assert.equal(llmEvidenceLabel(undefined), "No grounded evidence image available");
    assert.equal(llmEvidenceLabel(NaN), "No grounded evidence image available");
  });
});

test("llmPassPlainLanguage", async (t) => {
  await t.test("a pass that never ran reads plainly, without fabricating anything", () => {
    assert.equal(llmPassPlainLanguage("fine", undefined), "This pass did not run.");
    assert.equal(llmPassPlainLanguage("fine", null), "This pass did not run.");
  });

  await t.test("a confirmed coarse pass reports both a start and an end estimate", () => {
    const text = llmPassPlainLanguage("coarse", {
      submitted_count: 12,
      submitted_range_s: [0.0, 26.0],
      status: "confirmed",
      start_s: 4.0,
      end_s: 21.5,
      confidence: 0.9,
      reason_codes: [],
      raw_notes: "",
    });
    assert.match(text, /12 frames/);
    assert.match(text, /0\.0s to 26\.0s/);
    assert.match(text, /start at t = 4\.0s/);
    assert.match(text, /end at t = 21\.5s/);
    assert.match(text, /confidence 0\.90/);
  });

  await t.test("a confirmed fine pass reports a single start boundary", () => {
    const text = llmPassPlainLanguage("fine", {
      submitted_count: 19,
      submitted_range_s: [2.5, 5.5],
      status: "confirmed",
      start_s: 4.0,
      end_s: 4.0,
      confidence: 0.9,
      reason_codes: [],
      raw_notes: "",
    });
    assert.match(text, /a start at t = 4\.0s/);
    assert.doesNotMatch(text, /end at/);
  });

  await t.test("a confirmed end-coarse pass reports a candidate break", () => {
    const text = llmPassPlainLanguage("end_coarse", {
      submitted_count: 53,
      submitted_range_s: [4.0, 26.0],
      status: "confirmed",
      start_s: 5.5,
      end_s: 5.5,
      confidence: 0.9,
      reason_codes: [],
      raw_notes: "",
    });
    assert.match(text, /a candidate break at t = 5\.5s/);
  });

  await t.test("a confirmed end-validate pass reports a confirmed break", () => {
    const text = llmPassPlainLanguage("end_validate", {
      submitted_count: 19,
      submitted_range_s: [1.5, 11.5],
      status: "confirmed",
      start_s: 5.8,
      end_s: 5.8,
      confidence: 0.85,
      reason_codes: [],
      raw_notes: "",
    });
    assert.match(text, /a confirmed break at t = 5\.8s/);
  });

  await t.test("an abstained pass names its reason codes instead of a timestamp", () => {
    const text = llmPassPlainLanguage("end_validate", {
      submitted_count: 19,
      submitted_range_s: [1.5, 11.5],
      status: "abstain",
      start_s: null,
      end_s: null,
      confidence: 0.0,
      reason_codes: ["no_break_found"],
      raw_notes: "",
    });
    assert.match(text, /did not produce a confident answer/i);
    assert.match(text, /no_break_found/);
  });

  await t.test("raw_notes, when present, is appended verbatim", () => {
    const text = llmPassPlainLanguage("end_validate", {
      submitted_count: 1,
      submitted_range_s: [1.0, 1.0],
      status: "abstain",
      start_s: null,
      end_s: null,
      confidence: 0.0,
      reason_codes: ["no_break_found"],
      raw_notes: "no sustained drop observed",
    });
    assert.match(text, /no sustained drop observed/);
  });

  await t.test("a pass with exactly one submitted frame uses the singular", () => {
    const text = llmPassPlainLanguage("fine", {
      submitted_count: 1,
      submitted_range_s: [4.0, 4.0],
      status: "confirmed",
      start_s: 4.0,
      end_s: 4.0,
      confidence: 0.9,
      reason_codes: [],
      raw_notes: "",
    });
    assert.match(text, /1 frame spanning/);
    assert.doesNotMatch(text, /1 frames/);
  });
});

test("llmEndCoarseToValidateExplanation", async (t) => {
  await t.test("connects the candidate to the window built around it", () => {
    const text = llmEndCoarseToValidateExplanation({
      end_coarse_candidate_s: 5.5,
      end_validation_window_s: [1.5, 11.5],
    });
    assert.match(text, /5\.5s/);
    assert.match(text, /1\.5s/);
    assert.match(text, /11\.5s/);
  });

  await t.test("returns an empty string when the candidate is missing", () => {
    assert.equal(
      llmEndCoarseToValidateExplanation({ end_validation_window_s: [1.5, 11.5] }),
      ""
    );
  });

  await t.test("returns an empty string when the window is missing", () => {
    assert.equal(llmEndCoarseToValidateExplanation({ end_coarse_candidate_s: 5.5 }), "");
  });

  await t.test("returns an empty string for a missing/empty derived object", () => {
    assert.equal(llmEndCoarseToValidateExplanation(undefined), "");
    assert.equal(llmEndCoarseToValidateExplanation({}), "");
  });
});
