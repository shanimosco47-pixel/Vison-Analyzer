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
