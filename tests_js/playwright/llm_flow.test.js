/*
 * Browser-level regression test for the complete Experimental LLM analysis
 * flow: upload -> add engine (Provider+Model dropdowns) -> run -> result
 * card (Start/End/Duration/Confidence/Uncertainty) -> "Frames sent to AI"
 * audit section -> Start/End evidence images -> Test/Edit/Delete -> a page
 * refresh that preserves configured engines.
 *
 * This is deliberately NOT part of `node --test tests_js/*.test.js` (the
 * fast, dependency-free unit gate) - it drives a real Chromium instance
 * against a real Flask server over a real socket, so it needs Playwright
 * and a Python interpreter with this repo's dependencies. Run it with:
 *
 *   NODE_PATH=/opt/node22/lib/node_modules node --test tests_js/playwright/llm_flow.test.js
 *
 * The server it drives (see e2e_server.py) never makes a live vendor call:
 * LLMEngineStore.build_provider is patched at the class level to a
 * deterministic StubTimingProvider (the same stubbing pattern
 * tests/test_llm_run_service.py already uses), so this test needs no API
 * key, costs nothing, and is fully reproducible offline.
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const path = require("node:path");
const fs = require("node:fs");
const os = require("node:os");

function requirePlaywright() {
  try {
    return require("playwright");
  } catch (err) {
    if (err.code !== "MODULE_NOT_FOUND") throw err;
    try {
      const { execSync } = require("node:child_process");
      const globalRoot = execSync("npm root -g").toString().trim();
      return require(path.join(globalRoot, "playwright"));
    } catch (fallbackErr) {
      throw new Error(
        "Could not load the 'playwright' package (checked local resolution and " +
          "the global npm root). This suite needs Playwright installed - see " +
          "this file's header comment for how to run it. " +
          `Original error: ${fallbackErr.message}`
      );
    }
  }
}

const { chromium } = requirePlaywright();

/** Clicks the first button matching `selector` whose text is exactly
 * `text` - a plain, dependency-free stand-in for Playwright's `:text()`
 * pseudo-class, which some builds resolve differently. */
async function clickButtonByText(page, selector, text) {
  const buttons = await page.$$(selector);
  for (const button of buttons) {
    const label = (await button.textContent()).trim();
    if (label === text) {
      await button.click();
      return;
    }
  }
  throw new Error(`No button matching "${selector}" with text "${text}" was found`);
}

const REPO_ROOT = path.resolve(__dirname, "..", "..");
const SERVER_SCRIPT = path.join(__dirname, "e2e_server.py");
const SCREENSHOT_DIR = path.join(os.tmpdir(), "llm-e2e-screenshots");
const STARTUP_TIMEOUT_MS = 45000;
const RUN_TIMEOUT_MS = 30000;

function startServer(scenario = "confirmed") {
  return new Promise((resolve, reject) => {
    const proc = spawn("python", [SERVER_SCRIPT, scenario], { cwd: REPO_ROOT });
    let stdoutBuf = "";
    let stderrBuf = "";
    let settled = false;

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        proc.kill();
        reject(
          new Error(
            `e2e_server.py did not report ready within ${STARTUP_TIMEOUT_MS}ms.\n` +
              `stdout:\n${stdoutBuf}\nstderr:\n${stderrBuf}`
          )
        );
      }
    }, STARTUP_TIMEOUT_MS);

    proc.stdout.on("data", (chunk) => {
      stdoutBuf += chunk.toString();
      const match = stdoutBuf.match(/LISTENING (\d+) (.+)/);
      if (match && !settled) {
        settled = true;
        clearTimeout(timer);
        resolve({ proc, port: Number(match[1]), videoPath: match[2].trim() });
      }
    });
    proc.stderr.on("data", (chunk) => {
      stderrBuf += chunk.toString();
    });
    proc.on("exit", (code) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(new Error(`e2e_server.py exited early (code ${code}).\nstderr:\n${stderrBuf}`));
      }
    });
    proc.on("error", (err) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(err);
      }
    });
  });
}

test("Experimental LLM analysis - full browser flow against a stub provider", async (t) => {
  fs.mkdirSync(SCREENSHOT_DIR, { recursive: true });

  const { proc: serverProc, port, videoPath } = await startServer();
  const baseUrl = `http://127.0.0.1:${port}`;

  const browser = await chromium.launch({ headless: true });
  const consoleErrors = [];
  const badResponses = [];

  t.after(async () => {
    await browser.close().catch(() => {});
    serverProc.kill();
  });

  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  page.on("console", (msg) => {
    if (msg.type() === "error") {
      // location().url is the failed resource's own URL for a browser-
      // generated "Failed to load resource" line (confirmed empirically -
      // Chromium reports the resource, not the calling script, for this
      // message type), which is what lets the final assertion tell an
      // expected 404 probe apart from a real console error by URL rather
      // than by matching brittle, generic message text.
      consoleErrors.push({ text: msg.text(), url: msg.location().url });
    }
  });
  page.on("pageerror", (err) => consoleErrors.push({ text: String(err), url: "" }));
  page.on("response", (response) => {
    if (response.status() >= 400) {
      badResponses.push(`${response.status()} ${response.url()}`);
    }
  });

  await t.test("upload a video", async () => {
    await page.goto(baseUrl, { waitUntil: "load" });
    await page.setInputFiles("#file-input", videoPath);
    await page.waitForSelector("#video-details:not(.hidden)", { timeout: 15000 });
    await page.waitForSelector("#step-llm:not(.disabled)");
  });

  await t.test("the page shows a build version matching the backend's own API", async () => {
    const shown = await page.$eval("#app-version", (el) => el.textContent.trim());
    const apiVersion = await page.evaluate(async () => {
      const res = await fetch("/api/version");
      return (await res.json()).app_version;
    });
    assert.equal(shown, `Build ${apiVersion}`);
    assert.notEqual(apiVersion, "");
  });

  await t.test("add an engine using the Provider + Model dropdowns", async () => {
    await page.click("#llm-add-engine");
    await page.waitForSelector("#llm-engine-form:not(.hidden)");

    await page.fill("#llm-engine-display-name", "Stub Engine");
    await page.selectOption("#llm-engine-provider", "openai");
    await page.waitForFunction(
      () => document.querySelector("#llm-engine-model-id").options.length > 0
    );
    const modelValue = await page.$eval(
      "#llm-engine-model-id option:not([disabled])",
      (opt) => opt.value
    );
    assert.ok(modelValue, "the provider's model dropdown must offer at least one option");
    await page.selectOption("#llm-engine-model-id", modelValue);

    await page.check('input[name="llm-credential-mode"][value="env_var"]');
    await page.fill("#llm-engine-env-var", "STUB_ENV_VAR_NOT_ACTUALLY_USED");

    await page.click("#llm-engine-save");
    await page.waitForSelector("#llm-engine-form", { state: "hidden" });
    await page.waitForSelector(".llm-engine-card");
  });

  await t.test("model dropdown is a real <select>, not free text", async () => {
    const tagName = await page.$eval("#llm-engine-model-id", (el) => el.tagName);
    assert.equal(tagName, "SELECT");
  });

  await t.test("run the engine and see a confirmed result card", async () => {
    await page.click(".llm-run-btn");
    await page.waitForSelector(".llm-result:not(.hidden)", { timeout: RUN_TIMEOUT_MS });

    const badgeTexts = await page.$$eval(".llm-result .badge", (els) =>
      els.map((el) => el.textContent)
    );
    assert.ok(
      badgeTexts.some((t) => /confirmed/i.test(t)),
      `expected a Confirmed badge, got: ${badgeTexts.join(", ")}`
    );

    const labels = await page.$$eval(".llm-result > .details-grid dt", (els) =>
      els.map((el) => el.textContent)
    );
    assert.deepEqual(labels, ["Start", "End", "Duration", "Confidence", "Uncertainty"]);
  });

  await t.test("Frames sent to AI section shows every pass's actual submitted frames", async () => {
    const summaryText = await page.$eval(".llm-frames-sent summary", (el) => el.textContent);
    assert.equal(summaryText, "Frames sent to AI");

    const rows = await page.$eval(".llm-frames-sent .details-grid", (dl) => {
      const dts = Array.from(dl.querySelectorAll("dt")).map((el) => el.textContent);
      const dds = Array.from(dl.querySelectorAll("dd")).map(
        (el) => el.childNodes[0] && el.childNodes[0].textContent
      );
      return dts.map((label, i) => [label, dds[i]]);
    });
    assert.deepEqual(
      rows.map(([label]) => label),
      ["Start coarse", "Start refine", "End coarse", "End validate"]
    );
    for (const [label, summary] of rows) {
      assert.notEqual(summary, "Not run", `pass "${label}" should have actually run`);
      assert.match(
        summary,
        /^\d+\.\ds to \d+\.\ds - \d+ frames?$/,
        `pass "${label}" summary "${summary}" should be a frame range/count`
      );
    }

    // Both the outer "Frames sent to AI" <details> and the per-pass nested
    // one start closed - open both before checking the expanded content.
    await page.click(".llm-frames-sent > summary");
    await page.click(".llm-frame-timestamps summary");
    const detailText = await page.$eval(".llm-frame-timestamps p", (el) => el.textContent);
    assert.match(detailText, /\d+\.\d+s/, "expanded detail should list individual timestamps");
  });

  await t.test("Start/End evidence images load with correct 0.1s timestamp labels", async () => {
    const figureCaptions = await page.$$eval(".llm-evidence-figure figcaption", (els) =>
      els.map((el) => el.textContent)
    );
    assert.deepEqual(figureCaptions, ["Start evidence", "End evidence"]);

    const imgHandles = await page.$$(".llm-evidence-image");
    assert.equal(imgHandles.length, 2, "both evidence images should be present (CONFIRMED run)");

    for (const img of imgHandles) {
      await page.waitForFunction(
        (el) => el.complete && el.naturalWidth > 0,
        img,
        { timeout: 10000 }
      );
      const src = await img.getAttribute("src");
      assert.match(src, /\/api\/llm-runs\/[^/]+\/evidence\/(start|end)$/);
    }

    const labels = await page.$$eval(".llm-evidence-timestamp-label", (els) =>
      els.map((el) => el.textContent)
    );
    assert.equal(labels.length, 2);
    for (const label of labels) {
      assert.match(label, /^t = \d+\.\ds$/, `evidence label "${label}" must show a 0.1s timestamp`);
    }
  });

  await t.test("desktop and narrow-viewport screenshots of the result card", async () => {
    // Taken here, right after the result card is fully populated - editing
    // the engine below rebuilds its card from scratch (loadLLMEngines()
    // re-renders every card) and would otherwise wipe the just-completed
    // result before it could be captured.
    await page.setViewportSize({ width: 1280, height: 900 });
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, "result-desktop.png"),
      fullPage: true,
    });
    await page.setViewportSize({ width: 375, height: 900 });
    await page.screenshot({
      path: path.join(SCREENSHOT_DIR, "result-narrow.png"),
      fullPage: true,
    });
    await page.setViewportSize({ width: 1280, height: 900 });
  });

  await t.test("Test button reports readiness without a live vendor call", async () => {
    await page.click(".llm-test-btn");
    await page.waitForSelector(".llm-test-result:not(.hidden)", { timeout: 10000 });
    const text = await page.$eval(".llm-test-result", (el) => el.textContent);
    assert.ok(text && text.length > 0);
  });

  await t.test("Edit preserves the selected model and lets the display name change", async () => {
    await clickButtonByText(page, ".llm-engine-card button", "Edit");
    await page.waitForSelector("#llm-engine-form:not(.hidden)");

    const selectedModel = await page.$eval("#llm-engine-model-id", (el) => el.value);
    assert.ok(selectedModel, "editing should preserve the previously selected (still valid) model");
    const hintHidden = await page.$eval("#llm-engine-model-hint", (el) =>
      el.classList.contains("hidden")
    );
    assert.ok(hintHidden, "a still-supported model must not show the reselect warning");

    await page.fill("#llm-engine-display-name", "Stub Engine (edited)");
    await page.click("#llm-engine-save");
    await page.waitForSelector("#llm-engine-form", { state: "hidden" });
    const name = await page.$eval(".llm-engine-name", (el) => el.textContent);
    assert.equal(name, "Stub Engine (edited)");
  });

  await t.test("a page refresh preserves the configured engine", async () => {
    await page.reload({ waitUntil: "load" });
    await page.waitForSelector(".llm-engine-card");
    const name = await page.$eval(".llm-engine-name", (el) => el.textContent);
    assert.equal(name, "Stub Engine (edited)");
    // A refresh drops in-page state (state.video, same as every other step
    // in this wizard) - #step-llm goes back to its pointer-events:none
    // "disabled" styling until a video is re-selected, exactly like steps
    // 2-5. Re-upload so the still-configured engine can be exercised again.
    await page.setInputFiles("#file-input", videoPath);
    await page.waitForSelector("#step-llm:not(.disabled)");
  });

  await t.test("Delete removes the engine", async () => {
    await clickButtonByText(page, ".llm-engine-card button", "Delete");
    await page.waitForFunction(
      () => document.querySelectorAll(".llm-engine-card").length === 0
    );
  });

  await t.test("no console errors or broken network requests occurred", () => {
    // The one expected exception: loadLatestAuditForEngine() deliberately
    // probes GET /api/llm-engines/<id>/latest-audit for every configured
    // engine on load, and a 404 (never run yet) is a normal, handled
    // outcome - not a bug - so it (and the console line Chromium logs for
    // any 4xx/5xx response, regardless of whether the page handled it)
    // is excluded here by URL, the same way a real operator would ignore
    // it rather than by matching the browser's generic, unrelated-looking
    // message text.
    const isExpectedLatestAuditProbe = (url) => /\/llm-engines\/[^/]+\/latest-audit$/.test(url);
    const unexpectedBadResponses = badResponses.filter((entry) => {
      const [status, url] = entry.split(" ", 2);
      return !(status === "404" && isExpectedLatestAuditProbe(url));
    });
    const unexpectedConsoleErrors = consoleErrors.filter(
      (entry) => !isExpectedLatestAuditProbe(entry.url)
    );
    assert.deepEqual(
      unexpectedConsoleErrors,
      [],
      `console errors: ${unexpectedConsoleErrors.map((e) => e.text).join(" | ")}`
    );
    assert.deepEqual(
      unexpectedBadResponses,
      [],
      `failed requests: ${unexpectedBadResponses.join(" | ")}`
    );
  });
});

test("Experimental LLM analysis - the wrong-candidate audit trail explains itself", async (t) => {
  // The exact shape a real operator reported: end-coarse nominates a
  // too-early candidate (5.5s), so the coarse contact sheet built around
  // it ([3.5, 7.5]s) never sees the clip's actual continuation past 17s,
  // and the run abstains. This proves the "How the AI decided"
  // section - not just the pipeline/audit-builder layer already covered
  // by test_pipeline_derived_decisions_survive_a_rejected_end_candidate
  // and test_build_audit_record_for_a_rejected_candidate_still_shows_the_chain
  // - actually explains that chain on the page itself, without DevTools.
  const { proc: serverProc, port, videoPath } = await startServer("wrong_candidate");
  const baseUrl = `http://127.0.0.1:${port}`;

  const browser = await chromium.launch({ headless: true });
  t.after(async () => {
    await browser.close().catch(() => {});
    serverProc.kill();
  });

  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

  await t.test("run the engine and reach an honest ABSTAIN, not a fabricated result", async () => {
    await page.goto(baseUrl, { waitUntil: "load" });
    await page.setInputFiles("#file-input", videoPath);
    await page.waitForSelector("#video-details:not(.hidden)", { timeout: 15000 });

    await page.click("#llm-add-engine");
    await page.waitForSelector("#llm-engine-form:not(.hidden)");
    await page.fill("#llm-engine-display-name", "Wrong Candidate Engine");
    await page.selectOption("#llm-engine-provider", "openai");
    await page.waitForFunction(
      () => document.querySelector("#llm-engine-model-id").options.length > 0
    );
    await page.check('input[name="llm-credential-mode"][value="env_var"]');
    await page.fill("#llm-engine-env-var", "STUB_ENV_VAR_NOT_ACTUALLY_USED");
    await page.click("#llm-engine-save");
    await page.waitForSelector("#llm-engine-form", { state: "hidden" });

    await page.click(".llm-run-btn");
    await page.waitForSelector(".llm-result:not(.hidden)", { timeout: RUN_TIMEOUT_MS });
    await page.waitForSelector(".llm-how-decided", { timeout: 10000 });

    const badgeTexts = await page.$$eval(".llm-result .badge", (els) =>
      els.map((el) => el.textContent)
    );
    assert.ok(
      badgeTexts.some((t) => /abstain/i.test(t)),
      `expected an Abstained badge, got: ${badgeTexts.join(", ")}`
    );
    // Never a fabricated evidence image for a run that didn't confirm.
    assert.equal(await page.$$(".llm-evidence-image").then((els) => els.length), 0);
  });

  await t.test("Frames sent to AI still shows the end-coarse and end-validate ranges", async () => {
    await page.click(".llm-frames-sent > summary");
    const rows = await page.$eval(".llm-frames-sent .details-grid", (dl) => {
      const dts = Array.from(dl.querySelectorAll("dt")).map((el) => el.textContent);
      const dds = Array.from(dl.querySelectorAll("dd")).map(
        (el) => el.childNodes[0] && el.childNodes[0].textContent
      );
      return Object.fromEntries(dts.map((label, i) => [label, dds[i]]));
    });
    assert.notEqual(rows["End coarse"], "Not run");
    assert.notEqual(rows["End validate"], "Not run");
    // The coarse contact sheet's own 9 panels, evenly spaced across
    // [3.5, 7.5]s (see the "How the AI decided" assertions below) - a
    // single bounded composite image, not an open-ended scan.
    assert.match(rows["End validate"], /3\.5s to 7\.5s/);
  });

  await t.test("How the AI decided explains the candidate -> window -> abstain chain", async () => {
    await page.click(".llm-how-decided > summary");
    const paragraphs = await page.$$eval(".llm-how-decided-pass", (blocks) =>
      blocks.map((block) => ({
        heading: block.querySelector("h5").textContent,
        text: Array.from(block.querySelectorAll("p"))
          .map((p) => p.textContent)
          .join(" "),
      }))
    );

    const endCoarse = paragraphs.find((p) => p.heading === "End coarse");
    assert.ok(endCoarse, "no End coarse block found");
    assert.match(endCoarse.text, /candidate break at t = 5\.5s/);
    assert.match(endCoarse.text, /3\.5s/);
    assert.match(endCoarse.text, /7\.5s/);

    const endValidate = paragraphs.find((p) => p.heading === "End validate");
    assert.ok(endValidate, "no End validate block found");
    assert.match(endValidate.text, /did not produce a confident answer/i);
    assert.match(endValidate.text, /no_break_found/);
    assert.match(endValidate.text, /no sustained drop observed/);
  });

  await t.test("Download audit report (JSON) produces the same chain as a file", async () => {
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.click(".llm-audit-actions a"),
    ]);
    assert.match(download.suggestedFilename(), /^llm-run-.+-audit\.json$/);
    const downloadPath = await download.path();
    const audit = JSON.parse(fs.readFileSync(downloadPath, "utf8"));
    assert.equal(audit["final_verdict"]["status"], "abstain");
    assert.equal(audit["derived"]["end_coarse_candidate_s"], 5.5);
    assert.deepEqual(audit["derived"]["end_validation_window_s"], [3.5, 7.5]);
    // The downloaded audit must be matchable against the running backend
    // and the page that produced it - all three carry the same build id.
    const shownVersion = await page.$eval("#app-version", (el) => el.textContent.trim());
    assert.equal(shownVersion, `Build ${audit["app_version"]}`);
  });

  await t.test("a page refresh restores the same explanation from the durable audit", async () => {
    await page.reload({ waitUntil: "load" });
    await page.waitForSelector(".llm-how-decided", { timeout: 10000 });
    const badgeTexts = await page.$$eval(".llm-result .badge", (els) =>
      els.map((el) => el.textContent)
    );
    assert.ok(badgeTexts.some((t) => /abstain/i.test(t)));
  });
});

test("Experimental LLM analysis - a candidate conflict explains itself and resolves correctly", async (t) => {
  // CODEX REAL-RUN AUDIT DIAGNOSIS regression (run
  // bcf2dc44cffe48f3bf3cd32de8e44e81, commit 4dfb2d1): unlike the
  // wrong_candidate scenario above (where the coarse pass's own end
  // estimate happens to equal the wrong candidate, so no conflict fires),
  // this scenario's coarse pass reports a genuinely different, independent
  // end estimate (21.0s) - triggering the bounded second validation call
  // and proving, on the page itself, that the pipeline never reports the
  // early false candidate (5.5s) as the answer, and that the conflict and
  // its resolution are explained without DevTools.
  const { proc: serverProc, port, videoPath } = await startServer("conflict");
  const baseUrl = `http://127.0.0.1:${port}`;

  const browser = await chromium.launch({ headless: true });
  t.after(async () => {
    await browser.close().catch(() => {});
    serverProc.kill();
  });

  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });

  await t.test("run the engine and confirm the later, correctly-grounded break", async () => {
    await page.goto(baseUrl, { waitUntil: "load" });
    await page.setInputFiles("#file-input", videoPath);
    await page.waitForSelector("#video-details:not(.hidden)", { timeout: 15000 });

    await page.click("#llm-add-engine");
    await page.waitForSelector("#llm-engine-form:not(.hidden)");
    await page.fill("#llm-engine-display-name", "Conflict Engine");
    await page.selectOption("#llm-engine-provider", "openai");
    await page.waitForFunction(
      () => document.querySelector("#llm-engine-model-id").options.length > 0
    );
    await page.check('input[name="llm-credential-mode"][value="env_var"]');
    await page.fill("#llm-engine-env-var", "STUB_ENV_VAR_NOT_ACTUALLY_USED");
    await page.click("#llm-engine-save");
    await page.waitForSelector("#llm-engine-form", { state: "hidden" });

    await page.click(".llm-run-btn");
    await page.waitForSelector(".llm-result:not(.hidden)", { timeout: RUN_TIMEOUT_MS });
    await page.waitForSelector(".llm-how-decided", { timeout: 10000 });

    const badgeTexts = await page.$$eval(".llm-result .badge", (els) =>
      els.map((el) => el.textContent)
    );
    assert.ok(
      badgeTexts.some((t) => /confirmed/i.test(t)),
      `expected a Confirmed badge, got: ${badgeTexts.join(", ")}`
    );
    const resultRows = await page.$eval(".llm-result > .details-grid", (dl) => {
      const dts = Array.from(dl.querySelectorAll("dt")).map((el) => el.textContent);
      const dds = Array.from(dl.querySelectorAll("dd")).map((el) => el.textContent);
      return Object.fromEntries(dts.map((label, i) => [label, dds[i]]));
    });
    const endValue = parseFloat(resultRows["End"]);
    assert.ok(Math.abs(endValue - 20.5) < 1.0, `expected End near 20.5s, got ${resultRows["End"]}`);
    assert.notEqual(endValue, 5.5);
  });

  await t.test("Frames sent to AI includes the End validate (conflict check) row", async () => {
    await page.click(".llm-frames-sent > summary");
    const rows = await page.$eval(".llm-frames-sent .details-grid", (dl) => {
      const dts = Array.from(dl.querySelectorAll("dt")).map((el) => el.textContent);
      const dds = Array.from(dl.querySelectorAll("dd")).map(
        (el) => el.childNodes[0] && el.childNodes[0].textContent
      );
      return Object.fromEntries(dts.map((label, i) => [label, dds[i]]));
    });
    assert.notEqual(rows["End validate (conflict check)"], undefined);
    assert.notEqual(rows["End validate (conflict check)"], "Not run");
  });

  await t.test("How the AI decided explains the conflict and its resolution", async () => {
    await page.click(".llm-how-decided > summary");
    const blocks = await page.$$eval(".llm-how-decided-pass", (els) =>
      els.map((block) => ({
        heading: block.querySelector("h5").textContent,
        text: Array.from(block.querySelectorAll("p"))
          .map((p) => p.textContent)
          .join(" "),
      }))
    );

    const conflictBlock = blocks.find((b) => b.heading === "Candidate conflict");
    assert.ok(conflictBlock, "no Candidate conflict block found");
    assert.match(conflictBlock.text, /5\.5s/);
    assert.match(conflictBlock.text, /21\.0s/);
    assert.match(conflictBlock.text, /coarse pass's own end estimate held up/);
  });

  await t.test("Download audit report (JSON) records the conflict resolution", async () => {
    const [download] = await Promise.all([
      page.waitForEvent("download"),
      page.click(".llm-audit-actions a"),
    ]);
    const downloadPath = await download.path();
    const audit = JSON.parse(fs.readFileSync(downloadPath, "utf8"));
    assert.equal(audit["final_verdict"]["status"], "confirmed");
    assert.equal(audit["derived"]["candidate_conflict"], true);
    assert.equal(audit["derived"]["conflict_resolution"], "coarse_estimate_confirmed");
    assert.equal(audit["derived"]["confirmed_end_source"], "coarse_estimate");
    assert.notEqual(audit["final_verdict"]["end_s"], 5.5);
    assert.ok(Math.abs(audit["final_verdict"]["end_s"] - 20.5) < 1.0);
  });
});
