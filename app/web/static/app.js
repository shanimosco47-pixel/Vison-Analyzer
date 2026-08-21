/*
 * Video Analyzer - browser logic.
 *
 * Deliberately dependency-free: one file, no build step, no external scripts
 * (the page's Content-Security-Policy would block them anyway).
 *
 * The flow mirrors the five steps in the page: upload -> choose mode ->
 * configure (including marking the region on a real frame) -> analyse with
 * live progress -> results that link back into the video.
 */

"use strict";

const POLL_INTERVAL_MS = 700;
const REVIEW_SEEK_STEP_S = 0.1;
const REVIEW_BOUNDARY_CONTEXT_S = 1.0;
const REVIEW_CANVAS_TARGET_PX = 480;

const state = {
  video: null,        // metadata of the uploaded video
  mode: null,         // selected analysis mode
  // Zahn requires two operator-marked anchors, each {x, y, t} in source
  // pixels / video-relative seconds: an early one where the outlet is
  // clearly visible, and a late one near the expected end of the stream.
  // A single click is not enough to bound tracking drift over a whole
  // clip - see app/analysis/zahn_detector.py's two-anchor contract.
  outletEarly: null,
  outletLate: null,
  markingAnchor: "early",  // which anchor the next Zahn click sets
  roi: null,           // {x, y, width, height} in source pixels (non-Zahn only)
  frameTime: 0,        // timestamp of the frame shown in the picker
  jobId: null,
  pollTimer: null,
  dragStart: null,
  review: {
    active: false,      // a Zahn result with a usable ROI is on screen
    roi: null,           // {x, y, width, height} in source pixels, from the analysis result
    flowStartS: null,
    flowEndS: null,
    frameCallbackId: null,
    rafId: null,
  },
};

const el = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ */
/* Small helpers                                                       */
/* ------------------------------------------------------------------ */

function showError(message) {
  el("error-text").textContent = message;
  el("error-banner").classList.remove("hidden");
}

function clearError() {
  el("error-banner").classList.add("hidden");
}

function enableStep(id, enabled) {
  el(id).classList.toggle("disabled", !enabled);
}

function formatDuration(seconds) {
  if (seconds === null || seconds === undefined) return "unknown";
  const total = Math.floor(seconds);
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const secs = total % 60;
  const stem = hours > 0
    ? `${hours}:${String(minutes).padStart(2, "0")}:${String(secs).padStart(2, "0")}`
    : `${minutes}:${String(secs).padStart(2, "0")}`;
  return `${stem} (${seconds.toFixed(1)} s)`;
}

function formatBytes(bytes) {
  if (!bytes) return "unknown";
  const megabytes = bytes / (1024 * 1024);
  return megabytes >= 1024
    ? `${(megabytes / 1024).toFixed(2)} GB`
    : `${megabytes.toFixed(1)} MB`;
}

/* ------------------------------------------------------------------ */
/* Pure logic for the Zahn review panel                                */
/*                                                                      */
/* Kept free of the DOM so it can be unit-tested directly from Node     */
/* (see tests_js/) without a browser. The module.exports guard at the   */
/* bottom of this file is a no-op in the browser, where `module` does   */
/* not exist.                                                           */
/* ------------------------------------------------------------------ */

/** Format elapsed video time with tenths precision, e.g. "00:16.3". */
function formatTimeTenths(seconds) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return "--:--.-";
  // Round to whole tenths *before* decomposing into hours/minutes/seconds, so a
  // value like 59.96 carries into the next minute ("01:00.0") instead of
  // rounding its seconds component alone up to an impossible "60.0".
  const totalTenths = Math.round(Math.max(0, seconds) * 10);
  const wholeSeconds = Math.floor(totalTenths / 10);
  const tenth = totalTenths % 10;
  const hours = Math.floor(wholeSeconds / 3600);
  const minutes = Math.floor((wholeSeconds % 3600) / 60);
  const secs = wholeSeconds % 60;
  const secsText = `${String(secs).padStart(2, "0")}.${tenth}`;
  const minutesText = String(minutes).padStart(2, "0");
  return hours > 0 ? `${hours}:${minutesText}:${secsText}` : `${minutesText}:${secsText}`;
}

/** Clamp a seek target to [0, duration]. An unknown/invalid duration only clamps below. */
function clampSeekTime(time, duration) {
  const numericTime = typeof time === "number" && Number.isFinite(time) ? time : 0;
  const upperBound = typeof duration === "number" && Number.isFinite(duration) && duration > 0
    ? duration
    : Infinity;
  return Math.min(Math.max(numericTime, 0), upperBound);
}

/** The clamped result of stepping the current time by a (possibly negative) delta. */
function stepSeekTime(currentTime, deltaSeconds, duration) {
  return clampSeekTime((currentTime || 0) + deltaSeconds, duration);
}

/** The clamped seek target that gives some lead-in before a detected boundary. */
function boundaryJumpTime(boundarySeconds, duration, contextSeconds = REVIEW_BOUNDARY_CONTEXT_S) {
  if (typeof boundarySeconds !== "number" || !Number.isFinite(boundarySeconds)) return null;
  return clampSeekTime(boundarySeconds - contextSeconds, duration);
}

/**
 * Whether an ROI is a usable rectangle to crop and enlarge.
 *
 * Bounds against the video's native dimensions are only checked when they
 * are supplied, so this can also validate an ROI before the video element
 * has finished loading metadata.
 */
function isValidRoi(roi, videoWidth, videoHeight) {
  if (!roi || typeof roi !== "object") return false;
  const { x, y, width, height } = roi;
  if (![x, y, width, height].every((value) => typeof value === "number" && Number.isFinite(value))) {
    return false;
  }
  if (width <= 0 || height <= 0) return false;
  if (typeof videoWidth === "number" && typeof videoHeight === "number"
    && videoWidth > 0 && videoHeight > 0) {
    if (x < 0 || y < 0 || x + width > videoWidth || y + height > videoHeight) return false;
  }
  return true;
}

/**
 * The internal pixel resolution for the enlarged grayscale canvas: the ROI's
 * aspect ratio, scaled so its longer side is `targetLongSidePx`.
 *
 * Returns null for a degenerate ROI so callers can show a clear message
 * instead of drawing to a zero-sized or distorted canvas.
 */
function computeReviewCanvasSize(roiWidth, roiHeight, targetLongSidePx) {
  if (!(roiWidth > 0) || !(roiHeight > 0) || !(targetLongSidePx > 0)) return null;
  const scale = targetLongSidePx / Math.max(roiWidth, roiHeight);
  return {
    width: Math.max(1, Math.round(roiWidth * scale)),
    height: Math.max(1, Math.round(roiHeight * scale)),
  };
}

/** The percentage position (0-100) of a video-relative time along its timeline. */
function timelinePercent(seconds, duration) {
  if (typeof seconds !== "number" || !Number.isFinite(seconds)) return null;
  if (typeof duration !== "number" || !Number.isFinite(duration) || duration <= 0) return null;
  return Math.min(100, Math.max(0, (seconds / duration) * 100));
}

/** Whether the Zahn review panel should be shown at all for this result summary. */
function shouldShowZahnReview(summary) {
  return Boolean(summary && summary.mode === "zahn_cup");
}

async function readError(response) {
  try {
    const payload = await response.json();
    return payload.error || `Request failed (${response.status}).`;
  } catch (err) {
    return `Request failed (${response.status}).`;
  }
}

async function getJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) throw new Error(await readError(response));
  return response.json();
}

/* ------------------------------------------------------------------ */
/* Step 1 - upload                                                     */
/* ------------------------------------------------------------------ */

function initUpload() {
  const dropzone = el("dropzone");
  const input = el("file-input");

  dropzone.addEventListener("click", () => input.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    if (input.files.length) uploadFile(input.files[0]);
  });

  ["dragenter", "dragover"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((name) =>
    dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (event) => {
    const file = event.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
}

function uploadFile(file) {
  clearError();
  resetAnalysis();
  const form = new FormData();
  form.append("file", file);

  const bar = el("upload-bar");
  el("upload-progress").classList.remove("hidden");
  bar.style.width = "0%";

  const request = new XMLHttpRequest();
  request.open("POST", "/api/videos");
  request.upload.addEventListener("progress", (event) => {
    if (event.lengthComputable) {
      bar.style.width = `${Math.round((event.loaded / event.total) * 100)}%`;
    }
  });
  request.addEventListener("load", () => {
    el("upload-progress").classList.add("hidden");
    let payload;
    try {
      payload = JSON.parse(request.responseText);
    } catch (err) {
      showError("The server returned an unreadable response to the upload.");
      return;
    }
    if (request.status >= 400) {
      showError(payload.error || "The video could not be uploaded.");
      return;
    }
    onVideoUploaded(payload);
  });
  request.addEventListener("error", () => {
    el("upload-progress").classList.add("hidden");
    showError("The upload failed. Check that the application is still running.");
  });
  request.send(form);
}

function onVideoUploaded(info) {
  state.video = info;
  el("meta-name").textContent = info.original_name;
  el("meta-duration").textContent = formatDuration(info.duration_s);
  el("meta-resolution").textContent = `${info.width} x ${info.height}`;
  el("meta-fps").textContent =
    `${info.fps.toFixed(2)} fps${info.fps_source === "timestamps" ? " (estimated)" : ""}`;
  el("meta-size").textContent = formatBytes(info.size_bytes);
  el("meta-codec").textContent = info.fourcc;

  const warnings = el("meta-warnings");
  if (info.warnings && info.warnings.length) {
    warnings.textContent = info.warnings.join(" ");
    warnings.classList.remove("hidden");
  } else {
    warnings.classList.add("hidden");
  }

  const preview = el("preview");
  preview.src = `/api/videos/${info.video_id}/media`;
  preview.load();

  el("video-details").classList.remove("hidden");
  enableStep("step-mode", true);
  enableStep("step-configure", true);
  enableStep("step-analyse", true);
  // A new video invalidates any anchors marked on the previous one - never
  // silently carry them over onto different footage.
  resetOutletAnchors();
  drawRegion();
  updateRegionSummary();
  updateRunButton();
}

/* ------------------------------------------------------------------ */
/* Step 2 - mode selection                                             */
/* ------------------------------------------------------------------ */

function initModes() {
  document.querySelectorAll('input[name="mode"]').forEach((radio) => {
    radio.addEventListener("change", () => {
      state.mode = radio.value;
      applyModeToSettings();
      updateRunButton();
    });
  });

  el("scan-sensitivity").addEventListener("input", (event) => {
    el("scan-sensitivity-value").textContent = event.target.value;
  });
  el("zahn-sensitivity").addEventListener("input", (event) => {
    el("zahn-sensitivity-value").textContent = event.target.value;
  });
}

function isZahn() {
  return state.mode === "zahn_cup";
}

/**
 * Clear both Zahn anchors and any drawn region on `currentState` in place.
 * Pure (no DOM) so it is directly unit-testable - see
 * tests_js/analysis-params.test.js - separately from resetOutletAnchors's
 * DOM sync below.
 */
function resetOutletAnchorsState(currentState) {
  currentState.outletEarly = null;
  currentState.outletLate = null;
  currentState.roi = null;
  currentState.markingAnchor = "early";
}

/**
 * Clear both Zahn anchors and any drawn region. Called on a mode change and
 * on a fresh video upload - an anchor marked on one video/mode must never be
 * silently carried over and applied to another.
 */
function resetOutletAnchors() {
  resetOutletAnchorsState(state);
  if (el("anchor-target-early")) el("anchor-target-early").checked = true;
}

function applyModeToSettings() {
  const zahn = isZahn();
  el("settings-zahn").classList.toggle("hidden", !zahn);
  el("settings-activity").classList.toggle("hidden", zahn);
  el("idle-pause-field").classList.toggle("hidden", state.mode !== "robot_activity");
  el("advanced-preroll-field").classList.toggle("hidden", zahn);
  el("advanced-start-persistence-field").classList.toggle("hidden", !zahn);
  el("anchor-toggle").classList.toggle("hidden", !zahn);

  el("picker-instruction").textContent = zahn
    ? "Mark the outlet on two frames: an early one where the cup and the outlet "
      + "hole are clearly visible, and a late one near where the stream is "
      + "expected to end. Play the preview to each frame, press “Use the frame "
      + "currently shown”, then click exactly on the outlet hole."
    : "Optional: drag a box around the machine or the area you care about. "
      + "Restricting the region makes the scan faster and ignores activity elsewhere.";

  el("frame-picker").classList.remove("hidden");
  resetOutletAnchors();
  drawRegion();
  updateRegionSummary();
  updateRunButton();
}

/* ------------------------------------------------------------------ */
/* Step 3 - frame picker                                               */
/* ------------------------------------------------------------------ */

function initFramePicker() {
  el("grab-frame").addEventListener("click", loadFrameFromPreview);
  el("clear-region").addEventListener("click", () => {
    resetOutletAnchors();
    drawRegion();
    updateRegionSummary();
    updateRunButton();
  });
  [el("anchor-target-early"), el("anchor-target-late")].forEach((radio) => {
    radio.addEventListener("change", () => {
      state.markingAnchor = radio.value;
    });
  });

  const canvas = el("frame-canvas");
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  canvas.addEventListener("pointerleave", () => {
    state.dragStart = null;
  });

  el("frame-image").addEventListener("load", () => {
    const image = el("frame-image");
    const canvasEl = el("frame-canvas");
    canvasEl.width = image.naturalWidth;
    canvasEl.height = image.naturalHeight;
    drawRegion();
  });
}

function loadFrameFromPreview() {
  if (!state.video) return;
  const time = el("preview").currentTime || 0;
  state.frameTime = time;
  el("frame-image").src =
    `/api/videos/${state.video.video_id}/frame?t=${encodeURIComponent(time.toFixed(3))}`;
  el("frame-time-label").textContent = `Frame at ${time.toFixed(2)} s`;
  el("clear-region").classList.remove("hidden");
}

/** Convert a pointer event into source-video pixel coordinates. */
function toSourceCoords(event) {
  const canvas = el("frame-canvas");
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: Math.round((event.clientX - rect.left) * scaleX),
    y: Math.round((event.clientY - rect.top) * scaleY),
  };
}

function onPointerDown(event) {
  if (!el("frame-image").src) return;
  state.dragStart = toSourceCoords(event);
}

function onPointerMove(event) {
  if (!state.dragStart) return;
  const current = toSourceCoords(event);
  drawRegion(rectFrom(state.dragStart, current));
}

function onPointerUp(event) {
  if (!state.dragStart) return;
  const current = toSourceCoords(event);
  const rect = rectFrom(state.dragStart, current);
  const isClick = rect.width < 8 || rect.height < 8;

  if (isZahn()) {
    // Zahn requires two clicked anchors, not a drawn region - a drag is
    // simply ignored here rather than treated as a region.
    if (isClick) {
      const anchor = { x: current.x, y: current.y, t: state.frameTime };
      if (state.markingAnchor === "late") {
        state.outletLate = anchor;
      } else {
        state.outletEarly = anchor;
        // Minimal, explicit auto-advance: once the early anchor is set,
        // the natural next step is the late one - only when it is not
        // already marked, so re-marking the early anchor never silently
        // steals focus away from a late anchor the operator is editing.
        if (!state.outletLate) {
          state.markingAnchor = "late";
          el("anchor-target-late").checked = true;
        }
      }
    }
  } else if (!isClick) {
    state.roi = rect;
  }
  state.dragStart = null;
  drawRegion();
  updateRegionSummary();
  updateRunButton();
}

function rectFrom(a, b) {
  return {
    x: Math.min(a.x, b.x),
    y: Math.min(a.y, b.y),
    width: Math.abs(a.x - b.x),
    height: Math.abs(a.y - b.y),
  };
}

function drawRegion(pendingRect) {
  const canvas = el("frame-canvas");
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  const lineWidth = Math.max(2, Math.round(canvas.width / 400));

  if (pendingRect) {
    context.strokeStyle = "#1f5fa8";
    context.setLineDash([8, 6]);
    context.lineWidth = lineWidth;
    context.strokeRect(pendingRect.x, pendingRect.y, pendingRect.width, pendingRect.height);
    context.setLineDash([]);
  }
  if (state.roi) {
    context.strokeStyle = "#ffaa00";
    context.lineWidth = lineWidth;
    context.strokeRect(state.roi.x, state.roi.y, state.roi.width, state.roi.height);
  }
  const radius = Math.max(6, Math.round(canvas.width / 90));
  [["#ffaa00", state.outletEarly, "E"], ["#1f9d55", state.outletLate, "L"]].forEach(
    ([color, anchor, label]) => {
      if (!anchor) return;
      context.strokeStyle = color;
      context.fillStyle = color;
      context.lineWidth = lineWidth;
      context.beginPath();
      context.arc(anchor.x, anchor.y, radius, 0, Math.PI * 2);
      context.stroke();
      context.beginPath();
      context.moveTo(anchor.x, anchor.y);
      context.lineTo(anchor.x, canvas.height);
      context.setLineDash([10, 8]);
      context.stroke();
      context.setLineDash([]);
      context.font = `${Math.max(12, Math.round(canvas.width / 45))}px sans-serif`;
      context.fillText(label, anchor.x + radius + 4, anchor.y - radius - 4);
    }
  );
}

function describeAnchor(anchor) {
  return `x=${anchor.x}, y=${anchor.y} (t=${anchor.t.toFixed(2)}s)`;
}

function updateRegionSummary() {
  const summary = el("region-summary");
  if (isZahn()) {
    const early = state.outletEarly ? describeAnchor(state.outletEarly) : "not yet marked";
    const late = state.outletLate ? describeAnchor(state.outletLate) : "not yet marked";
    summary.textContent = `Early anchor: ${early}. Late anchor: ${late}. `
      + "Both are required; select an anchor above and click again to replace it.";
  } else if (state.roi) {
    summary.textContent =
      `Region: ${state.roi.width} x ${state.roi.height} pixels at `
      + `(${state.roi.x}, ${state.roi.y}).`;
  } else {
    summary.textContent = "No region selected - the whole frame will be analysed.";
  }
}

/* ------------------------------------------------------------------ */
/* Step 4 - run the analysis                                           */
/* ------------------------------------------------------------------ */

function updateRunButton() {
  const zahnReady = Boolean(state.outletEarly && state.outletLate);
  const ready = Boolean(state.video && state.mode && (!isZahn() || zahnReady));
  el("run-analysis").disabled = !ready;
}

/**
 * Build the `params` payload sent to POST /api/analyses.
 *
 * Takes `currentState` and a `getValue` field-reader explicitly (both
 * default to the real page's `state`/`el(...).value`) so this can run - and
 * be asserted on - outside a browser: see tests_js/analysis-params.test.js.
 * Zahn always sends both operator-marked anchors as `(x, y, t)` -
 * `outlet`/`outlet_reference_s` for the early one, `outlet_end`/
 * `outlet_end_reference_s` for the late one (a supervisor review found the
 * single-anchor predecessor of this field was never even wired through at
 * all - see app/analysis/zahn_detector.py's anchor handling for what the
 * backend does with these once sent, including the ordering/bounds
 * validation this payload alone cannot guarantee).
 */
function collectParams(currentState = state, getValue = (id) => Number(el(id).value)) {
  const params = {};
  if (currentState.mode === "zahn_cup") {
    if (currentState.outletEarly) {
      params.outlet = { x: currentState.outletEarly.x, y: currentState.outletEarly.y };
      params.outlet_reference_s = currentState.outletEarly.t;
    }
    if (currentState.outletLate) {
      params.outlet_end = { x: currentState.outletLate.x, y: currentState.outletLate.y };
      params.outlet_end_reference_s = currentState.outletLate.t;
    }
    params.flow_end_persistence_s = getValue("zahn-end-persistence");
    params.flow_start_persistence_s = getValue("zahn-start-persistence");
    params.activity_threshold = getValue("zahn-sensitivity");
  } else {
    if (currentState.roi) params.roi = currentState.roi;
    params.shortest_event_s = getValue("shortest-event");
    params.min_event_duration_s = getValue("min-event");
    params.sensitivity = getValue("scan-sensitivity");
    const roll = getValue("roll-seconds");
    params.pre_roll_s = roll;
    params.post_roll_s = roll;
    if (currentState.mode === "robot_activity") {
      params.idle_pause_s = getValue("idle-pause");
    }
  }
  return params;
}

function initAnalysis() {
  el("run-analysis").addEventListener("click", startAnalysis);
  el("cancel-analysis").addEventListener("click", cancelAnalysis);
  el("error-dismiss").addEventListener("click", clearError);
}

async function startAnalysis() {
  clearError();
  resetResults();
  el("run-analysis").disabled = true;
  el("cancel-analysis").classList.remove("hidden");
  el("job-progress").classList.remove("hidden");
  setProgress(0, "Starting");

  try {
    const job = await getJSON("/api/analyses", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        video_id: state.video.video_id,
        mode: state.mode,
        params: collectParams(),
        recording_start: el("recording-start").value || null,
      }),
    });
    state.jobId = job.job_id;
    el("job-plan").textContent = describePlan(job.plan);
    state.pollTimer = setInterval(pollJob, POLL_INTERVAL_MS);
    pollJob();
  } catch (error) {
    showError(error.message);
    finishAnalysisUI();
  }
}

function describePlan(plan) {
  if (!plan) return "";
  const parts = [];
  if (plan.sampling_interval_s) {
    parts.push(`scan samples every ${plan.sampling_interval_s} s`);
  }
  if (plan.frame_step) parts.push(plan.frame_step);
  if (plan.estimated_samples) parts.push(`${plan.estimated_samples} samples`);
  if (plan.scan_scale) parts.push(`scan scale ${plan.scan_scale}`);
  return parts.length ? `Plan: ${parts.join(", ")}.` : "";
}

async function cancelAnalysis() {
  if (!state.jobId) return;
  try {
    await getJSON(`/api/analyses/${state.jobId}/cancel`, { method: "POST" });
  } catch (error) {
    showError(error.message);
  }
}

function setProgress(fraction, message) {
  el("job-bar").style.width = `${Math.round(fraction * 100)}%`;
  el("job-message").textContent = message;
}

async function pollJob() {
  if (!state.jobId) return;
  let job;
  try {
    job = await getJSON(`/api/analyses/${state.jobId}`);
  } catch (error) {
    stopPolling();
    showError(error.message);
    finishAnalysisUI();
    return;
  }

  setProgress(job.progress, job.message || job.status);
  if (job.status === "complete") {
    stopPolling();
    finishAnalysisUI();
    renderResults(job);
  } else if (job.status === "failed" || job.status === "cancelled") {
    stopPolling();
    finishAnalysisUI();
    if (job.status === "failed") showError(job.error || "The analysis failed.");
  }
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function finishAnalysisUI() {
  el("cancel-analysis").classList.add("hidden");
  updateRunButton();
}

function resetAnalysis() {
  stopPolling();
  state.jobId = null;
  el("job-progress").classList.add("hidden");
  resetResults();
}

function resetResults() {
  enableStep("step-results", false);
  ["headline", "result-notes", "events-holder", "trace-holder", "diagnostics-holder"].forEach((id) =>
    el(id).classList.add("hidden")
  );
  el("events-table").querySelector("tbody").innerHTML = "";
  teardownZahnReview();
}

/* ------------------------------------------------------------------ */
/* Step 5 - results                                                    */
/* ------------------------------------------------------------------ */

function renderResults(job) {
  enableStep("step-results", true);
  const result = job.result || {};
  const summary = job.summary || {};

  renderHeadline(job, summary);
  renderNotes(result.warnings || [], summary.reasons || []);
  renderTrace(result.trace);
  renderEvents(job);
  setupZahnReview(summary);

  el("diagnostics-json").textContent = JSON.stringify(
    { plan: job.plan, summary, diagnostics: result.diagnostics },
    null,
    2
  );
  el("diagnostics-holder").classList.remove("hidden");
  el("step-results").scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderHeadline(job, summary) {
  const headline = el("headline");
  headline.classList.remove("hidden", "ok", "review", "bad");

  if (summary.mode === "zahn_cup") {
    const status = summary.status;
    headline.classList.add(status === "confirmed" ? "ok" : status === "review" ? "review" : "bad");
    const value = summary.efflux_seconds !== null && summary.efflux_seconds !== undefined
      ? `${summary.efflux_seconds.toFixed(2)} s`
      : "No reliable measurement";
    headline.innerHTML = "";
    appendHeadline(headline, "Zahn cup efflux time", value, [
      summary.flow_start_s !== null && summary.flow_start_s !== undefined
        ? `Flow start ${summary.flow_start_s.toFixed(3)} s, flow end ${summary.flow_end_s.toFixed(3)} s`
        : "Flow start was never detected",
      `${summary.frames_analysed} frames analysed at ${summary.fps} fps`,
      `Confidence ${Math.round((summary.confidence || 0) * 100)}% (${statusText(status)})`,
      summary.end_confirmed ? "End of flow confirmed by sustained absence of liquid"
                            : "End of flow NOT confirmed",
    ]);
    return;
  }

  const counts = job.counts || { total: 0, review: 0 };
  headline.classList.add(counts.total ? "ok" : "review");
  headline.innerHTML = "";
  appendHeadline(headline, "Detected events", String(counts.total), [
    `${summary.candidate_windows || 0} candidate window(s) from the fast scan, `
      + `${summary.refined_events || 0} confirmed after detailed analysis`,
    `Scanned ${summary.samples_analysed || 0} samples every ${summary.sample_interval_s || "?"} s`
      + (summary.scan_speed_x_realtime ? ` (${summary.scan_speed_x_realtime}x faster than real time)` : ""),
    `Processing took ${summary.processing_seconds || 0} s`,
    counts.review ? `${counts.review} event(s) need review` : "",
  ]);
}

function appendHeadline(container, label, value, lines) {
  const labelEl = document.createElement("p");
  labelEl.className = "headline-label";
  labelEl.textContent = label;
  const valueEl = document.createElement("p");
  valueEl.className = "headline-value";
  valueEl.textContent = value;
  container.append(labelEl, valueEl);
  lines.filter(Boolean).forEach((line) => {
    const p = document.createElement("p");
    p.className = "headline-meta";
    p.textContent = line;
    container.append(p);
  });
}

function statusText(status) {
  if (status === "confirmed") return "automatic detection succeeded";
  if (status === "review") return "review recommended";
  return "detection failed";
}

function renderNotes(warnings, reasons) {
  const list = el("result-notes");
  list.innerHTML = "";
  const items = [...warnings, ...reasons];
  if (!items.length) {
    list.classList.add("hidden");
    return;
  }
  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = text;
    list.append(li);
  });
  list.classList.remove("hidden");
}

function renderTrace(trace) {
  if (!trace || !trace.times || trace.times.length < 2) {
    el("trace-holder").classList.add("hidden");
    return;
  }
  el("trace-holder").classList.remove("hidden");
  const canvas = el("trace-canvas");
  const width = canvas.clientWidth || 800;
  canvas.width = width;
  const height = canvas.height;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, width, height);

  const maxScore = Math.max(...trace.scores, 1e-6);
  const maxTime = trace.times[trace.times.length - 1] || 1;

  context.strokeStyle = "#1f5fa8";
  context.lineWidth = 1;
  context.beginPath();
  trace.times.forEach((time, index) => {
    const x = (time / maxTime) * width;
    const y = height - (trace.scores[index] / maxScore) * (height - 6) - 3;
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.stroke();
}

function renderEvents(job) {
  const log = job.event_log;
  const holder = el("events-holder");
  if (!log || !log.rows.length) {
    holder.classList.add("hidden");
    return;
  }
  holder.classList.remove("hidden");
  el("download-csv").href = `/api/analyses/${job.job_id}/events.csv`;
  el("clock-note").textContent = log.has_wall_clock
    ? "Clock times are derived from the recording start you supplied."
    : "No recording start was supplied, so only video-relative times are reported.";

  const body = el("events-table").querySelector("tbody");
  body.innerHTML = "";
  const events = (job.result && job.result.events) || [];

  log.rows.forEach((row, index) => {
    const tr = document.createElement("tr");
    const statusClass = {
      detected: "status-confirmed",
      "review recommended": "status-review",
      "detection failed": "status-failed",
    }[row.status] || "";

    [
      row.event,
      row.start_video_time,
      row.end_video_time,
      `${row.duration_s} s`,
      row.start_wall_clock || "—",
      `${row.confidence_pct}%`,
      row.status,
    ].forEach((text, column) => {
      const td = document.createElement("td");
      td.textContent = text;
      if (column === 6 && statusClass) td.className = statusClass;
      tr.append(td);
    });

    const startSeconds = events[index] ? events[index].start_s : null;
    if (startSeconds !== null) {
      tr.addEventListener("click", () => {
        const preview = el("preview");
        preview.currentTime = Math.max(0, startSeconds - 1);
        preview.play().catch(() => { /* autoplay may be blocked; seeking is enough */ });
        preview.scrollIntoView({ behavior: "smooth", block: "center" });
      });
    }
    body.append(tr);
  });
}

/* ------------------------------------------------------------------ */
/* Zahn review panel - synchronized grayscale ROI and boundary controls */
/*                                                                      */
/* The panel reads frames directly from the existing <video>, draws the */
/* analysis ROI (in source-pixel coordinates, as returned by the        */
/* analysis result) onto a canvas at an enlarged size, and desaturates   */
/* it with a CSS filter. No second video is created, uploaded, or       */
/* stored; nothing here issues a network request.                       */
/* ------------------------------------------------------------------ */

function initZahnReview() {
  el("review-back").addEventListener("click", () => reviewStep(-REVIEW_SEEK_STEP_S));
  el("review-forward").addEventListener("click", () => reviewStep(REVIEW_SEEK_STEP_S));
  el("review-jump-start").addEventListener("click", () => reviewJumpToBoundary(state.review.flowStartS));
  el("review-jump-end").addEventListener("click", () => reviewJumpToBoundary(state.review.flowEndS));
  el("review-timeline-track").addEventListener("click", (event) => reviewSeekFromTimelineClick(event));

  const preview = el("preview");
  preview.addEventListener("loadedmetadata", onReviewSourceReady);
  preview.addEventListener("play", startReviewSyncLoop);
  preview.addEventListener("pause", () => {
    stopReviewSyncLoop();
    drawReviewFrame();
    updateReviewReadouts();
  });
  preview.addEventListener("seeked", () => { drawReviewFrame(); updateReviewReadouts(); });
  preview.addEventListener("timeupdate", updateReviewReadouts);
}

function reviewStep(deltaSeconds) {
  const preview = el("preview");
  preview.currentTime = stepSeekTime(preview.currentTime, deltaSeconds, preview.duration);
}

function reviewJumpToBoundary(boundarySeconds) {
  const preview = el("preview");
  const target = boundaryJumpTime(boundarySeconds, preview.duration);
  if (target === null) return;
  preview.pause();
  preview.currentTime = target;
}

function reviewSeekFromTimelineClick(event) {
  const preview = el("preview");
  if (!Number.isFinite(preview.duration) || preview.duration <= 0) return;
  const rect = el("review-timeline-track").getBoundingClientRect();
  const fraction = rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0;
  preview.currentTime = clampSeekTime(fraction * preview.duration, preview.duration);
}

/** Called after a new analysis result arrives; decides whether to show the panel. */
function setupZahnReview(summary) {
  if (!shouldShowZahnReview(summary)) {
    teardownZahnReview();
    return;
  }

  const preview = el("preview");
  state.review.active = true;
  state.review.roi = summary.roi || null;
  state.review.flowStartS = typeof summary.flow_start_s === "number" ? summary.flow_start_s : null;
  state.review.flowEndS = typeof summary.flow_end_s === "number" ? summary.flow_end_s : null;

  el("zahn-review").classList.remove("hidden");
  el("review-jump-start").disabled = state.review.flowStartS === null;
  el("review-jump-end").disabled = state.review.flowEndS === null;
  el("review-start-value").textContent =
    state.review.flowStartS === null ? "not detected" : formatTimeTenths(state.review.flowStartS);
  el("review-end-value").textContent =
    state.review.flowEndS === null ? "not detected" : formatTimeTenths(state.review.flowEndS);

  if (preview.readyState >= 1) onReviewSourceReady();
  updateReviewBoundaryMarkers();

  // The preview may already be playing when a result activates the panel
  // (analysis can take a while, and nothing pauses the preview during it).
  // In that case no future `play` event will arrive to start the sync loop,
  // so start it explicitly rather than leaving the panel showing one frozen
  // frame until the user happens to pause and play again.
  if (!preview.paused) startReviewSyncLoop();
}

function teardownZahnReview() {
  state.review.active = false;
  state.review.roi = null;
  state.review.flowStartS = null;
  state.review.flowEndS = null;
  stopReviewSyncLoop();
  el("zahn-review").classList.add("hidden");
  el("review-unavailable").classList.add("hidden");
  el("review-canvas").classList.remove("hidden");
}

/** Sizes the canvas once the video's native dimensions and the ROI are both known. */
function onReviewSourceReady() {
  if (!state.review.active) return;
  const preview = el("preview");
  const roi = state.review.roi;
  const valid = isValidRoi(roi, preview.videoWidth, preview.videoHeight);
  const canvas = el("review-canvas");
  const unavailable = el("review-unavailable");

  if (!valid) {
    canvas.classList.add("hidden");
    unavailable.textContent =
      "The marked region for this result can't be shown here - it no longer matches "
      + "this video's dimensions.";
    unavailable.classList.remove("hidden");
    return;
  }

  const size = computeReviewCanvasSize(roi.width, roi.height, REVIEW_CANVAS_TARGET_PX);
  canvas.width = size.width;
  canvas.height = size.height;
  canvas.classList.remove("hidden");
  unavailable.classList.add("hidden");

  drawReviewFrame();
  updateReviewReadouts();
  updateReviewBoundaryMarkers();
}

/** Prefers frame-accurate sync via requestVideoFrameCallback; falls back to rAF + events. */
function startReviewSyncLoop() {
  if (!state.review.active) return;
  // Single-owner: whatever chain might already be running (from an earlier
  // `play`, possibly still in flight when this one fires - a `pause` and a
  // fast `play` can otherwise race, since a callback already queued by a
  // prior chain does not un-queue itself) is stopped before starting a new
  // one, so at most one redraw loop is ever active.
  stopReviewSyncLoop();
  const preview = el("preview");

  if (typeof preview.requestVideoFrameCallback === "function") {
    const onFrame = () => {
      drawReviewFrame();
      updateReviewReadouts();
      if (!preview.paused && !preview.ended) {
        state.review.frameCallbackId = preview.requestVideoFrameCallback(onFrame);
      }
    };
    state.review.frameCallbackId = preview.requestVideoFrameCallback(onFrame);
    return;
  }

  // Fallback for browsers without requestVideoFrameCallback: redraw every animation
  // frame while playing. `timeupdate`/`seeked` (wired in initZahnReview) keep the
  // readouts and drawing correct while paused or scrubbing.
  const loop = () => {
    drawReviewFrame();
    updateReviewReadouts();
    if (!preview.paused && !preview.ended) {
      state.review.rafId = requestAnimationFrame(loop);
    }
  };
  state.review.rafId = requestAnimationFrame(loop);
}

function stopReviewSyncLoop() {
  const preview = el("preview");
  if (state.review.frameCallbackId !== null && typeof preview.cancelVideoFrameCallback === "function") {
    preview.cancelVideoFrameCallback(state.review.frameCallbackId);
  }
  state.review.frameCallbackId = null;
  if (state.review.rafId !== null) {
    cancelAnimationFrame(state.review.rafId);
    state.review.rafId = null;
  }
}

function drawReviewFrame() {
  if (!state.review.active) return;
  const canvas = el("review-canvas");
  if (canvas.classList.contains("hidden")) return; // ROI invalid; nothing to draw
  const preview = el("preview");
  const roi = state.review.roi;

  try {
    const context = canvas.getContext("2d");
    context.drawImage(
      preview,
      roi.x, roi.y, roi.width, roi.height,
      0, 0, canvas.width, canvas.height
    );
  } catch (err) {
    // A frame that cannot be drawn yet (e.g. before the first frame decodes)
    // is not an error worth surfacing; the next callback tries again.
  }
}

function updateReviewReadouts() {
  if (!state.review.active) return;
  const preview = el("preview");
  el("review-time-value").textContent = formatTimeTenths(preview.currentTime);
  const playPercent = timelinePercent(preview.currentTime, preview.duration);
  el("review-playhead").style.left = playPercent === null ? "0%" : `${playPercent}%`;
}

function updateReviewBoundaryMarkers() {
  const preview = el("preview");
  const duration = Number.isFinite(preview.duration) ? preview.duration : null;

  [
    ["review-marker-start", state.review.flowStartS],
    ["review-marker-end", state.review.flowEndS],
  ].forEach(([id, seconds]) => {
    const marker = el(id);
    const percent = timelinePercent(seconds, duration);
    if (percent === null) {
      marker.classList.add("hidden");
    } else {
      marker.classList.remove("hidden");
      marker.style.left = `${percent}%`;
    }
  });
}

/* ------------------------------------------------------------------ */

function init() {
  initUpload();
  initModes();
  initFramePicker();
  initAnalysis();
  initZahnReview();
}

// Guarded so this file can also be `require()`-d from plain Node (see
// tests_js/) to unit-test the pure functions above without a DOM.
if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", init);
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    formatTimeTenths,
    clampSeekTime,
    stepSeekTime,
    boundaryJumpTime,
    isValidRoi,
    computeReviewCanvasSize,
    timelinePercent,
    shouldShowZahnReview,
    collectParams,
    resetOutletAnchorsState,
  };
}
