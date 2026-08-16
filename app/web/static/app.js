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

const state = {
  video: null,        // metadata of the uploaded video
  mode: null,         // selected analysis mode
  outlet: null,       // {x, y} in source pixels (Zahn)
  roi: null,          // {x, y, width, height} in source pixels
  frameTime: 0,       // timestamp of the frame shown in the picker
  jobId: null,
  pollTimer: null,
  dragStart: null,
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

function applyModeToSettings() {
  const zahn = isZahn();
  el("settings-zahn").classList.toggle("hidden", !zahn);
  el("settings-activity").classList.toggle("hidden", zahn);
  el("idle-pause-field").classList.toggle("hidden", state.mode !== "robot_activity");
  el("advanced-preroll-field").classList.toggle("hidden", zahn);
  el("advanced-start-persistence-field").classList.toggle("hidden", !zahn);

  el("picker-instruction").textContent = zahn
    ? "Play the preview until the cup and the outlet hole are clearly visible, press "
      + "“Use the frame currently shown”, then click exactly on the outlet hole. "
      + "You can also drag a box around the area the liquid falls through."
    : "Optional: drag a box around the machine or the area you care about. "
      + "Restricting the region makes the scan faster and ignores activity elsewhere.";

  el("frame-picker").classList.remove("hidden");
  state.outlet = null;
  state.roi = null;
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
    state.outlet = null;
    state.roi = null;
    drawRegion();
    updateRegionSummary();
    updateRunButton();
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

  if (isClick && isZahn()) {
    state.outlet = { x: current.x, y: current.y };
    state.roi = null;
  } else if (!isClick) {
    state.roi = rect;
    if (isZahn()) state.outlet = null;
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
  if (state.outlet) {
    const radius = Math.max(6, Math.round(canvas.width / 90));
    context.strokeStyle = "#ffaa00";
    context.lineWidth = lineWidth;
    context.beginPath();
    context.arc(state.outlet.x, state.outlet.y, radius, 0, Math.PI * 2);
    context.stroke();
    context.beginPath();
    context.moveTo(state.outlet.x, state.outlet.y);
    context.lineTo(state.outlet.x, canvas.height);
    context.setLineDash([10, 8]);
    context.stroke();
    context.setLineDash([]);
  }
}

function updateRegionSummary() {
  const summary = el("region-summary");
  if (state.outlet) {
    summary.textContent =
      `Outlet marked at x=${state.outlet.x}, y=${state.outlet.y}. `
      + "The analysis region extends downward from this point.";
  } else if (state.roi) {
    summary.textContent =
      `Region: ${state.roi.width} x ${state.roi.height} pixels at `
      + `(${state.roi.x}, ${state.roi.y}).`;
  } else {
    summary.textContent = isZahn()
      ? "No outlet marked yet - this is required for Zahn cup analysis."
      : "No region selected - the whole frame will be analysed.";
  }
}

/* ------------------------------------------------------------------ */
/* Step 4 - run the analysis                                           */
/* ------------------------------------------------------------------ */

function updateRunButton() {
  const ready = Boolean(state.video && state.mode && (!isZahn() || state.outlet || state.roi));
  el("run-analysis").disabled = !ready;
}

function collectParams() {
  const params = {};
  if (isZahn()) {
    if (state.roi) params.roi = state.roi;
    else params.outlet = state.outlet;
    params.flow_end_persistence_s = Number(el("zahn-end-persistence").value);
    params.flow_start_persistence_s = Number(el("zahn-start-persistence").value);
    params.activity_threshold = Number(el("zahn-sensitivity").value);
  } else {
    if (state.roi) params.roi = state.roi;
    params.shortest_event_s = Number(el("shortest-event").value);
    params.min_event_duration_s = Number(el("min-event").value);
    params.sensitivity = Number(el("scan-sensitivity").value);
    const roll = Number(el("roll-seconds").value);
    params.pre_roll_s = roll;
    params.post_roll_s = roll;
    if (state.mode === "robot_activity") {
      params.idle_pause_s = Number(el("idle-pause").value);
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

function init() {
  initUpload();
  initModes();
  initFramePicker();
  initAnalysis();
}

document.addEventListener("DOMContentLoaded", init);
