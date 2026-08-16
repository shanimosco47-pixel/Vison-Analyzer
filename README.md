# Video Analyzer

Local analysis of recorded video, for two jobs:

1. **Zahn cup viscosity** — measure the efflux time of a Zahn cup (including #4)
   from a side-on recording.
2. **Robot / machine activity** — find the handful of interesting moments in a
   six- or twelve-hour factory recording without analysing every frame.

Everything runs on your machine. No video, frame, thumbnail, metadata or result
is ever sent anywhere: the application makes no outbound network requests, the
page loads no external scripts or fonts, and there is no telemetry.

---

## 1. Installation

Requires **Python 3.10 or newer**. No Docker, no database, no GPU.

```bash
git clone <this repository>
cd Vison-Analyzer

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

That is the whole installation. To run the tests as well:

```bash
pip install -r requirements-dev.txt
```

## 2. Running it

```bash
python -m app.main
```

Then open <http://127.0.0.1:8000/> in a browser.

Useful flags (`python -m app.main --help` lists them all):

| Flag | Meaning |
| --- | --- |
| `--port 8080` | Listen on a different port |
| `--data-dir /data/analyzer` | Where uploads, logs and diagnostics are stored |
| `--max-upload-mb 16000` | Raise the upload limit for whole-shift recordings |
| `--save-diagnostics` | Write the frames at each event boundary, with the region drawn on them |
| `--log-level DEBUG` | Verbose logging while tuning |

The same settings can come from environment variables — see
[`config.example.env`](config.example.env).

The server binds to `127.0.0.1`, so it is reachable only from this machine.
That is deliberate: the prototype has **no authentication**, and plant footage
should not be served to a network by accident.

## 3. Using it

The page is five steps, in order.

**1 — Upload.** Drag a video in, or click to choose one. Filename, duration,
resolution, frame rate, size and codec are shown, and the video appears in a
preview player. If the frame rate had to be estimated (some exports do not
declare one), the page says so, because that makes every timing less certain.

**2 — Choose what to analyse.** Zahn cup viscosity, robot/machine activity, or
a generic motion scan.

**3 — Configure.** Only the settings relevant to the chosen mode are shown;
everything else is under *Advanced settings*.

*For Zahn cup:* play the preview until the cup and its outlet are clearly
visible, press **Use the frame currently shown**, then **click on the outlet
hole**. The analysis region is built extending downward from that point — the
only part of the picture that is decoded and analysed. You can also drag a box
if you prefer to define the region yourself.

*For robot/machine:* optionally drag a box around the machine, and say what the
shortest event that must not be missed is. That single number determines how
fast the scan runs (see §5).

**4 — Analyse.** Progress is reported continuously: *scanning video (2:14:30 of
12:00:00)*, *4 candidate windows found*, *refining candidate 3 of 4*.

**5 — Results.** The headline result first (the efflux time, or the number of
events), then the evidence: confidence, an activity-over-time plot, and the
event table. Clicking a row seeks the preview player to that moment. **Export
event log (CSV)** downloads the table.

## 4. The detection architecture

```
VideoReader ─ streaming, seekable, one frame in memory at a time
     │  FrameSample(index, timestamp_s, image, scale)
     ▼
ActivityScorer ─ turns one frame into one number, plus evidence
     ├── MotionActivityScorer   background model + adaptive noise threshold
     └── StreamActivityScorer   Zahn: shape analysis of the liquid in the ROI
     ▼
Temporal logic ─ time-based persistence and hysteresis turn numbers into spans
     ├── FlowStateMachine       Zahn start/end persistence
     └── find_active_intervals  hysteresis, gap bridging, minimum duration
     ▼
EventLog ─ video-relative and (optionally) wall-clock timestamps, CSV export
```

Scoring and *timing* are kept apart on purpose. The scorer answers "what is in
this frame?"; the temporal layer answers "was that an event?". A future
detector — a dip detector, a cycle counter, even a neural scorer — replaces the
first without touching the second.

### Coarse-to-fine, for long recordings

A twelve-hour recording at 25 FPS is 1.1 million frames. Analysing them all is
neither necessary nor affordable, so long-video modes work in stages:

**Stage A — coarse scan** ([`coarse_scan.py`](app/analysis/coarse_scan.py)).
Sample the whole recording cheaply and mark *candidate windows*. Two separate
optimisations are applied, and the code keeps them separate because they cost
different things:

* *Frame skipping* — decode one frame every `interval_s`. The big win, but it
  costs temporal resolution, so the interval is **derived from the shortest
  event that must not be missed**, never hard-coded:
  `interval = shortest_event_s / safety_factor` (default factor 3, minimum 2).
  Sampling once per event length would let an event fall between two samples;
  that is exactly the mistake the safety factor exists to prevent.
* *Resolution reduction* — analyse a downscaled frame (default 320 px wide).
  This keeps full time coverage and makes each sample ~20× cheaper at 1080p.

Candidate boundaries from this stage are deliberately **conservative** (pushed
out to the neighbouring sample), so the true edge is guaranteed to be inside
the window Stage B looks at.

**Stage B — refinement**
([`candidate_refinement.py`](app/analysis/candidate_refinement.py)). For each
candidate, seek directly to `[start − pre_roll, end + post_roll]` and re-analyse
that short span densely (every frame, at higher resolution) to place the
boundaries accurately. The rest of the recording is never decoded again, and
the roll-back is never shorter than one coarse sampling interval.

**Stage C — semantic analysis.** Only inside candidate windows, and only when
a mode needs it. Today every mode is deterministic computer vision; the seam
for object detection exists ([`optional_yolo.py`](app/models/optional_yolo.py))
and is unused.

### Zahn cup mode

1. The user marks the outlet; the analysis region extends downward from it. A
   surrounding *guard* band is decoded in the same crop and watched only for
   disturbances.
2. Each frame is compared against a background model of the empty region. The
   threshold is `max(min_abs_diff, 4 × measured noise σ)`, so a grainy or
   heavily compressed camera raises its own bar automatically.
3. Connected components are classified **by shape**: the stream is tall and
   thin, a falling drop is small and compact. A hand, a cloth or a sweeping
   shadow is wide, and does not qualify.
4. **Flow start** requires liquid *at the outlet band* continuously for
   `flow_start_persistence_s`. The reported start is the first frame of that
   run, not the moment persistence was satisfied.
5. **Flow end** requires a sustained *absence* of liquid — stream **and** drops
   — for `flow_end_persistence_s`. The reported end is the last frame with
   activity. A Zahn run ends stream → weakening stream → intermittent drops, so
   a single broken frame must never stop the clock, and a lone drop counts as
   liquid even though it covers only a few percent of the region.
6. A confidence is computed from measured evidence and the result is reported as
   **detected**, **review recommended** or **detection failed**. A failed
   detection returns no time at all.

Analysis stops as soon as the end of flow is confirmed — there is nothing to
learn from the rest of the recording.

## 5. Settings that matter, and how to tune them

Every threshold lives in [`app/config.py`](app/config.py) with a comment
explaining it. All time values are in **seconds** and converted to frames from
the video's real frame rate; there is no rule anywhere of the form "15 frames",
because that would mean different things at 15 FPS and 60 FPS.

### Zahn cup (`ZahnConfig`)

| Setting | Default | What it does / when to change it |
| --- | --- | --- |
| `flow_end_persistence_s` | 0.50 s | How long no liquid must be seen before timing stops. **Tune this first.** Raise it if timing stops between the final drops; lower it if the reported end drags past the last drop. |
| `flow_start_persistence_s` | 0.30 s | How long liquid must be visible at the outlet before timing starts. Raise it if reflections or a hand start the clock early. |
| `activity_threshold` | 0.12 | Fraction of the region's height the stream must cover to count as flowing. Lower for a thin or faint stream. |
| `min_abs_diff` | 12 | Grey-level floor. Raise it for a noisy camera if background texture is being read as liquid. |
| `min_drop_area_px` | 8 | Smallest blob that still counts as a drop. Raise it if compression artefacts extend the measurement past the real end. |
| `outlet_band_fraction` | 0.22 | Height of the band under the orifice used for start detection. |
| `disturbance_area_ratio` | 0.35 | Fraction of the guard band that must change for a frame to be discarded as disturbed. |
| `review_confidence` / `fail_confidence` | 0.70 / 0.40 | Where "detected" becomes "review recommended" becomes "failed". |

### Long recordings (`CoarseScanConfig`, `RefinementConfig`)

| Setting | Default | What it does / when to change it |
| --- | --- | --- |
| `shortest_event_s` | 15 s | The shortest event that must not be missed. Drives the sampling interval and therefore the scan time. |
| `safety_factor` | 3.0 | Samples that must fall inside the shortest event. Below 2 is rejected. |
| `sensitivity` | 0.5 | One slider from "fewer, stronger detections" to "catches weaker activity". Scales the adaptive thresholds. |
| `scan_width_px` | 320 | Scan resolution. Raise it if the thing you care about is small in frame. |
| `min_candidate_duration_s` | 1.0 s | Candidates shorter than this never reach the expensive stage. |
| `pre_roll_s` / `post_roll_s` | 5 s | How far around each candidate the dense pass looks (never less than one coarse interval). |
| `min_event_duration_s` | 1.0 s | Confirmed events must last at least this long. |
| `idle_pause_s` | 300 s | Robot mode: quiet spells longer than this, *between* two working periods, are reported as an extended pause. 0 turns it off. |

### Confidence

Confidence is computed from things that were actually measured, never asserted.
For Zahn: how far the stream exceeded the detection threshold at the start, how
continuous the stream was over the timed interval, its contrast against the
region's own noise, whether the end was confirmed, and what fraction of frames
were disturbed. An unconfirmed end caps confidence at 0.50; nothing ever
reaches 1.0.

## 6. Testing

```bash
pytest                    # 237 tests, about 55 s
ruff check app tests      # lint
ruff format --check app tests
mypy app                  # static types
```

The suite has four layers:

* **Unit tests** for time↔frame conversion, sampling plans, persistence timers,
  hysteresis, ROI validation, configuration validation, timestamp formatting
  and CSV export.
* **Synthetic sequence tests** that drive the state machines with scripted
  observations — no motion, a single flicker, continuous stream, intermittent
  final drops, stream then sustained absence, disturbance during flow — and
  assert the exact timestamps produced. These also pin the frame-rate
  independence of every persistence rule (10, 25 and 50 FPS must agree).
* **Integration tests** on real video files generated with known ground truth
  ([`tests/conftest.py`](tests/conftest.py)): a Zahn run whose efflux time is
  known to the frame, a scene with two motion intervals, and a recording in
  which nothing happens.
* **Web tests** through Flask's test client: upload, metadata, frame preview,
  range requests, job progress, cancellation, CSV export, path-traversal and
  oversize rejection, repeated analysis and browser refresh.

When a real Zahn or plant recording exposes a bug, add that case as a
regression test rather than only fixing the code.

## 7. Measured results

Run on this repository's synthetic footage and a generated 720p recording,
on an ordinary CPU with no GPU:

| Check | Result |
| --- | --- |
| Zahn cup, ground truth 17.50 s | measured **17.28 s** (−0.22 s), end confirmed, confidence 98 % |
| Zahn cup, region pointed at empty background | **detection failed**, no time reported |
| Two motion intervals (5–9 s, 18–23 s) | both found, boundaries within 0.6 s |
| Recording with nothing happening | 0 events, with an explicit "no activity found" message |
| Coarse scan, 10 min of 1280×720 @ 25 FPS | **1.8 s** — 325× faster than real time, 120 of 15 000 frames decoded (0.8 %) |
| Extrapolated to a 12-hour recording | ≈ 2.2 minutes of scanning |
| Peak memory during that scan | 135 MB, flat (132 MB before the scan) |
| Full browser flow in Chromium | upload → mark outlet → analyse → results → CSV → refresh → re-run, all clean |

Memory is bounded by construction: one frame is held at a time, and the coarse
pass stores one float per *sample* (a 12-hour scan at one sample per 5 s is
~8 600 floats), not per frame.

## 8. Known limitations

* **The Zahn thresholds have only been validated against synthetic footage.**
  The defaults are derived from the physics of the measurement, not from real
  cup videos. Expect to tune `flow_end_persistence_s`, `activity_threshold` and
  `min_drop_area_px` against your own recordings — that is what
  `--save-diagnostics` is for.
* **A static camera is assumed.** Camera movement is *detected* and those frames
  are discarded, but the software does not stabilise the image. A hand-held
  recording will produce many disturbed frames and low confidence.
* **The generic motion scan can miss thin objects.** A stream a couple of pixels
  wide is below the absolute area floor that keeps sensor noise from firing.
  Use a region of interest, or raise the sensitivity, when the thing of interest
  is small in frame.
* **Robot mode reports activity and pauses, not semantics.** It does not yet
  know what a "dip" or a "cycle" is; those need plant-specific geometry or
  object detection (§9).
* **Seek performance depends on the codec.** Stage B seeks directly into the
  file. On long-GOP H.264/H.265 exports each seek costs more than on the
  fixtures used here; verify Stage B timing on your own footage.
* **Variable-frame-rate recordings** are handled by estimating the frame rate
  from timestamps, and the UI says when this happened, but timings on such
  files are inherently less exact.
* **Job state lives in memory.** Restarting the server clears uploads and
  results. That is intentional for a local prototype; a database would be the
  first thing to add if results must survive a restart.
* **No authentication**, by design, for a loopback-only prototype. Do not expose
  it on a network without adding one.

## 9. Recommended next steps

In the order the measured results suggest:

1. **Record 5–10 real Zahn runs** — different liquids, lighting and
   backgrounds — and tune the thresholds against them with
   `--save-diagnostics`. Add each awkward case as a regression test. This is
   worth more than any algorithmic change.
2. **Validate Stage B on a real long recording.** The coarse pass is measured
   and fast; the seek cost of refinement on long-GOP footage is the one number
   this repository cannot produce without real files.
3. **A repeat-Zahn mode**: several cup runs in one recording, timed
   automatically. The coarse-to-fine machinery already supports it; it needs a
   detector that does not stop after the first run.
4. **Plant-specific detectors** (dip start/end, cycle counting) as
   `CoarseToFineDetector` subclasses, driven by ROI geometry — a dip is a
   crossing of a region boundary, which is deterministic and testable.
5. **Only then evaluate object detection.** It becomes worthwhile when the
   question is *what* moved (operator vs. robot), not *whether* something moved.
   Run it inside candidate windows only, and settle the licence question first
   (§10).

## 10. Dependencies and licensing

| Package | Licence | Why |
| --- | --- | --- |
| opencv-python-headless | Apache-2.0 | Video decoding, all vision work |
| numpy | BSD-3-Clause | Array maths |
| flask | BSD-3-Clause | Local web server |
| pandas | BSD-3-Clause | Optional DataFrame export of the event log |

All four are permissively licensed and safe for internal commercial use.

**Ultralytics YOLO is deliberately *not* a dependency.** It is published under
AGPL-3.0 with a separate commercial licence; AGPL obligations extend to
network-served applications, so it needs a licence review before it can be
required for internal commercial use. The adapter in
[`app/models/optional_yolo.py`](app/models/optional_yolo.py) imports it lazily
and loads weights from a local path only, so the application starts and runs
without it, and a differently licensed detector can replace it behind the same
small interface.

## 11. Security and privacy

* Uploaded files are treated as untrusted input. The browser's filename is never
  used as a path — files are stored as `<uuid4>.<extension>` in one directory —
  so path traversal is structurally impossible.
* The declared MIME type is ignored; the real check is whether the file decodes.
* The size limit is enforced while streaming to disk, and partial uploads are
  always removed, including when the connection drops.
* The page carries a strict Content-Security-Policy (`default-src 'self'`), so
  it cannot load or contact anything external.
* No `eval`, no shell execution, no filename or file content is ever executed.
* Errors shown in the browser are short sentences; tracebacks go to the server
  log only.
* Uploads are deleted after 24 hours of inactivity
  (`VISION_ANALYZER_RETENTION_HOURS`), enforced by a background sweep every
  5 minutes for as long as the server is up — including while nobody is using
  it, which is the case that would otherwise fill a disk. Files left behind by
  a previous run are cleared at start-up, and finished analysis records are
  dropped after 6 hours.

## 12. Project layout

```
app/
  main.py                     entry point and command-line flags
  config.py                   every threshold, documented, in seconds
  errors.py                   typed errors with user-facing messages
  video/
    metadata.py               probing; treats container claims as a hypothesis
    reader.py                 streaming reads, grab-vs-seek, ROI crop, downscale
    sampling.py               the only place seconds become frame indices
  analysis/
    base_detector.py          Detector / Scorer interfaces, Event, confidence
    temporal.py               hysteresis, persistence timers, interval maths
    motion_detector.py        background-model change scoring
    coarse_scan.py            Stage A
    candidate_refinement.py   Stage B
    coarse_to_fine.py         the reusable Stage A+B detector
    zahn_detector.py          Zahn cup scorer, flow state machine, detector
    robot_activity_detector.py  first plant-oriented detector
    registry.py               the one place that lists the modes
  models/optional_yolo.py     unused optional object-detection seam
  services/
    storage.py                safe upload handling
    analysis_service.py       background jobs, progress, cancellation
    event_log.py              timestamps, wall-clock mapping, CSV
    diagnostics.py            overlays and debug frames
  web/                        Flask routes, one HTML page, plain CSS/JS
scripts/benchmark_scan.py     throughput and memory measurement
tests/                        unit, synthetic, integration and web tests
```

### Adding a detector

Subclass `CoarseToFineDetector`, override `make_scorer()` and `label_for()`,
optionally `derive_additional_events()`, and add one line to
[`registry.py`](app/analysis/registry.py). The reader, scanner, refinement,
confidence, progress reporting, event log, CSV export and UI all work
unchanged — [`robot_activity_detector.py`](app/analysis/robot_activity_detector.py)
is a 150-line example.
