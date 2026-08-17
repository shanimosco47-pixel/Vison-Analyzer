# Frame-timing fixtures

Three small H.264/MP4 files used by `tests/test_variable_frame_rate.py`. They
exist because OpenCV's writer emits **constant-rate** containers only, so the
`tests/conftest.py` generators cannot produce a recording whose presentation
timestamps vary — and three successive versions of the frame-timing detector
shipped with the same defect intact precisely because it had only ever been
tested against scripted timestamp lists.

The content is synthetic: a white bar moving across a green gradient. **No
private, plant, or customer footage is involved.** Only the *timing* is
borrowed from a real recording measured during review — a `600/19` fps base
tick (31.667 ms) with 34 of its 638 intervals doubled, which a container
reports as a plausible 29.98 fps average.

| File | Frames | Timing | Expected |
|---|---|---|---|
| `vfr_distributed.mp4` | 639 | 34 doubled intervals spread evenly | **Accepted** — sampled drift spread ~30 ms, inside the 50 ms budget |
| `vfr_clustered_burst.mp4` | 639 | the same 34 doubled intervals in one burst at frames 60–126 | **Refused** (`timing_drift`) — the burst falls entirely between sampling windows, so every window looks perfect, yet the file carries ~964 ms of error on a measured duration |
| `cfr_late_start.mp4` | 639 | perfectly constant, container starts at 5 s | **Accepted** — a non-zero start must not read as drift |

`vfr_distributed.mp4` being *accepted* is deliberate: the policy bounds the
timing error rather than demanding a strictly constant rate, because genuine
constant-rate exports routinely drop a few frames. Its worst reported-timestamp
error is under one frame period.

Regenerate with `scripts/make_vfr_fixtures.py`, which needs PyAV in a throwaway
environment. **Running the tests does not** — nothing under `app/` or `tests/`
imports PyAV, and these files are read with the project's own OpenCV.
