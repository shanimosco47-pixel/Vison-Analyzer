"""OutletTracker state transitions, driven by synthetic frames.

No video decoding is involved - each test builds small numpy frames directly,
in the same spirit as ``test_zahn_state_machine.py``, so the tracked/
predicted/lost/reacquired contract is pinned down independently of the video
pipeline around it.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.analysis.outlet_tracker import OutletTracker, TrackerInitError, TrackState
from app.config import ZahnConfig

WIDTH, HEIGHT = 220, 220
PATCH = 60
BACKGROUND = 200


def _config(**overrides) -> ZahnConfig:
    cfg = ZahnConfig()
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _patch(seed: int, size: int = PATCH) -> np.ndarray:
    """A block of speckle: enough corner response for goodFeaturesToTrack."""
    rng = np.random.default_rng(seed)
    return rng.integers(40, 215, size=(size, size), dtype=np.uint8)


def _frame(cup_x: int, cup_y: int, patch: np.ndarray) -> np.ndarray:
    """The patch (the "cup") sits with its bottom edge at (cup_x, cup_y)."""
    img = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
    ph, pw = patch.shape
    y0, x0 = cup_y - ph, cup_x - pw // 2
    y0c, y1c = max(0, y0), min(HEIGHT, y0 + ph)
    x0c, x1c = max(0, x0), min(WIDTH, x0 + pw)
    if y1c > y0c and x1c > x0c:
        img[y0c:y1c, x0c:x1c] = patch[y0c - y0 : y1c - y0, x0c - x0 : x1c - x0]
    return img


def _blank() -> np.ndarray:
    return np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)


def _outlet(cup_x: int, cup_y: int) -> tuple[float, float]:
    return float(cup_x), float(cup_y)


class TestInitialisation:
    def test_no_texture_above_the_outlet_fails_to_start(self):
        with pytest.raises(TrackerInitError):
            OutletTracker(_config(), _blank(), _outlet(110, 110), patch_half_px=15)

    def test_textured_cup_initialises_tracked(self):
        frame = _frame(100, 90, _patch(seed=0))
        tracker = OutletTracker(_config(), frame, _outlet(100, 90), patch_half_px=15)
        assert tracker.state == TrackState.TRACKED


class TestTracking:
    def test_translation_is_tracked_with_low_error(self):
        patch = _patch(seed=1)
        tracker = OutletTracker(
            _config(), _frame(100, 90, patch), _outlet(100, 90), patch_half_px=15
        )
        cup_x = 100
        result = None
        for step in range(15):
            cup_x += 2  # a few pixels/frame, well inside the defaults
            result = tracker.update(_frame(cup_x, 90, patch), timestamp_s=(step + 1) / 30.0)
            assert result.state == TrackState.TRACKED
        assert abs(result.outlet_x - cup_x) < 3.0
        assert abs(result.outlet_y - 90) < 3.0

    def test_camera_and_cup_drift_together_is_still_tracked(self):
        """Both motions the field report described, superimposed."""
        patch = _patch(seed=2)
        cup_x, cup_y = 110, 95
        tracker = OutletTracker(
            _config(), _frame(cup_x, cup_y, patch), _outlet(cup_x, cup_y), patch_half_px=15
        )
        result = None
        for step in range(20):
            cup_x += 1 if step % 2 == 0 else 2
            cup_y += 1 if step % 3 == 0 else 0
            result = tracker.update(_frame(cup_x, cup_y, patch), timestamp_s=(step + 1) / 30.0)
            assert result.state == TrackState.TRACKED
        assert abs(result.outlet_x - cup_x) < 4.0
        assert abs(result.outlet_y - cup_y) < 4.0

    def test_offset_is_reported_relative_to_the_reference_frame(self):
        patch = _patch(seed=3)
        tracker = OutletTracker(
            _config(), _frame(100, 90, patch), _outlet(100, 90), patch_half_px=15
        )
        result = tracker.update(_frame(112, 90, patch), timestamp_s=1 / 30.0)
        assert result.offset_x == pytest.approx(12.0, abs=2.0)
        assert result.offset_y == pytest.approx(0.0, abs=2.0)

    def test_an_implausible_jump_is_not_trusted_outright(self):
        """A frame the fit nominally supports, but far outside the configured bound."""
        patch = _patch(seed=4)
        cfg = _config(track_max_frame_displacement_px=15.0, track_max_bridge_s=1.0)
        tracker = OutletTracker(cfg, _frame(100, 90, patch), _outlet(100, 90), patch_half_px=15)
        result = tracker.update(_frame(160, 90, patch), timestamp_s=1 / 30.0)
        assert result.state != TrackState.TRACKED


class TestGapsAndPrediction:
    def test_covered_cup_bridges_as_predicted_then_lost(self):
        """The cup vanishes from view; a short bridge, then lost."""
        patch = _patch(seed=5)
        cfg = _config(track_max_bridge_s=0.2)
        tracker = OutletTracker(cfg, _frame(100, 90, patch), _outlet(100, 90), patch_half_px=15)

        states = []
        for step in range(10):
            result = tracker.update(_blank(), timestamp_s=(step + 1) / 30.0)
            states.append(result.state)
        assert TrackState.PREDICTED in states
        assert states[-1] == TrackState.LOST

    def test_predicted_position_extrapolates_recent_motion(self):
        """A predicted frame is a bridge estimate: it must not freeze or reverse."""
        patch = _patch(seed=6)
        cfg = _config(track_max_bridge_s=1.0)
        cup_x = 100
        tracker = OutletTracker(cfg, _frame(cup_x, 90, patch), _outlet(cup_x, 90), patch_half_px=15)
        result = None
        for step in range(5):  # establish a steady rightward drift
            cup_x += 3
            result = tracker.update(_frame(cup_x, 90, patch), timestamp_s=(step + 1) / 30.0)
            assert result.state == TrackState.TRACKED
        last_tracked_x = result.outlet_x

        result = tracker.update(_blank(), timestamp_s=6 / 30.0)
        assert result.state == TrackState.PREDICTED
        assert result.outlet_x > last_tracked_x

    def test_predicted_frames_do_not_persist_as_tracked(self):
        """State is exactly what the caller needs to gate persistence on."""
        patch = _patch(seed=7)
        cfg = _config(track_max_bridge_s=0.5)
        tracker = OutletTracker(cfg, _frame(100, 90, patch), _outlet(100, 90), patch_half_px=15)
        result = tracker.update(_blank(), timestamp_s=1 / 30.0)
        assert result.state in (TrackState.PREDICTED, TrackState.LOST)
        assert result.state is not TrackState.TRACKED


class TestReacquisition:
    def test_reacquires_after_a_gap_at_the_true_new_position(self):
        patch = _patch(seed=8)
        cfg = _config(track_max_bridge_s=0.05)  # exhausts almost immediately
        cup_x, cup_y = 100, 90
        tracker = OutletTracker(
            cfg, _frame(cup_x, cup_y, patch), _outlet(cup_x, cup_y), patch_half_px=15
        )

        result = None
        for step in range(5):  # long enough to reach LOST
            result = tracker.update(_blank(), timestamp_s=(step + 1) * 0.05)
        assert result.state == TrackState.LOST

        new_x, new_y = cup_x + 25, cup_y + 10
        result = tracker.update(_frame(new_x, new_y, patch), timestamp_s=1.0)
        assert result.state == TrackState.TRACKED
        assert result.reacquired is True
        assert abs(result.outlet_x - new_x) < 3.0
        assert abs(result.outlet_y - new_y) < 3.0

    def test_reacquisition_is_verified_not_assumed(self):
        """A differently-textured cup must not be accepted as a reacquisition.

        This is the "reacquisition does not retroactively prove what happened
        during the gap" requirement at the tracker level: coming back with
        *something* is not enough, it has to look like the thing that was
        lost.
        """
        patch = _patch(seed=9)
        other_patch = _patch(seed=999)  # unrelated content
        cfg = _config(track_max_bridge_s=0.05, track_reacquire_min_correlation=0.8)
        cup_x, cup_y = 100, 90
        tracker = OutletTracker(
            cfg, _frame(cup_x, cup_y, patch), _outlet(cup_x, cup_y), patch_half_px=15
        )

        result = None
        for step in range(5):
            result = tracker.update(_blank(), timestamp_s=(step + 1) * 0.05)
        assert result.state == TrackState.LOST

        decoy = _frame(cup_x + 10, cup_y, other_patch)
        result = tracker.update(decoy, timestamp_s=1.0)
        assert result.state == TrackState.LOST
        assert result.reacquired is False

    def test_reacquisition_survives_a_gap_at_the_frame_edge(self):
        """The search window bound: a candidate outside the frame is rejected."""
        patch = _patch(seed=10)
        cfg = _config(track_max_bridge_s=0.05)
        tracker = OutletTracker(cfg, _frame(100, 90, patch), _outlet(100, 90), patch_half_px=15)
        for step in range(5):
            result = tracker.update(_blank(), timestamp_s=(step + 1) * 0.05)
        assert result.state == TrackState.LOST
        # No candidate anywhere in frame: stays lost, does not crash or guess.
        result = tracker.update(_blank(), timestamp_s=2.0)
        assert result.state == TrackState.LOST
        assert result.reacquired is False
