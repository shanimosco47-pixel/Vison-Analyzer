"""Motion scorer behaviour on synthetic frames.

These check the properties that keep factory footage from producing nonsense:
noise is not motion, a light change is not motion, and a moving object is.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from app.analysis.motion_detector import MotionActivityScorer, MotionScorerConfig
from app.errors import ConfigurationError
from app.video.reader import FrameSample

WIDTH, HEIGHT = 320, 240


def noisy_frame(rng: np.random.Generator, level: int = 120, sigma: float = 2.5) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH), float(level), dtype=np.float32)
    frame += rng.normal(0.0, sigma, frame.shape).astype(np.float32)
    return np.clip(frame, 0, 255).astype(np.uint8)


def sample(image: np.ndarray, index: int, fps: float = 25.0) -> FrameSample:
    return FrameSample(index=index, timestamp_s=index / fps, image=image)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(4242)


class TestStaticScene:
    def test_sensor_noise_is_not_motion(self, rng):
        scorer = MotionActivityScorer()
        scores = [scorer.score(sample(noisy_frame(rng), i)).value for i in range(40)]
        assert max(scores) < 0.001

    def test_first_frame_is_never_motion(self, rng):
        scorer = MotionActivityScorer()
        assert scorer.score(sample(noisy_frame(rng), 0)).value == 0.0

    def test_reset_forgets_the_background(self, rng):
        scorer = MotionActivityScorer()
        scorer.score(sample(noisy_frame(rng), 0))
        scorer.reset()
        assert scorer.score(sample(noisy_frame(rng, level=200), 1)).value == 0.0


class TestMovingObject:
    def test_a_moving_block_is_detected(self, rng):
        scorer = MotionActivityScorer()
        for index in range(10):
            scorer.score(sample(noisy_frame(rng), index))

        frame = noisy_frame(rng)
        cv2.rectangle(frame, (100, 80), (160, 140), 250, -1)
        result = scorer.score(sample(frame, 11))
        # 60x60 of 320x240 is about 4.7% of the frame.
        assert result.value == pytest.approx(0.047, abs=0.01)
        assert not result.disturbed

    def test_tiny_speckle_is_filtered_out(self, rng):
        scorer = MotionActivityScorer(MotionScorerConfig(min_blob_area_ratio=0.001))
        for index in range(10):
            scorer.score(sample(noisy_frame(rng), index))

        frame = noisy_frame(rng)
        for x, y in ((10, 10), (200, 30), (300, 200)):  # three single bright pixels
            frame[y, x] = 255
        assert scorer.score(sample(frame, 11)).value == 0.0

    def test_a_stationary_object_fades_into_the_background(self, rng):
        """A parked object must stop being reported as activity."""
        scorer = MotionActivityScorer(
            MotionScorerConfig(background_alpha_active=0.2, background_alpha_idle=0.2)
        )
        for index in range(10):
            scorer.score(sample(noisy_frame(rng), index))

        scores = []
        for index in range(60):
            frame = noisy_frame(rng)
            cv2.rectangle(frame, (100, 80), (160, 140), 250, -1)
            scores.append(scorer.score(sample(frame, 11 + index)).value)
        assert scores[0] > 0.02
        assert scores[-1] < 0.005


class TestDisturbance:
    def test_a_scene_wide_light_change_is_flagged_not_reported_as_motion(self, rng):
        scorer = MotionActivityScorer()
        for index in range(10):
            scorer.score(sample(noisy_frame(rng, level=120), index))

        result = scorer.score(sample(noisy_frame(rng, level=200), 11))
        assert result.disturbed
        assert result.value == 0.0

    def test_the_background_recovers_after_a_disturbance(self, rng):
        scorer = MotionActivityScorer()
        for index in range(10):
            scorer.score(sample(noisy_frame(rng, level=120), index))
        scorer.score(sample(noisy_frame(rng, level=200), 11))  # lights on

        # Once the new level is the norm, it is quiet again, not endless motion.
        scores = [scorer.score(sample(noisy_frame(rng, level=200), 12 + i)).value for i in range(5)]
        assert max(scores) < 0.001

    def test_camera_shake_is_a_disturbance(self, rng):
        scorer = MotionActivityScorer()
        base = noisy_frame(rng)
        textured = base.copy()
        for x in range(0, WIDTH, 8):  # vertical stripes: a textured scene
            cv2.line(textured, (x, 0), (x, HEIGHT), 240, 3)
        for index in range(10):
            scorer.score(sample(textured, index))

        shifted = np.roll(textured, 6, axis=1)  # the whole image moves
        assert scorer.score(sample(shifted, 11)).disturbed


class TestConfigValidation:
    @pytest.mark.parametrize(
        "config",
        [
            MotionScorerConfig(min_abs_diff=0),
            MotionScorerConfig(noise_sigma_multiplier=0),
            MotionScorerConfig(min_blob_area_ratio=1.5),
            MotionScorerConfig(background_alpha_idle=0.0),
            MotionScorerConfig(disturbance_area_ratio=0.0),
        ],
    )
    def test_invalid_configuration_rejected(self, config):
        with pytest.raises(ConfigurationError):
            config.validate()

    def test_noisy_camera_raises_its_own_threshold(self):
        """The same object on a grainy camera must not swamp the score."""
        clean_rng = np.random.default_rng(1)
        grainy_rng = np.random.default_rng(2)
        results = {}
        for name, rng, sigma in (("clean", clean_rng, 1.0), ("grainy", grainy_rng, 12.0)):
            scorer = MotionActivityScorer()
            for index in range(15):
                scorer.score(sample(noisy_frame(rng, sigma=sigma), index))
            results[name] = scorer.score(sample(noisy_frame(rng, sigma=sigma), 16)).value
        assert results["clean"] < 0.005
        assert results["grainy"] < 0.02
