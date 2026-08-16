"""Configuration and ROI validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.config import (
    ROI,
    AppConfig,
    CoarseScanConfig,
    RefinementConfig,
    ZahnConfig,
    apply_overrides,
)
from app.errors import ConfigurationError, InvalidROIError


class TestROI:
    def test_valid_roi_accepted(self):
        ROI(10, 10, 100, 100).validate(640, 480)

    @pytest.mark.parametrize(
        "roi",
        [
            ROI(-1, 0, 100, 100),  # off the left edge
            ROI(0, -5, 100, 100),  # off the top
            ROI(600, 0, 100, 100),  # runs past the right edge
            ROI(0, 400, 100, 200),  # runs past the bottom
            ROI(0, 0, 2, 50),  # too narrow to analyse
        ],
    )
    def test_invalid_roi_rejected(self, roi):
        with pytest.raises(InvalidROIError):
            roi.validate(640, 480)

    def test_clipping_keeps_the_visible_part(self):
        clipped = ROI(600, 400, 200, 200).clipped_to(640, 480)
        assert (clipped.x, clipped.y, clipped.width, clipped.height) == (600, 400, 40, 80)

    def test_clipping_an_off_screen_roi_fails(self):
        with pytest.raises(InvalidROIError):
            ROI(700, 500, 10, 10).clipped_to(640, 480)

    def test_scaling(self):
        scaled = ROI(100, 200, 40, 80).scaled(0.5)
        assert (scaled.x, scaled.y, scaled.width, scaled.height) == (50, 100, 20, 40)

    def test_scaling_never_produces_a_zero_side(self):
        assert ROI(0, 0, 4, 4).scaled(0.05).width == 1

    def test_round_trip_through_dict(self):
        roi = ROI(1, 2, 3, 4)
        assert ROI.from_dict(roi.to_dict()) == roi

    def test_malformed_dict_rejected(self):
        with pytest.raises(InvalidROIError):
            ROI.from_dict({"x": "left", "y": 0, "width": 10, "height": 10})


class TestCoarseScanConfig:
    def test_defaults_are_valid(self):
        CoarseScanConfig().validate()

    def test_safety_factor_below_two_rejected(self):
        config = CoarseScanConfig(safety_factor=1.0)
        with pytest.raises(ConfigurationError):
            config.validate()

    def test_sensitivity_out_of_range_rejected(self):
        with pytest.raises(ConfigurationError):
            CoarseScanConfig(sensitivity=1.5).validate()

    def test_inverted_interval_bounds_rejected(self):
        with pytest.raises(ConfigurationError):
            CoarseScanConfig(min_sample_interval_s=5.0, max_sample_interval_s=1.0).validate()

    def test_exit_threshold_above_enter_rejected(self):
        with pytest.raises(ConfigurationError):
            CoarseScanConfig(enter_sigma=2.0, exit_sigma=6.0).validate()


class TestZahnConfig:
    def test_defaults_are_valid(self):
        ZahnConfig().validate()

    def test_persistence_must_be_positive(self):
        with pytest.raises(ConfigurationError):
            ZahnConfig(flow_end_persistence_s=0.0).validate()

    def test_activity_threshold_bounds(self):
        with pytest.raises(ConfigurationError):
            ZahnConfig(activity_threshold=1.5).validate()

    def test_confidence_thresholds_must_be_ordered(self):
        with pytest.raises(ConfigurationError):
            ZahnConfig(fail_confidence=0.9, review_confidence=0.5).validate()


class TestRefinementConfig:
    def test_defaults_are_valid(self):
        RefinementConfig().validate()

    def test_negative_roll_rejected(self):
        with pytest.raises(ConfigurationError):
            RefinementConfig(pre_roll_s=-1.0).validate()


class TestApplyOverrides:
    def test_known_keys_are_applied_and_coerced(self):
        config = apply_overrides(CoarseScanConfig(), {"shortest_event_s": "30", "sensitivity": 0.8})
        assert config.shortest_event_s == 30.0
        assert config.sensitivity == 0.8

    def test_unknown_keys_are_ignored(self):
        """The UI sends one parameter bag; each config takes what it knows."""
        config = apply_overrides(CoarseScanConfig(), {"flow_end_persistence_s": 0.5})
        assert config == CoarseScanConfig()

    def test_ignored_keys_are_skipped(self):
        config = apply_overrides(
            CoarseScanConfig(), {"shortest_event_s": 30}, ignore={"shortest_event_s"}
        )
        assert config.shortest_event_s == CoarseScanConfig().shortest_event_s

    def test_invalid_value_is_reported(self):
        with pytest.raises(ConfigurationError):
            apply_overrides(CoarseScanConfig(), {"shortest_event_s": "not a number"})

    def test_none_values_do_not_override(self):
        config = apply_overrides(CoarseScanConfig(), {"shortest_event_s": None})
        assert config.shortest_event_s == CoarseScanConfig().shortest_event_s


class TestAppConfig:
    def test_defaults_are_valid(self):
        AppConfig().validate()

    def test_environment_overrides(self, tmp_path: Path):
        config = AppConfig.from_env(
            {
                "VISION_ANALYZER_DATA_DIR": str(tmp_path),
                "VISION_ANALYZER_MAX_UPLOAD_MB": "50",
                "VISION_ANALYZER_PORT": "9001",
                "VISION_ANALYZER_SAVE_DIAGNOSTICS": "yes",
            }
        )
        assert config.data_dir == tmp_path
        assert config.max_upload_mb == 50
        assert config.port == 9001
        assert config.save_diagnostics is True
        assert config.upload_dir == tmp_path / "uploads"

    def test_invalid_environment_value_rejected(self):
        with pytest.raises(ConfigurationError):
            AppConfig.from_env({"VISION_ANALYZER_PORT": "not-a-port"})

    def test_out_of_range_port_rejected(self):
        with pytest.raises(ConfigurationError):
            AppConfig.from_env({"VISION_ANALYZER_PORT": "70000"})

    def test_defaults_are_local_only(self):
        """A prototype must not listen on every interface by accident."""
        assert AppConfig().host == "127.0.0.1"
