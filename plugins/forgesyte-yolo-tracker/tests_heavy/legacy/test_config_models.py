"""Tests for model configuration loading from models.yaml."""

import os

import pytest

from forgesyte_yolo_tracker.configs import (MODEL_CONFIG_PATH, get_confidence,
                                            get_model_path, load_model_config)


class TestLoadModelConfig:
    """Tests for load_model_config function."""

    def test_load_model_config_returns_dict(self) -> None:
        """Verify load_model_config returns a dictionary."""
        config = load_model_config()
        assert isinstance(config, dict)

    def test_config_has_models_key(self) -> None:
        """Verify config has 'models' key."""
        config = load_model_config()
        assert "models" in config
        assert isinstance(config["models"], dict)

    def test_config_has_confidence_key(self) -> None:
        """Verify config has 'confidence' key."""
        config = load_model_config()
        assert "confidence" in config
        assert isinstance(config["confidence"], dict)

    def test_config_has_device_key(self) -> None:
        """Verify config has 'device' key."""
        config = load_model_config()
        assert "device" in config


class TestModelPaths:
    """Tests for get_model_path function."""

    def test_get_player_detection_model_path(self) -> None:
        """Verify player detection model name is returned."""
        model_path = get_model_path("player_detection")
        assert isinstance(model_path, str)
        assert model_path.endswith(".pt")

    def test_get_ball_detection_model_path(self) -> None:
        """Verify ball detection model name is returned."""
        model_path = get_model_path("ball_detection")
        assert isinstance(model_path, str)
        assert model_path.endswith(".pt")

    def test_get_pitch_detection_model_path(self) -> None:
        """Verify pitch detection model name is returned."""
        model_path = get_model_path("pitch_detection")
        assert isinstance(model_path, str)
        assert model_path.endswith(".pt")

    def test_invalid_model_raises_error(self) -> None:
        """Verify invalid model key raises KeyError."""
        with pytest.raises(KeyError):
            get_model_path("nonexistent_model")


class TestConfidenceValues:
    """Tests for get_confidence function."""

    def test_get_player_confidence(self) -> None:
        """Verify player confidence is valid float."""
        confidence = get_confidence("player")
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_get_ball_confidence(self) -> None:
        """Verify ball confidence is valid float."""
        confidence = get_confidence("ball")
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_get_pitch_confidence(self) -> None:
        """Verify pitch confidence is valid float."""
        confidence = get_confidence("pitch")
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0

    def test_invalid_task_raises_error(self) -> None:
        """Verify invalid task raises KeyError."""
        with pytest.raises(KeyError):
            get_confidence("invalid_task")


class TestConfigFileExists:
    """Tests for config file existence."""

    def test_models_yaml_exists(self) -> None:
        """Verify models.yaml file exists."""
        assert os.path.exists(MODEL_CONFIG_PATH)

    def test_models_yaml_is_readable(self) -> None:
        """Verify models.yaml can be read."""
        with open(MODEL_CONFIG_PATH, "r") as f:
            content = f.read()
        assert len(content) > 0


class TestConfigContent:
    """Tests for config file content validation."""

    def test_player_model_name_is_valid(self) -> None:
        """Verify player model name is a valid .pt file."""
        config = load_model_config()
        model_name = config["models"]["player_detection"]
        assert isinstance(model_name, str)
        assert model_name.endswith(".pt")

    def test_ball_model_name_is_valid(self) -> None:
        """Verify ball model name is a valid .pt file."""
        config = load_model_config()
        model_name = config["models"]["ball_detection"]
        assert isinstance(model_name, str)
        assert model_name.endswith(".pt")

    def test_pitch_model_name_is_valid(self) -> None:
        """Verify pitch model name is a valid .pt file."""
        config = load_model_config()
        model_name = config["models"]["pitch_detection"]
        assert isinstance(model_name, str)
        assert model_name.endswith(".pt")

    def test_confidence_values_are_reasonable(self) -> None:
        """Verify confidence values are within reasonable bounds."""
        config = load_model_config()
        assert 0.1 <= config["confidence"]["player"] <= 0.5
        assert 0.1 <= config["confidence"]["ball"] <= 0.5
        assert 0.1 <= config["confidence"]["pitch"] <= 0.5

    def test_device_value_is_valid(self) -> None:
        """Verify device value is a valid string."""
        config = load_model_config()
        assert isinstance(config["device"], str)
        assert config["device"] in ["cpu", "cuda", "mps"]
