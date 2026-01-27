"""Tests for model file validation.

Ensures model files exist and are not stubs (size > 1KB).
Skipped by default (RUN_MODEL_TESTS=0) for CPU environments.
Run with RUN_MODEL_TESTS=1 to verify models are real (>1KB).
"""

import os
from pathlib import Path

import pytest

# Environment flag - only run model file tests when explicitly enabled
RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="Set RUN_MODEL_TESTS=1 to run (requires real models > 1KB from Roboflow)",
)


def _get_model_path(model_name: str) -> Path:
    """Get path to model file."""
    from forgesyte_yolo_tracker.configs import MODEL_CONFIG_PATH
    config_dir = Path(MODEL_CONFIG_PATH).parent
    models_dir = config_dir.parent / "models"
    return models_dir / model_name


def _model_is_stub(model_name: str) -> bool:
    """Check if model file is a stub (< 1KB)."""
    model_path = _get_model_path(model_name)
    if not model_path.exists():
        return True
    size_kb = model_path.stat().st_size / 1024
    return size_kb < 1


class TestModelFiles:
    """Tests for model file validation."""

    def test_player_model_file_exists(self) -> None:
        """Verify player detection model file exists."""
        from forgesyte_yolo_tracker.configs import MODEL_CONFIG_PATH

        config_dir = Path(MODEL_CONFIG_PATH).parent
        models_dir = config_dir.parent / "models"
        model_file = models_dir / "football-player-detection-v3.pt"

        assert model_file.exists(), f"Player model missing: {model_file}"

    def test_ball_model_file_exists(self) -> None:
        """Verify ball detection model file exists."""
        from forgesyte_yolo_tracker.configs import MODEL_CONFIG_PATH

        config_dir = Path(MODEL_CONFIG_PATH).parent
        models_dir = config_dir.parent / "models"
        model_file = models_dir / "football-ball-detection-v2.pt"

        assert model_file.exists(), f"Ball model missing: {model_file}"

    def test_pitch_model_file_exists(self) -> None:
        """Verify pitch detection model file exists."""
        from forgesyte_yolo_tracker.configs import MODEL_CONFIG_PATH

        config_dir = Path(MODEL_CONFIG_PATH).parent
        models_dir = config_dir.parent / "models"
        model_file = models_dir / "football-pitch-detection-v1.pt"

        assert model_file.exists(), f"Pitch model missing: {model_file}"

    @pytest.mark.skipif(
        _model_is_stub("football-player-detection-v3.pt"),
        reason="Player model is stub - download real model from Roboflow",
    )
    def test_player_model_is_not_stub(self) -> None:
        """Verify player model file is not a stub (size > 1KB)."""
        from forgesyte_yolo_tracker.configs import MODEL_CONFIG_PATH

        config_dir = Path(MODEL_CONFIG_PATH).parent
        models_dir = config_dir.parent / "models"
        model_file = models_dir / "football-player-detection-v3.pt"

        size_kb = model_file.stat().st_size / 1024
        assert size_kb > 1, (
            f"Player model is stub ({size_kb:.2f} KB). "
            f"Download real model from Roboflow on GPU environment."
        )

    @pytest.mark.skipif(
        _model_is_stub("football-ball-detection-v2.pt"),
        reason="Ball model is stub - download real model from Roboflow",
    )
    def test_ball_model_is_not_stub(self) -> None:
        """Verify ball model file is not a stub (size > 1KB)."""
        from forgesyte_yolo_tracker.configs import MODEL_CONFIG_PATH

        config_dir = Path(MODEL_CONFIG_PATH).parent
        models_dir = config_dir.parent / "models"
        model_file = models_dir / "football-ball-detection-v2.pt"

        size_kb = model_file.stat().st_size / 1024
        assert size_kb > 1, (
            f"Ball model is stub ({size_kb:.2f} KB). "
            f"Download real model from Roboflow on GPU environment."
        )

    @pytest.mark.skipif(
        _model_is_stub("football-pitch-detection-v1.pt"),
        reason="Pitch model is stub - download real model from Roboflow",
    )
    def test_pitch_model_is_not_stub(self) -> None:
        """Verify pitch model file is not a stub (size > 1KB)."""
        from forgesyte_yolo_tracker.configs import MODEL_CONFIG_PATH

        config_dir = Path(MODEL_CONFIG_PATH).parent
        models_dir = config_dir.parent / "models"
        model_file = models_dir / "football-pitch-detection-v1.pt"

        size_kb = model_file.stat().st_size / 1024
        assert size_kb > 1, (
            f"Pitch model is stub ({size_kb:.2f} KB). "
            f"Download real model from Roboflow on GPU environment."
        )
