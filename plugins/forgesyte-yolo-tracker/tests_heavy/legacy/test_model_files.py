"""Tests for model file validation.

Ensures model files exist and are not stubs (size > 1KB).
Model existence tests run when models directory exists.
Stub size tests skip when models are stubs (< 1KB).
Run with RUN_MODEL_TESTS=1 to force-run all tests in CPU environments.
"""

import os
from pathlib import Path

import pytest

from forgesyte_yolo_tracker.configs import MODEL_CONFIG_PATH

# Environment flag - force-run all model tests in CPU environments
RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

# Check if models directory exists
MODELS_DIR = Path(MODEL_CONFIG_PATH).parent.parent / "models"
MODELS_EXIST = MODELS_DIR.exists()

# Skip entire module if models don't exist AND RUN_MODEL_TESTS is not set
pytestmark = pytest.mark.skipif(
    not (MODELS_EXIST or RUN_MODEL_TESTS),
    reason="Models directory not found. Set RUN_MODEL_TESTS=1 to skip model existence checks.",
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
