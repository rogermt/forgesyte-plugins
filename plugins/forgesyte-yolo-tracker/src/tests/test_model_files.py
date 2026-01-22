"""Tests for model file validation.

Ensures model files exist and are not stubs (size > 1KB).
"""

from pathlib import Path


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
