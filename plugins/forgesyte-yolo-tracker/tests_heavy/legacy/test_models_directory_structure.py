"""Tests for models directory structure and path resolution.

Ensures models directory is correctly located at:
src/forgesyte_yolo_tracker/models/ (not src/models/)
"""

from pathlib import Path


class TestModelsDirectoryStructure:
    """Tests for models directory structure."""

    def test_models_dir_exists_at_correct_location(self) -> None:
        """Verify models directory exists at src/forgesyte_yolo_tracker/models/."""
        from forgesyte_yolo_tracker.configs import MODEL_CONFIG_PATH

        # CONFIG_PATH should be at src/forgesyte_yolo_tracker/configs/models.yaml
        config_path = Path(MODEL_CONFIG_PATH)
        expected_models_dir = config_path.parent.parent / "models"

        assert expected_models_dir.exists(), (
            f"Models directory should exist at {expected_models_dir}, "
            f"not at src/models/"
        )

    def test_models_directory_is_sibling_of_configs(self) -> None:
        """Verify models/ is sibling to configs/ under forgesyte_yolo_tracker/."""
        from forgesyte_yolo_tracker.configs import MODEL_CONFIG_PATH

        config_dir = Path(MODEL_CONFIG_PATH).parent
        models_dir = config_dir.parent / "models"

        # Both should be under forgesyte_yolo_tracker
        assert config_dir.name == "configs"
        assert models_dir.name == "models"
        assert config_dir.parent == models_dir.parent

    def test_all_inference_modules_use_correct_models_path(self) -> None:
        """Verify all inference modules resolve models path correctly."""
        from forgesyte_yolo_tracker.inference.player_detection import MODEL_PATH as pd_path
        from forgesyte_yolo_tracker.inference.player_tracking import MODEL_PATH as pt_path
        from forgesyte_yolo_tracker.inference.ball_detection import MODEL_PATH as bd_path
        from forgesyte_yolo_tracker.inference.pitch_detection import MODEL_PATH as pit_path
        from forgesyte_yolo_tracker.inference.radar import PLAYER_MODEL_PATH as r_player
        from forgesyte_yolo_tracker.inference.radar import PITCH_MODEL_PATH as r_pitch

        # All should contain "forgesyte_yolo_tracker/models"
        paths = [pd_path, pt_path, bd_path, pit_path, r_player, r_pitch]
        for path in paths:
            assert "forgesyte_yolo_tracker" in path, (
                f"Path should contain 'forgesyte_yolo_tracker': {path}"
            )
            assert "models" in path, f"Path should contain 'models': {path}"
            # Ensure it's not using wrong src/models path
            assert "/src/models" not in path, (
                f"Path incorrectly points to src/models: {path}"
            )

    def test_all_video_modules_use_correct_models_path(self) -> None:
        """Verify all video modules resolve models path correctly."""
        from forgesyte_yolo_tracker.video.player_detection_video import MODEL_PATH as pd_path
        from forgesyte_yolo_tracker.video.player_tracking_video import MODEL_PATH as pt_path
        from forgesyte_yolo_tracker.video.ball_detection_video import MODEL_PATH as bd_path
        from forgesyte_yolo_tracker.video.pitch_detection_video import MODEL_PATH as pit_path
        from forgesyte_yolo_tracker.video.radar_video import PLAYER_MODEL_PATH as r_player
        from forgesyte_yolo_tracker.video.radar_video import PITCH_MODEL_PATH as r_pitch

        # All should contain "forgesyte_yolo_tracker/models"
        paths = [pd_path, pt_path, bd_path, pit_path, r_player, r_pitch]
        for path in paths:
            assert "forgesyte_yolo_tracker" in path, (
                f"Path should contain 'forgesyte_yolo_tracker': {path}"
            )
            assert "models" in path, f"Path should contain 'models': {path}"
            # Ensure it's not using wrong src/models path
            assert "/src/models" not in path, (
                f"Path incorrectly points to src/models: {path}"
            )
