"""Tests for video module model path resolution.

Verifies that all video modules use correct relative paths from configs
instead of hardcoded paths.
"""

from pathlib import Path


class TestVideoModelPaths:
    """Tests for video module model path resolution."""

    def test_player_detection_video_model_path_is_resolved(self) -> None:
        """Verify player detection video uses config-based model path."""
        from forgesyte_yolo_tracker.video.player_detection_video import MODEL_PATH

        assert isinstance(MODEL_PATH, str)
        assert MODEL_PATH.endswith(".pt")
        assert "models" in MODEL_PATH
        assert not MODEL_PATH.startswith("src/")

    def test_player_tracking_video_model_path_is_resolved(self) -> None:
        """Verify player tracking video uses config-based model path."""
        from forgesyte_yolo_tracker.video.player_tracking_video import MODEL_PATH

        assert isinstance(MODEL_PATH, str)
        assert MODEL_PATH.endswith(".pt")
        assert "models" in MODEL_PATH
        assert not MODEL_PATH.startswith("src/")

    def test_ball_detection_video_model_path_is_resolved(self) -> None:
        """Verify ball detection video uses config-based model path."""
        from forgesyte_yolo_tracker.video.ball_detection_video import MODEL_PATH

        assert isinstance(MODEL_PATH, str)
        assert MODEL_PATH.endswith(".pt")
        assert "models" in MODEL_PATH
        assert not MODEL_PATH.startswith("src/")

    def test_pitch_detection_video_model_path_is_resolved(self) -> None:
        """Verify pitch detection video uses config-based model path."""
        from forgesyte_yolo_tracker.video.pitch_detection_video import MODEL_PATH

        assert isinstance(MODEL_PATH, str)
        assert MODEL_PATH.endswith(".pt")
        assert "models" in MODEL_PATH
        assert not MODEL_PATH.startswith("src/")

    def test_radar_video_player_model_path_is_resolved(self) -> None:
        """Verify radar video player model uses config-based path."""
        from forgesyte_yolo_tracker.video.radar_video import PLAYER_MODEL_PATH

        assert isinstance(PLAYER_MODEL_PATH, str)
        assert PLAYER_MODEL_PATH.endswith(".pt")
        assert "models" in PLAYER_MODEL_PATH
        assert not PLAYER_MODEL_PATH.startswith("src/")

    def test_radar_video_pitch_model_path_is_resolved(self) -> None:
        """Verify radar video pitch model uses config-based path."""
        from forgesyte_yolo_tracker.video.radar_video import PITCH_MODEL_PATH

        assert isinstance(PITCH_MODEL_PATH, str)
        assert PITCH_MODEL_PATH.endswith(".pt")
        assert "models" in PITCH_MODEL_PATH
        assert not PITCH_MODEL_PATH.startswith("src/")

    def test_all_video_model_paths_use_absolute_path(self) -> None:
        """Verify all video model paths are absolute."""
        from forgesyte_yolo_tracker.video.player_detection_video import MODEL_PATH as pd_path
        from forgesyte_yolo_tracker.video.player_tracking_video import MODEL_PATH as pt_path
        from forgesyte_yolo_tracker.video.ball_detection_video import MODEL_PATH as bd_path
        from forgesyte_yolo_tracker.video.pitch_detection_video import MODEL_PATH as pit_path
        from forgesyte_yolo_tracker.video.radar_video import PLAYER_MODEL_PATH as r_player
        from forgesyte_yolo_tracker.video.radar_video import PITCH_MODEL_PATH as r_pitch

        for path in [pd_path, pt_path, bd_path, pit_path, r_player, r_pitch]:
            p = Path(path)
            assert p.is_absolute(), f"Path not absolute: {path}"
