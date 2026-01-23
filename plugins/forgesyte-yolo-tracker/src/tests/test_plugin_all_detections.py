"""Test plugin analyze method runs all three detections."""

from unittest.mock import patch
import numpy as np

try:
    from app.models import AnalysisResult
except ImportError:
    # For testing outside FastAPI context
    class AnalysisResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)


class TestPluginAllDetections:
    """Tests for plugin.analyze() returning all three detection types."""

    @patch("forgesyte_yolo_tracker.plugin.detect_pitch_json")
    @patch("forgesyte_yolo_tracker.plugin.detect_ball_json")
    @patch("forgesyte_yolo_tracker.plugin.detect_players_json")
    def test_analyze_returns_all_three_detections(
        self, mock_players, mock_ball, mock_pitch
    ) -> None:
        """Verify analyze() runs players, ball, and pitch detection."""
        # Setup mocks
        mock_players.return_value = {"detections": [], "count": 0, "classes": {}}
        mock_ball.return_value = {"ball": [], "ball_detected": False}
        mock_pitch.return_value = {"keypoints": [], "pitch_detected": False}

        # Create fake image data
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        image_data = frame.tobytes()

        from forgesyte_yolo_tracker.plugin import Plugin

        plugin = Plugin()
        result = plugin.analyze(image_data)

        # Verify all three functions were called
        mock_players.assert_called_once()
        mock_ball.assert_called_once()
        mock_pitch.assert_called_once()

        # Verify result structure
        assert isinstance(result, AnalysisResult)
        assert result.extra is not None

    @patch("forgesyte_yolo_tracker.plugin.detect_pitch_json")
    @patch("forgesyte_yolo_tracker.plugin.detect_ball_json")
    @patch("forgesyte_yolo_tracker.plugin.detect_players_json")
    def test_analyze_extra_contains_all_results(
        self, mock_players, mock_ball, mock_pitch
    ) -> None:
        """Verify extra field contains results from all three detections."""
        # Setup mocks with distinct data
        mock_players.return_value = {
            "detections": [{"class": "player"}],
            "count": 1,
            "classes": {"player": 1},
        }
        mock_ball.return_value = {"ball": [{"x": 100}], "ball_detected": True}
        mock_pitch.return_value = {
            "keypoints": [{"x": 0, "y": 0}],
            "pitch_detected": True,
        }

        # Create fake image data
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        image_data = frame.tobytes()

        from forgesyte_yolo_tracker.plugin import Plugin

        plugin = Plugin()
        result = plugin.analyze(image_data)

        # Verify all three results are in extra (default behavior)
        assert "players" in result.extra
        assert "ball" in result.extra
        assert "pitch" in result.extra
        assert result.extra["players"]["count"] == 1
        assert result.extra["ball"]["ball_detected"] is True
        assert result.extra["pitch"]["pitch_detected"] is True

    @patch("forgesyte_yolo_tracker.plugin.detect_players_json")
    def test_analyze_respects_detections_option(self, mock_players) -> None:
        """Verify analyze() respects detections option."""
        mock_players.return_value = {"count": 0, "classes": {}}

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        image_data = frame.tobytes()

        from forgesyte_yolo_tracker.plugin import Plugin

        plugin = Plugin()
        
        # Only run players
        result = plugin.analyze(image_data, options={"detections": ["players"]})
        
        assert "players" in result.extra
        assert "ball" not in result.extra
        assert "pitch" not in result.extra

    @patch("forgesyte_yolo_tracker.plugin.detect_pitch_json")
    @patch("forgesyte_yolo_tracker.plugin.detect_ball_json")
    @patch("forgesyte_yolo_tracker.plugin.detect_players_json")
    def test_analyze_default_runs_all_detections(
        self, mock_players, mock_ball, mock_pitch
    ) -> None:
        """Verify default behavior runs all three detections."""
        mock_players.return_value = {"count": 0}
        mock_ball.return_value = {"detected": False}
        mock_pitch.return_value = {"detected": False}

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        image_data = frame.tobytes()

        from forgesyte_yolo_tracker.plugin import Plugin

        plugin = Plugin()
        result = plugin.analyze(image_data)  # No options, default behavior
        
        # Verify all three were called
        mock_players.assert_called_once()
        mock_ball.assert_called_once()
        mock_pitch.assert_called_once()
