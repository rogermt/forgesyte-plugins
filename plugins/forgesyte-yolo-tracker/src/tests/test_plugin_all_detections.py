"""Test plugin analyze method runs all three detections."""

from unittest.mock import patch, MagicMock
import numpy as np
import cv2


def create_valid_image_bytes() -> bytes:
    """Create valid image bytes that cv2.imdecode can parse."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
    # Use cv2.imencode to create valid PNG bytes that cv2.imdecode can parse
    return cv2.imencode('.png', frame)[1].tobytes()


def make_analysis_result(
    text: str = "",
    blocks=None,
    confidence: float = 1.0,
    language=None,
    error=None,
    extra=None,
) -> dict:
    """Factory for AnalysisResult dict (mirrors conftest.py and test_plugin.py)."""
    if blocks is None:
        blocks = []
    return {
        "text": text,
        "blocks": blocks,
        "confidence": confidence,
        "language": language,
        "error": error,
        "extra": extra,
    }


class TestPluginAllDetections:
    """Tests for plugin.analyze() returning all three detection types."""

    @patch('forgesyte_yolo_tracker.plugin.detect_pitch_json')
    @patch('forgesyte_yolo_tracker.plugin.detect_ball_json')
    @patch('forgesyte_yolo_tracker.plugin.detect_players_json')
    @patch('forgesyte_yolo_tracker.plugin.AnalysisResult')
    def test_analyze_returns_all_three_detections(
        self,
        mock_result: MagicMock,
        mock_players: MagicMock,
        mock_ball: MagicMock,
        mock_pitch: MagicMock,
    ) -> None:
        """Verify analyze() runs players, ball, and pitch detection."""
        # Setup mocks
        mock_result.side_effect = make_analysis_result
        mock_players.return_value = {"detections": [], "count": 0, "classes": {}}
        mock_ball.return_value = {"ball": [], "ball_detected": False}
        mock_pitch.return_value = {"keypoints": [], "pitch_detected": False}

        # Create valid image data that cv2.imdecode can parse
        image_data = create_valid_image_bytes()

        # Import and instantiate INSIDE the with block
        from forgesyte_yolo_tracker.plugin import Plugin
        plugin = Plugin()
        result = plugin.analyze(image_data)

        # Verify all three functions were called
        mock_players.assert_called_once()
        mock_ball.assert_called_once()
        mock_pitch.assert_called_once()

        # Verify result is a dict with extra field
        assert isinstance(result, dict)
        assert "extra" in result
        assert result["extra"] is not None

    @patch('forgesyte_yolo_tracker.plugin.detect_pitch_json')
    @patch('forgesyte_yolo_tracker.plugin.detect_ball_json')
    @patch('forgesyte_yolo_tracker.plugin.detect_players_json')
    @patch('forgesyte_yolo_tracker.plugin.AnalysisResult')
    def test_analyze_extra_contains_all_results(
        self,
        mock_result: MagicMock,
        mock_players: MagicMock,
        mock_ball: MagicMock,
        mock_pitch: MagicMock,
    ) -> None:
        """Verify extra field contains results from all three detections."""
        # Setup mocks with distinct data
        mock_result.side_effect = make_analysis_result
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

        # Create valid image data
        image_data = create_valid_image_bytes()

        # Import and instantiate INSIDE the with block
        from forgesyte_yolo_tracker.plugin import Plugin
        plugin = Plugin()
        result = plugin.analyze(image_data)

        # Verify all three results are in extra (default behavior)
        assert isinstance(result, dict)
        assert "extra" in result
        assert "players" in result["extra"]
        assert "ball" in result["extra"]
        assert "pitch" in result["extra"]
        assert result["extra"]["players"]["count"] == 1
        assert result["extra"]["ball"]["ball_detected"] is True
        assert result["extra"]["pitch"]["pitch_detected"] is True

    @patch('forgesyte_yolo_tracker.plugin.detect_players_json')
    @patch('forgesyte_yolo_tracker.plugin.AnalysisResult')
    def test_analyze_respects_detections_option(
        self,
        mock_result: MagicMock,
        mock_players: MagicMock,
    ) -> None:
        """Verify analyze() respects detections option."""
        mock_result.side_effect = make_analysis_result
        mock_players.return_value = {"count": 0, "classes": {}}

        # Create valid image data
        image_data = create_valid_image_bytes()

        # Import and instantiate INSIDE the with block
        from forgesyte_yolo_tracker.plugin import Plugin
        plugin = Plugin()

        # Only run players
        result = plugin.analyze(image_data, options={"detections": ["players"]})

        assert isinstance(result, dict)
        assert "extra" in result
        assert "players" in result["extra"]
        assert "ball" not in result["extra"]
        assert "pitch" not in result["extra"]

    def test_analyze_default_runs_all_detections(self) -> None:
        """Verify default behavior runs all three detections."""
        image_data = create_valid_image_bytes()

        with patch('forgesyte_yolo_tracker.plugin.detect_players_json') as mock_players, \
             patch('forgesyte_yolo_tracker.plugin.detect_ball_json') as mock_ball, \
             patch('forgesyte_yolo_tracker.plugin.detect_pitch_json') as mock_pitch:
            
            mock_players.return_value = {"count": 0}
            mock_ball.return_value = {"detected": False}
            mock_pitch.return_value = {"detected": False}

            # Import and instantiate INSIDE the with block
            from forgesyte_yolo_tracker.plugin import Plugin
            plugin = Plugin()
            result = plugin.analyze(image_data)  # No options, default behavior

            # Verify all three were called
            mock_players.assert_called_once()
            mock_ball.assert_called_once()
            mock_pitch.assert_called_once()
