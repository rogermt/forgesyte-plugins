"""Tests for inference modules using mocked YOLO models.

These tests verify inference function structure and behavior without
loading actual YOLO models (safe to run on CPU).
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


class TestPlayerDetectionMocked:
    """Tests for player detection with mocked YOLO."""

    @patch("forgesyte_yolo_tracker.inference.player_detection.YOLO")
    def test_detect_players_json_returns_dict(self, mock_yolo_class: MagicMock) -> None:
        """Verify player detection returns proper dict structure."""
        # Setup mock
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model

        # Create mock detections
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = []
        mock_result.boxes.conf = []
        mock_result.boxes.cls = []
        mock_model.predict.return_value = [mock_result]

        # Test
        from forgesyte_yolo_tracker.inference.player_detection import detect_players_json

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        result = detect_players_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "detections" in result
        assert "count" in result
        assert "classes" in result

    @patch("forgesyte_yolo_tracker.inference.player_detection.YOLO")
    def test_detect_players_json_with_annotated_returns_dict(
        self, mock_yolo_class: MagicMock
    ) -> None:
        """Verify player detection with annotated returns proper dict."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = []
        mock_result.boxes.conf = []
        mock_result.boxes.cls = []
        mock_model.predict.return_value = [mock_result]

        from forgesyte_yolo_tracker.inference.player_detection import (
            detect_players_json_with_annotated_frame,
        )

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        result = detect_players_json_with_annotated_frame(frame, device="cpu")

        assert isinstance(result, dict)
        assert "detections" in result
        assert "annotated_frame_base64" in result


class TestBallDetectionMocked:
    """Tests for ball detection with mocked YOLO."""

    @patch("forgesyte_yolo_tracker.inference.ball_detection.YOLO")
    def test_detect_ball_json_returns_dict(self, mock_yolo_class: MagicMock) -> None:
        """Verify ball detection returns proper dict structure."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = []
        mock_result.boxes.conf = []
        mock_result.boxes.cls = []
        mock_model.predict.return_value = [mock_result]

        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        result = detect_ball_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "detections" in result
        assert "ball_detected" in result


class TestPitchDetectionMocked:
    """Tests for pitch detection with mocked YOLO."""

    @patch("forgesyte_yolo_tracker.inference.pitch_detection.YOLO")
    def test_detect_pitch_json_returns_dict(self, mock_yolo_class: MagicMock) -> None:
        """Verify pitch detection returns proper dict structure."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_result = MagicMock()
        mock_result.keypoints = MagicMock()
        mock_result.keypoints.xy = []
        mock_result.boxes = MagicMock()
        mock_result.boxes.conf = []
        mock_model.predict.return_value = [mock_result]

        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        result = detect_pitch_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "keypoints" in result
        assert "pitch_detected" in result


class TestTeamClassificationMocked:
    """Tests for team classification with mocked YOLO."""

    @patch("forgesyte_yolo_tracker.inference.team_classification.YOLO")
    def test_classify_teams_json_returns_dict(
        self, mock_yolo_class: MagicMock
    ) -> None:
        """Verify team classification returns proper dict structure."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = []
        mock_result.boxes.conf = []
        mock_result.boxes.cls = []
        mock_model.predict.return_value = [mock_result]

        from forgesyte_yolo_tracker.inference.team_classification import (
            classify_teams_json,
        )

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        result = classify_teams_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "detections" in result
        assert "team_ids" in result


class TestRadarMocked:
    """Tests for radar with mocked YOLO."""

    @patch("forgesyte_yolo_tracker.inference.radar.YOLO")
    def test_generate_radar_json_returns_dict(self, mock_yolo_class: MagicMock) -> None:
        """Verify radar returns proper dict structure."""
        mock_model = MagicMock()
        mock_yolo_class.return_value = mock_model
        mock_result = MagicMock()
        mock_result.boxes = MagicMock()
        mock_result.boxes.xyxy = []
        mock_result.boxes.conf = []
        mock_result.boxes.cls = []
        mock_result.keypoints = MagicMock()
        mock_result.keypoints.xy = []
        mock_model.predict.return_value = [mock_result]

        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100
        result = generate_radar_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "radar_points" in result
