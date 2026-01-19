"""Tests for player detection inference module."""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS, reason="Set RUN_MODEL_TESTS=1 to run (requires YOLO model)"
)


class TestPlayerDetectionJSON:
    """Tests for detect_players_json function."""

    def test_returns_dict_with_detections(self) -> None:
        """Verify returns dictionary with detections key."""
        from forgesyte_yolo_tracker.inference.player_detection import detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "detections" in result

    def test_returns_count(self) -> None:
        """Verify returns count of detections."""
        from forgesyte_yolo_tracker.inference.player_detection import detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        assert "count" in result
        assert isinstance(result["count"], int)

    def test_returns_classes(self) -> None:
        """Verify returns class counts."""
        from forgesyte_yolo_tracker.inference.player_detection import detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        assert "classes" in result
        assert "player" in result["classes"]
        assert "goalkeeper" in result["classes"]
        assert "referee" in result["classes"]

    def test_detections_have_xyxy(self) -> None:
        """Verify each detection has xyxy coordinates."""
        from forgesyte_yolo_tracker.inference.player_detection import detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        for det in result["detections"]:
            assert "xyxy" in det
            assert len(det["xyxy"]) == 4

    def test_detections_have_confidence(self) -> None:
        """Verify each detection has confidence score."""
        from forgesyte_yolo_tracker.inference.player_detection import detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        for det in result["detections"]:
            assert "confidence" in det
            assert 0.0 <= det["confidence"] <= 1.0

    def test_detections_have_class_id(self) -> None:
        """Verify each detection has class_id."""
        from forgesyte_yolo_tracker.inference.player_detection import detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        for det in result["detections"]:
            assert "class_id" in det
            assert det["class_id"] in [0, 1, 2]  # player, goalkeeper, referee


class TestPlayerDetectionJSONWithAnnotated:
    """Tests for detect_players_json_with_annotated function."""

    def test_returns_dict(self) -> None:
        """Verify returns dictionary."""
        from forgesyte_yolo_tracker.inference.player_detection import (
            detect_players_json_with_annotated,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated(frame, device="cpu")

        assert isinstance(result, dict)

    def test_returns_annotated_frame_base64(self) -> None:
        """Verify returns base64 encoded annotated frame."""
        from forgesyte_yolo_tracker.inference.player_detection import (
            detect_players_json_with_annotated,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated(frame, device="cpu")

        assert "annotated_frame_base64" in result
        assert isinstance(result["annotated_frame_base64"], str)

    def test_annotated_frame_is_valid_base64(self) -> None:
        """Verify annotated_frame_base64 is valid base64."""
        import base64

        from forgesyte_yolo_tracker.inference.player_detection import (
            detect_players_json_with_annotated,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated(frame, device="cpu")

        decoded = base64.b64decode(result["annotated_frame_base64"])
        assert len(decoded) > 0


class TestPlayerDetectionModelCache:
    """Tests for model caching."""

    def test_model_is_cached(self) -> None:
        """Verify model is cached after first call."""
        from forgesyte_yolo_tracker.inference.player_detection import (
            get_player_detection_model,
        )

        with patch("forgesyte_yolo_tracker.inference.player_detection.YOLO") as mock_yolo:
            mock_instance = MagicMock()
            mock_yolo.return_value = mock_instance

            model1 = get_player_detection_model(device="cpu")
            model2 = get_player_detection_model(device="cpu")

            # Should be the same cached model
            assert model1 is model2
