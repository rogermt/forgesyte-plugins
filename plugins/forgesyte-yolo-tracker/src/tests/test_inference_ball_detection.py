"""Tests for ball detection inference module."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from tests.constants import RUN_MODEL_TESTS, MODELS_EXIST


pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS or not MODELS_EXIST,
    reason="Set RUN_MODEL_TESTS=1 AND download models to run",
)


class TestBallDetectionJSON:
    """Tests for detect_ball_json function."""

    def test_returns_dict_with_detections(self) -> None:
        """Verify returns dictionary with detections key."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "detections" in result

    def test_returns_count(self) -> None:
        """Verify returns count of detections."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert "count" in result
        assert isinstance(result["count"], int)

    def test_returns_ball(self) -> None:
        """Verify returns ball key with primary detection."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert "ball" in result

    def test_returns_ball_detected(self) -> None:
        """Verify returns ball_detected boolean."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert "ball_detected" in result
        assert isinstance(result["ball_detected"], bool)

    def test_detections_have_xyxy(self) -> None:
        """Verify each detection has xyxy coordinates."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        for det in result["detections"]:
            assert "xyxy" in det
            assert len(det["xyxy"]) == 4

    def test_detections_have_confidence(self) -> None:
        """Verify each detection has confidence score."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        for det in result["detections"]:
            assert "confidence" in det
            assert 0.0 <= det["confidence"] <= 1.0

    def test_ball_detected_matches_ball(self) -> None:
        """Verify ball_detected matches if ball exists."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert (result["ball_detected"] is True) == (result["ball"] is not None)


class TestBallDetectionJSONWithAnnotated:
    """Tests for detect_ball_json_with_annotated_frame function."""

    def test_returns_dict(self) -> None:
        """Verify returns dictionary."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated_frame(frame, device="cpu")

        assert isinstance(result, dict)

    def test_returns_annotated_frame_base64(self) -> None:
        """Verify returns base64 encoded annotated frame."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated_frame(frame, device="cpu")

        assert "annotated_frame_base64" in result
        assert isinstance(result["annotated_frame_base64"], str)

    def test_annotated_frame_is_valid_base64(self) -> None:
        """Verify annotated_frame_base64 is valid base64."""
        import base64

        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated_frame(frame, device="cpu")

        decoded = base64.b64decode(result["annotated_frame_base64"])
        assert len(decoded) > 0


class TestBallDetectionModelCache:
    """Tests for model caching."""

    def test_model_is_cached(self) -> None:
        """Verify model is cached after first call."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            get_ball_detection_model,
        )

        with patch("forgesyte_yolo_tracker.inference.ball_detection.YOLO") as mock_yolo:
            mock_instance = MagicMock()
            mock_yolo.return_value = mock_instance

            model1 = get_ball_detection_model(device="cpu")
            model2 = get_ball_detection_model(device="cpu")

            # Should be the same cached model
            assert model1 is model2
