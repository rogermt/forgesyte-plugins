"""Tests for ball detection inference module."""

import os
import pytest
import numpy as np


RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS, reason="Set RUN_MODEL_TESTS=1 to run (requires YOLO model)"
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

    def test_returns_ball_key(self) -> None:
        """Verify returns ball key with primary detection."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert "ball" in result

    def test_ball_has_xyxy_and_confidence(self) -> None:
        """Verify ball entry has xyxy and confidence."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        if result["ball"] is not None:
            assert "xyxy" in result["ball"]
            assert "confidence" in result["ball"]


class TestBallDetectionJSONWithAnnotated:
    """Tests for detect_ball_json_with_annotated function."""

    def test_returns_annotated_frame_base64(self) -> None:
        """Verify returns base64 encoded annotated frame."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json_with_annotated

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated(frame, device="cpu")

        assert "annotated_frame_base64" in result
        assert isinstance(result["annotated_frame_base64"], str)
