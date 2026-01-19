"""Tests for pitch detection inference module."""

import os
import pytest
import numpy as np


RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS, reason="Set RUN_MODEL_TESTS=1 to run (requires YOLO model)"
)


class TestPitchDetectionJSON:
    """Tests for detect_pitch_json function."""

    def test_returns_dict_with_keypoints(self) -> None:
        """Verify returns dictionary with keypoints."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "keypoints" in result

    def test_returns_pitch_polygon(self) -> None:
        """Verify returns pitch polygon."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        assert "pitch_polygon" in result

    def test_keypoints_have_xy_confidence_name(self) -> None:
        """Verify each keypoint has xy, confidence, and name."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        for kp in result.get("keypoints", []):
            assert "xy" in kp
            assert "confidence" in kp
            assert "name" in kp


class TestPitchDetectionJSONWithAnnotated:
    """Tests for detect_pitch_json_with_annotated function."""

    def test_returns_annotated_frame_base64(self) -> None:
        """Verify returns base64 encoded annotated frame."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            detect_pitch_json_with_annotated,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json_with_annotated(frame, device="cpu")

        assert "annotated_frame_base64" in result
        assert isinstance(result["annotated_frame_base64"], str)
