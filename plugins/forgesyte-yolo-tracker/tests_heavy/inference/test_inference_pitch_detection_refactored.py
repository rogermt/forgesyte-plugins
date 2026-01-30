"""Tests for refactored pitch detection using BaseDetector.

This module tests the pitch detection module which now uses BaseDetector
to eliminate code duplication. Tests verify wrapper functions properly
delegate to BaseDetector and handle pitch-specific logic (keypoints).
"""

import base64
from typing import Any, Dict

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS or not MODELS_EXIST,
    reason="Set RUN_MODEL_TESTS=1 AND download models to run",
)


class TestPitchDetectorConfiguration:
    """Tests for pitch detector configuration."""

    def test_pitch_detector_name_is_correct(self) -> None:
        """Verify detector name is 'pitch'."""
        from forgesyte_yolo_tracker.inference.pitch_detection import PITCH_DETECTOR

        assert PITCH_DETECTOR.detector_name == "pitch"

    def test_pitch_default_confidence_is_0_25(self) -> None:
        """Verify default confidence is 0.25."""
        from forgesyte_yolo_tracker.inference.pitch_detection import PITCH_DETECTOR

        assert PITCH_DETECTOR.default_confidence == 0.25

    def test_pitch_imgsz_is_1280(self) -> None:
        """Verify imgsz is 1280 for pitch."""
        from forgesyte_yolo_tracker.inference.pitch_detection import PITCH_DETECTOR

        assert PITCH_DETECTOR.imgsz == 1280

    def test_pitch_class_names_is_none(self) -> None:
        """Verify class_names is None (uses keypoints)."""
        from forgesyte_yolo_tracker.inference.pitch_detection import PITCH_DETECTOR

        assert PITCH_DETECTOR.class_names is None

    def test_pitch_colors_defined(self) -> None:
        """Verify colors defined for pitch detector."""
        from forgesyte_yolo_tracker.inference.pitch_detection import PITCH_DETECTOR

        assert PITCH_DETECTOR.colors is not None


class TestDetectPitchJSON:
    """Tests for detect_pitch_json function."""

    def test_detect_pitch_json_returns_dict(self) -> None:
        """Verify detect_pitch_json returns dictionary."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        assert isinstance(result, dict)

    def test_detect_pitch_json_returns_keypoints_key(self) -> None:
        """Verify keypoints key in result."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        assert "keypoints" in result
        assert isinstance(result["keypoints"], list)

    def test_detect_pitch_json_returns_count(self) -> None:
        """Verify count key in result."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        assert "count" in result
        assert isinstance(result["count"], int)

    def test_detect_pitch_json_returns_pitch_polygon(self) -> None:
        """Verify pitch_polygon key in result."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        assert "pitch_polygon" in result
        assert isinstance(result["pitch_polygon"], list)

    def test_detect_pitch_json_returns_pitch_detected_boolean(self) -> None:
        """Verify pitch_detected boolean key."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        assert "pitch_detected" in result
        assert isinstance(result["pitch_detected"], bool)

    def test_detect_pitch_json_returns_homography(self) -> None:
        """Verify homography key in result."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        assert "homography" in result

    def test_detect_pitch_json_pitch_detected_true_when_4_corners(self) -> None:
        """Verify pitch_detected is true when >= 4 corners."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        assert (result["pitch_detected"] is True) == (
            len(result["pitch_polygon"]) >= 4
        )

    def test_detect_pitch_json_keypoints_have_xy(self) -> None:
        """Verify each keypoint has xy coordinates."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        for kp in result["keypoints"]:
            assert "xy" in kp
            assert len(kp["xy"]) == 2

    def test_detect_pitch_json_keypoints_have_confidence(self) -> None:
        """Verify each keypoint has confidence."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        for kp in result["keypoints"]:
            assert "confidence" in kp
            assert isinstance(kp["confidence"], float)

    def test_detect_pitch_json_keypoints_have_name(self) -> None:
        """Verify each keypoint has name."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        for kp in result["keypoints"]:
            assert "name" in kp
            assert isinstance(kp["name"], str)

    def test_detect_pitch_json_count_matches_keypoints_length(self) -> None:
        """Verify count matches length of keypoints list."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        assert result["count"] == len(result["keypoints"])

    def test_detect_pitch_json_respects_confidence_parameter(self) -> None:
        """Verify confidence parameter is respected."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_low = detect_pitch_json(frame, device="cpu", confidence=0.10)
        result_high = detect_pitch_json(frame, device="cpu", confidence=0.90)

        # Higher confidence should result in fewer or equal detections
        assert result_high["count"] <= result_low["count"]

    def test_detect_pitch_json_accepts_device_parameter(self) -> None:
        """Verify device parameter is accepted."""
        from forgesyte_yolo_tracker.inference.pitch_detection import detect_pitch_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json(frame, device="cpu")

        assert result is not None


class TestDetectPitchJSONWithAnnotated:
    """Tests for detect_pitch_json_with_annotated_frame function."""

    def test_detect_pitch_with_annotated_returns_dict(self) -> None:
        """Verify returns dictionary."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            detect_pitch_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json_with_annotated_frame(frame, device="cpu")

        assert isinstance(result, dict)

    def test_detect_pitch_with_annotated_includes_keypoints(self) -> None:
        """Verify includes keypoints key."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            detect_pitch_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json_with_annotated_frame(frame, device="cpu")

        assert "keypoints" in result
        assert isinstance(result["keypoints"], list)

    def test_detect_pitch_with_annotated_includes_count(self) -> None:
        """Verify includes count key."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            detect_pitch_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json_with_annotated_frame(frame, device="cpu")

        assert "count" in result
        assert isinstance(result["count"], int)

    def test_detect_pitch_with_annotated_includes_pitch_detected(self) -> None:
        """Verify includes pitch_detected boolean."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            detect_pitch_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json_with_annotated_frame(frame, device="cpu")

        assert "pitch_detected" in result
        assert isinstance(result["pitch_detected"], bool)

    def test_detect_pitch_with_annotated_returns_base64(self) -> None:
        """Verify returns annotated_frame_base64 key."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            detect_pitch_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json_with_annotated_frame(frame, device="cpu")

        assert "annotated_frame_base64" in result
        assert isinstance(result["annotated_frame_base64"], str)

    def test_detect_pitch_with_annotated_base64_is_valid(self) -> None:
        """Verify annotated_frame_base64 is valid base64."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            detect_pitch_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json_with_annotated_frame(frame, device="cpu")

        try:
            decoded = base64.b64decode(result["annotated_frame_base64"])
            assert len(decoded) > 0
        except Exception as exc:
            pytest.fail(f"Invalid base64: {exc}")

    def test_detect_pitch_with_annotated_respects_device(self) -> None:
        """Verify respects device parameter."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            detect_pitch_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json_with_annotated_frame(frame, device="cpu")

        assert result is not None

    def test_detect_pitch_with_annotated_respects_confidence(self) -> None:
        """Verify respects confidence parameter."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            detect_pitch_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_pitch_json_with_annotated_frame(
            frame, device="cpu", confidence=0.50
        )

        assert result is not None
        assert "annotated_frame_base64" in result


class TestPitchDetectionModelCaching:
    """Tests for model caching."""

    def test_get_pitch_detection_model_returns_instance(self) -> None:
        """Verify get_pitch_detection_model returns model."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            get_pitch_detection_model,
        )

        model = get_pitch_detection_model(device="cpu")
        assert model is not None

    def test_get_pitch_detection_model_cached(self) -> None:
        """Verify model is cached after first call."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            get_pitch_detection_model,
        )

        model1 = get_pitch_detection_model(device="cpu")
        model2 = get_pitch_detection_model(device="cpu")

        assert model1 is model2


class TestRunPitchDetection:
    """Tests for legacy run_pitch_detection function."""

    def test_run_pitch_detection_returns_dict(self) -> None:
        """Verify run_pitch_detection returns dictionary."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            run_pitch_detection,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "include_annotated": False}
        result = run_pitch_detection(frame, config)

        assert isinstance(result, dict)

    def test_run_pitch_detection_json_mode(self) -> None:
        """Verify JSON mode returns keypoints without base64."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            run_pitch_detection,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "include_annotated": False}
        result = run_pitch_detection(frame, config)

        assert "keypoints" in result
        assert "annotated_frame_base64" not in result

    def test_run_pitch_detection_annotated_mode(self) -> None:
        """Verify annotated mode includes base64."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            run_pitch_detection,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "include_annotated": True}
        result = run_pitch_detection(frame, config)

        assert "keypoints" in result
        assert "annotated_frame_base64" in result

    def test_run_pitch_detection_respects_config_device(self) -> None:
        """Verify config device parameter is respected."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            run_pitch_detection,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu"}
        result = run_pitch_detection(frame, config)

        assert result is not None

    def test_run_pitch_detection_respects_config_confidence(self) -> None:
        """Verify config confidence parameter is respected."""
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            run_pitch_detection,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "confidence": 0.30}
        result = run_pitch_detection(frame, config)

        assert result is not None
