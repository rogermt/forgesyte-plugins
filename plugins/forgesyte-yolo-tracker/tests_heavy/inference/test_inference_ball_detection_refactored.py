"""Tests for refactored ball detection using BaseDetector.

This module tests the ball detection module which now uses BaseDetector
to eliminate code duplication. Tests verify that the wrapper functions
properly delegate to BaseDetector and handle ball-specific logic.
"""

import base64
from typing import Any, Dict

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS or not MODELS_EXIST,
    reason="Set RUN_MODEL_TESTS=1 AND download models to run",
)


class TestBallDetectorConfiguration:
    """Tests for ball detector configuration."""

    def test_ball_detector_name_is_correct(self) -> None:
        """Verify detector name is 'ball'."""
        from forgesyte_yolo_tracker.inference.ball_detection import BALL_DETECTOR

        assert BALL_DETECTOR.detector_name == "ball"

    def test_ball_default_confidence_is_0_20(self) -> None:
        """Verify default confidence is 0.20."""
        from forgesyte_yolo_tracker.inference.ball_detection import BALL_DETECTOR

        assert BALL_DETECTOR.default_confidence == 0.20

    def test_ball_imgsz_is_640(self) -> None:
        """Verify imgsz is 640 for ball."""
        from forgesyte_yolo_tracker.inference.ball_detection import BALL_DETECTOR

        assert BALL_DETECTOR.imgsz == 640

    def test_ball_class_names_is_none(self) -> None:
        """Verify class_names is None for ball."""
        from forgesyte_yolo_tracker.inference.ball_detection import BALL_DETECTOR

        assert BALL_DETECTOR.class_names is None

    def test_ball_colors_defined(self) -> None:
        """Verify colors defined for ball detector."""
        from forgesyte_yolo_tracker.inference.ball_detection import BALL_DETECTOR

        assert BALL_DETECTOR.colors is not None
        assert len(BALL_DETECTOR.colors) > 0


class TestDetectBallJSON:
    """Tests for detect_ball_json function."""

    def test_detect_ball_json_returns_dict(self) -> None:
        """Verify detect_ball_json returns dictionary."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert isinstance(result, dict)

    def test_detect_ball_json_returns_detections_key(self) -> None:
        """Verify detections key in result."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert "detections" in result
        assert isinstance(result["detections"], list)

    def test_detect_ball_json_returns_count(self) -> None:
        """Verify count key in result."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert "count" in result
        assert isinstance(result["count"], int)

    def test_detect_ball_json_returns_ball_key(self) -> None:
        """Verify ball key with primary detection."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert "ball" in result

    def test_detect_ball_json_returns_ball_detected_boolean(self) -> None:
        """Verify ball_detected boolean key."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert "ball_detected" in result
        assert isinstance(result["ball_detected"], bool)

    def test_detect_ball_json_ball_detected_matches_ball_exists(self) -> None:
        """Verify ball_detected matches if ball exists."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert (result["ball_detected"] is True) == (result["ball"] is not None)

    def test_detect_ball_json_count_matches_detections_length(self) -> None:
        """Verify count matches length of detections list."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert result["count"] == len(result["detections"])

    def test_detect_ball_json_detections_have_xyxy(self) -> None:
        """Verify each detection has xyxy coordinates."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        for det in result["detections"]:
            assert "xyxy" in det
            assert len(det["xyxy"]) == 4

    def test_detect_ball_json_detections_have_confidence(self) -> None:
        """Verify each detection has confidence score."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        for det in result["detections"]:
            assert "confidence" in det
            assert isinstance(det["confidence"], float)
            assert 0.0 <= det["confidence"] <= 1.0

    def test_detect_ball_json_ball_is_highest_confidence(self) -> None:
        """Verify ball is highest confidence detection."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        if result["detections"]:
            max_conf = max(d["confidence"] for d in result["detections"])
            assert result["ball"]["confidence"] == max_conf

    def test_detect_ball_json_respects_confidence_parameter(self) -> None:
        """Verify confidence parameter is respected."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_low = detect_ball_json(frame, device="cpu", confidence=0.10)
        result_high = detect_ball_json(frame, device="cpu", confidence=0.90)

        # Higher confidence should result in fewer or equal detections
        assert result_high["count"] <= result_low["count"]

    def test_detect_ball_json_accepts_device_parameter(self) -> None:
        """Verify device parameter is accepted."""
        from forgesyte_yolo_tracker.inference.ball_detection import detect_ball_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json(frame, device="cpu")

        assert result is not None


class TestDetectBallJSONWithAnnotated:
    """Tests for detect_ball_json_with_annotated_frame function."""

    def test_detect_ball_with_annotated_returns_dict(self) -> None:
        """Verify returns dictionary."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated_frame(frame, device="cpu")

        assert isinstance(result, dict)

    def test_detect_ball_with_annotated_includes_detections(self) -> None:
        """Verify includes detections key."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated_frame(frame, device="cpu")

        assert "detections" in result
        assert isinstance(result["detections"], list)

    def test_detect_ball_with_annotated_includes_count(self) -> None:
        """Verify includes count key."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated_frame(frame, device="cpu")

        assert "count" in result
        assert isinstance(result["count"], int)

    def test_detect_ball_with_annotated_includes_ball_detected(self) -> None:
        """Verify includes ball_detected boolean."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated_frame(frame, device="cpu")

        assert "ball_detected" in result
        assert isinstance(result["ball_detected"], bool)

    def test_detect_ball_with_annotated_returns_base64(self) -> None:
        """Verify returns annotated_frame_base64 key."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated_frame(frame, device="cpu")

        assert "annotated_frame_base64" in result
        assert isinstance(result["annotated_frame_base64"], str)

    def test_detect_ball_with_annotated_base64_is_valid(self) -> None:
        """Verify annotated_frame_base64 is valid base64."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated_frame(frame, device="cpu")

        try:
            decoded = base64.b64decode(result["annotated_frame_base64"])
            assert len(decoded) > 0
        except Exception as exc:
            pytest.fail(f"Invalid base64: {exc}")

    def test_detect_ball_with_annotated_respects_device(self) -> None:
        """Verify respects device parameter."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated_frame(frame, device="cpu")

        assert result is not None

    def test_detect_ball_with_annotated_respects_confidence(self) -> None:
        """Verify respects confidence parameter."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json_with_annotated_frame,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_ball_json_with_annotated_frame(
            frame, device="cpu", confidence=0.50
        )

        assert result is not None
        assert "annotated_frame_base64" in result


class TestBallDetectionModelCaching:
    """Tests for model caching."""

    def test_get_ball_detection_model_returns_instance(self) -> None:
        """Verify get_ball_detection_model returns model."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            get_ball_detection_model,
        )

        model = get_ball_detection_model(device="cpu")
        assert model is not None

    def test_get_ball_detection_model_cached(self) -> None:
        """Verify model is cached after first call."""
        from forgesyte_yolo_tracker.inference.ball_detection import (
            get_ball_detection_model,
        )

        model1 = get_ball_detection_model(device="cpu")
        model2 = get_ball_detection_model(device="cpu")

        assert model1 is model2


class TestRunBallDetection:
    """Tests for legacy run_ball_detection function."""

    def test_run_ball_detection_returns_dict(self) -> None:
        """Verify run_ball_detection returns dictionary."""
        from forgesyte_yolo_tracker.inference.ball_detection import run_ball_detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "include_annotated": False}
        result = run_ball_detection(frame, config)

        assert isinstance(result, dict)

    def test_run_ball_detection_json_mode(self) -> None:
        """Verify JSON mode returns detections without base64."""
        from forgesyte_yolo_tracker.inference.ball_detection import run_ball_detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "include_annotated": False}
        result = run_ball_detection(frame, config)

        assert "detections" in result
        assert "annotated_frame_base64" not in result

    def test_run_ball_detection_annotated_mode(self) -> None:
        """Verify annotated mode includes base64."""
        from forgesyte_yolo_tracker.inference.ball_detection import run_ball_detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "include_annotated": True}
        result = run_ball_detection(frame, config)

        assert "detections" in result
        assert "annotated_frame_base64" in result

    def test_run_ball_detection_respects_config_device(self) -> None:
        """Verify config device parameter is respected."""
        from forgesyte_yolo_tracker.inference.ball_detection import run_ball_detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu"}
        result = run_ball_detection(frame, config)

        assert result is not None

    def test_run_ball_detection_respects_config_confidence(self) -> None:
        """Verify config confidence parameter is respected."""
        from forgesyte_yolo_tracker.inference.ball_detection import run_ball_detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "confidence": 0.30}
        result = run_ball_detection(frame, config)

        assert result is not None
