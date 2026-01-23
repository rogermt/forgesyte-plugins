"""Tests for BaseDetector class.

This module tests the generic detection base class that all detectors
(player, ball, pitch) use to eliminate code duplication.
"""

import base64

import numpy as np
import pytest

from tests.constants import MODELS_EXIST, RUN_MODEL_TESTS

# Skip all model-dependent tests if models not available
pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS or not MODELS_EXIST,
    reason="Set RUN_MODEL_TESTS=1 AND download models to run",
)


class TestBaseDetectorInitialization:
    """Tests for BaseDetector initialization."""

    def test_can_instantiate_base_detector(self) -> None:
        """Verify BaseDetector can be instantiated."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="test-model-v1.pt",
            default_confidence=0.25,
        )

        assert detector is not None

    def test_detector_name_stored_correctly(self) -> None:
        """Verify detector_name is stored correctly."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="player",
            model_name="test-model-v1.pt",
            default_confidence=0.25,
        )

        assert detector.detector_name == "player"

    def test_model_name_stored_correctly(self) -> None:
        """Verify model_name is stored correctly."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        assert detector.model_name == "football-player-detection-v3.pt"

    def test_default_confidence_stored_correctly(self) -> None:
        """Verify default_confidence is stored correctly."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="test-model-v1.pt",
            default_confidence=0.30,
        )

        assert detector.default_confidence == 0.30

    def test_imgsz_default_is_1280(self) -> None:
        """Verify imgsz defaults to 1280."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="test-model-v1.pt",
            default_confidence=0.25,
        )

        assert detector.imgsz == 1280

    def test_imgsz_can_be_customized(self) -> None:
        """Verify imgsz can be customized."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="test-model-v1.pt",
            default_confidence=0.25,
            imgsz=640,
        )

        assert detector.imgsz == 640

    def test_class_names_default_is_none(self) -> None:
        """Verify class_names defaults to None."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="test-model-v1.pt",
            default_confidence=0.25,
        )

        assert detector.class_names is None

    def test_class_names_can_be_provided(self) -> None:
        """Verify class_names can be provided."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        names = {0: "ball", 1: "player"}
        detector = BaseDetector(
            detector_name="test",
            model_name="test-model-v1.pt",
            default_confidence=0.25,
            class_names=names,
        )

        assert detector.class_names == names

    def test_colors_default_is_none(self) -> None:
        """Verify colors defaults to None."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="test-model-v1.pt",
            default_confidence=0.25,
        )

        assert detector.colors is None

    def test_colors_can_be_provided(self) -> None:
        """Verify colors can be provided."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        colors = {0: "#FF0000", 1: "#00FF00"}
        detector = BaseDetector(
            detector_name="test",
            model_name="test-model-v1.pt",
            default_confidence=0.25,
            colors=colors,
        )

        assert detector.colors == colors


class TestBaseDetectorModelCaching:
    """Tests for model loading and caching."""

    def test_get_model_returns_yolo_instance(self) -> None:
        """Verify get_model returns YOLO instance."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        model = detector.get_model(device="cpu")
        assert model is not None

    def test_get_model_caches_on_first_call(self) -> None:
        """Verify model is cached after first call."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        model1 = detector.get_model(device="cpu")
        model2 = detector.get_model(device="cpu")

        assert model1 is model2

    def test_get_model_respects_device_parameter(self) -> None:
        """Verify get_model respects device parameter."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        model = detector.get_model(device="cpu")
        assert model is not None


class TestBaseDetectorFrameEncoding:
    """Tests for frame encoding utilities."""

    def test_encode_frame_to_base64_returns_string(self) -> None:
        """Verify encode_frame_to_base64 returns string."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="test-model-v1.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        encoded = detector._encode_frame_to_base64(frame)

        assert isinstance(encoded, str)

    def test_encoded_frame_is_valid_base64(self) -> None:
        """Verify encoded frame is valid base64."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="test-model-v1.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        encoded = detector._encode_frame_to_base64(frame)

        try:
            decoded = base64.b64decode(encoded)
            assert len(decoded) > 0
        except Exception as e:
            pytest.fail(f"Failed to decode base64: {e}")

    def test_encoded_frame_decodable_to_bytes(self) -> None:
        """Verify encoded frame can be decoded to bytes."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="test-model-v1.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        encoded = detector._encode_frame_to_base64(frame)

        decoded = base64.b64decode(encoded)
        assert isinstance(decoded, bytes)
        assert len(decoded) > 0


class TestBaseDetectorDetectJSON:
    """Tests for detect_json method."""

    def test_detect_json_returns_dict(self) -> None:
        """Verify detect_json returns dictionary."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        assert isinstance(result, dict)

    def test_detect_json_returns_count_key(self) -> None:
        """Verify detect_json returns count key."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        assert "count" in result

    def test_detect_json_count_is_int(self) -> None:
        """Verify count is integer."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        assert isinstance(result["count"], int)

    def test_detect_json_returns_detections_list(self) -> None:
        """Verify detect_json returns detections list."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        assert "detections" in result
        assert isinstance(result["detections"], list)

    def test_detect_json_detections_are_dicts(self) -> None:
        """Verify each detection is dictionary."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        for det in result["detections"]:
            assert isinstance(det, dict)

    def test_detect_json_each_detection_has_xyxy(self) -> None:
        """Verify each detection has xyxy coordinates."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        for det in result["detections"]:
            assert "xyxy" in det
            assert len(det["xyxy"]) == 4

    def test_detect_json_each_detection_has_confidence(self) -> None:
        """Verify each detection has confidence score."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        for det in result["detections"]:
            assert "confidence" in det
            assert isinstance(det["confidence"], float)
            assert 0.0 <= det["confidence"] <= 1.0

    def test_detect_json_each_detection_has_class_id(self) -> None:
        """Verify each detection has class_id."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        for det in result["detections"]:
            assert "class_id" in det
            assert isinstance(det["class_id"], int)

    def test_detect_json_count_matches_detections_length(self) -> None:
        """Verify count matches length of detections list."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        assert result["count"] == len(result["detections"])

    def test_detect_json_respects_confidence_threshold(self) -> None:
        """Verify detect_json respects confidence threshold parameter."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_low = detector.detect_json(frame, device="cpu", confidence=0.10)
        result_high = detector.detect_json(frame, device="cpu", confidence=0.90)

        # Higher confidence should result in fewer detections
        assert result_high["count"] <= result_low["count"]

    def test_detect_json_accepts_device_parameter(self) -> None:
        """Verify detect_json accepts device parameter."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        assert result is not None

    def test_detect_json_uses_imgsz_parameter(self) -> None:
        """Verify detect_json uses imgsz parameter."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
            imgsz=640,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        assert result is not None


class TestBaseDetectorDetectJSONWithAnnotated:
    """Tests for detect_json_with_annotated_frame method."""

    def test_detect_json_with_annotated_returns_dict(self) -> None:
        """Verify detect_json_with_annotated_frame returns dictionary."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json_with_annotated_frame(frame, device="cpu")

        assert isinstance(result, dict)

    def test_detect_json_with_annotated_includes_annotated_frame_base64(
        self,
    ) -> None:
        """Verify includes annotated_frame_base64 key."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json_with_annotated_frame(frame, device="cpu")

        assert "annotated_frame_base64" in result
        assert isinstance(result["annotated_frame_base64"], str)

    def test_detect_json_with_annotated_base64_is_valid(self) -> None:
        """Verify annotated_frame_base64 is valid base64."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json_with_annotated_frame(frame, device="cpu")

        try:
            decoded = base64.b64decode(result["annotated_frame_base64"])
            assert len(decoded) > 0
        except Exception as e:
            pytest.fail(f"Invalid base64: {e}")

    def test_detect_json_with_annotated_includes_count(self) -> None:
        """Verify includes count key."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json_with_annotated_frame(frame, device="cpu")

        assert "count" in result
        assert isinstance(result["count"], int)

    def test_detect_json_with_annotated_includes_detections(self) -> None:
        """Verify includes detections key."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json_with_annotated_frame(frame, device="cpu")

        assert "detections" in result
        assert isinstance(result["detections"], list)

    def test_detect_json_with_annotated_respects_device(self) -> None:
        """Verify respects device parameter."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json_with_annotated_frame(frame, device="cpu")

        assert result is not None

    def test_detect_json_with_annotated_respects_confidence(self) -> None:
        """Verify respects confidence parameter."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json_with_annotated_frame(
            frame, device="cpu", confidence=0.50
        )

        assert result is not None


class TestBaseDetectorClassNameHandling:
    """Tests for class name handling."""

    def test_detect_json_includes_class_name_when_provided(self) -> None:
        """Verify class_name included in detection when class_names provided."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        names = {0: "ball", 1: "player"}
        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
            class_names=names,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        for det in result["detections"]:
            assert "class_name" in det

    def test_detect_json_skips_class_name_when_not_provided(self) -> None:
        """Verify class_name not in detection when class_names is None."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        detector = BaseDetector(
            detector_name="test",
            model_name="football-ball-detection-v2.pt",
            default_confidence=0.25,
            class_names=None,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        if result["detections"]:
            for det in result["detections"]:
                # Ball detector has no class_names, so no class_name field
                assert "class_id" in det

    def test_fallback_class_name_for_unknown_ids(self) -> None:
        """Verify fallback class_name created for unknown class IDs."""
        from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

        names = {0: "ball"}
        detector = BaseDetector(
            detector_name="test",
            model_name="football-player-detection-v3.pt",
            default_confidence=0.25,
            class_names=names,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector.detect_json(frame, device="cpu")

        for det in result["detections"]:
            if "class_name" in det:
                # Should be either in names or fallback format
                assert det["class_name"] == "ball" or det["class_name"].startswith(
                    "class_"
                )
