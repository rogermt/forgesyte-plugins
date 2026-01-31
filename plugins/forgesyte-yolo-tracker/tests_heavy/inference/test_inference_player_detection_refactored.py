"""Tests for refactored player detection using BaseDetector.

This module tests the player detection module which now uses BaseDetector
to eliminate code duplication. Tests verify that the wrapper functions
properly delegate to BaseDetector and handle player-specific logic.
"""

import base64
from typing import Any, Dict

import numpy as np
import pytest

from tests_heavy.constants import MODELS_EXIST, RUN_MODEL_TESTS

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS or not MODELS_EXIST,
    reason="Set RUN_MODEL_TESTS=1 AND download models to run",
)


class TestPlayerDetectorConfiguration:
    """Tests for player detector configuration."""

    def test_player_detector_name_is_correct(self) -> None:
        """Verify detector name is 'player'."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            PLAYER_DETECTOR

        assert PLAYER_DETECTOR.detector_name == "player"

    def test_player_default_confidence_is_0_25(self) -> None:
        """Verify default confidence is 0.25."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            PLAYER_DETECTOR

        assert PLAYER_DETECTOR.default_confidence == 0.25

    def test_player_imgsz_is_1280(self) -> None:
        """Verify imgsz is 1280 for players."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            PLAYER_DETECTOR

        assert PLAYER_DETECTOR.imgsz == 1280

    def test_player_class_names_has_4_classes(self) -> None:
        """Verify 4 class names defined."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            PLAYER_DETECTOR

        assert PLAYER_DETECTOR.class_names is not None
        assert len(PLAYER_DETECTOR.class_names) == 4

    def test_player_class_names_includes_ball(self) -> None:
        """Verify 'ball' in class names."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            PLAYER_DETECTOR

        assert PLAYER_DETECTOR.class_names is not None
        assert "ball" in PLAYER_DETECTOR.class_names.values()

    def test_player_class_names_includes_goalkeeper(self) -> None:
        """Verify 'goalkeeper' in class names."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            PLAYER_DETECTOR

        assert PLAYER_DETECTOR.class_names is not None
        assert "goalkeeper" in PLAYER_DETECTOR.class_names.values()

    def test_player_class_names_includes_player(self) -> None:
        """Verify 'player' in class names."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            PLAYER_DETECTOR

        assert PLAYER_DETECTOR.class_names is not None
        assert "player" in PLAYER_DETECTOR.class_names.values()

    def test_player_class_names_includes_referee(self) -> None:
        """Verify 'referee' in class names."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            PLAYER_DETECTOR

        assert PLAYER_DETECTOR.class_names is not None
        assert "referee" in PLAYER_DETECTOR.class_names.values()

    def test_player_colors_defined(self) -> None:
        """Verify colors defined for player detector."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            PLAYER_DETECTOR

        assert PLAYER_DETECTOR.colors is not None
        assert len(PLAYER_DETECTOR.colors) > 0


class TestDetectPlayersJSON:
    """Tests for detect_players_json function."""

    def test_detect_players_json_returns_dict(self) -> None:
        """Verify detect_players_json returns dictionary."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        assert isinstance(result, dict)

    def test_detect_players_json_returns_detections_key(self) -> None:
        """Verify detections key in result."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        assert "detections" in result
        assert isinstance(result["detections"], list)

    def test_detect_players_json_returns_count(self) -> None:
        """Verify count key in result."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        assert "count" in result
        assert isinstance(result["count"], int)

    def test_detect_players_json_returns_classes(self) -> None:
        """Verify classes key in result."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        assert "classes" in result
        assert isinstance(result["classes"], dict)

    def test_detect_players_json_classes_has_all_4_keys(self) -> None:
        """Verify classes dict has all 4 class names."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        classes = result["classes"]
        assert "ball" in classes
        assert "goalkeeper" in classes
        assert "player" in classes
        assert "referee" in classes

    def test_detect_players_json_count_matches_detections_length(self) -> None:
        """Verify count matches length of detections list."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        assert result["count"] == len(result["detections"])

    def test_detect_players_json_detections_have_xyxy(self) -> None:
        """Verify each detection has xyxy coordinates."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        for det in result["detections"]:
            assert "xyxy" in det
            assert len(det["xyxy"]) == 4

    def test_detect_players_json_detections_have_confidence(self) -> None:
        """Verify each detection has confidence score."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        for det in result["detections"]:
            assert "confidence" in det
            assert isinstance(det["confidence"], float)
            assert 0.0 <= det["confidence"] <= 1.0

    def test_detect_players_json_detections_have_class_name(self) -> None:
        """Verify each detection has class_name."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        for det in result["detections"]:
            assert "class_name" in det
            assert det["class_name"] in ["ball", "goalkeeper", "player", "referee"]

    def test_detect_players_json_respects_confidence_parameter(self) -> None:
        """Verify confidence parameter is respected."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result_low = detect_players_json(frame, device="cpu", confidence=0.10)
        result_high = detect_players_json(frame, device="cpu", confidence=0.90)

        # Higher confidence should result in fewer or equal detections
        assert result_high["count"] <= result_low["count"]

    def test_detect_players_json_accepts_device_parameter(self) -> None:
        """Verify device parameter is accepted."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")

        assert result is not None


class TestDetectPlayersJSONWithAnnotated:
    """Tests for detect_players_json_with_annotated_frame function."""

    def test_detect_players_with_annotated_returns_dict(self) -> None:
        """Verify returns dictionary."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json_with_annotated_frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated_frame(frame, device="cpu")

        assert isinstance(result, dict)

    def test_detect_players_with_annotated_includes_detections(self) -> None:
        """Verify includes detections key."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json_with_annotated_frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated_frame(frame, device="cpu")

        assert "detections" in result
        assert isinstance(result["detections"], list)

    def test_detect_players_with_annotated_includes_count(self) -> None:
        """Verify includes count key."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json_with_annotated_frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated_frame(frame, device="cpu")

        assert "count" in result
        assert isinstance(result["count"], int)

    def test_detect_players_with_annotated_includes_classes(self) -> None:
        """Verify includes classes dict."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json_with_annotated_frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated_frame(frame, device="cpu")

        assert "classes" in result
        assert isinstance(result["classes"], dict)

    def test_detect_players_with_annotated_returns_base64(self) -> None:
        """Verify returns annotated_frame_base64 key."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json_with_annotated_frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated_frame(frame, device="cpu")

        assert "annotated_frame_base64" in result
        assert isinstance(result["annotated_frame_base64"], str)

    def test_detect_players_with_annotated_base64_is_valid(self) -> None:
        """Verify annotated_frame_base64 is valid base64."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json_with_annotated_frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated_frame(frame, device="cpu")

        try:
            decoded = base64.b64decode(result["annotated_frame_base64"])
            assert len(decoded) > 0
        except Exception as exc:
            pytest.fail(f"Invalid base64: {exc}")

    def test_detect_players_with_annotated_respects_device(self) -> None:
        """Verify respects device parameter."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json_with_annotated_frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated_frame(frame, device="cpu")

        assert result is not None

    def test_detect_players_with_annotated_respects_confidence(self) -> None:
        """Verify respects confidence parameter."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            detect_players_json_with_annotated_frame

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated_frame(frame, device="cpu", confidence=0.50)

        assert result is not None
        assert "annotated_frame_base64" in result


class TestPlayerDetectionModelCaching:
    """Tests for model caching."""

    def test_get_player_detection_model_returns_instance(self) -> None:
        """Verify get_player_detection_model returns model."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            get_player_detection_model

        model = get_player_detection_model(device="cpu")
        assert model is not None

    def test_get_player_detection_model_cached(self) -> None:
        """Verify model is cached after first call."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            get_player_detection_model

        model1 = get_player_detection_model(device="cpu")
        model2 = get_player_detection_model(device="cpu")

        assert model1 is model2


class TestRunPlayerDetection:
    """Tests for legacy run_player_detection function."""

    def test_run_player_detection_returns_dict(self) -> None:
        """Verify run_player_detection returns dictionary."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            run_player_detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "include_annotated": False}
        result = run_player_detection(frame, config)

        assert isinstance(result, dict)

    def test_run_player_detection_json_mode(self) -> None:
        """Verify JSON mode returns detections without base64."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            run_player_detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "include_annotated": False}
        result = run_player_detection(frame, config)

        assert "detections" in result
        assert "annotated_frame_base64" not in result

    def test_run_player_detection_annotated_mode(self) -> None:
        """Verify annotated mode includes base64."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            run_player_detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "include_annotated": True}
        result = run_player_detection(frame, config)

        assert "detections" in result
        assert "annotated_frame_base64" in result

    def test_run_player_detection_respects_config_device(self) -> None:
        """Verify config device parameter is respected."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            run_player_detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu"}
        result = run_player_detection(frame, config)

        assert result is not None

    def test_run_player_detection_respects_config_confidence(self) -> None:
        """Verify config confidence parameter is respected."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            run_player_detection

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        config: Dict[str, Any] = {"device": "cpu", "confidence": 0.50}
        result = run_player_detection(frame, config)

        assert result is not None
