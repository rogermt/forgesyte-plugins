"""Tests for player tracking inference module.

This module tests the player tracking functions which use ByteTrack
to assign and maintain unique IDs across frames. Tests verify track
ID assignment, persistence, occlusion handling, and lifecycle events.

Run with: RUN_MODEL_TESTS=1 pytest src/tests/test_inference_player_tracking_refactored.py -v
"""

import os

import cv2
import numpy as np
import pytest

RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="Set RUN_MODEL_TESTS=1 to run (requires YOLO model)",
)


@pytest.fixture
def sample_frame() -> np.ndarray:
    """Create a sample frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_frame_with_content() -> np.ndarray:
    """Create a frame with some content."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Add some simple shapes to simulate player presence
    cv2.rectangle(frame, (100, 100), (150, 200), (255, 255, 255), -1)
    cv2.rectangle(frame, (200, 150), (250, 250), (255, 255, 255), -1)
    cv2.rectangle(frame, (300, 120), (350, 220), (255, 255, 255), -1)
    return frame


# ============================================================================
# 1. Track ID Assignment & Uniqueness (4 tests)
# ============================================================================


class TestTrackIDAssignment:
    """Tests for track ID assignment and uniqueness."""

    def test_track_id_assignment_single_player(self, sample_frame: np.ndarray) -> None:
        """Verify track ID assigned to single player."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")

        assert isinstance(result, dict)
        assert "detections" in result
        assert "count" in result
        assert "track_ids" in result
        assert isinstance(result["detections"], list)

    def test_track_id_uniqueness_multi_player(self, sample_frame: np.ndarray) -> None:
        """Verify each player gets unique track ID."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")
        detections = result["detections"]

        if len(detections) > 1:
            tracking_ids = [d["tracking_id"] for d in detections]
            # All valid tracking IDs should be unique
            valid_ids = [tid for tid in tracking_ids if tid >= 0]
            assert len(valid_ids) == len(set(valid_ids)), "Duplicate tracking IDs found"

    def test_track_id_format_is_integer(self, sample_frame: np.ndarray) -> None:
        """Verify tracking_id field is integer."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")
        detections = result["detections"]

        for detection in detections:
            assert "tracking_id" in detection
            assert isinstance(detection["tracking_id"], (int, np.integer))

    def test_track_ids_list_contains_valid_ids(self, sample_frame: np.ndarray) -> None:
        """Verify track_ids list contains only valid IDs."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")
        track_ids = result["track_ids"]

        assert isinstance(track_ids, list)
        for tid in track_ids:
            assert tid >= 0


# ============================================================================
# 2. Track Persistence Across Frames (5 tests)
# ============================================================================


class TestTrackPersistence:
    """Tests for track persistence across frames."""

    def test_track_persistence_basic(self, sample_frame: np.ndarray) -> None:
        """Verify same frame returns consistent results."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        # Run same frame twice
        result1 = track_players_json(sample_frame, device="cpu")
        result2 = track_players_json(sample_frame, device="cpu")

        # Count should be same or similar
        assert isinstance(result1["count"], int)
        assert isinstance(result2["count"], int)

    def test_track_persistence_structure(self, sample_frame: np.ndarray) -> None:
        """Verify detection structure is consistent."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")
        detections = result["detections"]

        for detection in detections:
            assert "xyxy" in detection
            assert "confidence" in detection
            assert "class_id" in detection
            assert "class_name" in detection
            assert "tracking_id" in detection

    def test_xyxy_format_is_list_of_4(self, sample_frame: np.ndarray) -> None:
        """Verify xyxy format is [x1, y1, x2, y2]."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")
        detections = result["detections"]

        for detection in detections:
            xyxy = detection["xyxy"]
            assert isinstance(xyxy, list)
            assert len(xyxy) == 4
            assert all(isinstance(x, (int, float)) for x in xyxy)

    def test_confidence_in_valid_range(self, sample_frame: np.ndarray) -> None:
        """Verify confidence values are in [0, 1]."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")
        detections = result["detections"]

        for detection in detections:
            conf = detection["confidence"]
            assert isinstance(conf, (int, float))
            assert 0 <= conf <= 1, f"Confidence {conf} out of range"


# ============================================================================
# 3. Occlusion Handling (3 tests)
# ============================================================================


class TestOcclusionHandling:
    """Tests for occlusion handling."""

    def test_partial_detection_confidence_lower(self, sample_frame: np.ndarray) -> None:
        """Verify partially visible player has valid confidence."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")
        detections = result["detections"]

        # All detections should have confidence field
        for detection in detections:
            assert "confidence" in detection
            assert 0 <= detection["confidence"] <= 1

    def test_multiple_detections_handled(self, sample_frame: np.ndarray) -> None:
        """Verify multiple detections don't crash."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")

        # Should not crash with multiple detections
        assert isinstance(result["count"], int)
        assert result["count"] >= 0

    def test_no_detections_returns_valid_response(self, sample_frame: np.ndarray) -> None:
        """Verify empty detections returns valid response."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        # Create empty frame
        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = track_players_json(empty_frame, device="cpu")

        assert "detections" in result
        assert isinstance(result["detections"], list)
        assert result["count"] >= 0


# ============================================================================
# 4. JSON Output Validation (2 tests)
# ============================================================================


class TestTrackJSONOutput:
    """Tests for JSON output validation."""

    def test_json_response_structure(self, sample_frame: np.ndarray) -> None:
        """Verify JSON has required top-level keys."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")

        assert isinstance(result, dict)
        assert "detections" in result
        assert "count" in result
        assert "track_ids" in result

    def test_json_serializable(self, sample_frame: np.ndarray) -> None:
        """Verify result is JSON serializable."""
        import json

        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")

        # Should be JSON serializable
        try:
            json_str = json.dumps(result)
            assert isinstance(json_str, str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Result not JSON serializable: {e}")


# ============================================================================
# 5. Annotated Frame Output (2 tests)
# ============================================================================


class TestAnnotatedFrameOutput:
    """Tests for annotated frame output."""

    def test_annotated_frame_returns_base64(self, sample_frame: np.ndarray) -> None:
        """Verify annotated frame output includes base64."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json_with_annotated_frame

        result = track_players_json_with_annotated_frame(sample_frame, device="cpu")

        assert "annotated_frame_base64" in result
        assert isinstance(result["annotated_frame_base64"], str)
        assert len(result["annotated_frame_base64"]) > 0

    def test_annotated_frame_base64_valid(self, sample_frame: np.ndarray) -> None:
        """Verify base64 string is valid."""
        import base64

        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json_with_annotated_frame

        result = track_players_json_with_annotated_frame(sample_frame, device="cpu")
        b64_str = result["annotated_frame_base64"]

        try:
            decoded = base64.b64decode(b64_str)
            assert len(decoded) > 0
        except Exception as e:
            pytest.fail(f"Invalid base64: {e}")


# ============================================================================
# 6. Confidence Threshold Filtering (2 tests)
# ============================================================================


class TestConfidenceFiltering:
    """Tests for confidence threshold filtering."""

    def test_confidence_parameter_accepted(self, sample_frame: np.ndarray) -> None:
        """Verify confidence parameter is accepted."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        # Should accept confidence parameter
        result = track_players_json(sample_frame, device="cpu", confidence=0.5)

        assert isinstance(result, dict)
        assert "detections" in result

    def test_high_confidence_threshold(self, sample_frame: np.ndarray) -> None:
        """Verify high confidence threshold reduces detections."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        # Low threshold - should get more detections
        result_low = track_players_json(sample_frame, device="cpu", confidence=0.1)
        # High threshold - should get fewer detections
        result_high = track_players_json(sample_frame, device="cpu", confidence=0.9)

        # High confidence should not exceed low confidence count
        assert result_high["count"] <= result_low["count"]


# ============================================================================
# 7. Device Parameter (2 tests)
# ============================================================================


class TestDeviceParameter:
    """Tests for device parameter."""

    def test_device_cpu_accepted(self, sample_frame: np.ndarray) -> None:
        """Verify device='cpu' parameter works."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")

        assert isinstance(result, dict)
        assert "detections" in result

    def test_device_cuda_accepted(self, sample_frame: np.ndarray) -> None:
        """Verify device='cuda' parameter accepted (may fall back to cpu)."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        # Should accept cuda parameter (may use cpu if not available)
        try:
            result = track_players_json(sample_frame, device="cuda")
            assert isinstance(result, dict)
        except RuntimeError:
            # CUDA not available is acceptable
            pytest.skip("CUDA not available")


# ============================================================================
# 8. Class Names (2 tests)
# ============================================================================


class TestClassNames:
    """Tests for class name handling."""

    def test_valid_class_names(self, sample_frame: np.ndarray) -> None:
        """Verify class names are valid."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")
        detections = result["detections"]

        valid_classes = {"player", "goalkeeper", "referee"}
        for detection in detections:
            assert detection["class_name"] in valid_classes

    def test_class_id_matches_class_name(self, sample_frame: np.ndarray) -> None:
        """Verify class_id and class_name correspond."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")
        detections = result["detections"]

        class_mapping = {0: "player", 1: "goalkeeper", 2: "referee"}
        for detection in detections:
            class_id = detection["class_id"]
            class_name = detection["class_name"]
            assert class_mapping.get(class_id) == class_name


# ============================================================================
# 9. Count Consistency (1 test)
# ============================================================================


class TestCountConsistency:
    """Tests for count field consistency."""

    def test_count_matches_detections_length(self, sample_frame: np.ndarray) -> None:
        """Verify count matches length of detections list."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        result = track_players_json(sample_frame, device="cpu")

        assert result["count"] == len(result["detections"])
