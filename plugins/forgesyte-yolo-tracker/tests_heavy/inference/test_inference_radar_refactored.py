"""Tests for radar visualization inference module.

This module tests the radar generation functions which create bird's-eye
view visualizations of player and ball positions. Tests verify radar
dimensions, pitch mapping, position visualization, and base64 encoding.

Run with: RUN_MODEL_TESTS=1 pytest src/tests/test_inference_radar_refactored.py -v
"""

import base64
import os
from io import BytesIO

import numpy as np
import pytest

try:
    from PIL import Image
except ImportError:
    Image = None

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
def sample_radar_points() -> list:
    """Create sample radar points."""
    return [
        {"xy": [300, 150], "tracking_id": 1, "team_id": 0, "type": "player"},
        {"xy": [350, 180], "tracking_id": 2, "team_id": 0, "type": "player"},
        {"xy": [250, 120], "tracking_id": 3, "team_id": 1, "type": "player"},
        {"xy": [320, 150], "tracking_id": 4, "team_id": -1, "type": "goalkeeper"},
    ]


# ============================================================================
# 1. Radar Frame Generation (4 tests)
# ============================================================================


class TestRadarFrameGeneration:
    """Tests for radar frame generation."""

    def test_radar_json_returns_dict(self, sample_frame: np.ndarray) -> None:
        """Verify radar_json returns dictionary."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")

        assert isinstance(result, dict)

    def test_radar_has_required_keys(self, sample_frame: np.ndarray) -> None:
        """Verify radar response has required keys."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")

        assert "radar_points" in result
        assert "radar_size" in result
        assert isinstance(result["radar_points"], list)
        assert isinstance(result["radar_size"], list)

    def test_radar_size_is_600x300(self, sample_frame: np.ndarray) -> None:
        """Verify radar size is 600x300 (width, height)."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")
        radar_size = result["radar_size"]

        assert len(radar_size) == 2
        assert radar_size[0] == 600  # width
        assert radar_size[1] == 300  # height

    def test_radar_points_is_list(self, sample_frame: np.ndarray) -> None:
        """Verify radar_points is a list."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")

        assert isinstance(result["radar_points"], list)


# ============================================================================
# 2. Radar Point Structure (5 tests)
# ============================================================================


class TestRadarPointStructure:
    """Tests for radar point structure."""

    def test_radar_point_has_xy(self, sample_frame: np.ndarray) -> None:
        """Verify each radar point has xy coordinates."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")
        points = result["radar_points"]

        for point in points:
            assert "xy" in point
            assert isinstance(point["xy"], list)
            assert len(point["xy"]) == 2

    def test_radar_point_xy_in_bounds(self, sample_frame: np.ndarray) -> None:
        """Verify radar point coordinates are within bounds."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")
        points = result["radar_points"]
        width, height = result["radar_size"]

        for point in points:
            x, y = point["xy"]
            assert 0 <= x <= width, f"X={x} out of bounds [0, {width}]"
            assert 0 <= y <= height, f"Y={y} out of bounds [0, {height}]"

    def test_radar_point_has_tracking_id(self, sample_frame: np.ndarray) -> None:
        """Verify radar points have tracking_id."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")
        points = result["radar_points"]

        for point in points:
            assert "tracking_id" in point
            assert isinstance(point["tracking_id"], (int, np.integer))

    def test_radar_point_has_team_id(self, sample_frame: np.ndarray) -> None:
        """Verify radar points have team_id."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")
        points = result["radar_points"]

        for point in points:
            assert "team_id" in point
            assert point["team_id"] in [-1, 0, 1]

    def test_radar_point_has_type(self, sample_frame: np.ndarray) -> None:
        """Verify radar points have type field."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")
        points = result["radar_points"]

        valid_types = {"player", "goalkeeper", "ball"}
        for point in points:
            assert "type" in point
            assert point["type"] in valid_types


# ============================================================================
# 3. Annotated Radar Frame (2 tests)
# ============================================================================


class TestAnnotatedRadarFrame:
    """Tests for annotated radar frame output."""

    def test_annotated_radar_returns_base64(self, sample_frame: np.ndarray) -> None:
        """Verify annotated radar includes base64."""
        from forgesyte_yolo_tracker.inference.radar import \
            radar_json_with_annotated_frame

        result = radar_json_with_annotated_frame(sample_frame, device="cpu")

        assert "radar_base64" in result
        assert isinstance(result["radar_base64"], str)
        assert len(result["radar_base64"]) > 0

    def test_annotated_radar_base64_valid(self, sample_frame: np.ndarray) -> None:
        """Verify base64 string is valid."""
        from forgesyte_yolo_tracker.inference.radar import \
            radar_json_with_annotated_frame

        result = radar_json_with_annotated_frame(sample_frame, device="cpu")
        b64_str = result["radar_base64"]

        try:
            decoded = base64.b64decode(b64_str)
            assert len(decoded) > 0
        except Exception as e:
            pytest.fail(f"Invalid base64: {e}")


# ============================================================================
# 4. Base64 Decoding (3 tests)
# ============================================================================


class TestBase64Encoding:
    """Tests for base64 encoding and decoding."""

    def test_base64_decodes_to_png(self, sample_frame: np.ndarray) -> None:
        """Verify base64 decodes to PNG image."""
        if Image is None:
            pytest.skip("PIL not available")

        from forgesyte_yolo_tracker.inference.radar import \
            radar_json_with_annotated_frame

        result = radar_json_with_annotated_frame(sample_frame, device="cpu")
        b64_str = result["radar_base64"]

        try:
            decoded = base64.b64decode(b64_str)
            img = Image.open(BytesIO(decoded))
            assert img.format in ["PNG", "JPEG", None]  # None for raw data
        except Exception as e:
            pytest.fail(f"Could not decode base64 to image: {e}")

    def test_base64_string_not_empty(self, sample_frame: np.ndarray) -> None:
        """Verify base64 string has content."""
        from forgesyte_yolo_tracker.inference.radar import \
            radar_json_with_annotated_frame

        result = radar_json_with_annotated_frame(sample_frame, device="cpu")
        b64_str = result["radar_base64"]

        assert len(b64_str) > 100, "Base64 string too short"

    def test_base64_decode_length(self, sample_frame: np.ndarray) -> None:
        """Verify decoded base64 has reasonable size."""
        from forgesyte_yolo_tracker.inference.radar import \
            radar_json_with_annotated_frame

        result = radar_json_with_annotated_frame(sample_frame, device="cpu")
        b64_str = result["radar_base64"]

        decoded = base64.b64decode(b64_str)
        # Radar 600x300 PNG should be at least a few KB
        assert len(decoded) > 1000, f"Decoded data too small: {len(decoded)} bytes"


# ============================================================================
# 5. Confidence Threshold Filtering (2 tests)
# ============================================================================


class TestRadarConfidenceFiltering:
    """Tests for confidence threshold filtering."""

    def test_confidence_parameter_accepted(self, sample_frame: np.ndarray) -> None:
        """Verify confidence parameter is accepted."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu", confidence=0.5)

        assert isinstance(result, dict)
        assert "radar_points" in result

    def test_confidence_levels_affect_detection(self, sample_frame: np.ndarray) -> None:
        """Verify different confidence levels can affect results."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        # Low threshold - may get more detections
        result_low = generate_radar_json(sample_frame, device="cpu", confidence=0.1)
        # High threshold - may get fewer detections
        result_high = generate_radar_json(sample_frame, device="cpu", confidence=0.9)

        # Both should have valid structure
        assert isinstance(result_low["radar_points"], list)
        assert isinstance(result_high["radar_points"], list)


# ============================================================================
# 6. Device Parameter (2 tests)
# ============================================================================


class TestRadarDeviceParameter:
    """Tests for device parameter."""

    def test_device_cpu_accepted(self, sample_frame: np.ndarray) -> None:
        """Verify device='cpu' parameter works."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")

        assert isinstance(result, dict)
        assert "radar_points" in result

    def test_device_cuda_accepted(self, sample_frame: np.ndarray) -> None:
        """Verify device='cuda' parameter accepted (may fall back to cpu)."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        # Should accept cuda parameter (may use cpu if not available)
        try:
            result = generate_radar_json(sample_frame, device="cuda")
            assert isinstance(result, dict)
        except RuntimeError:
            # CUDA not available is acceptable
            pytest.skip("CUDA not available")


# ============================================================================
# 7. JSON Serialization (2 tests)
# ============================================================================


class TestRadarJSONSerialization:
    """Tests for JSON serialization."""

    def test_radar_json_serializable(self, sample_frame: np.ndarray) -> None:
        """Verify radar JSON is serializable."""
        import json

        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")

        try:
            json_str = json.dumps(result)
            assert isinstance(json_str, str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Result not JSON serializable: {e}")

    def test_annotated_radar_json_serializable(self, sample_frame: np.ndarray) -> None:
        """Verify annotated radar JSON is serializable."""
        import json

        from forgesyte_yolo_tracker.inference.radar import \
            radar_json_with_annotated_frame

        result = radar_json_with_annotated_frame(sample_frame, device="cpu")

        try:
            json_str = json.dumps(result)
            assert isinstance(json_str, str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Result not JSON serializable: {e}")


# ============================================================================
# 8. Radar Point Coordinates (3 tests)
# ============================================================================


class TestRadarPointCoordinates:
    """Tests for radar point coordinate format."""

    def test_xy_coordinates_are_numbers(self, sample_frame: np.ndarray) -> None:
        """Verify xy coordinates are numeric."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")
        points = result["radar_points"]

        for point in points:
            x, y = point["xy"]
            assert isinstance(x, (int, float, np.number))
            assert isinstance(y, (int, float, np.number))

    def test_xy_not_nan(self, sample_frame: np.ndarray) -> None:
        """Verify xy coordinates are not NaN."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")
        points = result["radar_points"]

        for point in points:
            x, y = point["xy"]
            assert not np.isnan(x), "X coordinate is NaN"
            assert not np.isnan(y), "Y coordinate is NaN"

    def test_xy_not_infinite(self, sample_frame: np.ndarray) -> None:
        """Verify xy coordinates are not infinite."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")
        points = result["radar_points"]

        for point in points:
            x, y = point["xy"]
            assert not np.isinf(x), "X coordinate is infinite"
            assert not np.isinf(y), "Y coordinate is infinite"


# ============================================================================
# 9. Empty Frame Handling (2 tests)
# ============================================================================


class TestEmptyFrameHandling:
    """Tests for handling empty frames."""

    def test_empty_frame_returns_valid_response(self) -> None:
        """Verify empty frame returns valid response."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = generate_radar_json(empty_frame, device="cpu")

        assert isinstance(result, dict)
        assert "radar_points" in result
        assert isinstance(result["radar_points"], list)

    def test_empty_frame_with_annotated(self) -> None:
        """Verify empty frame returns annotated frame."""
        from forgesyte_yolo_tracker.inference.radar import \
            radar_json_with_annotated_frame

        empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = radar_json_with_annotated_frame(empty_frame, device="cpu")

        assert isinstance(result, dict)
        assert "radar_base64" in result
        assert isinstance(result["radar_base64"], str)


# ============================================================================
# 10. Point Type Validation (2 tests)
# ============================================================================


class TestPointTypeValidation:
    """Tests for point type field validation."""

    def test_point_types_are_valid(self, sample_frame: np.ndarray) -> None:
        """Verify point types are in valid set."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")
        points = result["radar_points"]

        valid_types = {"player", "goalkeeper", "ball"}
        for point in points:
            assert point["type"] in valid_types, f"Invalid type: {point['type']}"

    def test_goalkeeper_type_has_valid_team(self, sample_frame: np.ndarray) -> None:
        """Verify goalkeeper points have valid team_id."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        result = generate_radar_json(sample_frame, device="cpu")
        points = result["radar_points"]

        for point in points:
            if point["type"] == "goalkeeper":
                # Goalkeepers should have team_id -1 or valid team
                assert point["team_id"] in [-1, 0, 1]
