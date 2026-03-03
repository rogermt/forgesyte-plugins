"""Tests for JSON sanitizer module.

Phase 1 of v0.10.0 implementation.
Tests for converting NumPy types to JSON-safe Python types.
"""

import numpy as np
import json


class TestSanitizeJson:
    """Unit tests for JSON sanitizer."""

    def test_sanitize_numpy_float32(self) -> None:
        """Test np.float32 → Python float conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.float32(1.23)
        result = sanitize_json(input_val)

        # Use approximate equality due to float32 precision
        assert abs(result - 1.23) < 1e-6
        assert type(result) is float  # noqa: E721
        assert not isinstance(result, np.floating)

    def test_sanitize_numpy_float64(self) -> None:
        """Test np.float64 → Python float conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.float64(3.14159)
        result = sanitize_json(input_val)

        assert result == 3.14159
        assert type(result) is float  # noqa: E721

    def test_sanitize_numpy_int32(self) -> None:
        """Test np.int32 → Python int conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.int32(42)
        result = sanitize_json(input_val)

        assert result == 42
        assert type(result) is int  # noqa: E721
        assert not isinstance(result, np.integer)

    def test_sanitize_numpy_int64(self) -> None:
        """Test np.int64 → Python int conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.int64(9999999999)
        result = sanitize_json(input_val)

        assert result == 9999999999
        assert type(result) is int  # noqa: E721

    def test_sanitize_numpy_array(self) -> None:
        """Test np.ndarray → Python list conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.array([1, 2, 3])
        result = sanitize_json(input_val)

        assert result == [1, 2, 3]
        assert type(result) is list  # noqa: E721

    def test_sanitize_numpy_array_of_floats(self) -> None:
        """Test np.ndarray of floats → list of Python floats."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.array([1.1, 2.2, 3.3], dtype=np.float32)
        result = sanitize_json(input_val)

        # Use approximate equality due to float32 precision
        assert len(result) == 3
        assert abs(result[0] - 1.1) < 1e-6
        assert abs(result[1] - 2.2) < 1e-6
        assert abs(result[2] - 3.3) < 1e-6
        assert all(type(x) is float for x in result)  # noqa: E721

    def test_sanitize_nested_dict(self) -> None:
        """Test nested dict with NumPy values."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = {
            "a": np.float32(1.0),
            "b": np.int64(2),
            "c": "string",
        }
        result = sanitize_json(input_val)

        assert result == {"a": 1.0, "b": 2, "c": "string"}
        assert type(result["a"]) is float  # noqa: E721
        assert type(result["b"]) is int  # noqa: E721

    def test_sanitize_nested_list(self) -> None:
        """Test nested list with NumPy values."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = [np.float32(1.0), np.int64(2), "string"]
        result = sanitize_json(input_val)

        assert result == [1.0, 2, "string"]
        assert type(result[0]) is float  # noqa: E721
        assert type(result[1]) is int  # noqa: E721

    def test_sanitize_deeply_nested(self) -> None:
        """Test deeply nested structure matching plugin output."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = {
            "frames": [
                {
                    "detections": {
                        "tracked_objects": [
                            {
                                "track_id": np.int64(1),
                                "center": np.array([np.float32(100.5), np.float32(200.5)]),
                                "xyxy": np.array([50, 100, 150, 300]),
                            }
                        ]
                    }
                }
            ]
        }
        result = sanitize_json(input_val)

        # Verify structure
        assert "frames" in result
        assert len(result["frames"]) == 1
        obj = result["frames"][0]["detections"]["tracked_objects"][0]

        # Verify types
        assert type(obj["track_id"]) is int  # noqa: E721
        assert type(obj["center"]) is list  # noqa: E721
        assert type(obj["center"][0]) is float  # noqa: E721
        assert type(obj["xyxy"]) is list  # noqa: E721

    def test_sanitize_2d_array(self) -> None:
        """Test 2D NumPy array conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.array([[1, 2], [3, 4]])
        result = sanitize_json(input_val)

        assert result == [[1, 2], [3, 4]]
        assert type(result) is list  # noqa: E721
        assert type(result[0]) is list  # noqa: E721

    def test_sanitize_preserves_python_primitives(self) -> None:
        """Test that Python primitives pass through unchanged."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = {
            "int": 42,
            "float": 3.14,
            "str": "hello",
            "bool": True,
            "none": None,
        }
        result = sanitize_json(input_val)

        assert result == input_val

    def test_sanitize_empty_structures(self) -> None:
        """Test empty dict and list."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        assert sanitize_json({}) == {}
        assert sanitize_json([]) == []

    def test_json_serializable_after_sanitize(self) -> None:
        """Test that sanitized output is JSON-serializable."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = {
            "numpy_float": np.float32(1.23),
            "numpy_int": np.int64(42),
            "numpy_array": np.array([1, 2, 3]),
            "nested": {
                "values": np.array([np.float32(1.0), np.float32(2.0)])
            }
        }
        result = sanitize_json(input_val)

        # Should not raise
        json_str = json.dumps(result)

        # Should be deserializable
        loaded = json.loads(json_str)
        # Use approximate equality for float32 precision
        assert abs(loaded["numpy_float"] - 1.23) < 1e-6
        assert loaded["numpy_int"] == 42

    def test_sanitize_tuple_becomes_list(self) -> None:
        """Test that tuples are converted to lists."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = (np.int64(1), np.int64(2), np.int64(3))
        result = sanitize_json(input_val)

        assert result == [1, 2, 3]
        assert type(result) is list  # noqa: E721

    def test_sanitize_xyxy_bounding_box(self) -> None:
        """Test typical bounding box array (xyxy format)."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = {
            "xyxy": np.array([50.5, 100.25, 150.75, 200.5], dtype=np.float32)
        }
        result = sanitize_json(input_val)

        assert result["xyxy"] == [50.5, 100.25, 150.75, 200.5]
        assert all(type(x) is float for x in result["xyxy"])  # noqa: E721

    def test_sanitize_center_coordinates(self) -> None:
        """Test center coordinate calculation result."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        xyxy = np.array([100.0, 200.0, 300.0, 400.0])
        center = np.array([
            float((xyxy[0] + xyxy[2]) / 2),
            float((xyxy[1] + xyxy[3]) / 2)
        ], dtype=np.float32)

        result = sanitize_json({"center": center})

        assert result["center"] == [200.0, 300.0]
        assert type(result["center"][0]) is float  # noqa: E721


class TestSanizeJsonEdgeCases:
    """Edge case tests for JSON sanitizer."""

    def test_sanitize_numpy_bool(self) -> None:
        """Test np.bool_ conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.bool_(True)
        result = sanitize_json(input_val)

        assert result is True
        assert type(result) is bool  # noqa: E721

    def test_sanitize_large_array(self) -> None:
        """Test large array performance."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.random.rand(1000, 4).astype(np.float32)
        result = sanitize_json(input_val)

        assert len(result) == 1000
        assert len(result[0]) == 4
        assert all(type(x) is float for row in result for x in row)  # noqa: E721

    def test_sanitize_numpy_uint8(self) -> None:
        """Test np.uint8 → Python int conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.uint8(255)
        result = sanitize_json(input_val)

        assert result == 255
        assert type(result) is int  # noqa: E721

    def test_sanitize_numpy_uint16(self) -> None:
        """Test np.uint16 → Python int conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.uint16(65535)
        result = sanitize_json(input_val)

        assert result == 65535
        assert type(result) is int  # noqa: E721

    def test_sanitize_numpy_int8(self) -> None:
        """Test np.int8 → Python int conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.int8(-128)
        result = sanitize_json(input_val)

        assert result == -128
        assert type(result) is int  # noqa: E721

    def test_sanitize_numpy_int16(self) -> None:
        """Test np.int16 → Python int conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.int16(-32768)
        result = sanitize_json(input_val)

        assert result == -32768
        assert type(result) is int  # noqa: E721

    def test_sanitize_numpy_float16(self) -> None:
        """Test np.float16 → Python float conversion."""
        from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json

        input_val = np.float16(1.5)
        result = sanitize_json(input_val)

        assert result == 1.5
        assert type(result) is float  # noqa: E721
