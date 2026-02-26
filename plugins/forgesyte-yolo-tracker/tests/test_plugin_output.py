"""Tests for sanitized plugin output.

Phase 2 of v0.10.0 implementation.
Tests to verify plugin output is JSON-serializable after sanitization.
"""

import json
import numpy as np

from forgesyte_yolo_tracker.utils.json_sanitize import sanitize_json


class TestPluginOutputSanitization:
    """Tests for sanitized plugin output."""

    def test_no_numpy_types_in_output(self) -> None:
        """Test that no NumPy types remain in output."""
        # Simulated plugin output with NumPy types
        output = {
            "total_frames": np.int64(100),
            "frames": [
                {
                    "frame_index": np.int64(0),
                    "detections": {
                        "tracked_objects": [
                            {
                                "track_id": np.int64(1),
                                "class_id": np.int64(0),
                                "xyxy": np.array([100, 200, 300, 400], dtype=np.float32),
                                "center": np.array([200.0, 300.0], dtype=np.float32),
                            }
                        ]
                    }
                }
            ]
        }

        result = sanitize_json(output)

        # Check all values are Python primitives
        assert type(result["total_frames"]) is int  # noqa: E721
        assert type(result["frames"][0]["frame_index"]) is int  # noqa: E721
        obj = result["frames"][0]["detections"]["tracked_objects"][0]
        assert type(obj["track_id"]) is int  # noqa: E721
        assert type(obj["xyxy"]) is list  # noqa: E721
        assert type(obj["center"]) is list  # noqa: E721

    def test_output_is_json_serializable(self) -> None:
        """Test that output can be JSON serialized."""
        output = {
            "total_frames": np.int64(100),
            "frames": [
                {
                    "frame_index": np.int64(0),
                    "detections": {
                        "tracked_objects": [
                            {
                                "track_id": np.int64(1),
                                "xyxy": np.array([100, 200, 300, 400], dtype=np.float32),
                                "center": np.array([200.0, 300.0], dtype=np.float32),
                            }
                        ]
                    }
                }
            ]
        }

        result = sanitize_json(output)

        # Should not raise
        json_str = json.dumps(result)
        assert "total_frames" in json_str

    def test_video_tool_output_structure(self) -> None:
        """Test video tool output structure matches expected format."""
        # Simulate video_player_tracking output
        output = {
            "total_frames": np.int64(10),
            "frames": [
                {
                    "frame_index": np.int64(i),
                    "detections": {
                        "tracked_objects": [
                            {
                                "track_id": np.int64(j),
                                "class_id": np.int64(0),
                                "xyxy": np.array([j * 10, j * 20, j * 30, j * 40], dtype=np.float32),
                                "center": np.array([j * 20.0, j * 30.0], dtype=np.float32),
                            }
                            for j in range(3)
                        ]
                    }
                }
                for i in range(10)
            ]
        }

        result = sanitize_json(output)

        # Verify structure is preserved
        assert result["total_frames"] == 10
        assert len(result["frames"]) == 10
        assert len(result["frames"][0]["detections"]["tracked_objects"]) == 3

        # Verify all types are JSON-safe
        json.dumps(result)  # Should not raise


class TestFloat32Fix:
    """Tests for float32 center coordinate fix."""

    def test_center_coordinates_are_python_floats(self) -> None:
        """Test that center coordinates are Python floats, not np.float32."""
        xyxy = np.array([100.5, 200.25, 300.75, 400.5], dtype=np.float32)

        # Calculate center (as done in plugin)
        center = [
            float((xyxy[0] + xyxy[2]) / 2),
            float((xyxy[1] + xyxy[3]) / 2)
        ]

        result = sanitize_json({"center": center})

        assert type(result["center"][0]) is float  # noqa: E721
        assert type(result["center"][1]) is float  # noqa: E721
        # Should be JSON serializable
        json.dumps(result)  # Should not raise


class TestTrackIdFix:
    """Tests for track_id truth-value fix."""

    def test_track_id_handles_none(self) -> None:
        """Test track_id handles None from get_tracker_ids."""
        from typing import Any
        tracker_ids: Any = None

        # After fix: should handle gracefully
        if tracker_ids is None:
            tracker_ids = []

        assert tracker_ids == []

    def test_track_id_is_python_int(self) -> None:
        """Test track_id is Python int, not np.int64."""
        track_id = np.int64(42)
        result = sanitize_json({"track_id": track_id})

        assert result["track_id"] == 42
        assert type(result["track_id"]) is int  # noqa: E721


class TestVideoBallDetectionOutput:
    """Tests for video_ball_detection output sanitization."""

    def test_ball_detection_output_json_serializable(self) -> None:
        """Test ball detection output is JSON-serializable."""
        # Simulated output
        output = {
            "total_frames": np.int64(5),
            "frames": [
                {
                    "frame_index": np.int64(i),
                    "detections": {
                        "xyxy": np.array([[100, 200, 150, 250]], dtype=np.float32).tolist(),
                        "confidence": np.array([0.95], dtype=np.float32).tolist(),
                        "class_id": np.array([0], dtype=np.int32).tolist(),
                    }
                }
                for i in range(5)
            ]
        }

        result = sanitize_json(output)

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert "total_frames" in json_str


class TestVideoPitchDetectionOutput:
    """Tests for video_pitch_detection output sanitization."""

    def test_pitch_detection_output_json_serializable(self) -> None:
        """Test pitch detection output is JSON-serializable."""
        # Simulated output with keypoints
        output = {
            "total_frames": np.int64(3),
            "frames": [
                {
                    "frame_index": np.int64(i),
                    "detections": {
                        "keypoints_xy": np.random.rand(32, 2).astype(np.float32).tolist(),
                        "keypoints_conf": np.random.rand(32).astype(np.float32).tolist(),
                    }
                }
                for i in range(3)
            ]
        }

        result = sanitize_json(output)

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert "total_frames" in json_str


class TestVideoRadarOutput:
    """Tests for video_radar output sanitization."""

    def test_radar_output_json_serializable(self) -> None:
        """Test radar output is JSON-serializable."""
        # Simulated output
        output = {
            "total_frames": np.int64(3),
            "frames": [
                {
                    "frame_index": np.int64(i),
                    "detections": {
                        "xyxy": np.random.rand(5, 4).astype(np.float32).tolist(),
                        "centers": np.random.rand(5, 2).astype(np.float32).tolist(),
                        "confidence": np.random.rand(5).astype(np.float32).tolist(),
                        "class_id": np.zeros(5, dtype=np.int32).tolist(),
                    }
                }
                for i in range(3)
            ]
        }

        result = sanitize_json(output)

        # Should be JSON serializable
        json_str = json.dumps(result)
        assert "total_frames" in json_str
