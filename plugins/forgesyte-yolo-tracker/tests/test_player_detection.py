"""Tests for player detection module."""

import numpy as np

from forgesyte_yolo_tracker.inference.player_detection import (
    detect_players,
    filter_detections,
)


class TestPlayerDetection:
    """Test player detection functionality."""

    def test_detect_players_returns_dict(self) -> None:
        """Test that detect_players returns a dictionary."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players(frame, model=None)

        assert isinstance(result, dict)
        assert "detections" in result
        assert "count" in result
        assert "classes" in result

    def test_detect_players_structure(self) -> None:
        """Test the structure of detection results."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players(frame, model=None)

        assert isinstance(result["detections"], list)
        assert isinstance(result["count"], int)
        assert isinstance(result["classes"], dict)
        assert "player" in result["classes"]
        assert "goalkeeper" in result["classes"]
        assert "referee" in result["classes"]

    def test_filter_detections_returns_dict(self) -> None:
        """Test that filter_detections returns a dictionary."""
        detections = {
            "detections": [],
            "count": 0,
            "classes": {"player": 0, "goalkeeper": 0, "referee": 0},
        }
        result = filter_detections(detections)

        assert isinstance(result, dict)

    def test_filter_detections_with_class_ids(self) -> None:
        """Test filtering by class IDs."""
        detections = {
            "detections": [],
            "count": 0,
            "classes": {"player": 0, "goalkeeper": 0, "referee": 0},
        }
        result = filter_detections(detections, class_ids=[0, 2])

        assert isinstance(result, dict)

    def test_custom_imgsz(self) -> None:
        """Test detection with custom image size."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players(frame, model=None, imgsz=1280)

        assert isinstance(result, dict)
        assert "detections" in result

    def test_custom_confidence(self) -> None:
        """Test detection with custom confidence threshold."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players(frame, model=None, confidence=0.7)

        assert isinstance(result, dict)
        assert "detections" in result
