"""Tests for player tracking module."""

import numpy as np
import pytest

from forgesyte_yolo_tracker.inference.player_tracking import (
    track_players,
    update_tracks,
)


class TestPlayerTracking:
    """Test player tracking functionality."""

    def test_track_players_returns_dict(self) -> None:
        """Test that track_players returns a dictionary."""
        detections = {
            "detections": [],
            "count": 0,
            "classes": {"player": 0, "goalkeeper": 0, "referee": 0},
        }
        result = track_players(detections)

        assert isinstance(result, dict)
        assert "tracks" in result
        assert "track_ids" in result
        assert "detections" in result

    def test_track_players_with_state(self) -> None:
        """Test tracking with previous state."""
        detections = {
            "detections": [],
            "count": 0,
            "classes": {"player": 0, "goalkeeper": 0, "referee": 0},
        }
        tracker_state = {
            "active_tracks": [],
            "inactive_tracks": [],
        }
        result = track_players(detections, tracker_state=tracker_state)

        assert isinstance(result, dict)
        assert isinstance(result["tracks"], list)
        assert isinstance(result["track_ids"], list)

    def test_update_tracks_returns_dict(self) -> None:
        """Test that update_tracks returns a dictionary."""
        current_detections = {
            "detections": [],
            "count": 0,
            "classes": {"player": 0, "goalkeeper": 0, "referee": 0},
        }
        previous_tracks = {
            "active_tracks": [],
            "inactive_tracks": [],
        }
        result = update_tracks(current_detections, previous_tracks)

        assert isinstance(result, dict)
        assert "active_tracks" in result
        assert "inactive_tracks" in result
        assert "current_detections" in result

    def test_update_tracks_structure(self) -> None:
        """Test the structure of updated tracking state."""
        current_detections = {
            "detections": [],
            "count": 0,
            "classes": {"player": 0, "goalkeeper": 0, "referee": 0},
        }
        previous_tracks = {
            "active_tracks": [],
            "inactive_tracks": [],
        }
        result = update_tracks(current_detections, previous_tracks)

        assert isinstance(result["active_tracks"], list)
        assert isinstance(result["inactive_tracks"], list)

    def test_custom_max_age(self) -> None:
        """Test tracking with custom max age parameter."""
        current_detections = {
            "detections": [],
            "count": 0,
            "classes": {"player": 0, "goalkeeper": 0, "referee": 0},
        }
        previous_tracks = {
            "active_tracks": [],
            "inactive_tracks": [],
        }
        result = update_tracks(current_detections, previous_tracks, max_age=60)

        assert isinstance(result, dict)
