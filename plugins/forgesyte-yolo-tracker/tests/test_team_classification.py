"""Tests for team classification module."""

import numpy as np

from forgesyte_yolo_tracker.inference.team_classification import (
    classify_teams,
    resolve_goalkeeper_teams,
)


class TestTeamClassification:
    """Test team classification functionality."""

    def test_classify_teams_returns_dict(self) -> None:
        """Test that classify_teams returns a dictionary."""
        player_crops = [
            np.zeros((100, 50, 3), dtype=np.uint8),
            np.zeros((100, 50, 3), dtype=np.uint8),
        ]
        result = classify_teams(player_crops)

        assert isinstance(result, dict)
        assert "team_ids" in result
        assert "team_colors" in result
        assert "confidence" in result

    def test_classify_teams_structure(self) -> None:
        """Test the structure of team classification results."""
        player_crops = [np.zeros((100, 50, 3), dtype=np.uint8)]
        result = classify_teams(player_crops)

        assert isinstance(result["team_ids"], list)
        assert isinstance(result["team_colors"], dict)
        assert isinstance(result["confidence"], list)

    def test_classify_teams_with_model(self) -> None:
        """Test team classification with a model."""
        player_crops = [np.zeros((100, 50, 3), dtype=np.uint8)]
        result = classify_teams(player_crops, team_model=None)

        assert isinstance(result, dict)
        assert "team_ids" in result

    def test_resolve_goalkeeper_teams_returns_array(self) -> None:
        """Test that resolve_goalkeeper_teams returns numpy array."""
        players = {
            "detections": [
                {"center": [100, 100]},
                {"center": [200, 200]},
            ]
        }
        player_team_ids = np.array([0, 1])
        goalkeepers = {
            "detections": [
                {"center": [120, 120]},
            ]
        }

        result = resolve_goalkeeper_teams(players, player_team_ids, goalkeepers)

        assert isinstance(result, np.ndarray)

    def test_resolve_goalkeeper_teams_with_multiple_goalkeepers(self) -> None:
        """Test resolving multiple goalkeepers."""
        players = {
            "detections": [
                {"center": [100, 100]},
                {"center": [200, 200]},
            ]
        }
        player_team_ids = np.array([0, 1])
        goalkeepers = {
            "detections": [
                {"center": [120, 120]},
                {"center": [180, 180]},
            ]
        }

        result = resolve_goalkeeper_teams(players, player_team_ids, goalkeepers)

        assert isinstance(result, np.ndarray)
        assert len(result) == 2

    def test_classify_teams_empty_crops(self) -> None:
        """Test team classification with empty crops."""
        player_crops: list = []
        result = classify_teams(player_crops)

        assert isinstance(result, dict)
        assert isinstance(result["team_ids"], list)
