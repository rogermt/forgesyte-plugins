"""Tests for team classification inference module."""

import os
import pytest
import numpy as np


RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS, reason="Set RUN_MODEL_TESTS=1 to run (requires TeamClassifier)"
)


class TestTeamClassificationJSON:
    """Tests for classify_teams_json function."""

    def test_returns_dict_with_team_ids(self) -> None:
        """Verify returns dictionary with team_ids."""
        from forgesyte_yolo_tracker.inference.team_classification import classify_teams_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = classify_teams_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "team_ids" in result

    def test_returns_team_counts(self) -> None:
        """Verify returns team counts."""
        from forgesyte_yolo_tracker.inference.team_classification import classify_teams_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = classify_teams_json(frame, device="cpu")

        assert "team_counts" in result
        assert "team_a" in result["team_counts"]
        assert "team_b" in result["team_counts"]

    def test_team_ids_are_0_or_1(self) -> None:
        """Verify team_ids are 0 or 1."""
        from forgesyte_yolo_tracker.inference.team_classification import classify_teams_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = classify_teams_json(frame, device="cpu")

        for team_id in result.get("team_ids", []):
            assert team_id in [0, 1]


class TestTeamClassificationJSONWithAnnotated:
    """Tests for classify_teams_json_with_annotated function."""

    def test_returns_annotated_frame_base64(self) -> None:
        """Verify returns base64 encoded annotated frame."""
        from forgesyte_yolo_tracker.inference.team_classification import (
            classify_teams_json_with_annotated,
        )

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = classify_teams_json_with_annotated(frame, device="cpu")

        assert "annotated_frame_base64" in result
        assert isinstance(result["annotated_frame_base64"], str)
