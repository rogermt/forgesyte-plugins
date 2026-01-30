"""Tests for radar inference module."""

import pytest
import numpy as np



pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS or not MODELS_EXIST,
    reason="Set RUN_MODEL_TESTS=1 AND download models to run",
)


class TestRadarJSON:
    """Tests for generate_radar_json function."""

    def test_returns_dict_with_radar_points(self) -> None:
        """Verify returns dictionary with radar_points."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = generate_radar_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "radar_points" in result

    def test_radar_points_have_xy_tracking_id_team_id(self) -> None:
        """Verify each radar point has required fields."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = generate_radar_json(frame, device="cpu")

        for point in result.get("radar_points", []):
            assert "xy" in point
            assert "tracking_id" in point
            assert "team_id" in point
            assert "type" in point


class TestRadarJSONWithAnnotated:
    """Tests for generate_radar_json_with_annotated function."""

    def test_returns_radar_base64(self) -> None:
        """Verify returns base64 encoded radar image."""
        from forgesyte_yolo_tracker.inference.radar import generate_radar_json_with_annotated

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = generate_radar_json_with_annotated(frame, device="cpu")

        assert "radar_base64" in result
        assert isinstance(result["radar_base64"], str)
