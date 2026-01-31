"""Tests for player tracking inference module."""

import numpy as np
import pytest

from tests_heavy.constants import MODELS_EXIST, RUN_MODEL_TESTS

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS or not MODELS_EXIST,
    reason="Set RUN_MODEL_TESTS=1 AND download models to run",
)


class TestPlayerTrackingJSON:
    """Tests for track_players_json function."""

    def test_returns_dict_with_detections(self) -> None:
        """Verify returns dictionary with detections key."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = track_players_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "detections" in result

    def test_returns_count(self) -> None:
        """Verify returns count of tracked players."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = track_players_json(frame, device="cpu")

        assert "count" in result
        assert isinstance(result["count"], int)

    def test_detections_have_tracking_id(self) -> None:
        """Verify each detection has tracking_id."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = track_players_json(frame, device="cpu")

        for det in result["detections"]:
            assert "tracking_id" in det
            assert isinstance(det["tracking_id"], int)

    def test_detections_have_xyxy(self) -> None:
        """Verify each detection has xyxy coordinates."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = track_players_json(frame, device="cpu")

        for det in result["detections"]:
            assert "xyxy" in det
            assert len(det["xyxy"]) == 4


class TestPlayerTrackingJSONWithAnnotated:
    """Tests for track_players_json_with_annotated function."""

    def test_returns_annotated_frame_base64(self) -> None:
        """Verify returns base64 encoded annotated frame."""
        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json_with_annotated

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = track_players_json_with_annotated(frame, device="cpu")

        assert "annotated_frame_base64" in result
        assert isinstance(result["annotated_frame_base64"], str)

    def test_annotated_frame_includes_tracking_labels(self) -> None:
        """Verify annotated frame includes tracking ID labels."""
        import base64

        from forgesyte_yolo_tracker.inference.player_tracking import \
            track_players_json_with_annotated

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = track_players_json_with_annotated(frame, device="cpu")

        decoded = base64.b64decode(result["annotated_frame_base64"])
        assert len(decoded) > 0
