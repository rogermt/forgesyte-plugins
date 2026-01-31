"""Pytest configuration for contract tests — patches inference to prevent YOLO loading."""

import sys
from unittest.mock import MagicMock

# Patch inference modules BEFORE they can be imported
# This prevents YOLO, Torch, ByteTrack, OpenCV from loading during contract tests

def create_mock_module(name: str) -> MagicMock:
    """Create a mock module with common inference functions."""
    mock = MagicMock()
    mock.__name__ = name
    return mock


# Mock ultralytics (YOLO)
sys.modules["ultralytics"] = MagicMock()
sys.modules["ultralytics.YOLO"] = MagicMock()

# Mock torch and related
sys.modules["torch"] = MagicMock()
sys.modules["torch.nn"] = MagicMock()
sys.modules["torchvision"] = MagicMock()

# Mock supervision
sys.modules["supervision"] = MagicMock()

# Mock ByteTrack
sys.modules["ByteTrack"] = MagicMock()

# Now mock inference functions themselves
sys.modules["forgesyte_yolo_tracker.inference.player_detection"] = create_mock_module(
    "forgesyte_yolo_tracker.inference.player_detection"
)
sys.modules["forgesyte_yolo_tracker.inference.player_detection"].detect_players_json = MagicMock(
    return_value={"detections": [], "count": 0}
)
sys.modules["forgesyte_yolo_tracker.inference.player_detection"].detect_players_json_with_annotated_frame = MagicMock(
    return_value={"detections": [], "count": 0, "annotated_frame": ""}
)
sys.modules["forgesyte_yolo_tracker.inference.player_detection"].CLASS_NAMES = {
    0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"
}
sys.modules["forgesyte_yolo_tracker.inference.player_detection"].TEAM_COLORS = {
    0: "#FFD700", 1: "#00BFFF", 2: "#FF1493", 3: "#FF6347"
}

sys.modules["forgesyte_yolo_tracker.inference.ball_detection"] = create_mock_module(
    "forgesyte_yolo_tracker.inference.ball_detection"
)
sys.modules["forgesyte_yolo_tracker.inference.ball_detection"].detect_ball_json = MagicMock(
    return_value={"detections": [], "count": 0}
)
sys.modules["forgesyte_yolo_tracker.inference.ball_detection"].detect_ball_json_with_annotated_frame = MagicMock(
    return_value={"detections": [], "count": 0, "annotated_frame": ""}
)

sys.modules["forgesyte_yolo_tracker.inference.pitch_detection"] = create_mock_module(
    "forgesyte_yolo_tracker.inference.pitch_detection"
)
sys.modules["forgesyte_yolo_tracker.inference.pitch_detection"].detect_pitch_json = MagicMock(
    return_value={"pitch": None}
)
sys.modules["forgesyte_yolo_tracker.inference.pitch_detection"].detect_pitch_json_with_annotated_frame = MagicMock(
    return_value={"pitch": None, "annotated_frame": ""}
)

sys.modules["forgesyte_yolo_tracker.inference.player_tracking"] = create_mock_module(
    "forgesyte_yolo_tracker.inference.player_tracking"
)
sys.modules["forgesyte_yolo_tracker.inference.player_tracking"].track_players_json = MagicMock(
    return_value={"tracks": [], "count": 0}
)
sys.modules["forgesyte_yolo_tracker.inference.player_tracking"].track_players_json_with_annotated_frame = MagicMock(
    return_value={"tracks": [], "count": 0, "annotated_frame": ""}
)

sys.modules["forgesyte_yolo_tracker.inference.radar"] = create_mock_module(
    "forgesyte_yolo_tracker.inference.radar"
)
sys.modules["forgesyte_yolo_tracker.inference.radar"].generate_radar_json = MagicMock(
    return_value={"radar": None}
)
sys.modules["forgesyte_yolo_tracker.inference.radar"].radar_json_with_annotated_frame = MagicMock(
    return_value={"radar": None, "annotated_frame": ""}
)
