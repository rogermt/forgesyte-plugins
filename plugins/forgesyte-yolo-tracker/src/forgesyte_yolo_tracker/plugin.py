"""ForgeSyte YOLO Tracker Plugin.

Frame-based JSON tools for football analysis:
- player_detection
- player_tracking
- ball_detection
- team_classification
- pitch_detection
- radar
"""

import base64
from typing import Any, Dict

import cv2
import numpy as np
import torch

from app.models import AnalysisResult, PluginMetadata

from forgesyte_yolo_tracker.inference.player_detection import (
    detect_players_json,
    detect_players_json_with_annotated_frame,
)
from forgesyte_yolo_tracker.inference.player_tracking import (
    track_players_json,
    track_players_json_with_annotated_frame,
)
from forgesyte_yolo_tracker.inference.ball_detection import (
    detect_ball_json,
    detect_ball_json_with_annotated_frame,
)
from forgesyte_yolo_tracker.inference.team_classification import (
    classify_teams_json,
    classify_teams_json_with_annotated_frame,
)
from forgesyte_yolo_tracker.inference.pitch_detection import (
    detect_pitch_json,
    detect_pitch_json_with_annotated_frame,
)
from forgesyte_yolo_tracker.inference.radar import (
    generate_radar_json as radar_json,
    radar_json_with_annotated_frame,
)


def _decode_frame_base64(frame_b64: str) -> np.ndarray:
    """Decode base64-encoded image into a numpy BGR frame.

    Args:
        frame_b64: Base64 encoded image data

    Returns:
        Decoded image as numpy BGR array
    """
    data = base64.b64decode(frame_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def player_detection(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    """Detect players in a single frame.

    Args:
        frame_base64: Base64 encoded image
        device: Device to run model on ('cpu' or 'cuda')
        annotated: If True, return annotated frame

    Returns:
        Dictionary with detections, count, classes
    """
    frame = _decode_frame_base64(frame_base64)
    if annotated:
        return detect_players_json_with_annotated_frame(frame, device=device)
    return detect_players_json(frame, device=device)


def player_tracking(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    """Track players across frames using ByteTrack.

    Args:
        frame_base64: Base64 encoded image
        device: Device to run model on ('cpu' or 'cuda')
        annotated: If True, return annotated frame

    Returns:
        Dictionary with detections, count, track_ids
    """
    frame = _decode_frame_base64(frame_base64)
    if annotated:
        return track_players_json_with_annotated_frame(frame, device=device)
    return track_players_json(frame, device=device)


def ball_detection(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    """Detect the football in a single frame.

    Args:
        frame_base64: Base64 encoded image
        device: Device to run model on ('cpu' or 'cuda')
        annotated: If True, return annotated frame

    Returns:
        Dictionary with detections, ball, ball_detected
    """
    frame = _decode_frame_base64(frame_base64)
    if annotated:
        return detect_ball_json_with_annotated_frame(frame, device=device)
    return detect_ball_json(frame, device=device)


def team_classification(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    """Classify players into teams using SigLIP embeddings + UMAP + KMeans.

    Args:
        frame_base64: Base64 encoded image
        device: Device to run model on ('cpu' or 'cuda')
        annotated: If True, return annotated frame

    Returns:
        Dictionary with detections, team_ids, team_counts
    """
    frame = _decode_frame_base64(frame_base64)
    if annotated:
        return classify_teams_json_with_annotated_frame(frame, device=device)
    return classify_teams_json(frame, device=device)


def pitch_detection(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    """Detect pitch keypoints for homography mapping.

    Args:
        frame_base64: Base64 encoded image
        device: Device to run model on ('cpu' or 'cuda')
        annotated: If True, return annotated frame

    Returns:
        Dictionary with keypoints, pitch_polygon, pitch_detected
    """
    frame = _decode_frame_base64(frame_base64)
    if annotated:
        return detect_pitch_json_with_annotated_frame(frame, device=device)
    return detect_pitch_json(frame, device=device)


def radar(frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
    """Generate radar (bird's-eye) view of player positions.

    Args:
        frame_base64: Base64 encoded image
        device: Device to run model on ('cpu' or 'cuda')
        annotated: If True, return radar image

    Returns:
        Dictionary with radar_points, radar_size, radar_base64 (if annotated)
    """
    frame = _decode_frame_base64(frame_base64)
    if annotated:
        return radar_json_with_annotated_frame(frame, device=device)
    return radar_json(frame, device=device)


class Plugin:
    """ForgeSyte YOLO Tracker plugin (old interface)."""

    name: str = "yolo-tracker"
    version: str = "0.1.0"

    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name=self.name,
            description="YOLO-based football analysis plugin",
            version=self.version,
            inputs=["image"],
            outputs=["json"],
            config_schema={
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
                "confidence": {"type": "number", "default": 0.25},
            },
        )

    def analyze(self, image_data: bytes, options: Dict[str, Any] | None = None) -> AnalysisResult:
        """Legacy analyze method — defaults to player detection."""
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        frame_b64 = base64.b64encode(image_data).decode("utf-8")
        frame = _decode_frame_base64(frame_b64)

        result = detect_players_json(frame, device=device)

        return AnalysisResult(
            text="",
            blocks=[],
            confidence=1.0,
            language=None,
            error=None,
            extra=result,
        )

    def on_load(self) -> None:
        """Called when plugin is loaded."""
        print("YOLO Tracker plugin loaded")

    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        print("YOLO Tracker plugin unloaded")
