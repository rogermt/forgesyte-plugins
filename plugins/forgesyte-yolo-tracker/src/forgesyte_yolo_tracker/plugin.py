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
from typing import Dict, Any

import cv2
import numpy as np

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
    radar_json,
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
