"""Radar inference module.

Provides JSON and JSON+Base64 modes for radar/bird's-eye view generation:
- generate_radar_json(): Returns mapped radar points
- radar_json_with_annotated_frame(): Returns data + radar image

Radar uses ViewTransformer to map frame coordinates to pitch coordinates.
"""

import base64
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration
from forgesyte_yolo_tracker.utils import ViewTransformer

PLAYER_MODEL_PATH = "src/forgesyte_yolo_tracker/models/football-player-detection-v3.pt"
PITCH_MODEL_PATH = "src/forgesyte_yolo_tracker/models/football-pitch-detection-v1.pt"
CONFIG = SoccerPitchConfiguration()

TEAM_A_COLOR = (0, 191, 255)  # DeepSkyBlue BGR
TEAM_B_COLOR = (255, 20, 147)  # DeepPink BGR
GK_COLOR = (0, 215, 255)  # Gold BGR
BALL_COLOR = (135, 206, 250)  # Light blue BGR

DEFAULT_CONFIDENCE = 0.25

_player_model: Optional[YOLO] = None
_pitch_model: Optional[YOLO] = None
_view_transformer: Optional[ViewTransformer] = None


def get_player_detection_model(device: str = "cpu") -> YOLO:
    """Get or create cached YOLO model."""
    global _player_model
    if _player_model is None:
        _player_model = YOLO(PLAYER_MODEL_PATH).to(device=device)
    return _player_model


def get_pitch_detection_model(device: str = "cpu") -> YOLO:
    """Get or create cached pitch detection model."""
    global _pitch_model
    if _pitch_model is None:
        _pitch_model = YOLO(PITCH_MODEL_PATH).to(device=device)
    return _pitch_model


def get_view_transformer(
    source: np.ndarray,
    target: np.ndarray,
) -> ViewTransformer:
    """Create ViewTransformer for coordinate mapping."""
    return ViewTransformer(source, target)


def _encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode frame to base64 PNG."""
    _, buffer = cv2.imencode(".png", frame)
    return base64.b64encode(buffer).decode("utf-8")


def _encode_radar_to_base64(radar: np.ndarray) -> str:
    """Encode radar image to base64 PNG."""
    _, buffer = cv2.imencode(".png", radar)
    return base64.b64encode(buffer).decode("utf-8")


def _create_radar_image(
    radar_points: List[Dict[str, Any]],
    radar_size: Tuple[int, int] = (600, 300),
) -> np.ndarray:
    """Create radar visualization image.

    Args:
        radar_points: List of radar point dicts
        radar_size: (width, height) of radar image

    Returns:
        Radar image as numpy array
    """
    radar_w, radar_h = radar_size
    radar = np.zeros((radar_h, radar_w, 3), dtype=np.uint8)

    # Draw pitch outline
    cv2.rectangle(radar, (0, 0), (radar_w, radar_h), (50, 50, 50), 2)

    # Draw center line
    cv2.line(radar, (radar_w // 2, 0), (radar_w // 2, radar_h), (100, 100, 100), 1)

    # Draw center circle
    cv2.circle(radar, (radar_w // 2, radar_h // 2), 20, (100, 100, 100), 1)

    # Draw points
    for point in radar_points:
        x, y = point["xy"]
        team_id = point.get("team_id", -1)
        point_type = point.get("type", "player")

        if point_type == "ball":
            color = BALL_COLOR
            radius = 6
        elif team_id == 0:
            color = TEAM_A_COLOR
            radius = 8
        elif team_id == 1:
            color = TEAM_B_COLOR
            radius = 8
        elif point_type == "goalkeeper":
            color = GK_COLOR
            radius = 8
        else:
            color = (128, 128, 128)
            radius = 6

        cv2.circle(radar, (int(x), int(y)), radius, color, -1)

    return radar


def generate_radar_json(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Generate radar data - JSON mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with:
        - radar_points: List of mapped points with xy, tracking_id, team_id, type
        - radar_size: Radar dimensions (width, height)
        - radar_base64: Optional rendered radar image
    """
    player_model = get_player_detection_model(device=device)
    pitch_model = get_pitch_detection_model(device=device)

    player_result = player_model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
    pitch_result = pitch_model(frame, imgsz=1280, conf=confidence, verbose=False)[0]

    player_detections = sv.Detections.from_ultralytics(player_result)

    radar_points = []
    radar_w, radar_h = CONFIG.radar_resolution

    if pitch_result.keypoints is not None and pitch_result.keypoints.xy is not None:
        keypoints_xy = pitch_result.keypoints.xy.cpu().numpy()[0]
        keypoints_conf = (
            pitch_result.keypoints.conf.cpu().numpy()[0]
            if pitch_result.keypoints.conf is not None
            else None
        )

        valid_kp_indices = []
        for i, conf in enumerate(keypoints_conf):
            if conf > confidence * 0.5:
                valid_kp_indices.append(i)

        if len(valid_kp_indices) >= 4:
            src_pts = np.array([keypoints_xy[i] for i in valid_kp_indices[:4]], dtype=np.float32)
            tgt_pts = np.array([CONFIG.vertices[i] for i in valid_kp_indices[:4]], dtype=np.float32)

            try:
                transformer = get_view_transformer(src_pts, tgt_pts)

                for i in range(len(player_detections)):
                    xyxy = player_detections.xyxy[i]
                    track_id = (
                        int(player_detections.track_id[i])
                        if player_detections.track_id is not None
                        else -1
                    )
                    cls = int(player_detections.class_id[i])

                    center_x = float((xyxy[0] + xyxy[2]) / 2)
                    center_y = float((xyxy[1] + xyxy[3]) / 2)

                    transformed = transformer.transform_points(
                        np.array([[center_x, center_y]], dtype=np.float32)
                    )
                    rx, ry = CONFIG.world_to_radar(transformed[0][0], transformed[0][1])

                    point = {
                        "xy": [rx, ry],
                        "tracking_id": track_id,
                        "team_id": -1,
                        "type": "goalkeeper" if cls == 1 else "player",
                    }
                    radar_points.append(point)
            except Exception:
                pass

    return {
        "radar_points": radar_points,
        "radar_size": [radar_w, radar_h],
    }


def radar_json_with_annotated_frame(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Generate radar data - JSON+Base64 mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with radar_points, radar_size, radar_base64
    """
    player_model = get_player_detection_model(device=device)
    pitch_model = get_pitch_detection_model(device=device)

    player_result = player_model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
    pitch_result = pitch_model(frame, imgsz=1280, conf=confidence, verbose=False)[0]

    player_detections = sv.Detections.from_ultralytics(player_result)

    radar_points = []
    radar_w, radar_h = CONFIG.radar_resolution

    if pitch_result.keypoints is not None and pitch_result.keypoints.xy is not None:
        keypoints_xy = pitch_result.keypoints.xy.cpu().numpy()[0]
        keypoints_conf = (
            pitch_result.keypoints.conf.cpu().numpy()[0]
            if pitch_result.keypoints.conf is not None
            else None
        )

        valid_kp_indices = []
        for i, conf in enumerate(keypoints_conf):
            if conf > confidence * 0.5:
                valid_kp_indices.append(i)

        if len(valid_kp_indices) >= 4:
            src_pts = np.array([keypoints_xy[i] for i in valid_kp_indices[:4]], dtype=np.float32)
            tgt_pts = np.array([CONFIG.vertices[i] for i in valid_kp_indices[:4]], dtype=np.float32)

            try:
                transformer = get_view_transformer(src_pts, tgt_pts)

                for i in range(len(player_detections)):
                    xyxy = player_detections.xyxy[i]
                    track_id = (
                        int(player_detections.track_id[i])
                        if player_detections.track_id is not None
                        else -1
                    )
                    cls = int(player_detections.class_id[i])

                    center_x = float((xyxy[0] + xyxy[2]) / 2)
                    center_y = float((xyxy[1] + xyxy[3]) / 2)

                    transformed = transformer.transform_points(
                        np.array([[center_x, center_y]], dtype=np.float32)
                    )
                    rx, ry = CONFIG.world_to_radar(transformed[0][0], transformed[0][1])

                    point = {
                        "xy": [rx, ry],
                        "tracking_id": track_id,
                        "team_id": -1,
                        "type": "goalkeeper" if cls == 1 else "player",
                    }
                    radar_points.append(point)
            except Exception:
                pass

    radar_image = _create_radar_image(radar_points, (radar_w, radar_h))

    return {
        "radar_points": radar_points,
        "radar_size": [radar_w, radar_h],
        "radar_base64": _encode_radar_to_base64(radar_image),
    }


def run_radar(frame: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function for plugin.py compatibility."""
    device = config.get("device", "cpu")
    confidence = config.get("confidence", DEFAULT_CONFIDENCE)
    include_annotated = config.get("include_annotated", False)

    if include_annotated:
        return radar_json_with_annotated_frame(frame, device=device, confidence=confidence)
    return generate_radar_json(frame, device=device, confidence=confidence)
