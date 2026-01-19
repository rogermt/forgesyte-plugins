"""Pitch detection inference module.

Provides JSON and JSON+Base64 modes for pitch detection:
- detect_pitch_json(): Returns keypoints and pitch polygon
- detect_pitch_json_with_annotated_frame(): Returns data + annotated frame
"""

import base64
from typing import Any, Dict, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

MODEL_PATH = "src/forgesyte_yolo_tracker/models/football-pitch-detection-v1.pt"
CONFIG = SoccerPitchConfiguration()

PITCH_COLOR = sv.Color.from_hex("#00FF00")
KEYPOINT_COLOR = sv.Color.from_hex("#FF0000")

DEFAULT_CONFIDENCE = 0.25
DEFAULT_NMS = 0.45

_model: Optional[YOLO] = None


def get_pitch_detection_model(device: str = "cpu") -> YOLO:
    """Get or create cached YOLO model."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH).to(device=device)
    return _model


def _encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode frame to base64 JPEG."""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def detect_pitch_json(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Detect pitch in a frame - JSON mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with:
        - keypoints: List of keypoint dicts with xy, confidence, name
        - pitch_polygon: List of polygon vertices
        - pitch_detected: Boolean indicating success
        - homography: Optional homography matrix for coordinate transform
    """
    model = get_pitch_detection_model(device=device)
    result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]

    keypoint_list = []
    if result.keypoints is not None and result.keypoints.xy is not None:
        keypoints_xy = result.keypoints.xy.cpu().numpy()[0]
        keypoints_conf = (
            result.keypoints.conf.cpu().numpy()[0] if result.keypoints.conf is not None else None
        )

        for i, (x, y) in enumerate(keypoints_xy):
            kp = {
                "xy": [float(x), float(y)],
                "confidence": float(keypoints_conf[i]) if keypoints_conf is not None else 1.0,
                "name": CONFIG.keypoint_names.get(i, f"keypoint_{i}"),
            }
            keypoint_list.append(kp)

    pitch_polygon = []
    valid_keypoints = [kp for kp in keypoint_list if kp["confidence"] > confidence * 0.5]
    if len(valid_keypoints) >= 4:
        # Use first 4 corner keypoints for polygon
        corner_names = [
            "bottom_left_corner",
            "top_left_corner",
            "top_right_corner",
            "bottom_right_corner",
        ]
        for name in corner_names:
            for kp in valid_keypoints:
                if kp["name"] == name:
                    pitch_polygon.append(kp["xy"])
                    break

    homography = None
    if len(pitch_polygon) >= 4:
        src_pts = np.array([kp["xy"] for kp in valid_keypoints[:4]], dtype=np.float32)
        tgt_pts = np.array([CONFIG.vertices[i] for i in range(4)], dtype=np.float32)
        try:
            m, _ = cv2.findHomography(src_pts, tgt_pts)
            if m is not None:
                homography = m.tolist()
        except Exception:
            pass

    return {
        "keypoints": keypoint_list,
        "pitch_polygon": pitch_polygon,
        "pitch_detected": len(pitch_polygon) >= 4,
        "homography": homography,
    }


def detect_pitch_json_with_annotated_frame(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Detect pitch in a frame - JSON+Base64 mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with keypoints, pitch_polygon, annotated_frame_base64
    """
    model = get_pitch_detection_model(device=device)
    result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]

    keypoint_list = []
    if result.keypoints is not None and result.keypoints.xy is not None:
        keypoints_xy = result.keypoints.xy.cpu().numpy()[0]
        keypoints_conf = (
            result.keypoints.conf.cpu().numpy()[0] if result.keypoints.conf is not None else None
        )

        for i, (x, y) in enumerate(keypoints_xy):
            kp = {
                "xy": [float(x), float(y)],
                "confidence": float(keypoints_conf[i]) if keypoints_conf is not None else 1.0,
                "name": CONFIG.keypoint_names.get(i, f"keypoint_{i}"),
            }
            keypoint_list.append(kp)

    pitch_polygon = []
    valid_keypoints = [kp for kp in keypoint_list if kp["confidence"] > confidence * 0.5]
    if len(valid_keypoints) >= 4:
        corner_names = [
            "bottom_left_corner",
            "top_left_corner",
            "top_right_corner",
            "bottom_right_corner",
        ]
        for name in corner_names:
            for kp in valid_keypoints:
                if kp["name"] == name:
                    pitch_polygon.append(kp["xy"])
                    break

    # Create annotated frame
    annotated = frame.copy()

    # Draw keypoints
    for kp in keypoint_list:
        x, y = int(kp["xy"][0]), int(kp["xy"][1])
        cv2.circle(annotated, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            kp["name"].replace("_", " "),
            (x + 5, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
        )

    # Draw pitch polygon
    if len(pitch_polygon) >= 4:
        pts = np.array(pitch_polygon, dtype=np.int32)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    return {
        "keypoints": keypoint_list,
        "pitch_polygon": pitch_polygon,
        "pitch_detected": len(pitch_polygon) >= 4,
        "annotated_frame_base64": _encode_frame_to_base64(annotated),
    }


def run_pitch_detection(frame: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function for plugin.py compatibility."""
    device = config.get("device", "cpu")
    confidence = config.get("confidence", DEFAULT_CONFIDENCE)
    include_annotated = config.get("include_annotated", False)

    if include_annotated:
        return detect_pitch_json_with_annotated_frame(frame, device=device, confidence=confidence)
    return detect_pitch_json(frame, device=device, confidence=confidence)
