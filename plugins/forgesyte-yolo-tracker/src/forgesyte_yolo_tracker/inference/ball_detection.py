"""Ball detection inference module.

Provides JSON and JSON+Base64 modes for ball detection:
- detect_ball_json(): Returns detection data with ball position
- detect_ball_json_with_annotated_frame(): Returns data + annotated frame
"""

import base64
from typing import Any, Dict, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

MODEL_PATH = "src/forgesyte_yolo_tracker/models/football-ball-detection-v2.pt"

BALL_COLOR = sv.Color.from_hex("#FF6347")
DEFAULT_CONFIDENCE = 0.20
DEFAULT_NMS = 0.10

_model: Optional[YOLO] = None


def get_ball_detection_model(device: str = "cpu") -> YOLO:
    """Get or create cached YOLO model."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH).to(device=device)
    return _model


def _encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode frame to base64 JPEG."""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def detect_ball_json(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Detect ball in a frame - JSON mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with:
        - detections: List of ball detections
        - ball: Primary ball detection (or None)
        - ball_detected: Boolean indicating if ball was found
    """
    model = get_ball_detection_model(device=device)
    result = model(frame, imgsz=640, conf=confidence, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    detection_list = []
    primary_ball = None

    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        conf = float(detections.confidence[i])

        detection_data = {
            "xyxy": xyxy.tolist(),
            "confidence": conf,
        }
        detection_list.append(detection_data)

        if primary_ball is None or conf > primary_ball["confidence"]:
            primary_ball = detection_data

    return {
        "detections": detection_list,
        "ball": primary_ball,
        "ball_detected": primary_ball is not None,
    }


def detect_ball_json_with_annotated_frame(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Detect ball in a frame - JSON+Base64 mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with detections, ball, and annotated_frame_base64
    """
    model = get_ball_detection_model(device=device)
    result = model(frame, imgsz=640, conf=confidence, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    detection_list = []
    primary_ball = None

    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        conf = float(detections.confidence[i])

        detection_data = {
            "xyxy": xyxy.tolist(),
            "confidence": conf,
        }
        detection_list.append(detection_data)

        if primary_ball is None or conf > primary_ball["confidence"]:
            primary_ball = detection_data

    # Create annotated frame
    box_annotator = sv.BoxAnnotator(color=BALL_COLOR, thickness=2)

    annotated = frame.copy()
    annotated = box_annotator.annotate(annotated, detections)

    return {
        "detections": detection_list,
        "ball": primary_ball,
        "ball_detected": primary_ball is not None,
        "annotated_frame_base64": _encode_frame_to_base64(annotated),
    }


def run_ball_detection(frame: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function for plugin.py compatibility."""
    device = config.get("device", "cpu")
    confidence = config.get("confidence", DEFAULT_CONFIDENCE)
    include_annotated = config.get("include_annotated", False)

    if include_annotated:
        return detect_ball_json_with_annotated_frame(frame, device=device, confidence=confidence)
    return detect_ball_json(frame, device=device, confidence=confidence)
