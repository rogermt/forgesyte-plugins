"""Ball detection inference module.

Provides JSON and JSON+Base64 modes for ball detection:
- detect_ball_json(): Returns structured detection data
- detect_ball_json_with_annotated_frame(): Returns data + annotated frame
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from forgesyte_yolo_tracker.configs import get_model_path, get_confidence

logger = logging.getLogger(__name__)

MODEL_NAME = get_model_path("ball_detection")
MODEL_PATH = str(Path(__file__).parent.parent / "models" / MODEL_NAME)
logger.debug(f"🔍 Ball detection model: {MODEL_NAME}")
logger.debug(f"🔍 Model path: {MODEL_PATH}")

DEFAULT_CONFIDENCE = get_confidence("ball")
logger.debug(f"🔍 Default confidence threshold: {DEFAULT_CONFIDENCE}")

BALL_COLOR = sv.Color.from_hex("#FF6347")
DEFAULT_NMS = 0.10

_model: Optional[YOLO] = None


def get_ball_detection_model(device: str = "cpu") -> YOLO:
    """Get or create cached YOLO model.

    Args:
        device: Device to run model on ('cpu' or 'cuda')

    Returns:
        YOLO model instance
    """
    global _model
    if _model is None:
        logger.info(f"📦 Loading ball detection model from: {MODEL_PATH}")
        model_file = Path(MODEL_PATH)
        model_size_kb = model_file.stat().st_size / 1024 if model_file.exists() else 0
        logger.info(f"📦 Model file size: {model_size_kb:.2f} KB")
        if model_size_kb < 1:
            logger.warning(f"⚠️  Model is a stub ({model_size_kb:.2f} KB)! Replace with real model.")
        _model = YOLO(MODEL_PATH).to(device=device)
        logger.info(f"✅ Model loaded successfully on device: {device}")
    return _model


def _encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode frame to base64 JPEG.

    Args:
        frame: Input image frame

    Returns:
        Base64 encoded JPEG string
    """
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
        - detections: List of detection dicts with xyxy, confidence
        - count: Total number of detections
        - ball: Primary ball detection
        - ball_detected: Boolean
    """
    logger.info(f"🎬 Starting ball detection (device={device}, confidence={confidence})")
    logger.debug(f"🎬 Frame shape: {frame.shape}")
    
    model = get_ball_detection_model(device=device)
    logger.info(f"🔫 Running YOLO inference...")
    result = model(frame, imgsz=640, conf=confidence, verbose=False)[0]
    logger.info(f"🔫 Inference complete. Processing results...")
    
    detections = sv.Detections.from_ultralytics(result)
    logger.info(f"📊 Found {len(detections)} detections")

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

    logger.info(f"✅ Detection complete: {len(detection_list)} balls found")
    logger.info(f"✅ Primary ball: {primary_ball['confidence'] if primary_ball else 'None'}")
    
    return {
        "detections": detection_list,
        "count": len(detection_list),
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
        Dictionary with:
        - detections: List of detection dicts
        - count: Total number of detections
        - ball: Primary ball detection
        - ball_detected: Boolean
        - annotated_frame_base64: Base64 encoded annotated frame
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
        "count": len(detection_list),
        "ball": primary_ball,
        "ball_detected": primary_ball is not None,
        "annotated_frame_base64": _encode_frame_to_base64(annotated),
    }


def run_ball_detection(frame: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function for plugin.py compatibility.

    Args:
        frame: Input image frame
        config: Configuration dictionary

    Returns:
        Detection results
    """
    device = config.get("device", "cpu")
    confidence = config.get("confidence", DEFAULT_CONFIDENCE)
    include_annotated = config.get("include_annotated", False)

    if include_annotated:
        return detect_ball_json_with_annotated_frame(frame, device=device, confidence=confidence)
    return detect_ball_json(frame, device=device, confidence=confidence)
