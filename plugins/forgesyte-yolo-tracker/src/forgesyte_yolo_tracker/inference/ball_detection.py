"""Ball detection inference module.

Provides JSON and JSON+Base64 modes for ball detection using BaseDetector:
- detect_ball_json(): Returns structured detection data
- detect_ball_json_with_annotated_frame(): Returns data + annotated frame

This module uses BaseDetector to eliminate code duplication across detectors.
Ball-specific configuration (single object detection) is applied via wrappers.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from forgesyte_yolo_tracker.configs import get_confidence, get_model_path
from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

logger = logging.getLogger(__name__)

# Model path for ball detection (full path for test compatibility)
MODEL_NAME = get_model_path("ball_detection")
MODEL_PATH: str = str(Path(__file__).parent.parent / "models" / MODEL_NAME)

# Ball detector instance with configuration
BALL_DETECTOR = BaseDetector(
    detector_name="ball",
    model_name=get_model_path("ball_detection"),
    default_confidence=get_confidence("ball"),
    imgsz=640,
    class_names=None,  # Ball has no classes, single object detection
    colors={0: "#FF6347"},  # Single red color
)


def detect_ball_json(
    frame: np.ndarray[Any, Any],
    device: str = "cpu",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect ball in a frame - JSON mode.

    Executes YOLO inference and returns structured detection results.
    Adds ball-specific fields: primary ball selection and ball_detected flag.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold (uses default if None)

    Returns:
        Dictionary with:
        - detections: List of detection dicts with xyxy, confidence
        - count: Total number of detections
        - ball: Primary ball detection (highest confidence)
        - ball_detected: Boolean indicating if ball found
    """
    if confidence is None:
        confidence = BALL_DETECTOR.default_confidence

    result = BALL_DETECTOR.detect_json(frame, device=device, confidence=confidence)

    # Add ball-specific logic: select primary ball (highest confidence)
    primary_ball: Optional[Dict[str, Any]] = None
    if result["detections"]:
        primary_ball = max(result["detections"], key=lambda x: x["confidence"])

    result["ball"] = primary_ball
    result["ball_detected"] = primary_ball is not None

    return result


def detect_ball_json_with_annotated_frame(
    frame: np.ndarray[Any, Any],
    device: str = "cpu",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect ball in a frame - JSON+Base64 mode.

    Executes YOLO inference, creates annotated frame with bounding boxes,
    and returns all data plus base64 encoded frame.
    Adds ball-specific fields: primary ball selection and ball_detected flag.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold (uses default if None)

    Returns:
        Dictionary with:
        - detections: List of detection dicts
        - count: Total number of detections
        - ball: Primary ball detection (highest confidence)
        - ball_detected: Boolean indicating if ball found
        - annotated_frame_base64: Base64 encoded annotated frame
    """
    if confidence is None:
        confidence = BALL_DETECTOR.default_confidence

    result = BALL_DETECTOR.detect_json_with_annotated_frame(
        frame, device=device, confidence=confidence
    )

    # Add ball-specific logic: select primary ball (highest confidence)
    primary_ball: Optional[Dict[str, Any]] = None
    if result["detections"]:
        primary_ball = max(result["detections"], key=lambda x: x["confidence"])

    result["ball"] = primary_ball
    result["ball_detected"] = primary_ball is not None

    return result


def get_ball_detection_model(device: str = "cpu") -> Any:
    """Get or create cached YOLO ball detection model.

    Returns the cached model instance from BALL_DETECTOR.
    Model is loaded on first access and cached for subsequent calls.

    Args:
        device: Device to run model on ('cpu' or 'cuda')

    Returns:
        YOLO model instance
    """
    return BALL_DETECTOR.get_model(device=device)


def run_ball_detection(
    frame: np.ndarray[Any, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Legacy function for plugin.py compatibility.

    Delegates to either detect_ball_json or detect_ball_json_with_annotated_frame
    based on include_annotated config flag.

    Args:
        frame: Input image frame
        config: Configuration dictionary with keys:
            - device: 'cpu' or 'cuda' (default: 'cpu')
            - confidence: Detection threshold (default: 0.20)
            - include_annotated: True to include annotated frame (default: False)

    Returns:
        Detection results dictionary
    """
    device = config.get("device", "cpu")
    confidence = config.get("confidence", BALL_DETECTOR.default_confidence)
    include_annotated = config.get("include_annotated", False)

    if include_annotated:
        return detect_ball_json_with_annotated_frame(
            frame, device=device, confidence=confidence
        )
    return detect_ball_json(frame, device=device, confidence=confidence)
