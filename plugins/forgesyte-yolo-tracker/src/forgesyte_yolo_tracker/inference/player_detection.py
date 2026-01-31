"""Player detection inference module.

Provides JSON and JSON+Base64 modes for player detection using BaseDetector:
- detect_players_json(): Returns structured detection data
- detect_players_json_with_annotated_frame(): Returns data + annotated frame

This module uses BaseDetector to eliminate code duplication across detectors.
Player-specific configuration is defined once and reused via wrapper functions.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from forgesyte_yolo_tracker.configs import get_confidence, get_model_path
from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

logger = logging.getLogger(__name__)

# Class names matching trained model's 4-class structure
CLASS_NAMES: Dict[int, str] = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}

# Team colors for each class
TEAM_COLORS: Dict[int, str] = {
    0: "#FFFFFF",  # Ball: White
    1: "#FFD700",  # Goalkeeper: Gold
    2: "#00BFFF",  # Player: DeepSkyBlue
    3: "#FF6347",  # Referee: Tomato
}

# Model path for player detection (full path for test compatibility)
MODEL_NAME = get_model_path("player_detection")
MODEL_PATH: str = str(Path(__file__).parent.parent / "models" / MODEL_NAME)

# Player detector instance with configuration
PLAYER_DETECTOR = BaseDetector(
    detector_name="player",
    model_name=MODEL_PATH,
    default_confidence=get_confidence("player"),
    imgsz=1280,
    class_names=CLASS_NAMES,
    colors=TEAM_COLORS,
)


def detect_players_json(
    frame: np.ndarray[Any, Any],
    device: str = "cpu",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect players in a frame - JSON mode.

    Executes YOLO inference and returns structured detection results
    including player, goalkeeper, ball, and referee detections.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold (uses default if None)

    Returns:
        Dictionary with:
        - detections: List of detection dicts with xyxy, confidence, class_id, class_name
        - count: Total number of detections
        - classes: Dict of class_name -> count
    """
    if confidence is None:
        confidence = PLAYER_DETECTOR.default_confidence

    try:
        return PLAYER_DETECTOR.detect_json(frame, device=device, confidence=confidence)
    except Exception as e:
        logger.error(f"detect_players_json failed: {e}", exc_info=True)
        raise


def detect_players_json_with_annotated_frame(
    frame: np.ndarray[Any, Any],
    device: str = "cpu",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect players in a frame - JSON+Base64 mode.

    Executes YOLO inference, creates annotated frame with bounding boxes
    and labels, and returns all data plus base64 encoded frame.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold (uses default if None)

    Returns:
        Dictionary with:
        - detections: List of detection dicts
        - count: Total number of detections
        - classes: Dict of class_name -> count
        - annotated_frame_base64: Base64 encoded annotated frame
    """
    if confidence is None:
        confidence = PLAYER_DETECTOR.default_confidence

    return PLAYER_DETECTOR.detect_json_with_annotated_frame(
        frame, device=device, confidence=confidence
    )


def get_player_detection_model(device: str = "cpu") -> Any:
    """Get or create cached YOLO player detection model.

    Returns the cached model instance from PLAYER_DETECTOR.
    Model is loaded on first access and cached for subsequent calls.

    Args:
        device: Device to run model on ('cpu' or 'cuda')

    Returns:
        YOLO model instance
    """
    return PLAYER_DETECTOR.get_model(device=device)


def run_player_detection(frame: np.ndarray[Any, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function for plugin.py compatibility.

    Delegates to either detect_players_json or detect_players_json_with_annotated_frame
    based on include_annotated config flag.

    Args:
        frame: Input image frame
        config: Configuration dictionary with keys:
            - device: 'cpu' or 'cuda' (default: 'cpu')
            - confidence: Detection threshold (default: 0.25)
            - include_annotated: True to include annotated frame (default: False)

    Returns:
        Detection results dictionary
    """
    device = config.get("device", "cpu")
    confidence = config.get("confidence", PLAYER_DETECTOR.default_confidence)
    include_annotated = config.get("include_annotated", False)

    if include_annotated:
        return detect_players_json_with_annotated_frame(frame, device=device, confidence=confidence)
    return detect_players_json(frame, device=device, confidence=confidence)
