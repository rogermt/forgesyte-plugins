"""Player detection inference module.

Provides JSON and JSON+Base64 modes for player detection:
- detect_players_json(): Returns structured detection data
- detect_players_json_with_annotated(): Returns data + annotated frame
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration
from forgesyte_yolo_tracker.configs import get_model_path, get_confidence

logger = logging.getLogger(__name__)

MODEL_NAME = get_model_path("player_detection")
MODEL_PATH = str(Path(__file__).parent.parent / "models" / MODEL_NAME)
logger.debug(f"🔍 Player detection model: {MODEL_NAME}")
logger.debug(f"🔍 Model path: {MODEL_PATH}")

CONFIG = SoccerPitchConfiguration()
DEFAULT_CONFIDENCE = get_confidence("player")
logger.debug(f"🔍 Default confidence threshold: {DEFAULT_CONFIDENCE}")

CLASS_NAMES = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
TEAM_COLORS = {
    0: "#FFFFFF",  # Ball: White
    1: "#FFD700",  # Goalkeeper: Gold
    2: "#00BFFF",  # Player: DeepSkyBlue
    3: "#FF6347",  # Referee: Tomato
}
DEFAULT_NMS = 0.45

_model: Optional[YOLO] = None


def get_player_detection_model(device: str = "cpu") -> YOLO:
    """Get or create cached YOLO model.

    Args:
        device: Device to run model on ('cpu' or 'cuda')

    Returns:
        YOLO model instance
    """
    global _model
    if _model is None:
        logger.info(f"📦 Loading player detection model from: {MODEL_PATH}")
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


def _create_annotators() -> tuple:
    """Create box and label annotators.

    Returns:
        Tuple of (BoxAnnotator, LabelAnnotator)
    """
    colors = sv.ColorPalette.from_hex(list(TEAM_COLORS.values()))
    box_annotator = sv.BoxAnnotator(
        color=colors,
        thickness=2,
    )
    label_annotator = sv.LabelAnnotator(
        color=colors,
        text_color=sv.Color.from_hex("#FFFFFF"),
        text_padding=5,
        text_thickness=1,
    )
    return box_annotator, label_annotator


def detect_players_json(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Detect players in a frame - JSON mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with:
        - detections: List of detection dicts with xyxy, confidence, class_id, class_name
        - count: Total number of detections
        - classes: Dict of class_name -> count
    """
    logger.info(f"🎬 Starting player detection (device={device}, confidence={confidence})")
    logger.debug(f"🎬 Frame shape: {frame.shape}")
    
    model = get_player_detection_model(device=device)
    logger.info(f"🔫 Running YOLO inference...")
    result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
    logger.info(f"🔫 Inference complete. Processing results...")
    
    detections = sv.Detections.from_ultralytics(result)
    logger.info(f"📊 Found {len(detections)} detections")

    detection_list = []
    class_counts = {"ball": 0, "goalkeeper": 0, "player": 0, "referee": 0}

    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        conf = float(detections.confidence[i])
        cls = int(detections.class_id[i])
        class_name = CLASS_NAMES.get(cls, f"class_{cls}")

        detection_list.append(
            {
                "xyxy": xyxy.tolist(),
                "confidence": conf,
                "class_id": cls,
                "class_name": class_name,
            }
        )

        if class_name in class_counts:
            class_counts[class_name] += 1

    logger.info(f"✅ Detection complete: {len(detection_list)} objects found")
    logger.info(f"✅ Class breakdown: {class_counts}")
    
    return {
        "detections": detection_list,
        "count": len(detection_list),
        "classes": class_counts,
    }


def detect_players_json_with_annotated_frame(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Detect players in a frame - JSON+Base64 mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with:
        - detections: List of detection dicts
        - count: Total number of detections
        - classes: Dict of class_name -> count
        - annotated_frame_base64: Base64 encoded annotated frame
    """
    model = get_player_detection_model(device=device)
    result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    detection_list = []
    class_counts = {"ball": 0, "goalkeeper": 0, "player": 0, "referee": 0}

    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        conf = float(detections.confidence[i])
        cls = int(detections.class_id[i])
        class_name = CLASS_NAMES.get(cls, f"class_{cls}")

        detection_list.append(
            {
                "xyxy": xyxy.tolist(),
                "confidence": conf,
                "class_id": cls,
                "class_name": class_name,
            }
        )

        if class_name in class_counts:
            class_counts[class_name] += 1

    # Create annotated frame
    box_annotator, label_annotator = _create_annotators()
    labels = [CLASS_NAMES.get(int(cls), f"class_{cls}") for cls in detections.class_id]

    annotated = frame.copy()
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)

    return {
        "detections": detection_list,
        "count": len(detection_list),
        "classes": class_counts,
        "annotated_frame_base64": _encode_frame_to_base64(annotated),
    }


def run_player_detection(frame: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
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
        return detect_players_json_with_annotated_frame(frame, device=device, confidence=confidence)
    return detect_players_json(frame, device=device, confidence=confidence)
