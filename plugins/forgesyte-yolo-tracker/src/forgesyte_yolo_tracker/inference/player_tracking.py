"""Player tracking inference module.

Provides JSON and JSON+Base64 modes for player tracking with ByteTrack:
- track_players_json(): Returns tracked detection data with IDs
- track_players_json_with_annotated_frame(): Returns data + annotated frame with labels
"""

import base64
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

MODEL_PATH = "src/forgesyte_yolo_tracker/models/football-player-detection-v3.pt"
CONFIG = SoccerPitchConfiguration()

CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee"}
TRACK_COLORS = sv.ColorPalette.from_hex(["#00BFFF", "#FFD700", "#FF6347"])
DEFAULT_CONFIDENCE = 0.25
DEFAULT_NMS = 0.45

_model: Optional[YOLO] = None
_tracker: Optional[sv.ByteTrack] = None


def get_player_detection_model(device: str = "cpu") -> YOLO:
    """Get or create cached YOLO model."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH).to(device=device)
    return _model


def get_tracker() -> sv.ByteTrack:
    """Get or create ByteTrack instance."""
    global _tracker
    if _tracker is None:
        _tracker = sv.ByteTrack(
            track_thresh=DEFAULT_CONFIDENCE,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30,
        )
    return _tracker


def _encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode frame to base64 JPEG."""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def _create_annotators() -> Tuple[sv.BoxAnnotator, sv.LabelAnnotator]:
    """Create box and label annotators with tracking support."""
    box_annotator = sv.BoxAnnotator(
        color=TRACK_COLORS,
        thickness=2,
    )
    label_annotator = sv.LabelAnnotator(
        color=TRACK_COLORS,
        text_color=sv.Color.from_hex("#FFFFFF"),
        text_padding=5,
        text_thickness=1,
    )
    return box_annotator, label_annotator


def track_players_json(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Track players in a frame - JSON mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with:
        - detections: List of detection dicts with xyxy, confidence, class_id, tracking_id
        - count: Total number of tracked players
        - track_ids: List of unique tracking IDs
    """
    model = get_player_detection_model(device=device)
    tracker = get_tracker()

    result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    detections = tracker.update_with_detections(detections)

    detection_list = []
    track_ids_set = set()

    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        conf = float(detections.confidence[i])
        cls = int(detections.class_id[i])
        track_id = int(detections.track_id[i]) if detections.track_id is not None else -1

        class_name = CLASS_NAMES.get(cls, f"class_{cls}")

        detection_list.append(
            {
                "xyxy": xyxy.tolist(),
                "confidence": conf,
                "class_id": cls,
                "class_name": class_name,
                "tracking_id": track_id,
            }
        )

        if track_id >= 0:
            track_ids_set.add(track_id)

    return {
        "detections": detection_list,
        "count": len(detection_list),
        "track_ids": sorted(list(track_ids_set)),
    }


def track_players_json_with_annotated_frame(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Track players in a frame - JSON+Base64 mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with detections, count, track_ids, and annotated_frame_base64
    """
    model = get_player_detection_model(device=device)
    tracker = get_tracker()

    result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    detections = tracker.update_with_detections(detections)

    detection_list = []
    track_ids_set = set()

    for i in range(len(detections)):
        xyxy = detections.xyxy[i]
        conf = float(detections.confidence[i])
        cls = int(detections.class_id[i])
        track_id = int(detections.track_id[i]) if detections.track_id is not None else -1

        class_name = CLASS_NAMES.get(cls, f"class_{cls}")

        detection_list.append(
            {
                "xyxy": xyxy.tolist(),
                "confidence": conf,
                "class_id": cls,
                "class_name": class_name,
                "tracking_id": track_id,
            }
        )

        if track_id >= 0:
            track_ids_set.add(track_id)

    # Create annotated frame with tracking labels
    box_annotator, label_annotator = _create_annotators()

    labels = [
        f"#{track_id if track_id >= 0 else '?'} {CLASS_NAMES.get(int(cls), f'class_{cls}')}"
        for track_id, cls in zip(detections.track_id or [-1] * len(detections), detections.class_id)
    ]

    annotated = frame.copy()
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)

    return {
        "detections": detection_list,
        "count": len(detection_list),
        "track_ids": sorted(list(track_ids_set)),
        "annotated_frame_base64": _encode_frame_to_base64(annotated),
    }


def run_player_tracking(frame: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function for plugin.py compatibility."""
    device = config.get("device", "cpu")
    confidence = config.get("confidence", DEFAULT_CONFIDENCE)
    include_annotated = config.get("include_annotated", False)

    if include_annotated:
        return track_players_json_with_annotated_frame(frame, device=device, confidence=confidence)
    return track_players_json(frame, device=device, confidence=confidence)
