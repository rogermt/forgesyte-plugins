"""Team classification inference module.

Provides JSON and JSON+Base64 modes for team classification using TeamClassifier:
- classify_teams_json(): Returns team assignments for players
- classify_teams_json_with_annotated_frame(): Returns data + annotated frame

Team classification uses on-the-fly clustering (collect crops → UMAP → KMeans → predict).
"""

import base64
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from forgesyte_yolo_tracker.configs import get_model_path, get_confidence
from forgesyte_yolo_tracker.utils import TeamClassifier

MODEL_NAME = get_model_path("player_detection")
MODEL_PATH = str(Path(__file__).parent.parent / "models" / MODEL_NAME)
DEFAULT_CONFIDENCE = get_confidence("player")

TEAM_A_COLOR = sv.Color.from_hex("#00BFFF")  # DeepSkyBlue
TEAM_B_COLOR = sv.Color.from_hex("#FF1493")  # DeepPink
GK_COLOR = sv.Color.from_hex("#FFD700")  # Gold

_model: Optional[YOLO] = None
_team_classifier: Optional[TeamClassifier] = None


def get_player_detection_model(device: str = "cpu") -> YOLO:
    """Get or create cached YOLO model."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH).to(device=device)
    return _model


def get_team_classifier(device: str = "cpu") -> TeamClassifier:
    """Get or create cached TeamClassifier."""
    global _team_classifier
    if _team_classifier is None:
        _team_classifier = TeamClassifier(device=device)
    return _team_classifier


def _encode_frame_to_base64(frame: np.ndarray) -> str:
    """Encode frame to base64 JPEG."""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")


def _get_team_color(team_id: int) -> sv.Color:
    """Get color for team ID."""
    if team_id == 0:
        return TEAM_A_COLOR
    return TEAM_B_COLOR


def classify_teams_json(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Classify players into teams - JSON mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with:
        - detections: Player detections with team_id
        - team_ids: List of team IDs (0=Team A, 1=Team B)
        - team_counts: Dict of team_a and team_b counts
    """
    model = get_player_detection_model(device=device)
    team_classifier = get_team_classifier(device=device)

    result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    player_detections = []
    player_crops = []

    for i in range(len(detections)):
        cls = int(detections.class_id[i])
        if cls == 0:  # Only classify players (not goalkeepers/referees)
            xyxy = detections.xyxy[i]
            player_detections.append(
                {
                    "xyxy": xyxy.tolist(),
                    "confidence": float(detections.confidence[i]),
                    "class_id": cls,
                }
            )
            # Crop player for team classification
            x1, y1, x2, y2 = xyxy.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                player_crops.append(crop)

    team_ids = []
    team_counts = {"team_a": 0, "team_b": 0}

    if len(player_crops) > 0:
        predictions = team_classifier.predict(player_crops)
        for pred in predictions:
            team_id = int(pred)
            team_ids.append(team_id)
            if team_id == 0:
                team_counts["team_a"] += 1
            else:
                team_counts["team_b"] += 1

    # Merge team_ids into detections
    crop_idx = 0
    for det in player_detections:
        if det["class_id"] == 0 and crop_idx < len(team_ids):
            det["team_id"] = team_ids[crop_idx]
            crop_idx += 1
        elif det["class_id"] == 0:
            det["team_id"] = -1  # No prediction
        else:
            det["team_id"] = -1  # Not a regular player

    return {
        "detections": player_detections,
        "team_ids": team_ids,
        "team_counts": team_counts,
    }


def classify_teams_json_with_annotated_frame(
    frame: np.ndarray,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Dict[str, Any]:
    """Classify players into teams - JSON+Base64 mode.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Returns:
        Dictionary with detections, team_ids, team_counts, annotated_frame_base64
    """
    model = get_player_detection_model(device=device)
    team_classifier = get_team_classifier(device=device)

    result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(result)

    player_detections = []
    player_crops = []
    crop_indices = []

    for i in range(len(detections)):
        cls = int(detections.class_id[i])
        if cls == 0:
            xyxy = detections.xyxy[i]
            player_detections.append(
                {
                    "xyxy": xyxy.tolist(),
                    "confidence": float(detections.confidence[i]),
                    "class_id": cls,
                }
            )
            crop_indices.append(i)
            x1, y1, x2, y2 = xyxy.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                player_crops.append(crop)

    team_ids = []
    team_counts = {"team_a": 0, "team_b": 0}

    if len(player_crops) > 0:
        predictions = team_classifier.predict(player_crops)
        for pred in predictions:
            team_id = int(pred)
            team_ids.append(team_id)
            if team_id == 0:
                team_counts["team_a"] += 1
            else:
                team_counts["team_b"] += 1

    # Create annotated frame with team colors
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.from_hex("#FFFFFF"),
        text_padding=5,
        text_thickness=1,
    )

    # Create custom colors for each detection based on team
    detection_colors = []
    crop_idx = 0
    for i in range(len(detections)):
        if i in crop_indices and crop_idx < len(team_ids):
            detection_colors.append(_get_team_color(team_ids[crop_idx]))
            crop_idx += 1
        elif int(detections.class_id[i]) == 1:  # Goalkeeper
            detection_colors.append(GK_COLOR)
        else:
            detection_colors.append(sv.Color.from_hex("#808080"))  # Referee/unknown

    # Update detections with team info
    crop_idx = 0
    for det in player_detections:
        if det["class_id"] == 0 and crop_idx < len(team_ids):
            det["team_id"] = team_ids[crop_idx]
            crop_idx += 1
        else:
            det["team_id"] = -1

    # Create labels with team info
    labels = []
    crop_idx = 0
    for i, det in enumerate(player_detections):
        if det["class_id"] == 0 and crop_idx < len(team_ids):
            labels.append(f"Team {'A' if team_ids[crop_idx] == 0 else 'B'} #{crop_idx}")
            crop_idx += 1
        elif det["class_id"] == 1:
            labels.append("GK")
        else:
            labels.append("Ref")

    annotated = frame.copy()
    annotated = box_annotator.annotate(annotated, detections, custom_colors=detection_colors)
    annotated = label_annotator.annotate(annotated, detections, labels=labels)

    return {
        "detections": player_detections,
        "team_ids": team_ids,
        "team_counts": team_counts,
        "annotated_frame_base64": _encode_frame_to_base64(annotated),
    }


def run_team_classification(frame: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy function for plugin.py compatibility."""
    device = config.get("device", "cpu")
    confidence = config.get("confidence", DEFAULT_CONFIDENCE)
    include_annotated = config.get("include_annotated", False)

    if include_annotated:
        return classify_teams_json_with_annotated_frame(frame, device=device, confidence=confidence)
    return classify_teams_json(frame, device=device, confidence=confidence)
