"""Video team classification module.

Roboflow-style full video processing:
- run_team_classification_video(): Process video and save annotated output
- run_team_classification_video_frames(): Generator yielding annotated frames
"""

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import supervision as sv
from ultralytics import YOLO

from forgesyte_yolo_tracker.utils import TeamClassifier
from forgesyte_yolo_tracker.configs import get_model_path

MODEL_NAME = get_model_path("player_detection")
MODEL_PATH = str(Path(__file__).parents[2] / "models" / MODEL_NAME)

TEAM_A_COLOR = sv.Color.from_hex("#00BFFF")
TEAM_B_COLOR = sv.Color.from_hex("#FF1493")
GK_COLOR = sv.Color.from_hex("#FFD700")
DEFAULT_CONFIDENCE = 0.25

_model: Optional[YOLO] = None
_classifier: Optional[TeamClassifier] = None


def get_model(device: str = "cpu") -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH).to(device=device)
    return _model


def get_classifier(device: str = "cpu") -> TeamClassifier:
    global _classifier
    if _classifier is None:
        _classifier = TeamClassifier(device=device)
    return _classifier


def _get_team_color(team_id: int) -> sv.Color:
    if team_id == 0:
        return TEAM_A_COLOR
    return TEAM_B_COLOR


def run_team_classification_video_frames(
    source_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Iterator[np.ndarray]:
    """Generate team classification frames from video."""
    model = get_model(device)
    classifier = get_classifier(device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(
        text_color=sv.Color.from_hex("#FFFFFF"),
        text_padding=5,
        text_thickness=1,
    )

    for frame in frame_generator:
        result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        player_indices = [i for i, cls in enumerate(detections.class_id) if int(cls) == 0]
        player_crops = []
        for i in player_indices:
            xyxy = detections.xyxy[i]
            x1, y1, x2, y2 = xyxy.astype(int)
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                player_crops.append(crop)

        team_predictions = classifier.predict(player_crops) if player_crops else []

        colors = []
        labels = []
        pred_idx = 0
        for i, cls in enumerate(detections.class_id):
            if int(cls) == 0 and pred_idx < len(team_predictions):
                team_id = int(team_predictions[pred_idx])
                colors.append(_get_team_color(team_id))
                labels.append(f"Team {'A' if team_id == 0 else 'B'}")
                pred_idx += 1
            elif int(cls) == 1:
                colors.append(GK_COLOR)
                labels.append("GK")
            else:
                colors.append(sv.Color.from_hex("#808080"))
                labels.append("Ref")

        annotated = frame.copy()
        annotated = box_annotator.annotate(annotated, detections, custom_colors=colors)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)

        yield annotated


def run_team_classification_video(
    source_video_path: str,
    target_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Process video and save team classification output."""
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in run_team_classification_video_frames(
            source_video_path=source_video_path,
            device=device,
            confidence=confidence,
        ):
            sink.write_frame(frame)
