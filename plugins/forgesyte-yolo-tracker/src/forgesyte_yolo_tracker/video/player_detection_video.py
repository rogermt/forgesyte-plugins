"""Video player detection module.

Roboflow-style full video processing:
- run_player_detection_video(): Process video and save annotated output
- run_player_detection_video_frames(): Generator yielding annotated frames
"""

from typing import Iterator, Optional

import numpy as np
import supervision as sv
from ultralytics import YOLO

from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

MODEL_PATH = "src/forgesyte_yolo_tracker/models/football-player-detection-v3.pt"
CONFIG = SoccerPitchConfiguration()

CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee"}
TEAM_COLORS = {
    0: "#00BFFF",  # Team A
    1: "#FFD700",  # Goalkeeper
    2: "#FF6347",  # Referee
}
DEFAULT_CONFIDENCE = 0.25

_model: Optional[YOLO] = None


def get_model(device: str = "cpu") -> YOLO:
    """Get or create cached YOLO model."""
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH).to(device=device)
    return _model


def run_player_detection_video_frames(
    source_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Iterator[np.ndarray]:
    """Generate annotated frames from video.

    Args:
        source_video_path: Path to input video
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold

    Yields:
        Annotated frames as numpy arrays
    """
    model = get_model(device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    colors = sv.ColorPalette.from_hex(list(TEAM_COLORS.values()))
    box_annotator = sv.BoxAnnotator(color=colors, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=colors,
        text_color=sv.Color.from_hex("#FFFFFF"),
        text_padding=5,
        text_thickness=1,
    )

    for frame in frame_generator:
        result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        labels = [CLASS_NAMES.get(int(cls), f"class_{cls}") for cls in detections.class_id]

        annotated = frame.copy()
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)

        yield annotated


def run_player_detection_video(
    source_video_path: str,
    target_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Process video and save annotated output.

    Args:
        source_video_path: Path to input video
        target_video_path: Path to output video
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold
    """
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in run_player_detection_video_frames(
            source_video_path=source_video_path,
            device=device,
            confidence=confidence,
        ):
            sink.write_frame(frame)
