"""Video ball detection module.

Roboflow-style full video processing:
- run_ball_detection_video(): Process video and save annotated output
- run_ball_detection_video_frames(): Generator yielding annotated frames
"""

from typing import Iterator, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

MODEL_PATH = "src/forgesyte_yolo_tracker/models/football-ball-detection-v2.pt"
BALL_COLOR = sv.Color.from_hex("#FF6347")
DEFAULT_CONFIDENCE = 0.20

_model: Optional[YOLO] = None


def get_model(device: str = "cpu") -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH).to(device=device)
    return _model


def run_ball_detection_video_frames(
    source_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Iterator[np.ndarray]:
    """Generate ball detection frames from video."""
    model = get_model(device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    box_annotator = sv.BoxAnnotator(color=BALL_COLOR, thickness=2)

    for frame in frame_generator:
        result = model(frame, imgsz=640, conf=confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated = frame.copy()
        annotated = box_annotator.annotate(annotated, detections)

        yield annotated


def run_ball_detection_video(
    source_video_path: str,
    target_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Process video and save ball detection output."""
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in run_ball_detection_video_frames(
            source_video_path=source_video_path,
            device=device,
            confidence=confidence,
        ):
            sink.write_frame(frame)
