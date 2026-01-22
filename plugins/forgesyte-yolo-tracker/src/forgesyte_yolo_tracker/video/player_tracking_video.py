"""Video player tracking module.

Roboflow-style full video processing with ByteTrack:
- run_player_tracking_video(): Process video and save annotated output
- run_player_tracking_video_frames(): Generator yielding annotated frames
"""

from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import supervision as sv
from ultralytics import YOLO

from forgesyte_yolo_tracker.configs import get_model_path

MODEL_NAME = get_model_path("player_detection")
MODEL_PATH = str(Path(__file__).parent.parent / "models" / MODEL_NAME)

CLASS_NAMES = {0: "player", 1: "goalkeeper", 2: "referee"}
TRACK_COLORS = sv.ColorPalette.from_hex(["#00BFFF", "#FFD700", "#FF6347"])
DEFAULT_CONFIDENCE = 0.25

_model: Optional[YOLO] = None
_tracker: Optional[sv.ByteTrack] = None


def get_model(device: str = "cpu") -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH).to(device=device)
    return _model


def get_tracker() -> sv.ByteTrack:
    global _tracker
    if _tracker is None:
        _tracker = sv.ByteTrack(
            track_thresh=DEFAULT_CONFIDENCE,
            track_buffer=30,
            match_thresh=0.8,
            frame_rate=30,
        )
    return _tracker


def run_player_tracking_video_frames(
    source_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Iterator[np.ndarray]:
    """Generate tracked frames from video."""
    model = get_model(device)
    tracker = get_tracker()
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    box_annotator = sv.BoxAnnotator(color=TRACK_COLORS, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=TRACK_COLORS,
        text_color=sv.Color.from_hex("#FFFFFF"),
        text_padding=5,
        text_thickness=1,
    )

    for frame in frame_generator:
        result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [
            f"#{int(tid) if tid else '?'} {CLASS_NAMES.get(int(cls), f'class_{cls}')}"
            for tid, cls in zip(detections.track_id or [-1] * len(detections), detections.class_id)
        ]

        annotated = frame.copy()
        annotated = box_annotator.annotate(annotated, detections)
        annotated = label_annotator.annotate(annotated, detections, labels=labels)

        yield annotated


def run_player_tracking_video(
    source_video_path: str,
    target_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Process video and save tracked output."""
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in run_player_tracking_video_frames(
            source_video_path=source_video_path,
            device=device,
            confidence=confidence,
        ):
            sink.write_frame(frame)
