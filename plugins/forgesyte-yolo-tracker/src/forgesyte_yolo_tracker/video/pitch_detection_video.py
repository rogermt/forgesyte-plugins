"""Video pitch detection module.

Roboflow-style full video processing:
- run_pitch_detection_video(): Process video and save annotated output
- run_pitch_detection_video_frames(): Generator yielding annotated frames
"""

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from forgesyte_yolo_tracker.configs import get_model_path
from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

MODEL_NAME = get_model_path("pitch_detection")
MODEL_PATH = str(Path(__file__).parent.parent / "models" / MODEL_NAME)
CONFIG = SoccerPitchConfiguration()

PITCH_COLOR = sv.Color.from_hex("#00FF00")
KEYPOINT_COLOR = sv.Color.from_hex("#FF0000")
DEFAULT_CONFIDENCE = 0.25

_model: Optional[YOLO] = None


def get_model(device: str = "cpu") -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_PATH).to(device=device)
    return _model


def run_pitch_detection_video_frames(
    source_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Iterator[np.ndarray]:
    """Generate pitch detection frames from video."""
    model = get_model(device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    for frame in frame_generator:
        result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
        annotated = frame.copy()

        if result.keypoints is not None and result.keypoints.xy is not None:
            keypoints_xy = result.keypoints.xy.cpu().numpy()[0]
            keypoints_conf = (
                result.keypoints.conf.cpu().numpy()[0]
                if result.keypoints.conf is not None
                else None
            )

            for i, (x, y) in enumerate(keypoints_xy):
                conf = keypoints_conf[i] if keypoints_conf is not None else 1.0
                if conf > confidence * 0.5:
                    x_int, y_int = int(x), int(y)
                    cv2.circle(annotated, (x_int, y_int), 5, (0, 0, 255), -1)
                    name = CONFIG.keypoint_names.get(i, f"kp_{i}")
                    cv2.putText(
                        annotated,
                        name.replace("_", " "),
                        (x_int + 5, y_int),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                    )

        yield annotated


def run_pitch_detection_video(
    source_video_path: str,
    target_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Process video and save pitch detection output."""
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in run_pitch_detection_video_frames(
            source_video_path=source_video_path,
            device=device,
            confidence=confidence,
        ):
            sink.write_frame(frame)
