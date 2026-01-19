"""Video radar module.

Roboflow-style full video processing:
- run_radar_video(): Process video and save output with radar overlay
- run_radar_video_frames(): Generator yielding frames with radar
"""

from typing import Iterator, Optional, Tuple

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration
from forgesyte_yolo_tracker.utils import ViewTransformer

PLAYER_MODEL_PATH = "src/forgesyte_yolo_tracker/models/football-player-detection-v3.pt"
PITCH_MODEL_PATH = "src/forgesyte_yolo_tracker/models/football-pitch-detection-v1.pt"
CONFIG = SoccerPitchConfiguration()

TEAM_A_COLOR = (0, 191, 255)
TEAM_B_COLOR = (255, 20, 147)
GK_COLOR = (0, 215, 255)
BALL_COLOR = (135, 206, 250)
DEFAULT_CONFIDENCE = 0.25

_player_model: Optional[YOLO] = None
_pitch_model: Optional[YOLO] = None


def get_player_model(device: str = "cpu") -> YOLO:
    global _player_model
    if _player_model is None:
        _player_model = YOLO(PLAYER_MODEL_PATH).to(device=device)
    return _player_model


def get_pitch_model(device: str = "cpu") -> YOLO:
    global _pitch_model
    if _pitch_model is None:
        _pitch_model = YOLO(PITCH_MODEL_PATH).to(device=device)
    return _pitch_model


def _create_radar_image(
    radar_points: list,
    radar_size: Tuple[int, int] = (600, 300),
) -> np.ndarray:
    """Create radar visualization."""
    radar_w, radar_h = radar_size
    radar = np.zeros((radar_h, radar_w, 3), dtype=np.uint8)

    cv2.rectangle(radar, (0, 0), (radar_w, radar_h), (50, 50, 50), 2)
    cv2.line(radar, (radar_w // 2, 0), (radar_w // 2, radar_h), (100, 100, 100), 1)
    cv2.circle(radar, (radar_w // 2, radar_h // 2), 20, (100, 100, 100), 1)

    for point in radar_points:
        x, y = point["xy"]
        team_id = point.get("team_id", -1)
        point_type = point.get("type", "player")

        if point_type == "ball":
            color = BALL_COLOR
            radius = 6
        elif team_id == 0:
            color = TEAM_A_COLOR
            radius = 8
        elif team_id == 1:
            color = TEAM_B_COLOR
            radius = 8
        elif point_type == "goalkeeper":
            color = GK_COLOR
            radius = 8
        else:
            color = (128, 128, 128)
            radius = 6

        cv2.circle(radar, (int(x), int(y)), radius, color, -1)

    return radar


def run_radar_video_frames(
    source_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> Iterator[np.ndarray]:
    """Generate frames with radar overlay from video."""
    player_model = get_player_model(device)
    pitch_model = get_pitch_model(device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    radar_w, radar_h = CONFIG.radar_resolution

    for frame in frame_generator:
        player_result = player_model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
        pitch_result = pitch_model(frame, imgsz=1280, conf=confidence, verbose=False)[0]

        player_detections = sv.Detections.from_ultralytics(player_result)
        radar_points = []

        if pitch_result.keypoints is not None and pitch_result.keypoints.xy is not None:
            keypoints_xy = pitch_result.keypoints.xy.cpu().numpy()[0]
            keypoints_conf = (
                pitch_result.keypoints.conf.cpu().numpy()[0]
                if pitch_result.keypoints.conf is not None
                else None
            )

            valid_kp_indices = [
                i for i, conf in enumerate(keypoints_conf) if conf > confidence * 0.5
            ]

            if len(valid_kp_indices) >= 4:
                src_pts = np.array(
                    [keypoints_xy[i] for i in valid_kp_indices[:4]], dtype=np.float32
                )
                tgt_pts = np.array(
                    [CONFIG.vertices[i] for i in valid_kp_indices[:4]], dtype=np.float32
                )

                try:
                    transformer = ViewTransformer(src_pts, tgt_pts)

                    for i in range(len(player_detections)):
                        xyxy = player_detections.xyxy[i]
                        cls = int(player_detections.class_id[i])

                        center_x = float((xyxy[0] + xyxy[2]) / 2)
                        center_y = float((xyxy[1] + xyxy[3]) / 2)

                        transformed = transformer.transform_points(
                            np.array([[center_x, center_y]], dtype=np.float32)
                        )
                        rx, ry = CONFIG.world_to_radar(transformed[0][0], transformed[0][1])

                        radar_points.append(
                            {
                                "xy": [rx, ry],
                                "team_id": -1,
                                "type": "goalkeeper" if cls == 1 else "player",
                            }
                        )
                except Exception:
                    pass

        radar_image = _create_radar_image(radar_points, (radar_w, radar_h))

        annotated = frame.copy()
        radar_h, radar_w = radar_image.shape[:2]
        annotated[-radar_h - 10 : -10, -radar_w - 10 : -10] = radar_image
        cv2.putText(
            annotated,
            "Radar",
            (annotated.shape[1] - radar_w, annotated.shape[0] - radar_h - 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        yield annotated


def run_radar_video(
    source_video_path: str,
    target_video_path: str,
    device: str = "cpu",
    confidence: float = DEFAULT_CONFIDENCE,
) -> None:
    """Process video and save with radar overlay."""
    video_info = sv.VideoInfo.from_video_path(source_video_path)
    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame in run_radar_video_frames(
            source_video_path=source_video_path,
            device=device,
            confidence=confidence,
        ):
            sink.write_frame(frame)
