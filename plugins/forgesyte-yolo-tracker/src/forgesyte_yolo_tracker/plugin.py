"""ForgeSyte YOLO Tracker Plugin — BasePlugin Architecture.

Frame-based JSON tools for football analysis:
- player_detection
- player_tracking
- ball_detection
- pitch_detection
- radar
"""

import io
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image

# Try to import BasePlugin from server, fallback for testing
try:
    from app.plugins.base import BasePlugin
except (ImportError, ModuleNotFoundError):
    # Fallback for standalone/test environments
    from abc import ABC

    class BasePlugin(ABC):  # type: ignore  # noqa: B024, F811
        """Fallback BasePlugin for testing."""

        name: str = ""
        tools: Dict[str, Any] = {}

        def run_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
            raise NotImplementedError  # pragma: no cover

        def on_load(self) -> None:  # pragma: no cover  # noqa: B027
            pass

        def on_unload(self) -> None:  # pragma: no cover  # noqa: B027
            pass


from forgesyte_yolo_tracker.inference.ball_detection import (
    detect_ball_json,
    detect_ball_json_with_annotated_frame,
)
from forgesyte_yolo_tracker.inference.pitch_detection import (
    detect_pitch_json,
    detect_pitch_json_with_annotated_frame,
)
from forgesyte_yolo_tracker.inference.player_detection import (
    detect_players_json,
    detect_players_json_with_annotated_frame,
)
from forgesyte_yolo_tracker.inference.player_tracking import (
    track_players_json,
    track_players_json_with_annotated_frame,
)
from forgesyte_yolo_tracker.inference.radar import generate_radar_json as radar_json
from forgesyte_yolo_tracker.inference.radar import radar_json_with_annotated_frame
from forgesyte_yolo_tracker.configs import load_model_config

logger = logging.getLogger(__name__)


def _get_default_device() -> str:
    """Get default device from config file.

    Returns:
        Device string from config (e.g., 'cuda' or 'cpu'), defaults to 'cpu'.
    """
    try:
        config = load_model_config()
        return config.get("device", "cpu")
    except Exception:
        return "cpu"


# ---------------------------------------------------------
# Image decoding helpers (Phase 12 contract: bytes input)
# ---------------------------------------------------------
def _decode_image_bytes(
    image_bytes: bytes, tool_name: str
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Decode raw image bytes to numpy array.

    Args:
        image_bytes: Raw image bytes (PNG, JPG, etc.)
        tool_name: Name of tool calling this (for error logging)

    Returns:
        (frame as numpy array, None) or (None, error_dict)
    """
    try:
        if not isinstance(image_bytes, (bytes, bytearray)):
            raise ValueError(f"Expected bytes, got {type(image_bytes).__name__}")

        # Decode bytes → PIL Image → numpy array
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame = np.array(image)

        return frame, None

    except Exception as e:
        msg = str(e)
        logger.warning(f"Image decode failed in {tool_name}: {msg}")
        return None, {
            "error": "invalid_image",
            "message": f"Failed to decode image: {msg}",
            "plugin": "yolo-tracker",
            "tool": tool_name,
        }


# ---------------------------------------------------------
# Tool functions (Phase 12 contract: accept image_bytes)
# ---------------------------------------------------------
def _tool_player_detection(
    image_bytes: bytes, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    frame, error = _decode_image_bytes(image_bytes, "player_detection")
    if error:
        return error
    if annotated and frame is not None:
        return detect_players_json_with_annotated_frame(frame, device=device)
    if frame is not None:
        return detect_players_json(frame, device=device)
    return {"error": "image_decode_failed"}


def _tool_player_tracking(
    image_bytes: bytes, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    frame, error = _decode_image_bytes(image_bytes, "player_tracking")
    if error:
        return error
    if annotated and frame is not None:
        return track_players_json_with_annotated_frame(frame, device=device)
    if frame is not None:
        return track_players_json(frame, device=device)
    return {"error": "image_decode_failed"}


def _tool_ball_detection(
    image_bytes: bytes, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    frame, error = _decode_image_bytes(image_bytes, "ball_detection")
    if error:
        return error
    if annotated and frame is not None:
        return detect_ball_json_with_annotated_frame(frame, device=device)
    if frame is not None:
        return detect_ball_json(frame, device=device)
    return {"error": "image_decode_failed"}


def _tool_pitch_detection(
    image_bytes: bytes, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    frame, error = _decode_image_bytes(image_bytes, "pitch_detection")
    if error:
        return error
    if annotated and frame is not None:
        return detect_pitch_json_with_annotated_frame(frame, device=device)
    if frame is not None:
        return detect_pitch_json(frame, device=device)
    return {"error": "image_decode_failed"}


def _tool_radar(
    image_bytes: bytes, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    frame, error = _decode_image_bytes(image_bytes, "radar")
    if error:
        return error
    if annotated and frame is not None:
        return radar_json_with_annotated_frame(frame, device=device)
    if frame is not None:
        return radar_json(frame, device=device)
    return {"error": "image_decode_failed"}


def _tool_player_detection_video(
    video_path: str, output_path: str, device: str = "cpu"
) -> Dict[str, str]:
    from forgesyte_yolo_tracker.video.player_detection_video import run_player_detection_video

    run_player_detection_video(video_path, output_path, device=device)
    return {"status": "success", "output_path": output_path}


def _tool_player_tracking_video(
    video_path: str, output_path: str, device: str = "cpu"
) -> Dict[str, str]:
    from forgesyte_yolo_tracker.video.player_tracking_video import run_player_tracking_video

    run_player_tracking_video(video_path, output_path, device=device)
    return {"status": "success", "output_path": output_path}


def _tool_ball_detection_video(
    video_path: str, output_path: str, device: str = "cpu"
) -> Dict[str, str]:
    from forgesyte_yolo_tracker.video.ball_detection_video import run_ball_detection_video

    run_ball_detection_video(video_path, output_path, device=device)
    return {"status": "success", "output_path": output_path}


def _tool_pitch_detection_video(
    video_path: str, output_path: str, device: str = "cpu"
) -> Dict[str, str]:
    from forgesyte_yolo_tracker.video.pitch_detection_video import run_pitch_detection_video

    run_pitch_detection_video(video_path, output_path, device=device)
    return {"status": "success", "output_path": output_path}


def _tool_radar_video(video_path: str, output_path: str, device: str = "cpu") -> Dict[str, str]:
    from forgesyte_yolo_tracker.video.radar_video import run_radar_video

    run_radar_video(video_path, output_path, device=device)
    return {"status": "success", "output_path": output_path}


# ---------------------------------------------------------
# v0.9.5 Video Tool (JSON frame-level output)
# v0.9.6: Added progress_callback support with detailed logging
# ---------------------------------------------------------
def _tool_video_player_detection(
    video_path: str,
    device: str = "cpu",
    annotated: bool = False,
    progress_callback=None,
) -> Dict[str, Any]:
    """Run player detection on video frames, returning JSON results.

    Uses YOLO streaming inference for memory efficiency.
    Returns frame-level detections aggregated into a single JSON response.

    Args:
        video_path: Path to input video file
        device: Device to run model on ('cpu' or 'cuda')
        annotated: Whether to include annotated frames (not implemented in v0.9.5)
        progress_callback: Optional callback for progress updates (v0.9.6)

    Returns:
        Dict with 'frames' array and 'summary' object
    """
    from pathlib import Path

    from forgesyte_yolo_tracker.configs import get_model_path

    # Lazy import to avoid loading YOLO at module load time
    from ultralytics import YOLO

    logger.info(f"Starting video_player_detection: device={device}")

    # Construct model path
    MODEL_NAME = get_model_path("player_detection")
    MODEL_PATH = str(Path(__file__).parent / "models" / MODEL_NAME)


    # Get total frames using OpenCV (for progress tracking)
    import cv2
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if total_frames <= 0:
        total_frames = 100  # Fallback heuristic

    logger.info(f"Video: {total_frames} frames at {fps} fps")

    # Load model and set device
    model = YOLO(MODEL_PATH).to(device=device)

    frame_results: list = []
    frame_index = 0

    # Use YOLO streaming inference for memory efficiency

    results = model(video_path, stream=True, verbose=False)

    for result in results:
        # Extract detections from result
        boxes = result.boxes

        # Check if boxes has detections by examining xyxy shape
        if boxes is not None:
            xyxy_array = boxes.xyxy.cpu().numpy()
            if len(xyxy_array) > 0:
                xyxy = xyxy_array.tolist()
                confidence = boxes.conf.cpu().numpy().tolist()
                class_id = boxes.cls.cpu().numpy().tolist()
            else:
                xyxy = []
                confidence = []
                class_id = []
        else:
            xyxy = []
            confidence = []
            class_id = []

        frame_results.append(
            {
                "frame_index": frame_index,
                "detections": {
                    "xyxy": xyxy,
                    "confidence": confidence,
                    "class_id": class_id,
                },
            }
        )

        # Call progress callback if provided (v0.9.6)
        if progress_callback:
            progress_callback(frame_index + 1, total_frames)


        frame_index += 1

    # Calculate summary
    total_detections = sum(len(f["detections"]["xyxy"]) for f in frame_results)

    logger.info(f"Completed: {frame_index} frames, {total_detections} detections")

    return {
        "frames": frame_results,
        "summary": {
            "total_frames": frame_index,
            "total_detections": total_detections,
        },
    }


# ---------------------------------------------------------
# Plugin class — FINAL, CORRECT, LOADER-COMPATIBLE
# ---------------------------------------------------------

# ---------------------------------------------------------
# v0.9.7: Shared video tool helper
# ---------------------------------------------------------
def _run_video_tool(
    model,
    video_path: str,
    progress_callback=None,
    frame_handler=None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Shared helper for video tools with progress tracking."""
    import cv2

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        total_frames = 100

    logger.info(f"Video: {total_frames} frames")

    frame_results = []
    frame_index = 0

    results = model(video_path, stream=True, verbose=False)

    for result in results:
        if frame_handler:
            frame_data = frame_handler(result)
        else:
            boxes = result.boxes
            if boxes is not None and len(boxes.xyxy) > 0:
                frame_data = {
                    "xyxy": boxes.xyxy.cpu().numpy().tolist(),
                    "confidence": boxes.conf.cpu().numpy().tolist(),
                    "class_id": boxes.cls.cpu().numpy().tolist(),
                }
            else:
                frame_data = {"xyxy": [], "confidence": [], "class_id": []}

        frame_results.append({
            "frame_index": frame_index,
            "detections": frame_data,
        })

        if progress_callback:
            progress_callback(frame_index + 1, total_frames)

        frame_index += 1

    logger.info(f"Completed: {frame_index} frames")

    return {
        "total_frames": frame_index,
        "frames": frame_results,
    }


def _tool_video_ball_detection(
    video_path: str,
    device: str = "cpu",
    progress_callback=None,
) -> Dict[str, Any]:
    """Run ball detection on video frames, returning JSON results."""
    from pathlib import Path
    from forgesyte_yolo_tracker.configs import get_model_path
    from ultralytics import YOLO

    MODEL_NAME = get_model_path("ball_detection")
    MODEL_PATH = str(Path(__file__).parent / "models" / MODEL_NAME)
    model = YOLO(MODEL_PATH).to(device=device)

    def handle_frame(result):
        boxes = result.boxes
        if boxes is not None and len(boxes.xyxy) > 0:
            return {
                "xyxy": boxes.xyxy.cpu().numpy().tolist(),
                "confidence": boxes.conf.cpu().numpy().tolist(),
                "class_id": boxes.cls.cpu().numpy().tolist(),
            }
        return {"xyxy": [], "confidence": [], "class_id": []}

    return _run_video_tool(
        model=model,
        video_path=video_path,
        progress_callback=progress_callback,
        frame_handler=handle_frame,
        device=device,
    )


def _tool_video_pitch_detection(
    video_path: str,
    device: str = "cpu",
    progress_callback=None,
) -> Dict[str, Any]:
    """Run pitch detection on video frames, returning JSON results."""
    from pathlib import Path
    from forgesyte_yolo_tracker.configs import get_model_path
    from ultralytics import YOLO

    MODEL_NAME = get_model_path("pitch_detection")
    MODEL_PATH = str(Path(__file__).parent / "models" / MODEL_NAME)
    model = YOLO(MODEL_PATH).to(device=device)

    def handle_frame(result):
        keypoints = result.keypoints
        if keypoints is not None and keypoints.xy is not None:
            xy = keypoints.xy.cpu().numpy()
            conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else None
            return {
                "keypoints_xy": xy.tolist() if len(xy) > 0 else [],
                "keypoints_conf": conf.tolist() if conf is not None and len(conf) > 0 else [],
            }
        return {"keypoints_xy": [], "keypoints_conf": []}

    return _run_video_tool(
        model=model,
        video_path=video_path,
        progress_callback=progress_callback,
        frame_handler=handle_frame,
        device=device,
    )


def _tool_video_radar(
    video_path: str,
    device: str = "cpu",
    progress_callback=None,
) -> Dict[str, Any]:
    """Run radar generation on video frames, returning JSON results."""
    from pathlib import Path
    from forgesyte_yolo_tracker.configs import get_model_path
    from ultralytics import YOLO

    MODEL_NAME = get_model_path("player_detection")
    MODEL_PATH = str(Path(__file__).parent / "models" / MODEL_NAME)
    model = YOLO(MODEL_PATH).to(device=device)

    def handle_frame(result):
        boxes = result.boxes
        if boxes is not None and len(boxes.xyxy) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            centers = [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in xyxy]
            return {
                "xyxy": xyxy.tolist(),
                "centers": centers,
                "confidence": boxes.conf.cpu().numpy().tolist(),
                "class_id": boxes.cls.cpu().numpy().tolist(),
            }
        return {"xyxy": [], "centers": [], "confidence": [], "class_id": []}

    return _run_video_tool(
        model=model,
        video_path=video_path,
        progress_callback=progress_callback,
        frame_handler=handle_frame,
        device=device,
    )


def _tool_video_player_tracking(
    video_path: str,
    device: str = "cpu",
    progress_callback=None,
) -> Dict[str, Any]:
    """Run player tracking on video frames with ByteTrack."""
    from pathlib import Path
    import cv2
    import supervision as sv
    from forgesyte_yolo_tracker.configs import get_model_path
    from ultralytics import YOLO

    MODEL_NAME = get_model_path("player_detection")
    MODEL_PATH = str(Path(__file__).parent / "models" / MODEL_NAME)
    model = YOLO(MODEL_PATH).to(device=device)

    tracker = sv.ByteTrack(
        track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        frame_rate=30,
    )

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if total_frames <= 0:
        total_frames = 100

    frame_results = []
    frame_index = 0

    results = model(video_path, stream=True, verbose=False)
    for result in results:
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        tracked_objects = []
        for tid, cls, xyxy in zip(
            detections.track_id or [],
            detections.class_id,
            detections.xyxy
        ):
            center = [(xyxy[0] + xyxy[2]) / 2, (xyxy[1] + xyxy[3]) / 2]
            tracked_objects.append({
                "track_id": int(tid) if tid else -1,
                "class_id": int(cls),
                "xyxy": xyxy.tolist(),
                "center": center,
            })

        frame_results.append({
            "frame_index": frame_index,
            "detections": {"tracked_objects": tracked_objects},
        })

        if progress_callback:
            progress_callback(frame_index + 1, total_frames)

        frame_index += 1

    return {
        "total_frames": frame_index,
        "frames": frame_results,
    }


class Plugin(BasePlugin):  # type: ignore[misc]
    """YOLO Tracker plugin with BasePlugin architecture."""

    name: str = "yolo-tracker"
    version: str = "0.2.0"
    description: str = "YOLO-based football analysis plugin"

    # CLASS-LEVEL tools dict (required by ForgeSyte loader contract)
    # Handler values are callables (no magic getattr resolution)
    tools: Dict[str, Dict[str, Any]] = {
        "player_detection": {
            "description": "Detect players in a frame",
            "input_schema": {
                "image_bytes": {"type": "string", "format": "binary"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "output_schema": {"result": {"type": "object"}},
            "handler": _tool_player_detection,
        },
        "player_tracking": {
            "description": "Track players across frames",
            "input_schema": {
                "image_bytes": {"type": "string", "format": "binary"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "output_schema": {"result": {"type": "object"}},
            "handler": _tool_player_tracking,
        },
        "ball_detection": {
            "description": "Detect the football",
            "input_schema": {
                "image_bytes": {"type": "string", "format": "binary"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "output_schema": {"result": {"type": "object"}},
            "handler": _tool_ball_detection,
        },
        "pitch_detection": {
            "description": "Detect pitch keypoints",
            "input_schema": {
                "image_bytes": {"type": "string", "format": "binary"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "output_schema": {"result": {"type": "object"}},
            "handler": _tool_pitch_detection,
        },
        "radar": {
            "description": "Generate radar (bird's-eye) view",
            "input_schema": {
                "image_bytes": {"type": "string", "format": "binary"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "output_schema": {"result": {"type": "object"}},
            "handler": _tool_radar,
        },
        "player_detection_video": {
            "description": "Detect players in a video",
            "input_schema": {
                "video_path": {"type": "string"},
                "output_path": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
            },
            "output_schema": {"status": {"type": "string"}},
            "handler": _tool_player_detection_video,
        },
        "player_tracking_video": {
            "description": "Track players in a video",
            "input_schema": {
                "video_path": {"type": "string"},
                "output_path": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
            },
            "output_schema": {"status": {"type": "string"}},
            "handler": _tool_player_tracking_video,
        },
        "ball_detection_video": {
            "description": "Detect ball in a video",
            "input_schema": {
                "video_path": {"type": "string"},
                "output_path": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
            },
            "output_schema": {"status": {"type": "string"}},
            "handler": _tool_ball_detection_video,
        },
        "pitch_detection_video": {
            "description": "Detect pitch in a video",
            "input_schema": {
                "video_path": {"type": "string"},
                "output_path": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
            },
            "output_schema": {"status": {"type": "string"}},
            "handler": _tool_pitch_detection_video,
        },
        "radar_video": {
            "description": "Generate radar overlay on video",
            "input_schema": {
                "video_path": {"type": "string"},
                "output_path": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
            },
            "output_schema": {"status": {"type": "string"}},
            "handler": _tool_radar_video,
        },
        # v0.9.5 Video Tool (JSON frame-level output)
        "video_player_detection": {
            "description": "Run player detection on video frames, returning JSON results",
            "input_schema": {
                "video_path": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "output_schema": {
                "frames": {"type": "array"},
                "summary": {"type": "object"},
            },
            "handler": _tool_video_player_detection,
        },
        # v0.9.7 Video Tools (JSON frame-level output + progress_callback)
        "video_ball_detection": {
            "description": "Run ball detection on video frames, returning JSON results",
            "input_schema": {
                "video_path": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
            },
            "output_schema": {
                "frames": {"type": "array"},
                "total_frames": {"type": "integer"},
            },
            "handler": _tool_video_ball_detection,
        },
        "video_pitch_detection": {
            "description": "Run pitch detection on video frames, returning JSON results",
            "input_schema": {
                "video_path": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
            },
            "output_schema": {
                "frames": {"type": "array"},
                "total_frames": {"type": "integer"},
            },
            "handler": _tool_video_pitch_detection,
        },
        "video_radar": {
            "description": "Run radar generation on video frames, returning JSON results",
            "input_schema": {
                "video_path": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
            },
            "output_schema": {
                "frames": {"type": "array"},
                "total_frames": {"type": "integer"},
            },
            "handler": _tool_video_radar,
        },
        "video_player_tracking": {
            "description": "Run player tracking on video frames with ByteTrack, returning JSON results",
            "input_schema": {
                "video_path": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
            },
            "output_schema": {
                "frames": {"type": "array"},
                "total_frames": {"type": "integer"},
            },
            "handler": _tool_video_player_tracking,
        },
    }

    # -------------------------------------------------------
    # Dispatcher
    # -------------------------------------------------------
    def run_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of tool to execute (must be from manifest)
            args: Tool arguments dict

        Returns:
            Tool result (dict with detections/keypoints/etc)

        Raises:
            ValueError: If tool name not found or invalid args
        """
        logger.info(f"run_tool: {tool_name}")

        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = self.tools[tool_name]["handler"]

        # v0.9.7 video tools (JSON frame-level output + progress_callback)
        v097_video_tools = {
            "video_player_detection",
            "video_ball_detection",
            "video_pitch_detection",
            "video_radar",
            "video_player_tracking",
        }
        if tool_name in v097_video_tools:
            return handler(
                video_path=args.get("video_path"),
                device=args.get("device", _get_default_device()),
                progress_callback=args.get("progress_callback"),
            )

        # Legacy video tools (output annotated video files)
        if tool_name.endswith("_video"):
            return handler(
                video_path=args.get("video_path"),
                output_path=args.get("output_path"),
                device=args.get("device", _get_default_device()),
            )

        # Frame tools use image_bytes (Phase 12 contract)
        image_bytes = args.get("image_bytes")
        if not isinstance(image_bytes, (bytes, bytearray)):
            return {
                "error": "invalid_image_bytes",
                "message": f"image_bytes must be bytes, got {type(image_bytes).__name__}",
            }

        return handler(
            image_bytes=image_bytes,
            device=args.get("device", _get_default_device()),
            annotated=args.get("annotated", False),
        )

    def __init__(self) -> None:
        """Initialize YOLO Tracker plugin."""
        super().__init__()  # Call BasePlugin __init__ for contract validation

    # -------------------------------------------------------
    # Lifecycle hooks
    # -------------------------------------------------------
    def on_load(self) -> None:
        logger.info("YOLO Tracker plugin loaded")

    def on_unload(self) -> None:
        logger.info("YOLO Tracker plugin unloaded")
