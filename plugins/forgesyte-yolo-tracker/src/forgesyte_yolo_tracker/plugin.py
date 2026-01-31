"""ForgeSyte YOLO Tracker Plugin — BasePlugin Architecture.

Frame-based JSON tools for football analysis:
- player_detection
- player_tracking
- ball_detection
- pitch_detection
- radar
"""

import base64
import logging
import re
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

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
    detect_ball_json, detect_ball_json_with_annotated_frame)
from forgesyte_yolo_tracker.inference.pitch_detection import (
    detect_pitch_json, detect_pitch_json_with_annotated_frame)
from forgesyte_yolo_tracker.inference.player_detection import (
    detect_players_json, detect_players_json_with_annotated_frame)
from forgesyte_yolo_tracker.inference.player_tracking import (
    track_players_json, track_players_json_with_annotated_frame)
from forgesyte_yolo_tracker.inference.radar import \
    generate_radar_json as radar_json
from forgesyte_yolo_tracker.inference.radar import \
    radar_json_with_annotated_frame

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Base64 helpers
# ---------------------------------------------------------
def _validate_base64(frame_b64: str) -> str:
    """Validate and normalize base64 input."""
    if frame_b64.startswith("data:image"):
        frame_b64 = frame_b64.split(",", 1)[-1]

    if not frame_b64 or not frame_b64.strip():
        raise ValueError("Empty base64 string after processing")

    if not re.match(r"^[A-Za-z0-9+/=]+$", frame_b64):
        raise ValueError("Invalid base64 characters detected")

    return frame_b64


def _decode_frame_base64_safe(
    frame_b64: str, tool_name: str
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Safely decode base64-encoded image with error handling."""
    try:
        cleaned_b64 = _validate_base64(frame_b64)
        data = base64.b64decode(cleaned_b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode image data")

        return frame, None

    except Exception as e:
        msg = str(e)
        logger.warning(f"Base64 decode failed in {tool_name}: {msg}")
        return None, {
            "error": "invalid_base64",
            "message": f"Failed to decode frame: {msg}",
            "plugin": "yolo-tracker",
            "tool": tool_name,
        }


# ---------------------------------------------------------
# Tool functions
# ---------------------------------------------------------
def _tool_player_detection(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    frame, error = _decode_frame_base64_safe(frame_base64, "player_detection")
    if error:
        return error
    if annotated and frame is not None:
        return detect_players_json_with_annotated_frame(frame, device=device)
    if frame is not None:
        return detect_players_json(frame, device=device)
    return {"error": "frame_decode_failed"}


def _tool_player_tracking(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    frame, error = _decode_frame_base64_safe(frame_base64, "player_tracking")
    if error:
        return error
    if annotated and frame is not None:
        return track_players_json_with_annotated_frame(frame, device=device)
    if frame is not None:
        return track_players_json(frame, device=device)
    return {"error": "frame_decode_failed"}


def _tool_ball_detection(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    frame, error = _decode_frame_base64_safe(frame_base64, "ball_detection")
    if error:
        return error
    if annotated and frame is not None:
        return detect_ball_json_with_annotated_frame(frame, device=device)
    if frame is not None:
        return detect_ball_json(frame, device=device)
    return {"error": "frame_decode_failed"}


def _tool_pitch_detection(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    frame, error = _decode_frame_base64_safe(frame_base64, "pitch_detection")
    if error:
        return error
    if annotated and frame is not None:
        return detect_pitch_json_with_annotated_frame(frame, device=device)
    if frame is not None:
        return detect_pitch_json(frame, device=device)
    return {"error": "frame_decode_failed"}


def _tool_radar(frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
    frame, error = _decode_frame_base64_safe(frame_base64, "radar")
    if error:
        return error
    if annotated and frame is not None:
        return radar_json_with_annotated_frame(frame, device=device)
    if frame is not None:
        return radar_json(frame, device=device)
    return {"error": "frame_decode_failed"}


def _tool_player_detection_video(
    video_path: str, output_path: str, device: str = "cpu"
) -> Dict[str, str]:
    from forgesyte_yolo_tracker.video.player_detection_video import \
        run_player_detection_video

    run_player_detection_video(video_path, output_path, device=device)
    return {"status": "success", "output_path": output_path}


def _tool_player_tracking_video(
    video_path: str, output_path: str, device: str = "cpu"
) -> Dict[str, str]:
    from forgesyte_yolo_tracker.video.player_tracking_video import \
        run_player_tracking_video

    run_player_tracking_video(video_path, output_path, device=device)
    return {"status": "success", "output_path": output_path}


def _tool_ball_detection_video(
    video_path: str, output_path: str, device: str = "cpu"
) -> Dict[str, str]:
    from forgesyte_yolo_tracker.video.ball_detection_video import \
        run_ball_detection_video

    run_ball_detection_video(video_path, output_path, device=device)
    return {"status": "success", "output_path": output_path}


def _tool_pitch_detection_video(
    video_path: str, output_path: str, device: str = "cpu"
) -> Dict[str, str]:
    from forgesyte_yolo_tracker.video.pitch_detection_video import \
        run_pitch_detection_video

    run_pitch_detection_video(video_path, output_path, device=device)
    return {"status": "success", "output_path": output_path}


def _tool_radar_video(video_path: str, output_path: str, device: str = "cpu") -> Dict[str, str]:
    from forgesyte_yolo_tracker.video.radar_video import run_radar_video

    run_radar_video(video_path, output_path, device=device)
    return {"status": "success", "output_path": output_path}


# ---------------------------------------------------------
# Plugin class — FINAL, CORRECT, LOADER-COMPATIBLE
# ---------------------------------------------------------
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
                "frame_base64": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "output_schema": {"result": {"type": "object"}},
            "handler": _tool_player_detection,
        },
        "player_tracking": {
            "description": "Track players across frames",
            "input_schema": {
                "frame_base64": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "output_schema": {"result": {"type": "object"}},
            "handler": _tool_player_tracking,
        },
        "ball_detection": {
            "description": "Detect the football",
            "input_schema": {
                "frame_base64": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "output_schema": {"result": {"type": "object"}},
            "handler": _tool_ball_detection,
        },
        "pitch_detection": {
            "description": "Detect pitch keypoints",
            "input_schema": {
                "frame_base64": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "output_schema": {"result": {"type": "object"}},
            "handler": _tool_pitch_detection,
        },
        "radar": {
            "description": "Generate radar (bird's-eye) view",
            "input_schema": {
                "frame_base64": {"type": "string"},
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
    }

    # -------------------------------------------------------
    # Dispatcher
    # -------------------------------------------------------
    def run_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = self.tools[tool_name]["handler"]

        # Video tools use different args
        if "video" in tool_name:
            return handler(
                video_path=args.get("video_path"),
                output_path=args.get("output_path"),
                device=args.get("device", "cpu"),
            )

        # Frame tools use frame_base64
        return handler(
            frame_base64=args.get("frame_base64"),
            device=args.get("device", "cpu"),
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
