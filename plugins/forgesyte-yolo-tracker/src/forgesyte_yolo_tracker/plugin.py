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


from forgesyte_yolo_tracker.configs import get_device, ConfigError
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

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Device resolution — Phase 12 strict governance
# ---------------------------------------------------------
def _resolve_device(options: Dict[str, Any]) -> str:
    """Resolve device for inference.

    Phase 12 governance:
    - If options contains 'device' and it's not None, use it
    - Else fall back to models.yaml device
    - If models.yaml missing or invalid, raise ConfigError

    Args:
        options: Options dict from API/server

    Returns:
        Device string: 'cpu' or 'cuda'

    Raises:
        ConfigError: If device resolution fails
    """
    # Explicit device in options takes priority
    if "device" in options and options["device"]:
        return str(options["device"])

    # Fall back to config-level device (models.yaml)
    try:
        return get_device()
    except ConfigError as e:
        logger.error(f"Device resolution failed: {e}")
        raise


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
    }

    # -------------------------------------------------------
    # Dispatcher
    # -------------------------------------------------------
    def run_tool(self, tool_name: str, args: Dict[str, Any]) -> Any:
        """Execute a tool by name with the given arguments.

        Phase 12 governance:
        - Device is resolved strictly: options → models.yaml
        - If neither provided, raise ConfigError
        - Device mismatch with inference modules is acceptable
          (inference modules keep device="cpu" default for backward compat)

        Args:
            tool_name: Name of tool to execute (must be from manifest)
            args: Tool arguments dict

        Returns:
            Tool result (dict with detections/keypoints/etc)

        Raises:
            ValueError: If tool name not found or invalid args
        """
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = self.tools[tool_name]["handler"]

        # Resolve device strictly (Phase 12)
        try:
            device = _resolve_device(args)
        except ConfigError as e:
            logger.error(f"Device resolution failed: {e}")
            return {
                "error": "device_resolution_failed",
                "message": str(e),
                "plugin": "yolo-tracker",
                "tool": tool_name,
            }

        # Video tools use different args
        if "video" in tool_name:
            return handler(
                video_path=args.get("video_path"),
                output_path=args.get("output_path"),
                device=device,
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
            device=device,
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
