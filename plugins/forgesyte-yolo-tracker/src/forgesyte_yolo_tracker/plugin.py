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
import torch

try:
    from app.plugins.base import BasePlugin
    from app.models import AnalysisResult, PluginMetadata
except ImportError:
    # Fallback for testing/standalone mode
    class BasePlugin:
        """Stub BasePlugin for testing."""
        name: str = ""
        version: str = ""
        description: str = ""
        tools: Dict[str, Any] = {}
        
        def run_tool(self, tool_name: str, args: dict) -> Dict[str, Any]:
            """Dispatch tool execution."""
            raise NotImplementedError
        
        def metadata(self) -> Dict[str, Any]:
            """Return plugin metadata."""
            return {
                "name": self.name,
                "version": self.version,
                "description": self.description,
            }
        
        def on_load(self) -> None:
            """Called when plugin is loaded."""
            pass
        
        def on_unload(self) -> None:
            """Called when plugin is unloaded."""
            pass
    
    class AnalysisResult:
        """Stub AnalysisResult for testing."""
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        
        def __contains__(self, key: str) -> bool:
            """Support 'in' operator for dict-like access."""
            return hasattr(self, key)
    
    class PluginMetadata(dict):
        """Stub PluginMetadata for testing (dict subclass)."""
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

from forgesyte_yolo_tracker.configs import get_default_detections
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
from forgesyte_yolo_tracker.inference.radar import (
    generate_radar_json as radar_json,
    radar_json_with_annotated_frame,
)

logger = logging.getLogger(__name__)


# -----
# Base64 helpers (unchanged)
# -----
def _validate_base64(frame_b64: str) -> str:
    """Validate and clean base64 input."""
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

        return (frame, None)

    except Exception as e:
        msg = str(e)
        logger.warning(f"Base64 decode failed in {tool_name}: {msg}")
        return (
            None,
            {
                "error": "invalid_base64",
                "message": f"Failed to decode frame: {msg}",
                "plugin": "yolo-tracker",
                "tool": tool_name,
            },
        )


def _decode_frame_base64(frame_b64: str) -> np.ndarray:
    """Decode base64-encoded image into a numpy BGR frame."""
    data = base64.b64decode(frame_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# -----
# Tool wrappers (unchanged)
# -----
def player_detection(frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
    """Detect players in a single frame."""
    frame, error = _decode_frame_base64_safe(frame_base64, "player_detection")
    if error:
        return error
    return (
        detect_players_json_with_annotated_frame(frame, device=device)
        if annotated
        else detect_players_json(frame, device=device)
    )


def player_tracking(frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
    """Track players across frames using ByteTrack."""
    frame, error = _decode_frame_base64_safe(frame_base64, "player_tracking")
    if error:
        return error
    return (
        track_players_json_with_annotated_frame(frame, device=device)
        if annotated
        else track_players_json(frame, device=device)
    )


def ball_detection(frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
    """Detect the football in a single frame."""
    frame, error = _decode_frame_base64_safe(frame_base64, "ball_detection")
    if error:
        return error
    return (
        detect_ball_json_with_annotated_frame(frame, device=device)
        if annotated
        else detect_ball_json(frame, device=device)
    )


def pitch_detection(frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
    """Detect pitch keypoints for homography mapping."""
    frame, error = _decode_frame_base64_safe(frame_base64, "pitch_detection")
    if error:
        return error
    return (
        detect_pitch_json_with_annotated_frame(frame, device=device)
        if annotated
        else detect_pitch_json(frame, device=device)
    )


def radar(frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
    """Generate radar (bird's-eye) view of player positions."""
    frame, error = _decode_frame_base64_safe(frame_base64, "radar")
    if error:
        return error
    return (
        radar_json_with_annotated_frame(frame, device=device)
        if annotated
        else radar_json(frame, device=device)
    )


# -----
# BasePlugin Implementation
# -----
class Plugin(BasePlugin):
    """YOLO Tracker plugin with BasePlugin architecture."""

    name = "yolo-tracker"
    version = "0.2.0"
    description = "YOLO-based football analysis plugin"

    # MCP tool definitions
    tools = {
        "player_detection": {
            "description": "Detect players in a frame",
            "inputs": {
                "frame_base64": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "outputs": {"result": {"type": "object"}},
            "handler": "player_detection",
        },
        "player_tracking": {
            "description": "Track players across frames",
            "inputs": {
                "frame_base64": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "outputs": {"result": {"type": "object"}},
            "handler": "player_tracking",
        },
        "ball_detection": {
            "description": "Detect the football",
            "inputs": {
                "frame_base64": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "outputs": {"result": {"type": "object"}},
            "handler": "ball_detection",
        },
        "pitch_detection": {
            "description": "Detect pitch keypoints",
            "inputs": {
                "frame_base64": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "outputs": {"result": {"type": "object"}},
            "handler": "pitch_detection",
        },
        "radar": {
            "description": "Generate radar (bird's-eye) view",
            "inputs": {
                "frame_base64": {"type": "string"},
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
            },
            "outputs": {"result": {"type": "object"}},
            "handler": "radar",
        },
    }

    # MCP dispatcher
    def run_tool(self, tool_name: str, args: dict) -> Dict[str, Any]:
        """Dispatch tool execution."""
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        handler = getattr(self, tool_name)
        return handler(
            frame_base64=args.get("frame_base64"),
            device=args.get("device", "cpu"),
            annotated=args.get("annotated", False),
        )

    # Metadata for UI
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            inputs=["image"],
            outputs=["json"],
            config_schema={
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
                "confidence": {"type": "number", "default": 0.25},
            },
        )

    def analyze(
        self, image_data: bytes, options: Optional[Dict[str, Any]] = None
    ) -> AnalysisResult:
        """Legacy compatibility wrapper for old tests.
        
        TODO: Refactor after design decision on backward compatibility.
        See: https://github.com/rogermt/forgesyte-plugins/issues/62
        """
        # Convert raw bytes → base64
        frame_b64 = base64.b64encode(image_data).decode("utf-8")

        # Default detection set
        options = options or {}
        detections = options.get("detections", get_default_detections())
        device = options.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        combined = {}

        if "players" in detections:
            combined["players"] = detect_players_json(frame_b64, device=device)

        if "ball" in detections:
            combined["ball"] = detect_ball_json(frame_b64, device=device)

        if "pitch" in detections:
            combined["pitch"] = detect_pitch_json(frame_b64, device=device)

        return AnalysisResult(
            text="",
            blocks=[],
            confidence=1.0,
            language=None,
            error=None,
            extra=combined,
        )

    # Lifecycle hooks
    def on_load(self) -> None:
        """Called when plugin is loaded."""
        logger.info("YOLO Tracker plugin loaded")

    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        logger.info("YOLO Tracker plugin unloaded")

    # Tool method wrappers (delegate to module functions)
    def player_detection(
        self, frame_base64: str, device: str = "cpu", annotated: bool = False
    ) -> Dict[str, Any]:
        """Detect players in a single frame."""
        return player_detection(frame_base64, device, annotated)

    def player_tracking(
        self, frame_base64: str, device: str = "cpu", annotated: bool = False
    ) -> Dict[str, Any]:
        """Track players across frames using ByteTrack."""
        return player_tracking(frame_base64, device, annotated)

    def ball_detection(
        self, frame_base64: str, device: str = "cpu", annotated: bool = False
    ) -> Dict[str, Any]:
        """Detect the football in a single frame."""
        return ball_detection(frame_base64, device, annotated)

    def pitch_detection(
        self, frame_base64: str, device: str = "cpu", annotated: bool = False
    ) -> Dict[str, Any]:
        """Detect pitch keypoints for homography mapping."""
        return pitch_detection(frame_base64, device, annotated)

    def radar(
        self, frame_base64: str, device: str = "cpu", annotated: bool = False
    ) -> Dict[str, Any]:
        """Generate radar (bird's-eye) view of player positions."""
        return radar(frame_base64, device, annotated)
