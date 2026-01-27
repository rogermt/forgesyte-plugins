"""ForgeSyte YOLO Tracker Plugin.

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

from app.models import AnalysisResult, PluginMetadata
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


def _validate_base64(frame_b64: str) -> str:
    """Validate and clean base64 input.

    Args:
        frame_b64: Raw base64 string or data URL

    Returns:
        Cleaned base64 string without data URL prefix

    Raises:
        ValueError: If input is not valid base64
    """
    # Strip data URL prefix if present
    if frame_b64.startswith("data:image"):
        frame_b64 = frame_b64.split(",", 1)[-1]

    # Validate it's not empty after stripping
    if not frame_b64 or not frame_b64.strip():
        raise ValueError("Empty base64 string after processing")

    # Validate characters are valid base64
    # Allow for padding and URL-safe variants
    if not re.match(r'^[A-Za-z0-9+/=]+$', frame_b64):
        raise ValueError("Invalid base64 characters detected")

    return frame_b64


def _decode_frame_base64_safe(
    frame_b64: str, tool_name: str
) -> Tuple[Optional[np.ndarray], Optional[Dict[str, Any]]]:
    """Safely decode base64-encoded image with error handling.

    Args:
        frame_b64: Base64 encoded image data
        tool_name: Name of the calling tool for error context

    Returns:
        Tuple of (frame, error_dict) where exactly one is None
        - On success: (frame, None)
        - On failure: (None, error_dict)
    """
    try:
        # Validate first
        cleaned_b64 = _validate_base64(frame_b64)
        # Then decode
        data = base64.b64decode(cleaned_b64)
        arr = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:
            raise ValueError("Failed to decode image data")

        return (frame, None)

    except ValueError as e:
        error_msg = str(e)
        logger.warning(
            f"Base64 validation failed in {tool_name}: {error_msg}"
        )
        return (None, {
            "error": "invalid_base64",
            "message": f"Failed to decode frame: {error_msg}",
            "plugin": "yolo-tracker",
            "tool": tool_name,
        })
    except Exception as e:
        error_msg = str(e)
        logger.warning(
            f"Base64 decode exception in {tool_name}: {error_msg}"
        )
        return (None, {
            "error": "invalid_base64",
            "message": f"Failed to decode frame: {error_msg}",
            "plugin": "yolo-tracker",
            "tool": tool_name,
        })


def _decode_frame_base64(frame_b64: str) -> np.ndarray:
    """Decode base64-encoded image into a numpy BGR frame.

    Args:
        frame_b64: Base64 encoded image data

    Returns:
        Decoded image as numpy BGR array
    """
    data = base64.b64decode(frame_b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def player_detection(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    """Detect players in a single frame.

    Args:
        frame_base64: Base64 encoded image
        device: Device to run model on ('cpu' or 'cuda')
        annotated: If True, return annotated frame

    Returns:
        Dictionary with detections, count, classes
    """
    frame, error = _decode_frame_base64_safe(frame_base64, "player_detection")
    if error:
        logger.error(f"Player detection failed: {error}")
        return error

    try:
        if annotated:
            return detect_players_json_with_annotated_frame(frame, device=device)
        return detect_players_json(frame, device=device)
    except Exception as e:
        logger.error(f"player_detection tool execution failed: {e}", exc_info=True)
        return {
            "error": "tool_execution_failed",
            "message": str(e),
            "plugin": "yolo-tracker",
            "tool": "player_detection",
        }


def player_tracking(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    """Track players across frames using ByteTrack.

    Args:
        frame_base64: Base64 encoded image
        device: Device to run model on ('cpu' or 'cuda')
        annotated: If True, return annotated frame

    Returns:
        Dictionary with detections, count, track_ids
    """
    frame, error = _decode_frame_base64_safe(frame_base64, "player_tracking")
    if error:
        return error

    if annotated:
        return track_players_json_with_annotated_frame(frame, device=device)
    return track_players_json(frame, device=device)


def ball_detection(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    """Detect the football in a single frame.

    Args:
        frame_base64: Base64 encoded image
        device: Device to run model on ('cpu' or 'cuda')
        annotated: If True, return annotated frame

    Returns:
        Dictionary with detections, ball, ball_detected
    """
    frame, error = _decode_frame_base64_safe(frame_base64, "ball_detection")
    if error:
        return error

    if annotated:
        return detect_ball_json_with_annotated_frame(frame, device=device)
    return detect_ball_json(frame, device=device)


def pitch_detection(
    frame_base64: str, device: str = "cpu", annotated: bool = False
) -> Dict[str, Any]:
    """Detect pitch keypoints for homography mapping.

    Args:
        frame_base64: Base64 encoded image
        device: Device to run model on ('cpu' or 'cuda')
        annotated: If True, return annotated frame

    Returns:
        Dictionary with keypoints, pitch_polygon, pitch_detected
    """
    frame, error = _decode_frame_base64_safe(frame_base64, "pitch_detection")
    if error:
        return error

    if annotated:
        return detect_pitch_json_with_annotated_frame(frame, device=device)
    return detect_pitch_json(frame, device=device)


def radar(frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
    """Generate radar (bird's-eye) view of player positions.

    Args:
        frame_base64: Base64 encoded image
        device: Device to run model on ('cpu' or 'cuda')
        annotated: If True, return radar image

    Returns:
        Dictionary with radar_points, radar_size, radar_base64 (if annotated)
    """
    frame, error = _decode_frame_base64_safe(frame_base64, "radar")
    if error:
        return error

    if annotated:
        return radar_json_with_annotated_frame(frame, device=device)
    return radar_json(frame, device=device)


class Plugin:
    """ForgeSyte YOLO Tracker plugin (old interface)."""

    name: str = "yolo-tracker"
    version: str = "0.1.0"

    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name=self.name,
            description="YOLO-based football analysis plugin",
            version=self.version,
            inputs=["image"],
            outputs=["json"],
            config_schema={
                "device": {"type": "string", "default": "cpu"},
                "annotated": {"type": "boolean", "default": False},
                "confidence": {"type": "number", "default": 0.25},
            },
        )

    def analyze(self, image_data: bytes, options: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """Analyze image with configurable detections: players, ball, pitch.

        Args:
            image_data: Raw image bytes
            options: Optional dict with:
                - detections: list of detection types to run (default: ["players", "ball", "pitch"])
                  Valid values: "players", "ball", "pitch"
                - device: "cpu" or "cuda" (default: auto-detect)

        Example:
            # Run only players and pitch
            options = {"detections": ["players", "pitch"]}

            # Run all (default)
            options = {"detections": ["players", "ball", "pitch"]}
        """
        options = options or {}

        # Determine device
        if "device" in options:
            device = options["device"]
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine which detections to run (from config or options override)
        if "detections" in options:
            requested_detections = options["detections"]
        else:
            requested_detections = get_default_detections()

        # Try to decode image_data directly as PNG/JPEG bytes first
        arr = np.frombuffer(image_data, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if frame is None:
            # Fallback: try base64 decode (legacy path)
            try:
                frame_b64 = base64.b64encode(image_data).decode("utf-8")
                frame, error = _decode_frame_base64_safe(frame_b64, "analyze")
                if error:
                    logger.warning(f"Base64 decode failed in analyze: {error}")
                    return AnalysisResult(
                        text="",
                        blocks=[],
                        confidence=0.0,
                        language=None,
                        error=error,
                        extra={},
                    )
            except Exception as e:
                logger.warning(f"Image decode failed in analyze: {e}")
                return AnalysisResult(
                    text="",
                    blocks=[],
                    confidence=0.0,
                    language=None,
                    error={"error": "decode_failed", "message": str(e)},
                    extra={},
                )

        combined = {}

        # Run requested detections
        if "players" in requested_detections:
            combined["players"] = detect_players_json(frame, device=device)

        if "ball" in requested_detections:
            combined["ball"] = detect_ball_json(frame, device=device)

        if "pitch" in requested_detections:
            combined["pitch"] = detect_pitch_json(frame, device=device)

        return AnalysisResult(
            text="",
            blocks=[],
            confidence=1.0,
            language=None,
            error=None,
            extra=combined,
        )

    def on_load(self) -> None:
        """Called when plugin is loaded."""
        print("YOLO Tracker plugin loaded")

    def on_unload(self) -> None:
        """Called when plugin is unloaded."""
        print("YOLO Tracker plugin unloaded")

    # Tool methods - delegate to module functions for MCP tool exposure
    # NOTE: Parameter name must match manifest.json ("frame_base64")
    def player_detection(self, frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
        """Detect players in a single frame."""
        return player_detection(frame_base64, device, annotated)

    def player_tracking(self, frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
        """Track players across frames using ByteTrack."""
        return player_tracking(frame_base64, device, annotated)

    def ball_detection(self, frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
        """Detect the football in a single frame."""
        return ball_detection(frame_base64, device, annotated)

    def pitch_detection(self, frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
        """Detect pitch keypoints for homography mapping."""
        return pitch_detection(frame_base64, device, annotated)

    def radar(self, frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
        """Generate radar (bird's-eye) view of player positions."""
        return radar(frame_base64, device, annotated)

