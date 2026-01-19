"""Main plugin class for YOLO Football Analysis."""

import base64
import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

# Import plugin interfaces from forgesyte server
try:
    from app.models import AnalysisResult, PluginMetadata
except ImportError:
    # Fallback for testing environments where app.models may not be available
    AnalysisResult = None  # type: ignore
    PluginMetadata = None  # type: ignore

from .inference.ball_detection import run_ball_detection
from .inference.pitch_detection import run_pitch_detection
from .inference.player_detection import run_player_detection
from .inference.player_tracking import run_player_tracking
from .inference.radar import run_radar
from .inference.team_classification import run_team_classification

logger = logging.getLogger(__name__)


def decode_image(image_b64: str) -> np.ndarray:
    """Decode base64-encoded image into a numpy BGR frame.

    Args:
        image_b64: Base64 encoded image data

    Returns:
        Decoded image as numpy BGR array

    Raises:
        ValueError: If image decoding fails
    """
    data = base64.b64decode(image_b64)
    arr = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image from base64.")
    return frame


def encode_frame(frame: np.ndarray) -> str:
    """Encode numpy BGR frame into base64 JPEG.

    Args:
        frame: Image frame as numpy BGR array

    Returns:
        Base64 encoded JPEG string

    Raises:
        ValueError: If image encoding fails
    """
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise ValueError("Failed to encode frame to JPEG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


class Plugin:
    """ForgeSyte YOLO Football Analysis Plugin.

    Exposes multiple tools:
      - yolo_player_detection
      - yolo_player_tracking
      - yolo_ball_detection
      - yolo_team_classification
      - yolo_pitch_detection
      - yolo_radar
    """

    def __init__(self) -> None:
        """Initialize the plugin.

        If you want persistent models/trackers, you can initialize them here
        and pass into the inference functions.
        """
        pass

    def on_load(self) -> None:
        """Called when the plugin is loaded.

        Use this for initialization that requires the full application context.
        """
        logger.info("YOLO Football Analysis plugin loaded")

    def on_unload(self) -> None:
        """Called when the plugin is unloaded.

        Use this for cleanup and resource release.
        """
        logger.info("YOLO Football Analysis plugin unloaded")

    def metadata(self) -> Dict[str, Any]:
        """Return plugin metadata.

        Returns:
            Dictionary with plugin metadata (name, version, config_schema, etc.)
        """
        return {
            "name": "forgesyte-yolo-tracker",
            "version": "0.1.0",
            "description": "YOLO-based football analysis tools for ForgeSyte.",
            "config_schema": {
                "confidence_threshold": {
                    "type": "number",
                    "default": 0.5,
                    "description": "Detection confidence threshold",
                },
                "max_detections": {
                    "type": "integer",
                    "default": 100,
                    "description": "Maximum number of detections to return",
                },
            },
        }

    def analyze(self, image_data: bytes, **kwargs: Any) -> Dict[str, Any]:
        """Analyze an image and return results.

        Args:
            image_data: Raw image bytes
            **kwargs: Additional analysis options

        Returns:
            Analysis results as AnalysisResult compatible dict
        """
        try:
            # Decode image
            arr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("Failed to decode image")

            # Return placeholder result
            return {
                "text": "YOLO tracker analysis not yet implemented",
                "blocks": [],
                "confidence": 0.0,
                "language": None,
                "error": "YOLO tracker plugin analysis methods under development",
            }

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            return {
                "text": "",
                "blocks": [],
                "confidence": 0.0,
                "language": None,
                "error": str(e),
            }

    # ---------------------------------------------------------
    #  YOLO PLAYER DETECTION
    # ---------------------------------------------------------
    def yolo_player_detection(
        self, image: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect players in an image.

        Args:
            image: Base64 encoded image
            config: Optional configuration dictionary

        Returns:
            Dictionary containing detection results and encoded frame
        """
        frame = decode_image(image)
        result = run_player_detection(frame, config or {})
        # Ensure frame is available to UI
        result.setdefault("frame", encode_frame(frame))
        return result

    # ---------------------------------------------------------
    #  YOLO PLAYER TRACKING (YOLO + ByteTrack)
    # ---------------------------------------------------------
    def yolo_player_tracking(
        self, image: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Track players across frames.

        Args:
            image: Base64 encoded image
            config: Optional configuration dictionary

        Returns:
            Dictionary containing tracking results and encoded frame
        """
        frame = decode_image(image)
        result = run_player_tracking(frame, config or {})
        result.setdefault("frame", encode_frame(frame))
        return result

    # ---------------------------------------------------------
    #  BALL DETECTION
    # ---------------------------------------------------------
    def yolo_ball_detection(
        self, image: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect ball in an image.

        Args:
            image: Base64 encoded image
            config: Optional configuration dictionary

        Returns:
            Dictionary containing ball detection results and encoded frame
        """
        frame = decode_image(image)
        result = run_ball_detection(frame, config or {})
        result.setdefault("frame", encode_frame(frame))
        return result

    # ---------------------------------------------------------
    #  TEAM CLASSIFICATION
    # ---------------------------------------------------------
    def yolo_team_classification(
        self, image: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Classify players into teams.

        Args:
            image: Base64 encoded image
            config: Optional configuration dictionary

        Returns:
            Dictionary containing team classification results and encoded frame
        """
        frame = decode_image(image)
        result = run_team_classification(frame, config or {})
        result.setdefault("frame", encode_frame(frame))
        return result

    # ---------------------------------------------------------
    #  PITCH DETECTION
    # ---------------------------------------------------------
    def yolo_pitch_detection(
        self, image: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect pitch lines and keypoints.

        Args:
            image: Base64 encoded image
            config: Optional configuration dictionary

        Returns:
            Dictionary containing pitch detection results and encoded frame
        """
        frame = decode_image(image)
        result = run_pitch_detection(frame, config or {})
        result.setdefault("frame", encode_frame(frame))
        return result

    # ---------------------------------------------------------
    #  RADAR VIEW
    # ---------------------------------------------------------
    def yolo_radar(self, image: str, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate radar bird's-eye view.

        Args:
            image: Base64 encoded image
            config: Optional configuration dictionary

        Returns:
            Dictionary containing radar results and encoded frame
        """
        frame = decode_image(image)
        result = run_radar(frame, config or {})
        # result["radar"] should be base64 PNG from your radar renderer
        result.setdefault("frame", encode_frame(frame))
        return result
