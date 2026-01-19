"""Main plugin class for YOLO Football Analysis."""

import base64
from typing import Any, Dict, Optional

import cv2
import numpy as np

from .inference.ball_detection import run_ball_detection
from .inference.pitch_detection import run_pitch_detection
from .inference.player_detection import run_player_detection
from .inference.player_tracking import run_player_tracking
from .inference.radar import run_radar
from .inference.team_classification import run_team_classification


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
    def yolo_radar(
        self, image: str, config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
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
