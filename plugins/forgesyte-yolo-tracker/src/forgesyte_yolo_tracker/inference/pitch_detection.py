"""Pitch detection inference module.

Provides JSON and JSON+Base64 modes for pitch detection using BaseDetector:
- detect_pitch_json(): Returns structured keypoint data
- detect_pitch_json_with_annotated_frame(): Returns data + annotated frame

This module uses BaseDetector for base inference, then applies pitch-specific
logic to extract keypoints, corners, and homography transformation matrix.
"""

import logging
from typing import Any, Dict, Optional

import cv2
import numpy as np

from forgesyte_yolo_tracker.configs import get_confidence, get_model_path
from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration
from forgesyte_yolo_tracker.inference._base_detector import BaseDetector

logger = logging.getLogger(__name__)

CONFIG = SoccerPitchConfiguration()

# Pitch detector instance with configuration
PITCH_DETECTOR = BaseDetector(
    detector_name="pitch",
    model_name=get_model_path("pitch_detection"),
    default_confidence=get_confidence("pitch"),
    imgsz=1280,
    class_names=None,  # Pitch uses keypoints, not classes
    colors={0: "#00FF00"},  # Green for pitch
)


def detect_pitch_json(
    frame: np.ndarray[Any, Any],
    device: str = "cpu",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect pitch keypoints in a frame - JSON mode.

    Executes YOLO keypoint inference and extracts pitch corners and
    homography transformation matrix for perspective transform.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold (uses default if None)

    Returns:
        Dictionary with:
        - keypoints: List of keypoint dicts with xy, confidence, name
        - count: Total number of keypoints
        - pitch_polygon: List of corner vertices (if 4+ found)
        - pitch_detected: Boolean indicating if full pitch found
        - homography: Optional homography matrix for perspective transform
    """
    if confidence is None:
        confidence = PITCH_DETECTOR.default_confidence

    model = PITCH_DETECTOR.get_model(device=device)
    logger.info("🔫 Running YOLO keypoint inference...")
    result = model(frame, imgsz=1280, conf=confidence, verbose=False)[0]
    logger.info("🔫 Inference complete. Processing keypoints...")

    keypoint_list: list[Dict[str, Any]] = []
    if result.keypoints is not None and result.keypoints.xy is not None:
        keypoints_xy = result.keypoints.xy.cpu().numpy()[0]
        keypoints_conf = (
            result.keypoints.conf.cpu().numpy()[0]
            if result.keypoints.conf is not None
            else None
        )

        for i, (x, y) in enumerate(keypoints_xy):
            kp: Dict[str, Any] = {
                "xy": [float(x), float(y)],
                "confidence": (
                    float(keypoints_conf[i]) if keypoints_conf is not None else 1.0
                ),
                "name": CONFIG.keypoint_names.get(i, f"keypoint_{i}"),
            }
            keypoint_list.append(kp)

    # Extract pitch corners
    pitch_polygon: list[list[float]] = []
    valid_keypoints = [kp for kp in keypoint_list if kp["confidence"] > confidence * 0.5]
    if len(valid_keypoints) >= 4:
        corner_names = [
            "bottom_left_corner",
            "top_left_corner",
            "top_right_corner",
            "bottom_right_corner",
        ]
        for name in corner_names:
            for kp in valid_keypoints:
                if kp["name"] == name:
                    pitch_polygon.append(kp["xy"])
                    break

    # Compute homography matrix
    homography: Optional[list[list[float]]] = None
    if len(pitch_polygon) >= 4:
        src_pts = np.array([kp["xy"] for kp in valid_keypoints[:4]], dtype=np.float32)
        tgt_pts = np.array(
            [CONFIG.vertices[i] for i in range(4)], dtype=np.float32
        )
        try:
            m, _ = cv2.findHomography(src_pts, tgt_pts)
            if m is not None:
                homography = m.tolist()
        except Exception as exc:
            logger.warning(f"Failed to compute homography: {exc}")

    logger.info(f"✅ Detection complete: {len(keypoint_list)} keypoints found")
    logger.info(f"✅ Pitch detected: {len(pitch_polygon) >= 4}")

    return {
        "keypoints": keypoint_list,
        "count": len(keypoint_list),
        "pitch_polygon": pitch_polygon,
        "pitch_detected": len(pitch_polygon) >= 4,
        "homography": homography,
    }


def detect_pitch_json_with_annotated_frame(
    frame: np.ndarray[Any, Any],
    device: str = "cpu",
    confidence: Optional[float] = None,
) -> Dict[str, Any]:
    """Detect pitch keypoints in a frame - JSON+Base64 mode.

    Executes YOLO keypoint inference, extracts pitch corners, and creates
    annotated frame with keypoint markers and pitch polygon.

    Args:
        frame: Input image frame (BGR format)
        device: Device to run model on ('cpu' or 'cuda')
        confidence: Detection confidence threshold (uses default if None)

    Returns:
        Dictionary with:
        - keypoints: List of keypoint dicts
        - count: Total number of keypoints
        - pitch_polygon: Corner vertices (if 4+ found)
        - pitch_detected: Boolean indicating if full pitch found
        - annotated_frame_base64: Base64 encoded annotated frame
    """
    if confidence is None:
        confidence = PITCH_DETECTOR.default_confidence

    result = detect_pitch_json(frame, device=device, confidence=confidence)

    # Create annotated frame
    annotated = frame.copy()

    # Draw keypoints
    for kp in result["keypoints"]:
        x, y = int(kp["xy"][0]), int(kp["xy"][1])
        cv2.circle(annotated, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            kp["name"].replace("_", " "),
            (x + 5, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
        )

    # Draw pitch polygon
    if len(result["pitch_polygon"]) >= 4:
        pts = np.array(result["pitch_polygon"], dtype=np.int32)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=2)

    result["annotated_frame_base64"] = PITCH_DETECTOR._encode_frame_to_base64(annotated)

    return result


def get_pitch_detection_model(device: str = "cpu") -> Any:
    """Get or create cached YOLO pitch detection model.

    Returns the cached model instance from PITCH_DETECTOR.
    Model is loaded on first access and cached for subsequent calls.

    Args:
        device: Device to run model on ('cpu' or 'cuda')

    Returns:
        YOLO model instance
    """
    return PITCH_DETECTOR.get_model(device=device)


def run_pitch_detection(
    frame: np.ndarray[Any, Any], config: Dict[str, Any]
) -> Dict[str, Any]:
    """Legacy function for plugin.py compatibility.

    Delegates to either detect_pitch_json or detect_pitch_json_with_annotated_frame
    based on include_annotated config flag.

    Args:
        frame: Input image frame
        config: Configuration dictionary with keys:
            - device: 'cpu' or 'cuda' (default: 'cpu')
            - confidence: Detection threshold (default: 0.25)
            - include_annotated: True to include annotated frame (default: False)

    Returns:
        Detection results dictionary
    """
    device = config.get("device", "cpu")
    confidence = config.get("confidence", PITCH_DETECTOR.default_confidence)
    include_annotated = config.get("include_annotated", False)

    if include_annotated:
        return detect_pitch_json_with_annotated_frame(
            frame, device=device, confidence=confidence
        )
    return detect_pitch_json(frame, device=device, confidence=confidence)
