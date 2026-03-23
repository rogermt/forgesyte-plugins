"""Video summary computation utilities.

Provides functions to compute aggregated metadata from video tool results.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


# Registry pattern for detection extraction (avoid elif chains)
_DETECTION_FIELD_MAP: Dict[str, str] = {
    "detections": "detections",
    "tracked_objects": "tracked_objects",
    "radar_points": "radar_points",
}


def _get_detections_from_frame(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract detections list from frame using registry lookup.

    Args:
        frame: Frame result dictionary

    Returns:
        List of detection dicts, or empty list if not found
    """
    for field_name in _DETECTION_FIELD_MAP.values():
        if field_name in frame:
            return frame[field_name]
    return []


def compute_video_summary(frames: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute aggregated summary metadata from video tool frames.

    Args:
        frames: List of frame results from video processing

    Returns:
        Dictionary with:
        - detection_count: Total detections across all frames
        - frame_count: Number of frames processed
    """
    total_detection_count = 0

    for frame in frames:
        detections = _get_detections_from_frame(frame)
        total_detection_count += len(detections)

    logger.info(
        f"Video summary: {total_detection_count} detections across {len(frames)} frames"
    )

    return {
        "detection_count": total_detection_count,
        "frame_count": len(frames),
    }
