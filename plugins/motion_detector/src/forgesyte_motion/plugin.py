"""Adaptive Motion Detection Plugin.

Optimized for high-frequency analysis with adaptive baseline learning.
"""

import io
import logging
import time
from typing import Any

import numpy as np
from app.models import AnalysisResult, PluginMetadata
from PIL import Image
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 1. Validated Data Models
# ---------------------------------------------------------------------------


class BoundingBox(BaseModel):  # type: ignore[misc]
    """Coordinates for a detected motion area."""

    x: int
    y: int
    width: int
    height: int


class MotionRegion(BaseModel):  # type: ignore[misc]
    """Metadata for a specific region exhibiting motion."""

    bbox: BoundingBox
    area: int
    center: dict[str, int]


# ---------------------------------------------------------------------------
# 2. Motion Detection Plugin
# ---------------------------------------------------------------------------


class Plugin:
    """
    Production-ready Motion Detection Plugin.
    Utilizes frame-to-frame differencing and an alpha-weighted adaptive baseline.
    """

    name: str = "motion_detector"
    version: str = "1.1.0"
    description: str = (
        "Detect motion between consecutive frames using adaptive learning"
    )

    def __init__(self) -> None:
        """Initialize frame storage and motion history."""
        self._previous_frame: np.ndarray | None = None
        self._frame_count: int = 0
        self._last_motion_time: float = 0
        self._motion_history: list[dict[str, Any]] = []

    def metadata(self) -> PluginMetadata:
        """Returns Pydantic-validated metadata for MCP discovery."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            inputs=["image"],
            outputs=["text", "blocks", "confidence"],
            config_schema={
                "threshold": {
                    "type": "float",
                    "default": 25.0,
                    "min": 1.0,
                    "max": 100.0,
                },
                "min_area": {
                    "type": "float",
                    "default": 0.01,
                    "min": 0.001,
                    "max": 0.5,
                },
                "blur_size": {"type": "integer", "default": 5},
                "reset_baseline": {"type": "boolean", "default": False},
            },
        )

    def analyze(
        self, image_bytes: bytes, options: dict[str, Any] | None = None
    ) -> AnalysisResult:
        """
        Calculates frame differences and updates adaptive baseline.
        Returns a universal AnalysisResult for ForgeSyte Core.
        """
        opts = options or {}
        self._frame_count += 1

        # Reset baseline if requested via configuration
        if opts.get("reset_baseline", False):
            self._previous_frame = None

        try:
            # 1. Preprocessing: Load as Grayscale and map to NumPy
            img = Image.open(io.BytesIO(image_bytes)).convert("L")
            current_frame = np.array(img, dtype=np.float32)

            # 2. Noise Reduction: Separable Gaussian Blur
            blur_size = opts.get("blur_size", 5)
            if blur_size > 1:
                current_frame = self._gaussian_blur(current_frame, blur_size)

            # 3. Initial State Management
            if (
                self._previous_frame is None
                or current_frame.shape != self._previous_frame.shape
            ):
                self._previous_frame = current_frame
                return AnalysisResult(
                    text="Baseline established",
                    blocks=[],
                    confidence=0.0,
                    language=None,
                    error=None,
                )

            # 4. Differencing and Thresholding
            diff = np.abs(current_frame - self._previous_frame)
            motion_mask = diff > opts.get("threshold", 25.0)

            # 5. Scoring
            motion_score = np.sum(motion_mask) / motion_mask.size
            motion_detected = motion_score >= opts.get("min_area", 0.01)

            # 6. Adaptive Baseline Update: Alpha learning rate of 0.1
            alpha = 0.1
            self._previous_frame = (
                alpha * current_frame + (1 - alpha) * self._previous_frame
            )

            # 7. Region Detection
            regions = self._find_motion_regions(motion_mask) if motion_detected else []

            # 8. History Management: Keep recent 100 events
            if motion_detected:
                self._last_motion_time = time.time()
                self._motion_history.append(
                    {"time": self._last_motion_time, "frame": self._frame_count}
                )

            self._motion_history = self._motion_history[-100:]
            # Note: recent_events calculation is kept logic-side but not currently
            # mapped to AnalysisResult as there isn't a clear field for it.
            # If needed, it could go into text or a custom block.

            # 9. Validated Output Construction (Universal AnalysisResult)
            return AnalysisResult(
                text="motion detected" if motion_detected else "",
                blocks=[r.model_dump() for r in regions],
                confidence=float(motion_score),
                language=None,
                error=None,
            )

        except Exception as e:
            logger.exception("Motion analysis failed", extra={"plugin": self.name})
            return AnalysisResult(
                text="",
                blocks=[],
                confidence=0.0,
                language=None,
                error=str(e),
            )

    def _gaussian_blur(self, img: np.ndarray, size: int) -> np.ndarray:
        """Standard Gaussian kernel generation and separable filter application."""
        x = np.arange(size) - size // 2
        kernel = np.exp(-(x**2) / (2 * (size / 4) ** 2))
        kernel = kernel / kernel.sum()

        # Apply separable convolution across both axes
        result = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), 0, img
        )
        return np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), 1, result
        )

    def _find_motion_regions(
        self, motion_mask: np.ndarray, min_size: int = 100
    ) -> list[MotionRegion]:
        """Identifies the bounding box of contiguous motion pixels."""
        rows = np.any(motion_mask, axis=1)
        cols = np.any(motion_mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return []

        row_indices = np.where(rows)[0]
        col_indices = np.where(cols)[0]

        y_min, y_max = row_indices[0], row_indices[-1]
        x_min, x_max = col_indices[0], col_indices[-1]
        area = (x_max - x_min) * (y_max - y_min)

        if area < min_size:
            return []

        return [
            MotionRegion(
                bbox=BoundingBox(
                    x=int(x_min),
                    y=int(y_min),
                    width=int(x_max - x_min),
                    height=int(y_max - y_min),
                ),
                area=int(area),
                center={
                    "x": int((x_min + x_max) / 2),
                    "y": int((y_min + y_max) / 2),
                },
            )
        ]

    def reset(self) -> None:
        """Reset internal detector state."""
        self._previous_frame = None
        self._frame_count = 0
        self._last_motion_time = 0
        self._motion_history = []

    def on_load(self) -> None:
        """Lifecycle hook: plugin initialized."""
        logger.info("Motion detector plugin loaded")

    def on_unload(self) -> None:
        """Lifecycle hook: cleanup and shutdown."""
        self.reset()
        logger.info("Motion detector plugin unloaded")
