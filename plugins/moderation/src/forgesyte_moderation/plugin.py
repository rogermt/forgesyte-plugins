"""
Content Moderation Plugin.

Analyzes images for unsafe content across NSFW, violence, and hate speech categories.
"""

import io
import logging
from typing import Any

try:
    import numpy as np
    from PIL import Image

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

from app.models import AnalysisResult, PluginMetadata
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validated Data Models (Internal)
# ---------------------------------------------------------------------------


class CategoryResult(BaseModel):  # type: ignore[misc]
    """Validated result for an individual moderation category."""

    category: str
    score: float = Field(ge=0.0, le=1.0)
    flagged: bool
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Moderation Plugin Implementation
# ---------------------------------------------------------------------------


class Plugin:
    """
    Content moderation plugin for safety detection.
    Analyzes images for unsafe content across NSFW, violence, and hate speech.
    """

    name: str = "moderation"
    version: str = "1.0.0"
    description: str = "Detect potentially unsafe or inappropriate content"

    def __init__(self) -> None:
        """Initialize the moderation plugin with default state."""
        self.sensitivity: str = "medium"

    def metadata(self) -> PluginMetadata:
        """Return plugin metadata validated by Pydantic."""
        return PluginMetadata(
            name=self.name,
            version=self.version,
            description=self.description,
            inputs=["image"],
            outputs=["safe", "categories", "confidence"],
            config_schema={
                "sensitivity": {
                    "type": "string",
                    "default": "medium",
                    "enum": ["low", "medium", "high"],
                    "description": "Detection sensitivity level",
                },
                "categories": {
                    "type": "array",
                    "default": ["nsfw", "violence", "hate"],
                    "description": "Categories to check",
                },
            },
        )

    def analyze(
        self, image_bytes: bytes, options: dict[str, Any] | None = None
    ) -> AnalysisResult:
        """
        Analyze image for content safety.
        Returns a universal AnalysisResult for ForgeSyte Core.
        """
        opts = options or {}

        # Fallback if PIL or NumPy are missing
        if not HAS_DEPS:
            return self._basic_analysis(image_bytes, opts)

        try:
            img = Image.open(io.BytesIO(image_bytes))

            # Extract configuration
            sensitivity = opts.get("sensitivity", self.sensitivity)
            categories = opts.get("categories", ["nsfw", "violence", "hate"])

            # Perform heuristic analysis
            analysis = self._analyze_content(img, categories, sensitivity)

            # Calculate overall safety verdict based on sensitivity thresholds
            threshold = self._get_threshold(sensitivity)
            is_safe = all(cat.score < threshold for cat in analysis["categories"])

            # Generate content hash for tracking/caching
            # (Note: content_hash is computed but not explicitly returned in
            # universal result, could be part of metadata if needed,
            # but keeping it simple for now)
            # content_hash = hashlib.md5(image_bytes).hexdigest()

            recommendation = self._get_recommendation(is_safe, analysis["categories"])

            # Map to universal AnalysisResult
            # text -> recommendation/summary
            # blocks -> detailed category results
            # confidence -> overall confidence
            return AnalysisResult(
                text=recommendation,
                blocks=[cat.model_dump() for cat in analysis["categories"]],
                confidence=analysis["overall_confidence"],
                language=None,
                error=None,
            )

        except Exception as e:
            logger.error("Moderation analysis failed", extra={"error": str(e)})
            return AnalysisResult(
                text="",
                blocks=[],
                confidence=0.0,
                language=None,
                error=str(e),
            )

    def _analyze_content(
        self, img: "Image.Image", categories: list[str], sensitivity: str
    ) -> dict[str, Any]:
        """Run analysis on each category and compute overall confidence."""
        arr = np.array(img.convert("RGB"))
        results: list[CategoryResult] = []
        threshold = self._get_threshold(sensitivity)

        for category in categories:
            score = self._calculate_placeholder_score(arr, category)
            # Confidence is higher when the score is near the extremes (0 or 1)
            confidence = 0.5 + (0.5 - abs(score - 0.5))

            results.append(
                CategoryResult(
                    category=category,
                    score=score,
                    flagged=score > threshold,
                    confidence=confidence,
                )
            )

        overall_conf = (
            sum(r.confidence for r in results) / len(results) if results else 0.0
        )
        return {"categories": results, "overall_confidence": overall_conf}

    def _calculate_placeholder_score(self, arr: "np.ndarray", category: str) -> float:
        """Heuristic-based safety scoring using color distribution."""
        if category == "nsfw":
            # Skin tone detection heuristic
            r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
            skin_like = (
                (r > 95)
                & (g > 40)
                & (b > 20)
                & (
                    (np.maximum(r, np.maximum(g, b)) - np.minimum(r, np.minimum(g, b)))
                    > 15
                )
                & (abs(r.astype(int) - g.astype(int)) > 15)
                & (r > g)
                & (r > b)
            )
            skin_ratio = np.mean(skin_like)
            return float(
                min(skin_ratio * 2, 1.0) * 0.3
            )  # Scaled to prevent false positives

        elif category == "violence":
            # Red-content ratio heuristic
            red_ratio = np.mean(arr[:, :, 0] > 150)
            return float(red_ratio * 0.2)

        elif category == "hate":
            return 0.05  # Placeholder for symbol detection

        return 0.1

    def _get_threshold(self, sensitivity: str) -> float:
        """Map sensitivity labels to numerical thresholds."""
        thresholds = {"low": 0.8, "medium": 0.5, "high": 0.3}
        return thresholds.get(sensitivity, 0.5)

    def _get_recommendation(
        self, is_safe: bool, categories: list[CategoryResult]
    ) -> str:
        """Generate actionable reviewer recommendations."""
        if is_safe:
            return "Content appears safe for general viewing"

        flagged = [c.category for c in categories if c.flagged]
        if flagged:
            return (
                f"Content flagged for: {', '.join(flagged)}. Manual review recommended."
            )

        return "Content may require review"

    def _basic_analysis(
        self, image_bytes: bytes, options: dict[str, Any]
    ) -> AnalysisResult:
        """Fallback analysis for environments without heavy dependencies."""
        return AnalysisResult(
            text="Warning: Full moderation requires PIL and numpy",
            blocks=[],
            confidence=0.0,
            language=None,
            error="Missing dependencies: PIL, numpy",
        )

    def on_load(self) -> None:
        """Lifecycle hook: logs load and placeholder warning."""
        logger.info("Moderation plugin loaded")
        logger.warning(
            "Using heuristic analysis - replace with real ML model for production",
            extra={"plugin_name": self.name},
        )

    def on_unload(self) -> None:
        """Lifecycle hook: logs shutdown."""
        logger.info("Moderation plugin unloaded")
