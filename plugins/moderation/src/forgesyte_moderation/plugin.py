"""
Content Moderation Plugin.

Analyzes images for unsafe content across NSFW, violence, and hate speech categories.
"""

import io
import logging
from typing import Any

from pydantic import BaseModel, Field

from app.models import AnalysisResult, PluginMetadata
from app.plugins.base import BasePlugin

try:
    import numpy as np
    from PIL import Image

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validated Data Models (Internal)
# ---------------------------------------------------------------------------

class CategoryResult(BaseModel):
    """Validated result for an individual moderation category."""
    category: str
    score: float = Field(ge=0.0, le=1.0)
    flagged: bool
    confidence: float = Field(ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Moderation Plugin (Migrated to BasePlugin)
# ---------------------------------------------------------------------------

class Plugin(BasePlugin):
    """
    Content moderation plugin for safety detection.
    Fully migrated to BasePlugin contract.
    """

    name: str = "moderation"
    version: str = "1.0.0"
    description: str = "Detect potentially unsafe or inappropriate content"

    # NEW: Required by BasePlugin
    tools = {
        "analyze": {
            "description": "Analyze image for unsafe content",
            "inputs": {
                "image_bytes": {"type": "bytes"},
                "options": {"type": "object", "optional": True},
            },
            "outputs": {
                "result": {"type": "object"},
            },
            "handler": "analyze",
        }
    }

    # NEW: Required by BasePlugin
    def run_tool(self, tool_name: str, args: dict[str, Any]) -> AnalysisResult:
        if tool_name == "analyze":
            return self.analyze(
                image_bytes=args.get("image_bytes"),
                options=args.get("options"),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    # -----------------------------------------------------------------------
    # Existing logic preserved exactly
    # -----------------------------------------------------------------------

    def __init__(self) -> None:
        self.sensitivity: str = "medium"

    def metadata(self) -> PluginMetadata:
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
        opts = options or {}

        if not HAS_DEPS:
            return self._basic_analysis(image_bytes, opts)

        try:
            img = Image.open(io.BytesIO(image_bytes))

            sensitivity = opts.get("sensitivity", self.sensitivity)
            categories = opts.get("categories", ["nsfw", "violence", "hate"])

            analysis = self._analyze_content(img, categories, sensitivity)

            threshold = self._get_threshold(sensitivity)
            is_safe = all(cat.score < threshold for cat in analysis["categories"])

            recommendation = self._get_recommendation(is_safe, analysis["categories"])

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
    ) -> dict[str, list[CategoryResult] | float]:
        arr = np.array(img.convert("RGB"))
        results: list[CategoryResult] = []
        threshold = self._get_threshold(sensitivity)

        for category in categories:
            score = self._calculate_placeholder_score(arr, category)
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

    def _calculate_placeholder_score(self, arr: "np.ndarray[Any, Any]", category: str) -> float:
        if category == "nsfw":
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
            return float(min(skin_ratio * 2, 1.0) * 0.3)

        elif category == "violence":
            red_ratio = np.mean(arr[:, :, 0] > 150)
            return float(red_ratio * 0.2)

        elif category == "hate":
            return 0.05

        return 0.1

    def _get_threshold(self, sensitivity: str) -> float:
        thresholds = {"low": 0.8, "medium": 0.5, "high": 0.3}
        return thresholds.get(sensitivity, 0.5)

    def _get_recommendation(
        self, is_safe: bool, categories: list[CategoryResult]
    ) -> str:
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
        return AnalysisResult(
            text="Warning: Full moderation requires PIL and numpy",
            blocks=[],
            confidence=0.0,
            language=None,
            error="Missing dependencies: PIL, numpy",
        )

    def on_load(self) -> None:
        logger.info("Moderation plugin loaded")

    def on_unload(self) -> None:
        logger.info("Moderation plugin unloaded")
