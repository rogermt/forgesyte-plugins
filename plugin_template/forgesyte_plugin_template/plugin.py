"""
ForgeSyte Plugin Template

This template provides a clean, production‑ready structure for building
ForgeSyte plugins. It mirrors the architecture of the OCR plugin:
- Strong metadata contract
- Structured analysis result
- Error handling
- Lifecycle hooks
- Configurable options
"""

import logging
from typing import Any, Optional

from app.models import AnalysisResult, PluginMetadata

logger = logging.getLogger(__name__)


class Plugin:
    """
    Template plugin for ForgeSyte.

    Replace the analyze() method with your actual logic.
    Update metadata() to describe your plugin’s inputs, outputs, and config.
    """

    name: str = "template_plugin"
    version: str = "1.0.0"

    def __init__(self) -> None:
        """Initialize plugin state here."""
        self.supported_modes = ["default"]

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------
    def metadata(self) -> PluginMetadata:
        """
        Return plugin metadata for discovery and configuration.

        Update:
        - description
        - inputs / outputs
        - config_schema
        """
        return PluginMetadata(
            name=self.name,
            description="Template plugin — replace with your description.",
            version=self.version,
            inputs=["image"],          # or ["text"], ["binary"], etc.
            outputs=["json"],          # or ["text"], ["regions"], etc.
            config_schema={
                "mode": {
                    "type": "string",
                    "default": "default",
                    "enum": self.supported_modes,
                    "description": "Processing mode for this plugin",
                }
            },
        )

    # ---------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------
    def analyze(
        self,
        image_bytes: bytes,
        options: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Main plugin logic.

        Args:
            image_bytes: Raw image data (or other input depending on plugin type)
            options: Configuration parameters from metadata.config_schema

        Returns:
            Dictionary from AnalysisResult.model_dump()
        """
        options = options or {}

        try:
            # -------------------------------------------------
            # TODO: Replace this with your actual logic
            # -------------------------------------------------
            result = AnalysisResult(
                text="",
                blocks=[],
                confidence=0.0,
                language=None,
                error="Template plugin has no implementation.",
            )
            return result.model_dump()

        except Exception as e:
            logger.error(
                "Plugin execution failed",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            result = AnalysisResult(
                text="",
                blocks=[],
                confidence=0.0,
                language=None,
                error=str(e),
            )
            return result.model_dump()

    # ---------------------------------------------------------
    # Lifecycle Hooks
    # ---------------------------------------------------------
    def on_load(self) -> None:
        """Called when plugin is loaded by PluginManager."""
        logger.info(f"{self.name} plugin loaded")

    def on_unload(self) -> None:
        """Called when plugin is unloaded by PluginManager."""
        logger.info(f"{self.name} plugin unloaded")
