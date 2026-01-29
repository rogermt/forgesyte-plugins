"""OCR Plugin - Extract text from images using Tesseract.

This plugin extracts text from images using pytesseract with configurable
language support and page segmentation modes. Returns text, text blocks with
bounding boxes, and confidence scores for each block.

Migrated to BasePlugin architecture (Milestone 1.5).
"""

import logging
from typing import Any, Optional

from .ocr_engine import OCREngine
from .schemas import ImageSize, OCRInput, OCROutput, TextBlock

logger = logging.getLogger(__name__)


class Plugin:
    """OCR plugin for text extraction from images using Tesseract.

    Implements BasePlugin contract:
    - name: plugin identifier
    - tools: dict of tool definitions with string handler names
    - run_tool(): executes tool by name
    - on_load/on_unload: lifecycle hooks
    """

    name = "ocr"

    def __init__(self) -> None:
        """Initialize plugin with tools dict."""
        self.engine = OCREngine()
        self.supported_languages: list[str] = ["eng", "fra", "deu", "spa", "ita"]

        # Tools dict with STRING handler names (BasePlugin contract)
        self.tools = {
            "analyze": {
                "description": "Extract text and blocks from an image using OCR",
                "input_schema": OCRInput.model_json_schema(),
                "output_schema": OCROutput.model_json_schema(),
                "handler": "analyze",  # STRING, not callable
            }
        }

    def run_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Execute a tool by name with the given arguments.

        Args:
            tool_name: Name of tool to execute
            args: Tool arguments dict

        Returns:
            Tool result (OCROutput)

        Raises:
            ValueError: If tool name not found
        """
        if tool_name == "analyze":
            image_bytes = args.get("image_bytes")
            if not isinstance(image_bytes, bytes):
                raise ValueError("image_bytes must be bytes")
            return self.analyze(
                image_bytes=image_bytes,
                options=args.get("options"),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    def analyze(
        self, image_bytes: bytes, options: Optional[dict[str, Any]] = None
    ) -> OCROutput:
        """Perform OCR analysis on image bytes.

        Args:
            image_bytes: Image bytes (PNG, JPG, etc.)
            options: OCR options dict

        Returns:
            OCROutput with extracted text, blocks, and metadata
        """
        return self.engine.analyze(image_bytes, options)

    def on_load(self) -> None:
        """Initialize OCR engine on plugin load."""
        self.engine.on_load()

    def on_unload(self) -> None:
        """Clean up on plugin unload."""
        logger.info("OCR plugin unloaded")


# For backwards compatibility with existing tests, expose classes at module level
__all__ = ["Plugin", "TextBlock", "ImageSize", "OCRInput", "OCROutput"]
