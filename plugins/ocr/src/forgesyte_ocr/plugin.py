"""OCR Plugin - Extract text from images using Tesseract.

This plugin extracts text from images using pytesseract with configurable
language support and page segmentation modes. Returns text, text blocks with
bounding boxes, and confidence scores for each block.
"""

import io
import logging
from typing import Any, Optional

from PIL import Image
from pydantic import BaseModel, Field

from app.plugins.base import BasePlugin
from app.models import AnalysisResult, PluginMetadata

logger = logging.getLogger(__name__)

# Try to import OCR dependencies
try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning("pytesseract not installed - OCR will use fallback")


class TextBlock(BaseModel):  # type: ignore[misc]
    text: str
    confidence: float
    bbox: dict[str, int] = Field(description="Bounding box with x, y, width, height")
    level: int
    block_num: int
    line_num: int


class ImageSize(BaseModel):  # type: ignore[misc]
    width: int
    height: int


class Plugin(BasePlugin):
    """OCR plugin for text extraction from images using Tesseract."""

    name: str = "ocr"
    version: str = "1.0.0"
    description: str = (
        "Extracts structured text and positions from images with confidence scores."
    )

    def __init__(self) -> None:
        self.supported_languages: list[str] = ["eng", "fra", "deu", "spa", "ita"]
        # Initialize tools after instance is created (so handler can reference self)
        self.tools = {
            "analyze": {
                "description": "Extract text and blocks from an image using OCR",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "image_bytes": {"type": "string", "format": "binary"},
                        "options": {"type": "object"},
                    },
                    "required": ["image_bytes"],
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "blocks": {"type": "array"},
                        "confidence": {"type": "number"},
                        "language": {"type": "string"},
                        "error": {"type": "string"},
                    },
                },
                "handler": self.analyze,
            }
        }
        super().__init__()

    def run_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
        """Execute a tool by name with the given arguments."""
        if tool_name == "analyze":
            return self.analyze(
                image_bytes=args.get("image_bytes"),
                options=args.get("options"),
            )
        raise ValueError(f"Unknown tool: {tool_name}")

    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name=self.name,
            description=self.description,
            version=self.version,
            inputs=["image"],
            outputs=["text", "blocks", "confidence"],
            config_schema={
                "language": {
                    "type": "string",
                    "default": "eng",
                    "enum": self.supported_languages,
                    "description": "OCR language code",
                },
                "psm": {
                    "type": "integer",
                    "default": 3,
                    "description": "Page segmentation mode (0-13)",
                },
            },
        )

    def analyze(
        self, image_bytes: bytes, options: Optional[dict[str, Any]] = None
    ) -> AnalysisResult:
        if not HAS_TESSERACT:
            return self._fallback_analyze(image_bytes, options)

        options = options or {}

        try:
            img = Image.open(io.BytesIO(image_bytes))

            if img.mode not in ("L", "RGB"):
                img = img.convert("RGB")

            lang = options.get("language", "eng")
            psm = options.get("psm", 3)
            config = f"--psm {psm}"

            text = pytesseract.image_to_string(img, lang=lang, config=config)

            data = pytesseract.image_to_data(
                img, lang=lang, config=config, output_type=pytesseract.Output.DICT
            )

            blocks = []
            n_boxes = len(data["level"])
            for i in range(n_boxes):
                conf = int(data["conf"][i])
                if conf > 0:
                    blocks.append(
                        TextBlock(
                            text=data["text"][i],
                            confidence=float(conf),
                            bbox={
                                "x": data["left"][i],
                                "y": data["top"][i],
                                "width": data["width"][i],
                                "height": data["height"][i],
                            },
                            level=data["level"][i],
                            block_num=data["block_num"][i],
                            line_num=data["line_num"][i],
                        )
                    )

            confidences = [b.confidence for b in blocks if b.confidence > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            blocks_dict = [
                {
                    "text": b.text,
                    "confidence": b.confidence,
                    "bbox": b.bbox,
                    "level": b.level,
                    "block_num": b.block_num,
                    "line_num": b.line_num,
                }
                for b in blocks
            ]

            return AnalysisResult(
                text=text.strip(),
                blocks=blocks_dict,
                confidence=avg_confidence / 100.0,
                language=lang,
                error=None,
            )

        except Exception as e:
            logger.error(
                "OCR execution failed",
                extra={"error": str(e), "error_type": type(e).__name__},
            )
            return AnalysisResult(
                text="",
                blocks=[],
                confidence=0.0,
                language=None,
                error=str(e),
            )

    def _fallback_analyze(
        self, image_bytes: bytes, options: Optional[dict[str, Any]] = None
    ) -> AnalysisResult:
        return AnalysisResult(
            text="",
            blocks=[],
            confidence=0.0,
            language=None,
            error="Tesseract not installed. Install with: pip install pytesseract",
        )

    def on_load(self) -> None:
        if HAS_TESSERACT:
            try:
                version = pytesseract.get_tesseract_version()
                logger.info(
                    "OCR plugin loaded successfully",
                    extra={"tesseract_version": str(version)},
                )
            except Exception as e:
                logger.warning(
                    "Tesseract binary not found or not accessible",
                    extra={"error": str(e)},
                )
        else:
            logger.warning("OCR plugin loaded without Tesseract support")

    def on_unload(self) -> None:
        logger.info("OCR plugin unloaded")
