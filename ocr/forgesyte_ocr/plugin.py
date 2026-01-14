"""OCR Plugin - Extract text from images using Tesseract.

This plugin extracts text from images using pytesseract with configurable
language support and page segmentation modes. Returns text, text blocks with
bounding boxes, and confidence scores for each block.
"""

import io
import logging
from typing import Any, Optional

from app.models import AnalysisResult, PluginMetadata
from PIL import Image
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Try to import OCR dependencies
try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning("pytesseract not installed - OCR will use fallback")


class TextBlock(BaseModel):
    """A single text block detected by OCR.

    Attributes:
        text: Extracted text content
        confidence: Confidence score (0-100)
        bbox: Bounding box coordinates
        level: OCR level (word, line, block, etc.)
        block_num: Block number
        line_num: Line number
    """

    text: str
    confidence: float
    bbox: dict[str, int] = Field(description="Bounding box with x, y, width, height")
    level: int
    block_num: int
    line_num: int


class ImageSize(BaseModel):
    """Image dimensions.

    Attributes:
        width: Image width in pixels
        height: Image height in pixels
    """

    width: int
    height: int


class OCRResponse(BaseModel):
    """OCR analysis response with validated data.

    Attributes:
        text: Full extracted text
        blocks: List of detected text blocks with positions
        confidence: Average confidence score across blocks
        language: Language used for OCR
        image_size: Dimensions of input image
        error: Error message if analysis failed, None on success
    """

    text: str
    blocks: list[TextBlock]
    confidence: float
    language: Optional[str] = None
    image_size: Optional[ImageSize] = None
    error: Optional[str] = None


class Plugin:
    """OCR plugin for text extraction from images using Tesseract.

    Extracts text with position information and confidence scores using
    Tesseract OCR with support for multiple languages and page segmentation modes.

    Attributes:
        name: Plugin identifier
        version: Plugin version
        supported_languages: List of supported language codes
    """

    name: str = "ocr"
    version: str = "1.0.0"

    def __init__(self) -> None:
        """Initialize the OCR plugin.

        Sets up supported language list and validates Tesseract availability.
        """
        self.supported_languages: list[str] = ["eng", "fra", "deu", "spa", "ita"]

    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for discovery and configuration.

        Returns:
            PluginMetadata with plugin info, inputs/outputs, and config schema
            for language and page segmentation mode.
        """
        return PluginMetadata(
            name=self.name,
            description=(
                "Extracts structured text and positions from images "
                "with confidence scores."
            ),
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
        """Extract text from an image using OCR.

        Performs OCR using Tesseract with configurable language and
        page segmentation mode. Returns extracted text and individual
        text blocks with position and confidence information.

        Args:
            image_bytes: Raw image data (PNG, JPEG, etc.).
            options: Configuration with language and PSM (page segmentation mode).

        Returns:
            AnalysisResult with extracted text, text blocks with bboxes, confidence,
            and image metadata. On error, returns AnalysisResult with error field set.
        """
        if not HAS_TESSERACT:
            return self._fallback_analyze(image_bytes, options)

        options = options or {}

        try:
            # Load and validate image
            img = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if necessary for Tesseract compatibility
            if img.mode not in ("L", "RGB"):
                img = img.convert("RGB")

            # Get OCR configuration
            lang = options.get("language", "eng")
            psm = options.get("psm", 3)
            config = f"--psm {psm}"

            # Extract text with Tesseract
            text = pytesseract.image_to_string(img, lang=lang, config=config)

            # Get detailed OCR data with confidence scores
            data = pytesseract.image_to_data(
                img, lang=lang, config=config, output_type=pytesseract.Output.DICT
            )

            # Build structured blocks with positions and confidence
            blocks = []
            n_boxes = len(data["level"])
            for i in range(n_boxes):
                conf = int(data["conf"][i])
                if conf > 0:  # Filter low confidence results
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

            # Calculate average confidence across all blocks
            confidences = [b.confidence for b in blocks if b.confidence > 0]
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

            # Convert blocks to dict format for AnalysisResult
            # Note: TextBlock confidence is 0-100, keep as-is for blocks detail
            blocks_dict = [
                {
                    "text": b.text,
                    "confidence": b.confidence,  # Keep 0-100 for granular info
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
                confidence=avg_confidence / 100.0,  # AnalysisResult expects 0.0-1.0
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
        """Fallback when Tesseract is not available.

        Returns placeholder results with diagnostic information.

        Args:
            image_bytes: Raw image data (unused in fallback).
            options: Configuration options (unused in fallback).

        Returns:
            AnalysisResult with empty results and installation instructions.
        """
        return AnalysisResult(
            text="",
            blocks=[],
            confidence=0.0,
            language=None,
            error="Tesseract not installed. Install with: pip install pytesseract",
        )

    def on_load(self) -> None:
        """Called when plugin is loaded by the plugin loader.

        Validates Tesseract availability and logs version information
        for observability and debugging.
        """
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
        """Called when plugin is unloaded by the plugin loader.

        Performs cleanup and logs shutdown for observability.
        """
        logger.info("OCR plugin unloaded")
