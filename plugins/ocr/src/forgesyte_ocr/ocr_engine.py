"""OCR Engine wrapper - isolated OCR execution."""

import io
import logging
from typing import Any, Optional

from PIL import Image

from .schemas import OCROutput, TextBlock

logger = logging.getLogger(__name__)

# Try to import OCR dependencies
try:
    import pytesseract

    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False
    logger.warning("pytesseract not installed - OCR will use fallback")


class OCREngine:
    """Isolated OCR engine for text extraction from images."""

    def __init__(self) -> None:
        self.supported_languages: list[str] = ["eng", "fra", "deu", "spa", "ita"]

    def analyze(
        self, image_bytes: bytes, options: Optional[dict[str, Any]] = None
    ) -> OCROutput:
        """Perform OCR analysis on image bytes.

        Args:
            image_bytes: Image bytes (PNG, JPG, etc.)
            options: OCR options dict with keys like 'language', 'psm'

        Returns:
            OCROutput with extracted text, blocks, confidence, and error info
        """
        if not HAS_TESSERACT:
            return self._fallback_analyze(image_bytes, options)

        options = options or {}

        try:
            img: Image.Image = Image.open(io.BytesIO(image_bytes))

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

            return OCROutput(
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
            return OCROutput(
                text="",
                blocks=[],
                confidence=0.0,
                language=None,
                error=str(e),
            )

    def _fallback_analyze(
        self, image_bytes: bytes, options: Optional[dict[str, Any]] = None
    ) -> OCROutput:
        """Fallback when Tesseract not available."""
        return OCROutput(
            text="",
            blocks=[],
            confidence=0.0,
            language=None,
            error="Tesseract not installed. Install with: pip install pytesseract",
        )

    def on_load(self) -> None:
        """Initialize OCR engine on plugin load."""
        if HAS_TESSERACT:
            try:
                version = pytesseract.get_tesseract_version()
                logger.info(
                    "OCR engine initialized successfully",
                    extra={"tesseract_version": str(version)},
                )
            except Exception as e:
                logger.warning(
                    "Tesseract binary not found or not accessible",
                    extra={"error": str(e)},
                )
        else:
            logger.warning("OCR engine initialized without Tesseract support")
