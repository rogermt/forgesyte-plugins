"""Pydantic schemas for OCR plugin input/output."""

from typing import Any, Optional

from pydantic import BaseModel, Field


class OCRInput(BaseModel):
    """Input schema for OCR analysis."""

    image_bytes: bytes = Field(description="Image bytes (PNG, JPG, etc.)")
    options: Optional[dict[str, Any]] = Field(
        default=None, description="OCR options (language, psm, etc.)"
    )


class TextBlock(BaseModel):
    """Bounding box and text data for a single OCR block."""

    text: str
    confidence: float
    bbox: dict[str, int] = Field(description="Bounding box with x, y, width, height")
    level: int
    block_num: int
    line_num: int


class OCROutput(BaseModel):
    """Output schema for OCR analysis."""

    text: str = Field(description="Full extracted text")
    blocks: list[dict[str, Any]] = Field(
        default_factory=list, description="Text blocks with bounding boxes"
    )
    confidence: float = Field(description="Average confidence score (0-1)")
    language: Optional[str] = Field(default=None, description="Language used for OCR")
    error: Optional[str] = Field(default=None, description="Error message if failed")


class ImageSize(BaseModel):
    """Image dimensions for OCR analysis."""

    width: int
    height: int
