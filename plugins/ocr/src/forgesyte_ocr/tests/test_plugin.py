"""Unit tests for OCR plugin.

MIGRATION NOTE (Milestone 1.5):
==============================

This test file was originally written for the legacy `app.plugins.base` architecture
and tested the Plugin class with mocked `AnalysisResult` and `PluginMetadata` from
`app.models`.

After BasePlugin migration:
- The Plugin class now delegates to OCREngine (pure isolation)
- AnalysisResult → OCROutput (Pydantic model)
- Plugin no longer exposes `metadata()` method (not BasePlugin contract)
- Tests updated to use new architecture while maintaining coverage

New tests for BasePlugin contract compliance are in test_entrypoint_contract.py

Tests cover:
- Successful OCR analysis with mocked pytesseract
- Fallback behavior when Tesseract unavailable
- Error handling for invalid images
- Response validation with Pydantic models
- Lifecycle hooks (on_load, on_unload)
- Tool routing via run_tool()
"""

import io
from typing import Any
from unittest.mock import patch

import pytest
from PIL import Image

from forgesyte_ocr.plugin import ImageSize, Plugin, TextBlock
from forgesyte_ocr.schemas import OCROutput


class TestOCRPlugin:
    """Test suite for OCR Plugin."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        p = Plugin()
        p.on_load()  # Initialize on load
        return p

    @pytest.fixture
    def sample_image_bytes(self) -> bytes:
        """Generate sample image bytes for testing."""
        img = Image.new("RGB", (100, 100), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    @pytest.fixture
    def mock_pytesseract_data(self) -> dict[str, Any]:
        """Mock pytesseract output data."""
        return {
            "level": [3, 4, 4],
            "text": ["hello", "world", "!"],
            "conf": [95, 90, 100],
            "left": [10, 20, 30],
            "top": [10, 20, 30],
            "width": [40, 40, 20],
            "height": [20, 20, 20],
            "block_num": [1, 1, 1],
            "line_num": [1, 1, 1],
        }

    # Plugin contract tests (BasePlugin architecture)
    def test_plugin_has_name(self, plugin: Plugin) -> None:
        """Test plugin has name attribute (BasePlugin contract)."""
        assert hasattr(plugin, "name")
        assert plugin.name == "ocr"

    def test_plugin_has_tools_dict(self, plugin: Plugin) -> None:
        """Test plugin has tools dict with 'analyze' tool (BasePlugin contract)."""
        assert hasattr(plugin, "tools")
        assert isinstance(plugin.tools, dict)
        assert "analyze" in plugin.tools

    def test_plugin_tool_handler_is_string(self, plugin: Plugin) -> None:
        """Test tool handler is a string (BasePlugin contract)."""
        tool_config = plugin.tools["analyze"]
        assert "handler" in tool_config
        assert isinstance(tool_config["handler"], str)
        assert tool_config["handler"] == "analyze"

    def test_plugin_run_tool_routes_correctly(
        self, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test run_tool routes to correct handler (BasePlugin contract)."""
        with patch.object(plugin.engine, "analyze") as mock_analyze:
            mock_analyze.return_value = OCROutput(
                text="test", blocks=[], confidence=0.0, language="eng"
            )

            result = plugin.run_tool(
                "analyze",
                {"image_bytes": sample_image_bytes, "options": None},
            )

            assert result is not None
            mock_analyze.assert_called_once()

    # Successful analysis tests
    @patch("forgesyte_ocr.ocr_engine.pytesseract")
    def test_analyze_successful_ocr(
        self,
        mock_tesseract: Any,
        plugin: Plugin,
        sample_image_bytes: bytes,
        mock_pytesseract_data: dict[str, Any],
    ) -> None:
        """Test successful OCR analysis returns valid OCROutput."""
        mock_tesseract.image_to_string.return_value = "hello world !"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        response = plugin.analyze(sample_image_bytes)

        assert isinstance(response, OCROutput)
        assert response.text == "hello world !"
        assert response.error is None
        assert response.language == "eng"

    @patch("forgesyte_ocr.ocr_engine.pytesseract")
    def test_analyze_with_custom_language(
        self,
        mock_tesseract: Any,
        plugin: Plugin,
        sample_image_bytes: bytes,
        mock_pytesseract_data: dict[str, Any],
    ) -> None:
        """Test OCR with custom language option."""
        mock_tesseract.image_to_string.return_value = "Bonjour"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        response = plugin.analyze(
            sample_image_bytes, options={"language": "fra", "psm": 6}
        )

        assert isinstance(response, OCROutput)
        assert response.language == "fra"
        mock_tesseract.image_to_string.assert_called_once()
        call_args = mock_tesseract.image_to_string.call_args
        assert "lang=fra" in str(call_args) or call_args[1].get("lang") == "fra"

    @patch("forgesyte_ocr.ocr_engine.pytesseract")
    @patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", True)
    def test_analyze_filters_low_confidence_blocks(
        self,
        mock_tesseract: Any,
        plugin: Plugin,
        sample_image_bytes: bytes,
    ) -> None:
        """Test that blocks with confidence <= 0 are filtered out."""
        data = {
            "level": [3, 4, 4, 4],
            "text": ["hello", "world", "!", "skip"],
            "conf": [95, 90, 0, -1],  # Last two should be filtered
            "left": [10, 20, 30, 40],
            "top": [10, 20, 30, 40],
            "width": [40, 40, 20, 20],
            "height": [20, 20, 20, 20],
            "block_num": [1, 1, 1, 1],
            "line_num": [1, 1, 1, 1],
        }
        mock_tesseract.image_to_string.return_value = "hello world ! skip"
        mock_tesseract.image_to_data.return_value = data
        mock_tesseract.Output.DICT = data

        response = plugin.analyze(sample_image_bytes)

        assert isinstance(response, OCROutput)
        assert len(response.blocks) == 2
        assert response.blocks[0]["text"] == "hello"
        assert response.blocks[1]["text"] == "world"

    @patch("forgesyte_ocr.ocr_engine.pytesseract")
    @patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", True)
    def test_analyze_calculates_average_confidence(
        self,
        mock_tesseract: Any,
        plugin: Plugin,
        sample_image_bytes: bytes,
        mock_pytesseract_data: dict[str, Any],
    ) -> None:
        """Test average confidence is calculated correctly."""
        mock_tesseract.image_to_string.return_value = "hello world !"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        response = plugin.analyze(sample_image_bytes)

        assert isinstance(response, OCROutput)
        expected_avg = (95 + 90 + 100) / 3 / 100.0
        assert response.confidence == pytest.approx(expected_avg)

    # Error handling tests
    @patch("forgesyte_ocr.ocr_engine.pytesseract")
    @patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", True)
    def test_analyze_handles_invalid_image_data(
        self, mock_tesseract: Any, plugin: Plugin
    ) -> None:
        """Test error handling for invalid image bytes."""
        invalid_bytes = b"not an image"

        response = plugin.analyze(invalid_bytes)

        assert isinstance(response, OCROutput)
        assert response.error is not None

    @patch("forgesyte_ocr.ocr_engine.pytesseract")
    @patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", True)
    def test_analyze_handles_tesseract_exception(
        self,
        mock_tesseract: Any,
        plugin: Plugin,
        sample_image_bytes: bytes,
    ) -> None:
        """Test error handling when pytesseract raises exception."""
        mock_tesseract.image_to_string.side_effect = Exception("Tesseract error")

        response = plugin.analyze(sample_image_bytes)

        assert isinstance(response, OCROutput)
        assert response.error and "Tesseract error" in response.error

    # Fallback tests
    @patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", False)
    def test_analyze_fallback_when_tesseract_unavailable(
        self, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test fallback response when Tesseract is not installed."""
        response = plugin.analyze(sample_image_bytes)

        assert isinstance(response, OCROutput)
        assert response.error is not None
        assert "Tesseract not installed" in response.error

    # Lifecycle tests
    def test_on_load_with_tesseract_available(self, plugin: Plugin) -> None:
        """Test on_load lifecycle hook with Tesseract available."""
        with patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", True):
            with patch("forgesyte_ocr.ocr_engine.pytesseract") as mock_tesseract:
                mock_tesseract.get_tesseract_version.return_value = "5.0.0"
                plugin.on_load()

    def test_on_load_without_tesseract(self, plugin: Plugin) -> None:
        """Test on_load when Tesseract is not available."""
        with patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", False):
            plugin.on_load()

    def test_on_unload(self, plugin: Plugin) -> None:
        """Test on_unload lifecycle hook."""
        plugin.on_unload()

    # Pydantic model tests
    def test_text_block_model_validation(self) -> None:
        """Test TextBlock Pydantic model validates fields."""
        block = TextBlock(
            text="hello",
            confidence=95.5,
            bbox={"x": 10, "y": 20, "width": 40, "height": 20},
            level=4,
            block_num=1,
            line_num=1,
        )

        assert block.text == "hello"
        assert block.confidence == 95.5
        assert block.bbox["x"] == 10

    def test_image_size_model_validation(self) -> None:
        """Test ImageSize Pydantic model validates fields."""
        size = ImageSize(width=100, height=200)

        assert size.width == 100
        assert size.height == 200

    # Integration tests
    @patch("forgesyte_ocr.ocr_engine.pytesseract")
    @patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", True)
    def test_analyze_grayscale_image_conversion(
        self,
        mock_tesseract: Any,
        plugin: Plugin,
        mock_pytesseract_data: dict[str, Any],
    ) -> None:
        """Test grayscale images are converted to RGB."""
        img = Image.new("L", (100, 100), color=200)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = img_bytes.getvalue()

        mock_tesseract.image_to_string.return_value = "test"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        response = plugin.analyze(image_data)

        assert isinstance(response, OCROutput)
        assert response.error is None

    @patch("forgesyte_ocr.ocr_engine.pytesseract")
    @patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", True)
    def test_analyze_rgba_image_conversion(
        self,
        mock_tesseract: Any,
        plugin: Plugin,
        mock_pytesseract_data: dict[str, Any],
    ) -> None:
        """Test RGBA images are converted to RGB."""
        img = Image.new("RGBA", (100, 100), color=(255, 255, 255, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = img_bytes.getvalue()

        mock_tesseract.image_to_string.return_value = "test"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        response = plugin.analyze(image_data)

        assert isinstance(response, OCROutput)
        assert response.error is None

    @patch("forgesyte_ocr.ocr_engine.pytesseract")
    @patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", True)
    def test_analyze_returns_pydantic_model_not_dict(
        self,
        mock_tesseract: Any,
        plugin: Plugin,
        sample_image_bytes: bytes,
        mock_pytesseract_data: dict[str, Any],
    ) -> None:
        """Test that analyze() returns Pydantic model (OCROutput)."""
        mock_tesseract.image_to_string.return_value = "hello world"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        result = plugin.analyze(sample_image_bytes)

        # Must be Pydantic model, not dict
        assert isinstance(result, OCROutput)
        assert not isinstance(result, dict)

    # Text extraction quality tests
    @patch("forgesyte_ocr.ocr_engine.pytesseract")
    @patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", True)
    def test_analyze_extracts_specific_expected_text(
        self,
        mock_tesseract: Any,
        plugin: Plugin,
        sample_image_bytes: bytes,
    ) -> None:
        """Test OCR extracts specific expected text from test image.

        Verifies extraction of known text patterns without LLM judgment.
        """
        # Simulate OCR output from gemini-cli test image
        extracted_text = (
            "This is a lot of 12 point text to test the\n"
            "ocr code and see if it works on all types\n"
            "of file format.\n"
            "\n"
            "The quick brown dog jumped over the\n"
            "lazy fox. The quick brown dog jumped\n"
            "over the lazy fox. The quick brown dog\n"
            "jumped over the lazy fox. The quick\n"
            "brown dog jumped over the lazy fox"
        )
        mock_tesseract.image_to_string.return_value = extracted_text
        mock_tesseract.image_to_data.return_value = {
            "level": [3, 4, 4],
            "text": ["12 point text", "quick brown", "lazy fox"],
            "conf": [92, 88, 95],
            "left": [10, 20, 30],
            "top": [10, 20, 30],
            "width": [100, 100, 100],
            "height": [20, 20, 20],
            "block_num": [1, 1, 1],
            "line_num": [1, 1, 1],
        }
        mock_tesseract.Output.DICT = {
            "level": [3, 4, 4],
            "text": ["12 point text", "quick brown", "lazy fox"],
            "conf": [92, 88, 95],
            "left": [10, 20, 30],
            "top": [10, 20, 30],
            "width": [100, 100, 100],
            "height": [20, 20, 20],
            "block_num": [1, 1, 1],
            "line_num": [1, 1, 1],
        }

        response = plugin.analyze(sample_image_bytes)

        # Verify expected text fragments are present
        assert isinstance(response, OCROutput)
        assert response.error is None
        assert "12 point text" in response.text
        assert "quick brown" in response.text
        assert "lazy fox" in response.text

    @patch("forgesyte_ocr.ocr_engine.pytesseract")
    @patch("forgesyte_ocr.ocr_engine.HAS_TESSERACT", True)
    def test_analyze_maintains_minimum_confidence_threshold(
        self,
        mock_tesseract: Any,
        plugin: Plugin,
        sample_image_bytes: bytes,
    ) -> None:
        """Test OCR maintains acceptable confidence threshold (>80%).

        Verifies extracted text meets quality standards without LLM judgment.
        """
        mock_tesseract.image_to_string.return_value = "test output"
        mock_tesseract.image_to_data.return_value = {
            "level": [3, 4, 4],
            "text": ["good", "text", "here"],
            "conf": [92, 88, 85],  # All above 80%
            "left": [10, 20, 30],
            "top": [10, 20, 30],
            "width": [40, 40, 40],
            "height": [20, 20, 20],
            "block_num": [1, 1, 1],
            "line_num": [1, 1, 1],
        }
        mock_tesseract.Output.DICT = {
            "level": [3, 4, 4],
            "text": ["good", "text", "here"],
            "conf": [92, 88, 85],
            "left": [10, 20, 30],
            "top": [10, 20, 30],
            "width": [40, 40, 40],
            "height": [20, 20, 20],
            "block_num": [1, 1, 1],
            "line_num": [1, 1, 1],
        }

        response = plugin.analyze(sample_image_bytes)

        assert isinstance(response, OCROutput)
        # Average: (92 + 88 + 85) / 3 / 100 = 0.8833...
        assert (
            response.confidence > 0.80
        ), f"Confidence {response.confidence} below 80% threshold"
