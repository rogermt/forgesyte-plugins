"""Unit tests for OCR plugin.

Tests cover:
- Successful OCR analysis with mocked pytesseract
- Fallback behavior when Tesseract unavailable
- Error handling for invalid images
- Response validation with Pydantic models
- Lifecycle hooks (on_load, on_unload)
- Metadata endpoint
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from forgesyte_ocr.plugin import Plugin, TextBlock, ImageSize, OCRResponse


class TestOCRPlugin:
    """Test suite for OCR Plugin."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance for testing."""
        return Plugin()

    @pytest.fixture
    def sample_image_bytes(self):
        """Generate sample image bytes for testing."""
        img = Image.new("RGB", (100, 100), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    @pytest.fixture
    def mock_pytesseract_data(self):
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

    # Metadata tests
    def test_metadata_returns_plugin_metadata(self, plugin):
        """Test metadata endpoint returns valid PluginMetadata."""
        metadata = plugin.metadata()
        assert metadata.name == "ocr"
        assert metadata.version == "1.0.0"
        assert "image" in metadata.inputs
        assert "text" in metadata.outputs
        assert "blocks" in metadata.outputs
        assert "confidence" in metadata.outputs

    def test_metadata_includes_config_schema(self, plugin):
        """Test metadata includes language and PSM configuration."""
        metadata = plugin.metadata()
        config = metadata.config_schema
        assert "language" in config
        assert "psm" in config
        assert config["language"]["default"] == "eng"
        assert config["psm"]["default"] == 3

    def test_metadata_supported_languages(self, plugin):
        """Test metadata includes list of supported languages."""
        metadata = plugin.metadata()
        languages = metadata.config_schema["language"]["enum"]
        assert "eng" in languages
        assert "fra" in languages
        assert len(languages) > 0

    # Successful analysis tests
    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    def test_analyze_successful_ocr(
        self, mock_tesseract, plugin, sample_image_bytes, mock_pytesseract_data
    ):
        """Test successful OCR analysis returns valid OCRResponse."""
        mock_tesseract.image_to_string.return_value = "hello world !"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data  # Mock Output.DICT

        response = plugin.analyze(sample_image_bytes)

        assert isinstance(response, OCRResponse)
        assert response.text == "hello world !"
        assert len(response.blocks) == 3
        assert response.error is None
        assert response.language == "eng"
        assert response.image_size is not None

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    def test_analyze_with_custom_language(
        self, mock_tesseract, plugin, sample_image_bytes, mock_pytesseract_data
    ):
        """Test OCR with custom language option."""
        mock_tesseract.image_to_string.return_value = "Bonjour"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        response = plugin.analyze(
            sample_image_bytes, options={"language": "fra", "psm": 6}
        )

        assert response.language == "fra"
        mock_tesseract.image_to_string.assert_called_once()
        call_args = mock_tesseract.image_to_string.call_args
        assert "lang=fra" in str(call_args) or call_args[1].get("lang") == "fra"

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    def test_analyze_filters_low_confidence_blocks(
        self, mock_tesseract, plugin, sample_image_bytes
    ):
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

        assert len(response.blocks) == 2  # Only 95 and 90 confidence blocks
        assert response.blocks[0].text == "hello"
        assert response.blocks[1].text == "world"

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    def test_analyze_calculates_average_confidence(
        self, mock_tesseract, plugin, sample_image_bytes, mock_pytesseract_data
    ):
        """Test average confidence is calculated correctly."""
        mock_tesseract.image_to_string.return_value = "hello world !"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        response = plugin.analyze(sample_image_bytes)

        # Average of 95, 90, 100
        expected_avg = (95 + 90 + 100) / 3
        assert response.confidence == expected_avg

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    def test_analyze_image_size_captured(
        self, mock_tesseract, plugin, sample_image_bytes, mock_pytesseract_data
    ):
        """Test image dimensions are captured in response."""
        mock_tesseract.image_to_string.return_value = "test"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        response = plugin.analyze(sample_image_bytes)

        assert response.image_size is not None
        assert response.image_size.width == 100
        assert response.image_size.height == 100

    # Error handling tests
    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    def test_analyze_handles_invalid_image_data(
        self, mock_tesseract, plugin
    ):
        """Test error handling for invalid image bytes."""
        invalid_bytes = b"not an image"

        response = plugin.analyze(invalid_bytes)

        assert isinstance(response, OCRResponse)
        assert response.error is not None
        assert response.text == ""
        assert len(response.blocks) == 0
        assert response.confidence == 0.0

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    def test_analyze_handles_tesseract_exception(
        self, mock_tesseract, plugin, sample_image_bytes
    ):
        """Test error handling when pytesseract raises exception."""
        mock_tesseract.image_to_string.side_effect = Exception("Tesseract error")

        response = plugin.analyze(sample_image_bytes)

        assert response.error is not None
        assert "Tesseract error" in response.error
        assert response.text == ""

    # Fallback tests
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", False)
    def test_analyze_fallback_when_tesseract_unavailable(
        self, plugin, sample_image_bytes
    ):
        """Test fallback response when Tesseract is not installed."""
        response = plugin.analyze(sample_image_bytes)

        assert isinstance(response, OCRResponse)
        assert response.text == ""
        assert len(response.blocks) == 0
        assert response.error is not None
        assert "Tesseract not installed" in response.error

    # Lifecycle tests
    def test_on_load_with_tesseract_available(self, plugin):
        """Test on_load lifecycle hook with Tesseract available."""
        with patch("forgesyte_ocr.plugin.HAS_TESSERACT", True):
            with patch("forgesyte_ocr.plugin.pytesseract") as mock_tesseract:
                mock_tesseract.get_tesseract_version.return_value = "5.0.0"
                plugin.on_load()
                # Should not raise exception

    def test_on_load_without_tesseract(self, plugin):
        """Test on_load when Tesseract is not available."""
        with patch("forgesyte_ocr.plugin.HAS_TESSERACT", False):
            plugin.on_load()
            # Should not raise exception

    def test_on_unload(self, plugin):
        """Test on_unload lifecycle hook."""
        plugin.on_unload()
        # Should not raise exception

    # Pydantic model tests
    def test_text_block_model_validation(self):
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

    def test_image_size_model_validation(self):
        """Test ImageSize Pydantic model validates fields."""
        size = ImageSize(width=100, height=200)

        assert size.width == 100
        assert size.height == 200

    def test_ocr_response_model_validation(self, sample_image_bytes):
        """Test OCRResponse Pydantic model validates and serializes."""
        block = TextBlock(
            text="test",
            confidence=90.0,
            bbox={"x": 0, "y": 0, "width": 50, "height": 20},
            level=4,
            block_num=1,
            line_num=1,
        )
        size = ImageSize(width=100, height=100)

        response = OCRResponse(
            text="test output",
            blocks=[block],
            confidence=90.0,
            language="eng",
            image_size=size,
            error=None,
        )

        assert response.text == "test output"
        assert len(response.blocks) == 1
        assert response.language == "eng"
        assert response.error is None

    def test_ocr_response_serialization(self):
        """Test OCRResponse can be serialized to dict/JSON."""
        response = OCRResponse(
            text="test",
            blocks=[],
            confidence=0.0,
            language="eng",
            image_size=ImageSize(width=100, height=100),
            error=None,
        )

        # Should be serializable to dict
        data = response.model_dump()
        assert data["text"] == "test"
        assert data["language"] == "eng"
        assert data["image_size"]["width"] == 100

    # Integration tests
    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    def test_analyze_grayscale_image_conversion(
        self, mock_tesseract, plugin, mock_pytesseract_data
    ):
        """Test grayscale images are converted to RGB."""
        # Create grayscale image
        img = Image.new("L", (100, 100), color=200)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = img_bytes.getvalue()

        mock_tesseract.image_to_string.return_value = "test"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        response = plugin.analyze(image_data)

        assert response.error is None
        assert response.text == "test"

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    def test_analyze_rgba_image_conversion(
        self, mock_tesseract, plugin, mock_pytesseract_data
    ):
        """Test RGBA images are converted to RGB."""
        # Create RGBA image
        img = Image.new("RGBA", (100, 100), color=(255, 255, 255, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = img_bytes.getvalue()

        mock_tesseract.image_to_string.return_value = "test"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        response = plugin.analyze(image_data)

        assert response.error is None
        assert response.text == "test"
