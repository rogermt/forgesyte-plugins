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
    @patch('forgesyte_ocr.plugin.PluginMetadata')
    def test_metadata_returns_plugin_metadata(self, mock_metadata_cls, plugin):
        """Test metadata endpoint returns valid PluginMetadata."""
        mock_instance = mock_metadata_cls.return_value
        mock_instance.name = "ocr"
        mock_instance.version = "1.0.0"
        mock_instance.inputs = ["image"]
        mock_instance.outputs = ["text", "blocks", "confidence"]
        
        metadata = plugin.metadata()
        
        assert metadata.name == "ocr"
        assert metadata.version == "1.0.0"
        assert "image" in metadata.inputs
        assert "text" in metadata.outputs

    @patch('forgesyte_ocr.plugin.PluginMetadata')
    def test_metadata_includes_config_schema(self, mock_metadata_cls, plugin):
        """Test metadata includes language and PSM configuration."""
        mock_instance = mock_metadata_cls.return_value
        mock_instance.config_schema = {
            "language": {"default": "eng", "enum": ["eng"]},
            "psm": {"default": 3}
        }
        
        metadata = plugin.metadata()
        config = metadata.config_schema
        assert "language" in config
        assert "psm" in config
        assert config["language"]["default"] == "eng"
        assert config["psm"]["default"] == 3

    # Successful analysis tests
    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    @patch("forgesyte_ocr.plugin.AnalysisResult")
    def test_analyze_successful_ocr(
        self, mock_analysis_cls, mock_tesseract, plugin, sample_image_bytes, mock_pytesseract_data
    ):
        """Test successful OCR analysis returns valid AnalysisResult."""
        mock_tesseract.image_to_string.return_value = "hello world !"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        expected_instance = mock_analysis_cls.return_value
        expected_instance.text = "hello world !"
        expected_instance.language = "eng"
        expected_instance.error = None

        response = plugin.analyze(sample_image_bytes)

        assert response == expected_instance
        assert response.text == "hello world !"
        assert response.error is None
        assert response.language == "eng"

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    @patch("forgesyte_ocr.plugin.AnalysisResult")
    def test_analyze_with_custom_language(
        self, mock_analysis_cls, mock_tesseract, plugin, sample_image_bytes, mock_pytesseract_data
    ):
        """Test OCR with custom language option."""
        mock_tesseract.image_to_string.return_value = "Bonjour"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        expected_instance = mock_analysis_cls.return_value
        expected_instance.language = "fra"

        response = plugin.analyze(
            sample_image_bytes, options={"language": "fra", "psm": 6}
        )

        assert response.language == "fra"
        mock_tesseract.image_to_string.assert_called_once()
        call_args = mock_tesseract.image_to_string.call_args
        assert "lang=fra" in str(call_args) or call_args[1].get("lang") == "fra"

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    @patch("forgesyte_ocr.plugin.AnalysisResult")
    def test_analyze_filters_low_confidence_blocks(
        self, mock_analysis_cls, mock_tesseract, plugin, sample_image_bytes
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
        
        plugin.analyze(sample_image_bytes)
        
        call_kwargs = mock_analysis_cls.call_args[1]
        blocks = call_kwargs["blocks"]
        
        assert len(blocks) == 2
        assert blocks[0]["text"] == "hello"
        assert blocks[1]["text"] == "world"

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    @patch("forgesyte_ocr.plugin.AnalysisResult")
    def test_analyze_calculates_average_confidence(
        self, mock_analysis_cls, mock_tesseract, plugin, sample_image_bytes, mock_pytesseract_data
    ):
        """Test average confidence is calculated correctly."""
        mock_tesseract.image_to_string.return_value = "hello world !"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        plugin.analyze(sample_image_bytes)
        
        call_kwargs = mock_analysis_cls.call_args[1]
        confidence = call_kwargs["confidence"]
        
        expected_avg = (95 + 90 + 100) / 3 / 100.0
        assert confidence == pytest.approx(expected_avg)

    # Error handling tests
    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    @patch("forgesyte_ocr.plugin.AnalysisResult")
    def test_analyze_handles_invalid_image_data(
        self, mock_analysis_cls, mock_tesseract, plugin
    ):
        """Test error handling for invalid image bytes."""
        invalid_bytes = b"not an image"
        
        expected_instance = mock_analysis_cls.return_value
        expected_instance.error = "Invalid image"

        response = plugin.analyze(invalid_bytes)

        assert response == expected_instance
        assert response.error is not None
        
        call_kwargs = mock_analysis_cls.call_args[1]
        assert call_kwargs["error"] is not None

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    @patch("forgesyte_ocr.plugin.AnalysisResult")
    def test_analyze_handles_tesseract_exception(
        self, mock_analysis_cls, mock_tesseract, plugin, sample_image_bytes
    ):
        """Test error handling when pytesseract raises exception."""
        mock_tesseract.image_to_string.side_effect = Exception("Tesseract error")
        
        expected_instance = mock_analysis_cls.return_value
        expected_instance.error = "Tesseract error"

        response = plugin.analyze(sample_image_bytes)

        assert response == expected_instance
        assert "Tesseract error" in response.error

    # Fallback tests
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", False)
    @patch("forgesyte_ocr.plugin.AnalysisResult")
    def test_analyze_fallback_when_tesseract_unavailable(
        self, mock_analysis_cls, plugin, sample_image_bytes
    ):
        """Test fallback response when Tesseract is not installed."""
        expected_instance = mock_analysis_cls.return_value
        expected_instance.error = "Tesseract not installed"
        
        response = plugin.analyze(sample_image_bytes)

        assert response == expected_instance
        assert response.error is not None
        
        call_kwargs = mock_analysis_cls.call_args[1]
        assert "Tesseract not installed" in call_kwargs["error"]

    # Lifecycle tests
    def test_on_load_with_tesseract_available(self, plugin):
        """Test on_load lifecycle hook with Tesseract available."""
        with patch("forgesyte_ocr.plugin.HAS_TESSERACT", True):
            with patch("forgesyte_ocr.plugin.pytesseract") as mock_tesseract:
                mock_tesseract.get_tesseract_version.return_value = "5.0.0"
                plugin.on_load()

    def test_on_load_without_tesseract(self, plugin):
        """Test on_load when Tesseract is not available."""
        with patch("forgesyte_ocr.plugin.HAS_TESSERACT", False):
            plugin.on_load()

    def test_on_unload(self, plugin):
        """Test on_unload lifecycle hook."""
        plugin.on_unload()

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
        """Test OCRResponse Pydantic model validates."""
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
        """Test OCRResponse can be serialized."""
        response = OCRResponse(
            text="test",
            blocks=[],
            confidence=0.0,
            language="eng",
            image_size=ImageSize(width=100, height=100),
            error=None,
        )

        data = response.model_dump()
        assert data["text"] == "test"
        assert data["language"] == "eng"
        assert data["image_size"]["width"] == 100

    # Integration tests
    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    @patch("forgesyte_ocr.plugin.AnalysisResult")
    def test_analyze_grayscale_image_conversion(
        self, mock_analysis_cls, mock_tesseract, plugin, mock_pytesseract_data
    ):
        """Test grayscale images are converted to RGB."""
        img = Image.new("L", (100, 100), color=200)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = img_bytes.getvalue()

        mock_tesseract.image_to_string.return_value = "test"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data
        
        expected_instance = mock_analysis_cls.return_value
        expected_instance.error = None

        response = plugin.analyze(image_data)

        assert response.error is None

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    @patch("forgesyte_ocr.plugin.AnalysisResult")
    def test_analyze_rgba_image_conversion(
        self, mock_analysis_cls, mock_tesseract, plugin, mock_pytesseract_data
    ):
        """Test RGBA images are converted to RGB."""
        img = Image.new("RGBA", (100, 100), color=(255, 255, 255, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = img_bytes.getvalue()

        mock_tesseract.image_to_string.return_value = "test"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data

        expected_instance = mock_analysis_cls.return_value
        expected_instance.error = None

        response = plugin.analyze(image_data)

        assert response.error is None

    @patch("forgesyte_ocr.plugin.pytesseract")
    @patch("forgesyte_ocr.plugin.HAS_TESSERACT", True)
    @patch("forgesyte_ocr.plugin.AnalysisResult")
    def test_analyze_returns_pydantic_model_not_dict(
        self, mock_analysis_cls, mock_tesseract, plugin, sample_image_bytes, mock_pytesseract_data
    ):
        """Test that analyze() returns Pydantic model (AnalysisResult)."""
        mock_tesseract.image_to_string.return_value = "hello world"
        mock_tesseract.image_to_data.return_value = mock_pytesseract_data
        mock_tesseract.Output.DICT = mock_pytesseract_data
        
        expected_instance = mock_analysis_cls.return_value

        result = plugin.analyze(sample_image_bytes)

        # Must be Pydantic model, not dict
        assert result == expected_instance
        assert not isinstance(result, dict)
