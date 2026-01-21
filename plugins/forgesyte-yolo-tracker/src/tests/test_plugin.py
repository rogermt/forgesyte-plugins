"""Unit tests for YOLO tracker plugin.

Tests cover:
- Plugin metadata endpoint
- analyze method (player detection)
- Lifecycle hooks (on_load/on_unload)
- Tool method availability (module-level functions)
- Error handling
"""

import io
from unittest.mock import patch, MagicMock

import pytest
from PIL import Image

from forgesyte_yolo_tracker.plugin import Plugin


def make_analysis_result(
    text="", blocks=None, confidence=1.0, language=None, error=None, extra=None
):
    """Factory for AnalysisResult dict."""
    if blocks is None:
        blocks = []
    return {
        "text": text,
        "blocks": blocks,
        "confidence": confidence,
        "language": language,
        "error": error,
        "extra": extra,
    }


class TestPluginMetadata:
    """Test suite for plugin metadata."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        return Plugin()

    def test_metadata_returns_dict(self, plugin: Plugin) -> None:
        """Test metadata endpoint returns a dictionary."""
        metadata = plugin.metadata()
        assert isinstance(metadata, dict)

    def test_metadata_has_required_fields(self, plugin: Plugin) -> None:
        """Test metadata includes required fields."""
        metadata = plugin.metadata()
        required_fields = ["name", "version", "description", "config_schema"]
        for field in required_fields:
            assert field in metadata, f"Missing required field: {field}"

    def test_metadata_name_is_yolo_tracker(self, plugin: Plugin) -> None:
        """Test metadata name is correct."""
        metadata = plugin.metadata()
        assert metadata["name"] == "yolo-tracker"

    def test_metadata_version_is_semantic(self, plugin: Plugin) -> None:
        """Test metadata version follows semantic versioning."""
        metadata = plugin.metadata()
        version = metadata["version"]
        parts = version.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_metadata_config_schema_includes_parameters(self, plugin: Plugin) -> None:
        """Test config schema includes expected parameters."""
        metadata = plugin.metadata()
        config = metadata["config_schema"]
        assert isinstance(config, dict)
        assert len(config) > 0

    def test_metadata_has_meaningful_description(self, plugin: Plugin) -> None:
        """Test metadata includes descriptive text."""
        metadata = plugin.metadata()
        description = metadata.get("description", "")
        assert len(description) > 0
        assert "YOLO" in description or "football" in description.lower()


class TestPluginAnalyze:
    """Test suite for plugin analyze method."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        return Plugin()

    @pytest.fixture
    def sample_image_bytes(self) -> bytes:
        """Generate sample image bytes for testing."""
        img = Image.new("RGB", (640, 480), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    @patch("forgesyte_yolo_tracker.plugin.detect_players_json")
    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_returns_analysis_result(
        self,
        mock_result: MagicMock,
        mock_detect: MagicMock,
        plugin: Plugin,
        sample_image_bytes: bytes,
    ) -> None:
        """Test analyze returns AnalysisResult object."""
        mock_detect.return_value = {"detections": [], "count": 0, "classes": {}}
        mock_result.side_effect = make_analysis_result
        result = plugin.analyze(sample_image_bytes)
        assert isinstance(result, dict)

    @patch("forgesyte_yolo_tracker.plugin.detect_players_json")
    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_returns_analysis_result_compatible_dict(
        self,
        mock_result: MagicMock,
        mock_detect: MagicMock,
        plugin: Plugin,
        sample_image_bytes: bytes,
    ) -> None:
        """Test analyze returns AnalysisResult compatible dictionary with required fields."""
        mock_detect.return_value = {"detections": [], "count": 0, "classes": {}}
        mock_result.side_effect = make_analysis_result
        result = plugin.analyze(sample_image_bytes)

        required_fields = ["text", "blocks", "confidence", "language", "error", "extra"]
        for field in required_fields:
            assert field in result, f"Missing AnalysisResult field: {field}"

    @patch("forgesyte_yolo_tracker.plugin.detect_players_json")
    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_with_options(
        self,
        mock_result: MagicMock,
        mock_detect: MagicMock,
        plugin: Plugin,
        sample_image_bytes: bytes,
    ) -> None:
        """Test analyze accepts options parameter."""
        mock_detect.return_value = {"detections": [], "count": 0, "classes": {}}
        mock_result.side_effect = make_analysis_result
        result = plugin.analyze(
            sample_image_bytes, {"confidence_threshold": 0.7, "max_detections": 50}
        )
        assert isinstance(result, dict)

    @patch("forgesyte_yolo_tracker.plugin.detect_players_json")
    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_handles_invalid_image_gracefully(
        self, mock_result: MagicMock, mock_detect: MagicMock, plugin: Plugin
    ) -> None:
        """Test analyze handles invalid image data without crashing."""
        mock_detect.return_value = {"detections": [], "count": 0, "classes": {}}
        mock_result.side_effect = make_analysis_result
        result = plugin.analyze(b"invalid image data")

        assert isinstance(result, dict)
        assert "error" in result

    @patch("forgesyte_yolo_tracker.plugin.detect_players_json")
    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_with_grayscale_image(
        self, mock_result: MagicMock, mock_detect: MagicMock, plugin: Plugin
    ) -> None:
        """Test analyze can handle grayscale images."""
        mock_detect.return_value = {"detections": [], "count": 0, "classes": {}}
        mock_result.side_effect = make_analysis_result
        img = Image.new("L", (640, 480), color=128)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")

        result = plugin.analyze(img_bytes.getvalue())
        assert isinstance(result, dict)

    @patch("forgesyte_yolo_tracker.plugin.detect_players_json")
    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_with_rgba_image(
        self, mock_result: MagicMock, mock_detect: MagicMock, plugin: Plugin
    ) -> None:
        """Test analyze can handle RGBA images."""
        mock_detect.return_value = {"detections": [], "count": 0, "classes": {}}
        mock_result.side_effect = make_analysis_result
        img = Image.new("RGBA", (640, 480), color=(255, 255, 255, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")

        result = plugin.analyze(img_bytes.getvalue())
        assert isinstance(result, dict)


class TestPluginLifecycle:
    """Test suite for plugin lifecycle hooks."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        return Plugin()

    def test_on_load_does_not_crash(self, plugin: Plugin) -> None:
        """Test on_load lifecycle hook executes without error."""
        plugin.on_load()

    def test_on_unload_does_not_crash(self, plugin: Plugin) -> None:
        """Test on_unload lifecycle hook executes without error."""
        plugin.on_unload()

    @patch("forgesyte_yolo_tracker.plugin.detect_players_json")
    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_plugin_lifecycle_sequence(
        self, mock_result: MagicMock, mock_detect: MagicMock, plugin: Plugin
    ) -> None:
        """Test complete plugin lifecycle: load -> analyze -> unload."""
        mock_detect.return_value = {"detections": [], "count": 0, "classes": {}}
        mock_result.side_effect = make_analysis_result

        img = Image.new("RGB", (640, 480), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        sample_image_bytes = img_bytes.getvalue()

        plugin.on_load()
        result = plugin.analyze(sample_image_bytes)
        plugin.on_unload()

        assert isinstance(result, dict)


class TestPluginDocstring:
    """Test suite for plugin documentation."""

    def test_plugin_class_has_docstring(self) -> None:
        """Test Plugin class has documentation."""
        assert Plugin.__doc__ is not None
        assert len(Plugin.__doc__) > 0
