"""Unit tests for YOLO tracker plugin.

Tests cover:
- Plugin metadata endpoint
- Placeholder analyze method (not yet implemented)
- Lifecycle hooks
- Tool method availability
- Error handling
"""

import io
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from forgesyte_yolo_tracker.plugin import Plugin


class TestPlugin:
    """Test suite for the YOLO Tracker Plugin."""

    @pytest.fixture  # type: ignore[untyped-decorator]
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        return Plugin()

    @pytest.fixture  # type: ignore[untyped-decorator]
    def sample_image_bytes(self) -> bytes:
        """Generate sample image bytes for testing."""
        img = Image.new("RGB", (640, 480), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    # Metadata tests
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
        assert metadata["name"] == "forgesyte-yolo-tracker"

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

    # Analyze method tests (placeholder implementation)
    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_returns_dict(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test analyze returns analysis result dictionary."""
        result = plugin.analyze(sample_image_bytes)
        assert isinstance(result, dict)

    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_returns_analysis_result_compatible_dict(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test analyze returns AnalysisResult compatible dictionary with required fields."""
        result = plugin.analyze(sample_image_bytes)
        
        required_fields = ["text", "blocks", "confidence", "language", "error"]
        for field in required_fields:
            assert field in result, f"Missing AnalysisResult field: {field}"

    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_with_config_options(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test analyze accepts config options like other plugins."""
        result = plugin.analyze(
            sample_image_bytes,
            confidence_threshold=0.7,
            max_detections=50
        )
        assert isinstance(result, dict)
        assert "error" in result

    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_handles_invalid_image_gracefully(
        self, mock_analysis_cls: Any, plugin: Plugin
    ) -> None:
        """Test analyze handles invalid image data without crashing."""
        result = plugin.analyze(b"invalid image data")
        
        # Should return error field, not crash
        assert isinstance(result, dict)
        assert "error" in result
        assert result["error"] is not None

    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_returns_placeholder_result(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test analyze returns placeholder result (implementation in progress)."""
        result = plugin.analyze(sample_image_bytes)
        
        # Current placeholder behavior
        assert result["text"] == "YOLO tracker analysis not yet implemented"
        assert result["error"] is not None
        assert "under development" in result["error"]

    # Lifecycle tests
    def test_on_load_does_not_crash(self, plugin: Plugin) -> None:
        """Test on_load lifecycle hook executes without error."""
        plugin.on_load()  # Should not raise

    def test_on_unload_does_not_crash(self, plugin: Plugin) -> None:
        """Test on_unload lifecycle hook executes without error."""
        plugin.on_unload()  # Should not raise

    def test_plugin_lifecycle_sequence(
        self, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test complete plugin lifecycle: load -> analyze -> unload."""
        plugin.on_load()
        result = plugin.analyze(sample_image_bytes)
        plugin.on_unload()
        
        assert isinstance(result, dict)

    # Tool method tests
    def test_has_all_required_tool_methods(self, plugin: Plugin) -> None:
        """Test plugin has all required tool methods from manifest."""
        tools = [
            "yolo_player_detection",
            "yolo_player_tracking",
            "yolo_ball_detection",
            "yolo_team_classification",
            "yolo_pitch_detection",
            "yolo_radar",
        ]
        
        for tool in tools:
            assert hasattr(plugin, tool), f"Missing tool method: {tool}"
            method = getattr(plugin, tool)
            assert callable(method), f"Tool not callable: {tool}"

    def test_tool_methods_accept_image_and_config(
        self, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test tool methods accept image bytes and optional config."""
        # Just verify they're callable with expected signature
        assert callable(plugin.yolo_player_detection)
        assert callable(plugin.yolo_player_tracking)
        
        # Methods should exist and have proper signature
        # (actual calls would need real inference functions)

    # Integration-like tests
    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_with_grayscale_image(
        self, mock_analysis_cls: Any, plugin: Plugin
    ) -> None:
        """Test analyze can handle grayscale images."""
        img = Image.new("L", (640, 480), color=128)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        
        result = plugin.analyze(img_bytes.getvalue())
        assert isinstance(result, dict)

    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_with_rgba_image(
        self, mock_analysis_cls: Any, plugin: Plugin
    ) -> None:
        """Test analyze can handle RGBA images."""
        img = Image.new("RGBA", (640, 480), color=(255, 255, 255, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        
        result = plugin.analyze(img_bytes.getvalue())
        assert isinstance(result, dict)

    # Documentation tests
    def test_metadata_has_meaningful_description(self, plugin: Plugin) -> None:
        """Test metadata includes descriptive text."""
        metadata = plugin.metadata()
        description = metadata.get("description", "")
        assert len(description) > 0
        assert "YOLO" in description or "football" in description.lower()

    def test_plugin_docstring_exists(self, plugin: Plugin) -> None:
        """Test plugin class has documentation."""
        assert Plugin.__doc__ is not None
        assert len(Plugin.__doc__) > 0
