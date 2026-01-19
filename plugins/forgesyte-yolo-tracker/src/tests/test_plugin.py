"""Unit tests for YOLO tracker plugin.

Tests cover:
- Plugin metadata endpoint
- Plugin analyze method
- Lifecycle hooks
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from forgesyte_yolo_tracker.plugin import Plugin


class TestPlugin:
    """Test suite for the YOLO Tracker Plugin."""

    @pytest.fixture  # type: ignore[untyped-decorator]
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        return Plugin()

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

    def test_metadata_name_correct(self, plugin: Plugin) -> None:
        """Test metadata name is correct."""
        metadata = plugin.metadata()
        assert metadata["name"] == "forgesyte-yolo-tracker"

    def test_metadata_version_correct(self, plugin: Plugin) -> None:
        """Test metadata version is correct."""
        metadata = plugin.metadata()
        assert metadata["version"] == "0.1.0"

    def test_metadata_config_schema_exists(self, plugin: Plugin) -> None:
        """Test metadata includes config schema."""
        metadata = plugin.metadata()
        assert isinstance(metadata["config_schema"], dict)
        assert len(metadata["config_schema"]) > 0

    # Analysis tests
    def test_analyze_accepts_bytes(self, plugin: Plugin) -> None:
        """Test analyze method accepts image bytes."""
        result = plugin.analyze(b"fake image data")
        assert isinstance(result, dict)

    def test_analyze_returns_analysis_result_dict(self, plugin: Plugin) -> None:
        """Test analyze returns AnalysisResult compatible dictionary."""
        result = plugin.analyze(b"fake image data")
        
        # Should have AnalysisResult fields
        required_fields = ["text", "blocks", "confidence", "language", "error"]
        for field in required_fields:
            assert field in result, f"Missing AnalysisResult field: {field}"

    def test_analyze_handles_invalid_image(self, plugin: Plugin) -> None:
        """Test analyze handles invalid image data gracefully."""
        result = plugin.analyze(b"invalid image data that cannot be decoded")
        
        # Should return error, not crash
        assert isinstance(result, dict)
        assert "error" in result

    def test_analyze_with_kwargs(self, plugin: Plugin) -> None:
        """Test analyze accepts optional kwargs."""
        result = plugin.analyze(
            b"fake image data",
            confidence_threshold=0.7,
            max_detections=50
        )
        assert isinstance(result, dict)

    @patch("forgesyte_yolo_tracker.plugin.AnalysisResult")
    def test_analyze_could_use_analysis_result_class(
        self, mock_analysis_cls: Any, plugin: Plugin
    ) -> None:
        """Test that analyze could instantiate AnalysisResult when available.
        
        This test documents that when AnalysisResult is available from app.models,
        the plugin should use it to create properly typed results.
        """
        # In production, AnalysisResult would be the real class
        # For now, it's mocked and plugin returns dict
        result = plugin.analyze(b"test image")
        
        # Plugin currently returns dict-compatible format
        assert isinstance(result, dict)
        assert "error" in result

    # Lifecycle tests
    def test_on_load(self, plugin: Plugin) -> None:
        """Test on_load lifecycle hook executes."""
        # Should not raise exception
        plugin.on_load()

    def test_on_unload(self, plugin: Plugin) -> None:
        """Test on_unload lifecycle hook executes."""
        # Should not raise exception
        plugin.on_unload()

    def test_lifecycle_sequence(self, plugin: Plugin) -> None:
        """Test plugin lifecycle (load -> analyze -> unload)."""
        # Should support full lifecycle
        plugin.on_load()
        result = plugin.analyze(b"test image")
        plugin.on_unload()
        
        assert isinstance(result, dict)

    # Tool method tests
    def test_has_all_tool_methods(self, plugin: Plugin) -> None:
        """Test plugin has all required tool methods."""
        tools = [
            "yolo_player_detection",
            "yolo_player_tracking",
            "yolo_ball_detection",
            "yolo_team_classification",
            "yolo_pitch_detection",
            "yolo_radar",
        ]
        
        for tool in tools:
            assert hasattr(plugin, tool), f"Missing tool: {tool}"
            assert callable(getattr(plugin, tool)), f"Tool not callable: {tool}"
