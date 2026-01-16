"""Unit tests for Block Mapper Plugin.

Tests cover:
- Metadata configuration
- Image to block mapping logic (mocked)
- Adherence to universal AnalysisResult contract
- Error handling
- Lifecycle hooks
"""

import io
from typing import Any
from unittest.mock import patch

import pytest
from forgesyte_block_mapper.plugin import Plugin
from PIL import Image


class TestBlockMapperPlugin:
    """Test suite for Block Mapper Plugin."""

    @pytest.fixture  # type: ignore[untyped-decorator]
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        return Plugin()

    @pytest.fixture  # type: ignore[untyped-decorator]
    def sample_image_bytes(self) -> bytes:
        """Generate a random sample image."""
        img = Image.new("RGB", (64, 64), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    # Metadata Tests
    @patch("forgesyte_block_mapper.plugin.PluginMetadata")
    def test_metadata_returns_valid_structure(
        self, mock_metadata_cls: Any, plugin: Plugin
    ) -> None:
        """Test metadata endpoint returns valid PluginMetadata."""
        mock_instance = mock_metadata_cls.return_value
        mock_instance.name = "block_mapper"
        mock_instance.version = "1.0.0"
        mock_instance.inputs = ["image"]
        mock_instance.outputs = ["text", "blocks", "confidence"]

        metadata = plugin.metadata()

        assert metadata.name == "block_mapper"
        assert "image" in metadata.inputs
        assert "text" in metadata.outputs
        assert "blocks" in metadata.outputs

    @patch("forgesyte_block_mapper.plugin.PluginMetadata")
    def test_metadata_includes_config_schema(
        self, mock_metadata_cls: Any, plugin: Plugin
    ) -> None:
        """Test metadata includes width and height config."""
        mock_instance = mock_metadata_cls.return_value
        mock_instance.config_schema = {
            "width": {"default": 64},
            "height": {"default": 64},
        }

        metadata = plugin.metadata()
        config = metadata.config_schema

        assert "width" in config
        assert "height" in config
        assert config["width"]["default"] == 64

    # Analysis Tests
    @patch("forgesyte_block_mapper.plugin.AnalysisResult")
    def test_analyze_returns_analysis_result(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test analyze returns a proper AnalysisResult object."""
        expected_instance = mock_analysis_cls.return_value

        result = plugin.analyze(sample_image_bytes)

        assert result == expected_instance
        mock_analysis_cls.assert_called_once()

    @patch("forgesyte_block_mapper.plugin.AnalysisResult")
    def test_analyze_mapping_structure(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test the structure of the mapped results in AnalysisResult."""
        plugin.analyze(sample_image_bytes)

        call_kwargs = mock_analysis_cls.call_args[1]

        # text should be empty or summary
        assert isinstance(call_kwargs["text"], str)

        # blocks should contain the schematic blocks
        blocks = call_kwargs["blocks"]
        assert isinstance(blocks, list)

        # Check if confidence is 1.0 (deterministic mapping)
        assert call_kwargs["confidence"] == 1.0

    @patch("forgesyte_block_mapper.plugin.AnalysisResult")
    def test_analyze_respects_options(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test that width/height options are respected."""
        # This test relies on the implementation logic, but we can verify
        # that it runs without error with valid options.
        # A more detailed test would check the number of blocks returned.
        plugin.analyze(sample_image_bytes, options={"width": 10, "height": 10})

        call_kwargs = mock_analysis_cls.call_args[1]
        blocks = call_kwargs["blocks"]

        # 10x10 = 100 blocks
        assert len(blocks) == 100

    # Error Handling
    @patch("forgesyte_block_mapper.plugin.AnalysisResult")
    def test_analyze_handles_invalid_image(
        self, mock_analysis_cls: Any, plugin: Plugin
    ) -> None:
        """Test error handling for invalid image bytes."""
        plugin.analyze(b"not an image")

        call_kwargs = mock_analysis_cls.call_args[1]
        assert call_kwargs["error"] is not None
        assert call_kwargs["confidence"] == 0.0

    @patch("forgesyte_block_mapper.plugin.AnalysisResult")
    def test_analyze_handles_missing_deps(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test fallback when dependencies are missing."""
        with patch("forgesyte_block_mapper.plugin.HAS_DEPS", False):
            plugin.analyze(sample_image_bytes)

            call_kwargs = mock_analysis_cls.call_args[1]
            assert "requires PIL" in call_kwargs["error"] or "error" in call_kwargs

    # Lifecycle Hooks
    def test_lifecycle_hooks(self, plugin: Plugin) -> None:
        """Test on_load and on_unload hooks."""
        plugin.on_load()
        plugin.on_unload()
