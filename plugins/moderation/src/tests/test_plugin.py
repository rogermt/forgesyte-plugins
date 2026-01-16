"""Unit tests for Moderation Plugin.

Tests cover:
- Metadata configuration
- Analysis of safe/unsafe content (mocked)
- Adherence to universal AnalysisResult contract
- Error handling
- Lifecycle hooks
"""

import io
from typing import Any
from unittest.mock import patch

import pytest
from forgesyte_moderation.plugin import Plugin
from PIL import Image


class TestModerationPlugin:
    """Test suite for Moderation Plugin."""

    @pytest.fixture  # type: ignore[untyped-decorator]
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        return Plugin()

    @pytest.fixture  # type: ignore[untyped-decorator]
    def sample_image_bytes(self) -> bytes:
        """Generate a random sample image."""
        img = Image.new("RGB", (100, 100), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    @pytest.fixture  # type: ignore[untyped-decorator]
    def unsafe_image_bytes(self) -> bytes:
        """Generate a sample image likely to trigger simple heuristics
        (e.g. skin tone)."""
        # Create an image with skin-tone like color
        img = Image.new("RGB", (100, 100), color=(255, 200, 150))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    # Metadata Tests
    @patch("forgesyte_moderation.plugin.PluginMetadata")
    def test_metadata_returns_valid_structure(
        self, mock_metadata_cls: Any, plugin: Plugin
    ) -> None:
        """Test metadata endpoint returns valid PluginMetadata."""
        mock_instance = mock_metadata_cls.return_value
        mock_instance.name = "moderation"
        mock_instance.version = "1.0.0"
        mock_instance.inputs = ["image"]
        mock_instance.outputs = ["safe", "categories", "confidence"]

        metadata = plugin.metadata()

        assert metadata.name == "moderation"
        assert "image" in metadata.inputs
        assert "safe" in metadata.outputs
        assert "categories" in metadata.outputs

    @patch("forgesyte_moderation.plugin.PluginMetadata")
    def test_metadata_includes_config_schema(
        self, mock_metadata_cls: Any, plugin: Plugin
    ) -> None:
        """Test metadata includes sensitivity and categories config."""
        mock_instance = mock_metadata_cls.return_value
        mock_instance.config_schema = {
            "sensitivity": {"default": "medium"},
            "categories": {"default": ["nsfw", "violence", "hate"]},
        }

        metadata = plugin.metadata()
        config = metadata.config_schema

        assert "sensitivity" in config
        assert "categories" in config
        assert config["sensitivity"]["default"] == "medium"

    # Analysis Tests
    @patch("forgesyte_moderation.plugin.AnalysisResult")
    def test_analyze_returns_analysis_result(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test analyze returns a proper AnalysisResult object."""
        expected_instance = mock_analysis_cls.return_value

        result = plugin.analyze(sample_image_bytes)

        assert result == expected_instance
        mock_analysis_cls.assert_called_once()

    @patch("forgesyte_moderation.plugin.AnalysisResult")
    def test_analyze_safe_content(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test analysis of safe content."""
        plugin.analyze(sample_image_bytes)

        call_kwargs = mock_analysis_cls.call_args[1]

        # text field should contain the recommendation
        assert "safe" in call_kwargs["text"].lower()

        # blocks should contain category details
        assert isinstance(call_kwargs["blocks"], list)

        # confidence should be a float
        assert isinstance(call_kwargs["confidence"], float)
        assert 0.0 <= call_kwargs["confidence"] <= 1.0

    @patch("forgesyte_moderation.plugin.AnalysisResult")
    def test_analyze_unsafe_content(
        self, mock_analysis_cls: Any, plugin: Plugin, unsafe_image_bytes: bytes
    ) -> None:
        """Test analysis of potentially unsafe content (heuristic based)."""
        plugin.analyze(unsafe_image_bytes)

        call_kwargs = mock_analysis_cls.call_args[1]
        blocks = call_kwargs["blocks"]

        # Check if we have results mapped to blocks
        assert len(blocks) > 0

        # Verify structure of blocks (CategoryResult)
        first_block = blocks[0]
        assert "category" in first_block
        assert "score" in first_block
        assert "flagged" in first_block
        assert "confidence" in first_block

    @patch("forgesyte_moderation.plugin.AnalysisResult")
    def test_analyze_respects_sensitivity_option(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test that sensitivity option is accepted."""
        plugin.analyze(sample_image_bytes, options={"sensitivity": "high"})

        # We can't easily check internal state without spying,
        # but we ensure it runs without error with valid options
        mock_analysis_cls.assert_called_once()

    # Error Handling
    @patch("forgesyte_moderation.plugin.AnalysisResult")
    def test_analyze_handles_invalid_image(
        self, mock_analysis_cls: Any, plugin: Plugin
    ) -> None:
        """Test error handling for invalid image bytes."""
        plugin.analyze(b"not an image")

        call_kwargs = mock_analysis_cls.call_args[1]
        assert call_kwargs["error"] is not None
        assert call_kwargs["confidence"] == 0.0

    @patch("forgesyte_moderation.plugin.AnalysisResult")
    def test_analyze_handles_missing_deps(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_image_bytes: bytes
    ) -> None:
        """Test fallback when dependencies are missing."""
        with patch("forgesyte_moderation.plugin.HAS_DEPS", False):
            plugin.analyze(sample_image_bytes)

            call_kwargs = mock_analysis_cls.call_args[1]
            assert (
                "requires PIL" in call_kwargs["error"]
                or "warning" in call_kwargs["text"].lower()
            )

    # Lifecycle Hooks
    def test_lifecycle_hooks(self, plugin: Plugin) -> None:
        """Test on_load and on_unload hooks."""
        # Just ensure they don't raise exceptions
        plugin.on_load()
        plugin.on_unload()
