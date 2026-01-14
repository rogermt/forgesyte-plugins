"""Unit tests for template plugin.

Tests cover:
- Metadata endpoint
- Template analyze method (error handling)
- Lifecycle hooks
"""

from unittest.mock import patch

import pytest

from forgesyte_plugin_template.plugin import Plugin


class TestTemplatePlugin:
    """Test suite for Template Plugin."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance for testing."""
        return Plugin()

    @patch('forgesyte_plugin_template.plugin.PluginMetadata')
    def test_metadata_returns_plugin_metadata(self, mock_metadata_cls, plugin):
        """Test metadata endpoint returns valid PluginMetadata."""
        mock_instance = mock_metadata_cls.return_value
        mock_instance.name = "template_plugin"
        mock_instance.version = "1.0.0"
        mock_instance.inputs = ["image"]
        mock_instance.outputs = ["json"]
        
        metadata = plugin.metadata()
        
        assert metadata.name == "template_plugin"
        assert metadata.version == "1.0.0"
        assert "image" in metadata.inputs
        assert "json" in metadata.outputs

    @patch('forgesyte_plugin_template.plugin.PluginMetadata')
    def test_metadata_includes_config_schema(self, mock_metadata_cls, plugin):
        """Test metadata includes configuration schema."""
        mock_instance = mock_metadata_cls.return_value
        mock_instance.config_schema = {
            "mode": {
                "type": "string",
                "default": "default",
                "enum": ["default"],
                "description": "Processing mode for this plugin",
            }
        }
        
        metadata = plugin.metadata()
        config = metadata.config_schema
        assert "mode" in config
        assert config["mode"]["default"] == "default"
        assert "default" in config["mode"]["enum"]

    @patch('forgesyte_plugin_template.plugin.AnalysisResult')
    def test_analyze_returns_template_error(self, mock_analysis_cls, plugin):
        """Test analyze returns template error response."""
        expected_instance = mock_analysis_cls.return_value
        expected_instance.error = "Template plugin has no implementation."
        expected_instance.text = ""
        expected_instance.blocks = []
        expected_instance.confidence = 0.0
        expected_instance.language = None

        result = plugin.analyze(b"dummy image bytes")

        assert result == expected_instance
        assert result.error == "Template plugin has no implementation."
        assert result.text == ""

    @patch('forgesyte_plugin_template.plugin.AnalysisResult')
    def test_analyze_with_options(self, mock_analysis_cls, plugin):
        """Test analyze handles options parameter."""
        expected_instance = mock_analysis_cls.return_value
        expected_instance.error = "Template plugin has no implementation."

        result = plugin.analyze(b"dummy", options={"mode": "default"})

        assert result == expected_instance
        assert result.error == "Template plugin has no implementation."

    def test_on_load(self, plugin):
        """Test on_load lifecycle hook."""
        plugin.on_load()
        # Should not raise exception

    def test_on_unload(self, plugin):
        """Test on_unload lifecycle hook."""
        plugin.on_unload()
        # Should not raise exception
