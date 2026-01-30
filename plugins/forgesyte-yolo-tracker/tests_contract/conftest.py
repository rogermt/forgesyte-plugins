"""Test configuration for template plugin tests."""

import sys
from typing import Any, Dict
from unittest.mock import MagicMock

# Mock the app module and its models to allow testing without the full app
mock_app = MagicMock()
sys.modules["app"] = mock_app

mock_models = MagicMock()
sys.modules["app.models"] = mock_models


def make_plugin_metadata(
    name: str = "yolo-tracker",
    description: str = "YOLO-based football analysis plugin",
    version: str = "0.1.0",
    inputs=None,
    outputs=None,
    config_schema=None,
) -> Dict[str, Any]:
    """Factory for PluginMetadata dict."""
    if inputs is None:
        inputs = ["image"]
    if outputs is None:
        outputs = ["json"]
    if config_schema is None:
        config_schema = {
            "device": {"type": "string", "default": "cpu"},
            "annotated": {"type": "boolean", "default": False},
            "confidence": {"type": "number", "default": 0.25},
        }
    return {
        "name": name,
        "description": description,
        "version": version,
        "inputs": inputs,
        "outputs": outputs,
        "config_schema": config_schema,
    }


def make_analysis_result(
    text: str = "",
    blocks=None,
    confidence: float = 1.0,
    language=None,
    error=None,
    extra=None,
) -> Dict[str, Any]:
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


# Configure PluginMetadata mock to return a proper dict
mock_models.PluginMetadata = make_plugin_metadata

# Configure AnalysisResult mock to return a proper dict
mock_models.AnalysisResult = make_analysis_result
