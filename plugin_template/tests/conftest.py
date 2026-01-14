"""Test configuration for template plugin tests."""

import sys
from unittest.mock import MagicMock

# Mock the app module and its models to allow testing without the full app
mock_app = MagicMock()
sys.modules["app"] = mock_app

mock_models = MagicMock()
sys.modules["app.models"] = mock_models

# Mock the model classes
mock_models.PluginMetadata = MagicMock
mock_models.AnalysisResult = MagicMock

# Configure AnalysisResult mock instance
mock_analysis_instance = MagicMock()
mock_analysis_instance.model_dump.return_value = {
    "text": "",
    "blocks": [],
    "confidence": 0.0,
    "language": None,
    "error": "Template plugin has no implementation.",
}
mock_models.AnalysisResult.return_value = mock_analysis_instance