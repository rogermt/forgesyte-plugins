"""Tests for plugin loading and metadata."""

import json
from pathlib import Path

import pytest

from forgesyte_yolo_tracker.plugin import YOLOTrackerPlugin, AnalysisMode


class TestPluginLoading:
    """Test plugin loading and initialization."""

    def test_plugin_instantiation(self) -> None:
        """Test that plugin can be instantiated."""
        plugin = YOLOTrackerPlugin()
        assert plugin is not None
        assert plugin.name == "YOLO Tracker"

    def test_get_metadata(self) -> None:
        """Test that plugin returns valid metadata."""
        plugin = YOLOTrackerPlugin()
        metadata = plugin.get_metadata()

        assert "name" in metadata
        assert "version" in metadata
        assert "description" in metadata
        assert "capabilities" in metadata
        assert isinstance(metadata["capabilities"], list)

    def test_plugin_version(self) -> None:
        """Test plugin version."""
        plugin = YOLOTrackerPlugin()
        assert plugin.version == "0.1.0"

    def test_plugin_capabilities(self) -> None:
        """Test that plugin reports all capabilities."""
        plugin = YOLOTrackerPlugin()
        expected_capabilities = [
            "player_detection",
            "player_tracking",
            "ball_detection",
            "team_classification",
            "pitch_detection",
            "radar_visualization",
        ]
        assert plugin.capabilities == expected_capabilities


class TestManifest:
    """Test manifest file validity."""

    def test_manifest_exists(self) -> None:
        """Test that manifest.json exists."""
        manifest_path = (
            Path(__file__).parent.parent
            / "src"
            / "forgesyte_yolo_tracker"
            / "manifest.json"
        )
        assert manifest_path.exists()

    def test_manifest_valid_json(self) -> None:
        """Test that manifest is valid JSON."""
        manifest_path = (
            Path(__file__).parent.parent
            / "src"
            / "forgesyte_yolo_tracker"
            / "manifest.json"
        )
        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "id" in manifest
        assert "name" in manifest
        assert "version" in manifest


class TestAnalysisModes:
    """Test analysis mode enumeration."""

    def test_all_modes_available(self) -> None:
        """Test that all analysis modes are defined."""
        modes = [mode for mode in AnalysisMode]
        assert len(modes) == 6
        assert AnalysisMode.PITCH_DETECTION in modes
        assert AnalysisMode.PLAYER_DETECTION in modes
        assert AnalysisMode.BALL_DETECTION in modes
        assert AnalysisMode.PLAYER_TRACKING in modes
        assert AnalysisMode.TEAM_CLASSIFICATION in modes
        assert AnalysisMode.RADAR in modes

    def test_mode_values(self) -> None:
        """Test that mode values are properly formatted."""
        assert AnalysisMode.PITCH_DETECTION.value == "pitch_detection"
        assert AnalysisMode.PLAYER_DETECTION.value == "player_detection"
        assert AnalysisMode.BALL_DETECTION.value == "ball_detection"
        assert AnalysisMode.PLAYER_TRACKING.value == "player_tracking"
        assert AnalysisMode.TEAM_CLASSIFICATION.value == "team_classification"
        assert AnalysisMode.RADAR.value == "radar"

    def test_supports_mode(self) -> None:
        """Test mode support checking."""
        plugin = YOLOTrackerPlugin()
        assert plugin.supports_mode("player_detection")
        assert plugin.supports_mode("ball_detection")
        assert plugin.supports_mode("pitch_detection")
        assert not plugin.supports_mode("invalid_mode")
