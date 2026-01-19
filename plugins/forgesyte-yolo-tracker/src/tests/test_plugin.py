"""Unit tests for YOLO tracker plugin.

Tests cover:
- Plugin initialization
- Lifecycle hooks (on_load, on_unload)
- Tool methods (placeholder/stub tests)
"""

import pytest
from forgesyte_yolo_tracker.plugin import Plugin, decode_image, encode_frame


class TestPlugin:
    """Test suite for the YOLO Tracker Plugin."""

    @pytest.fixture  # type: ignore[untyped-decorator]
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        return Plugin()

    # Initialization tests
    def test_plugin_initialization(self, plugin: Plugin) -> None:
        """Test plugin initializes without error."""
        assert plugin is not None
        assert isinstance(plugin, Plugin)

    # Lifecycle tests
    def test_on_load(self, plugin: Plugin) -> None:
        """Test on_load lifecycle hook."""
        plugin.on_load()

    def test_on_unload(self, plugin: Plugin) -> None:
        """Test on_unload lifecycle hook."""
        plugin.on_unload()

    # Tool method existence tests
    def test_has_yolo_player_detection(self, plugin: Plugin) -> None:
        """Test plugin has yolo_player_detection method."""
        assert hasattr(plugin, "yolo_player_detection")
        assert callable(plugin.yolo_player_detection)

    def test_has_yolo_player_tracking(self, plugin: Plugin) -> None:
        """Test plugin has yolo_player_tracking method."""
        assert hasattr(plugin, "yolo_player_tracking")
        assert callable(plugin.yolo_player_tracking)

    def test_has_yolo_ball_detection(self, plugin: Plugin) -> None:
        """Test plugin has yolo_ball_detection method."""
        assert hasattr(plugin, "yolo_ball_detection")
        assert callable(plugin.yolo_ball_detection)

    def test_has_yolo_team_classification(self, plugin: Plugin) -> None:
        """Test plugin has yolo_team_classification method."""
        assert hasattr(plugin, "yolo_team_classification")
        assert callable(plugin.yolo_team_classification)

    def test_has_yolo_pitch_detection(self, plugin: Plugin) -> None:
        """Test plugin has yolo_pitch_detection method."""
        assert hasattr(plugin, "yolo_pitch_detection")
        assert callable(plugin.yolo_pitch_detection)

    def test_has_yolo_radar(self, plugin: Plugin) -> None:
        """Test plugin has yolo_radar method."""
        assert hasattr(plugin, "yolo_radar")
        assert callable(plugin.yolo_radar)
