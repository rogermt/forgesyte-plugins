"""Unit tests for YOLO tracker plugin — BasePlugin Architecture.

MIGRATION NOTE (Issue #139):
============================

This test file was refactored from legacy plugin interface to BasePlugin architecture.

Legacy interface (removed):
- plugin.metadata() → plugin.tools dict (BasePlugin contract)
- plugin.analyze() → plugin.run_tool("tool_name", args={}) (BasePlugin contract)

New tests follow the OCR plugin pattern:
- Use plugin.run_tool() for tool execution
- Use plugin.tools dict for manifest data
- Validate JSON-safe output (no numpy, tensors, custom objects)
- Test lifecycle hooks (on_load, on_unload)

Tests maintain behavior coverage:
- Image format support (RGB, grayscale, RGBA)
- Tool parameter routing
- Error handling (invalid images, missing args)
- Tool availability and schema validation
"""

import io
from unittest.mock import patch

import pytest
from PIL import Image

from forgesyte_yolo_tracker.plugin import Plugin


class TestPluginTools:
    """Test BasePlugin tools dict contract."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        p = Plugin()
        p.on_load()  # Initialize on load
        return p

    def test_plugin_has_tools_dict(self, plugin: Plugin) -> None:
        """Test plugin has tools dict (BasePlugin contract)."""
        assert hasattr(plugin, "tools")
        assert isinstance(plugin.tools, dict)

    def test_plugin_has_player_detection_tool(self, plugin: Plugin) -> None:
        """Test player_detection tool is registered."""
        assert "player_detection" in plugin.tools
        tool = plugin.tools["player_detection"]
        assert "handler" in tool
        assert "input_schema" in tool
        assert "output_schema" in tool

    def test_plugin_has_ball_detection_tool(self, plugin: Plugin) -> None:
        """Test ball_detection tool is registered."""
        assert "ball_detection" in plugin.tools
        assert "handler" in plugin.tools["ball_detection"]

    def test_plugin_has_pitch_detection_tool(self, plugin: Plugin) -> None:
        """Test pitch_detection tool is registered."""
        assert "pitch_detection" in plugin.tools
        assert "handler" in plugin.tools["pitch_detection"]

    def test_plugin_has_player_tracking_tool(self, plugin: Plugin) -> None:
        """Test player_tracking tool is registered."""
        assert "player_tracking" in plugin.tools
        assert "handler" in plugin.tools["player_tracking"]

    def test_plugin_has_radar_tool(self, plugin: Plugin) -> None:
        """Test radar tool is registered."""
        assert "radar" in plugin.tools
        assert "handler" in plugin.tools["radar"]

    def test_tool_handler_is_callable(self, plugin: Plugin) -> None:
        """Test tool handlers are callables (BasePlugin contract)."""
        for tool_name, tool_config in plugin.tools.items():
            handler = tool_config.get("handler")
            assert callable(
                handler
            ), f"Tool '{tool_name}' handler must be callable, got {type(handler)}"


class TestPluginRunTool:
    """Test plugin.run_tool() method (BasePlugin contract)."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        p = Plugin()
        p.on_load()
        return p

    @pytest.fixture
    def sample_frame_base64(self) -> str:
        """Generate sample frame as base64 for testing."""
        import base64

        img = Image.new("RGB", (640, 480), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    def test_run_tool_player_detection_routing(
        self, plugin: Plugin, sample_frame_base64: str
    ) -> None:
        """Test run_tool routes to player_detection handler."""
        with patch(
            "forgesyte_yolo_tracker.plugin.detect_players_json",
            return_value={"detections": []},
        ):
            result = plugin.run_tool(
                "player_detection",
                args={
                    "frame_base64": sample_frame_base64,
                    "device": "cpu",
                },
            )

            assert isinstance(result, dict)

    def test_run_tool_invalid_tool_name_raises_error(
        self, plugin: Plugin, sample_frame_base64: str
    ) -> None:
        """Test run_tool raises ValueError for unknown tool."""
        with pytest.raises(ValueError, match="Unknown tool"):
            plugin.run_tool(
                "nonexistent_tool",
                args={"frame_base64": sample_frame_base64},
            )

    def test_run_tool_validates_args(self, plugin: Plugin) -> None:
        """Test run_tool handles missing frame_base64 (returns error dict)."""
        result = plugin.run_tool(
            "player_detection",
            args={},  # Missing frame_base64
        )
        # Tool returns error dict instead of raising
        assert isinstance(result, dict)
        assert "error" in result


class TestPluginAnalyzeBehavior:
    """Test analyze behavior using run_tool interface.

    These tests validate the behavior expected from analyze() but executed
    through run_tool(), which is the BasePlugin contract.
    """

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        p = Plugin()
        p.on_load()
        return p

    @pytest.fixture
    def sample_frame_base64(self) -> str:
        """Generate sample frame as base64."""
        import base64

        img = Image.new("RGB", (640, 480), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    @pytest.fixture
    def grayscale_frame_base64(self) -> str:
        """Generate grayscale frame as base64."""
        import base64

        img = Image.new("L", (640, 480), color=128)  # Grayscale
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    @pytest.fixture
    def rgba_frame_base64(self) -> str:
        """Generate RGBA frame as base64."""
        import base64

        img = Image.new("RGBA", (640, 480), color=(255, 255, 255, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    def test_run_tool_handles_rgb_image(self, plugin: Plugin, sample_frame_base64: str) -> None:
        """Test tool handles RGB images correctly."""
        with patch(
            "forgesyte_yolo_tracker.plugin.detect_players_json",
            return_value={"detections": []},
        ):
            result = plugin.run_tool(
                "player_detection",
                args={
                    "frame_base64": sample_frame_base64,
                    "device": "cpu",
                },
            )
            assert isinstance(result, dict)

    def test_run_tool_handles_grayscale_image(
        self, plugin: Plugin, grayscale_frame_base64: str
    ) -> None:
        """Test tool handles grayscale images gracefully."""
        with patch(
            "forgesyte_yolo_tracker.plugin.detect_players_json",
            return_value={"detections": []},
        ):
            # Should convert grayscale to RGB or handle appropriately
            result = plugin.run_tool(
                "player_detection",
                args={
                    "frame_base64": grayscale_frame_base64,
                    "device": "cpu",
                },
            )
            assert isinstance(result, dict)

    def test_run_tool_handles_rgba_image(self, plugin: Plugin, rgba_frame_base64: str) -> None:
        """Test tool handles RGBA images (with alpha channel)."""
        with patch(
            "forgesyte_yolo_tracker.plugin.detect_players_json",
            return_value={"detections": []},
        ):
            # Should convert RGBA to RGB or handle appropriately
            result = plugin.run_tool(
                "player_detection",
                args={
                    "frame_base64": rgba_frame_base64,
                    "device": "cpu",
                },
            )
            assert isinstance(result, dict)

    def test_run_tool_handles_invalid_image_gracefully(self, plugin: Plugin) -> None:
        """Test tool handles invalid image bytes gracefully (returns error dict)."""
        result = plugin.run_tool(
            "player_detection",
            args={
                "frame_base64": "invalid_base64!!!",
                "device": "cpu",
            },
        )
        # Tool returns error dict instead of raising
        assert isinstance(result, dict)
        assert "error" in result

    def test_run_tool_respects_options(self, plugin: Plugin, sample_frame_base64: str) -> None:
        """Test tool respects options parameter."""
        with patch(
            "forgesyte_yolo_tracker.plugin.detect_players_json_with_annotated_frame",
            return_value={"detections": []},
        ):
            result = plugin.run_tool(
                "player_detection",
                args={
                    "frame_base64": sample_frame_base64,
                    "device": "cpu",
                    "annotated": True,
                },
            )
            assert isinstance(result, dict)

    def test_run_tool_output_is_json_safe(self, plugin: Plugin, sample_frame_base64: str) -> None:
        """Test tool output is JSON-serializable."""
        import json

        with patch(
            "forgesyte_yolo_tracker.plugin.detect_players_json",
            return_value={
                "detections": [{"x1": 100, "y1": 200, "x2": 150, "y2": 350, "confidence": 0.92}]
            },
        ):
            result = plugin.run_tool(
                "player_detection",
                args={
                    "frame_base64": sample_frame_base64,
                    "device": "cpu",
                },
            )

            # Verify JSON serializable
            json_str = json.dumps(result)
            assert json_str is not None
            parsed = json.loads(json_str)
            assert parsed == result


class TestPluginLifecycle:
    """Test plugin lifecycle hooks (BasePlugin contract)."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        return Plugin()

    def test_on_load_does_not_crash(self, plugin: Plugin) -> None:
        """Test on_load lifecycle hook does not raise errors."""
        plugin.on_load()  # Should not raise

    def test_on_unload_does_not_crash(self, plugin: Plugin) -> None:
        """Test on_unload lifecycle hook does not raise errors."""
        plugin.on_unload()  # Should not raise

    def test_plugin_lifecycle_sequence(self, plugin: Plugin) -> None:
        """Test plugin lifecycle sequence (load → use → unload)."""
        plugin.on_load()  # Initialize
        assert hasattr(plugin, "tools")  # Tools should be available
        plugin.on_unload()  # Cleanup

    def test_plugin_class_has_docstring(self, plugin: Plugin) -> None:
        """Test plugin class has documentation."""
        assert plugin.__class__.__doc__ is not None
        assert len(plugin.__class__.__doc__) > 0
