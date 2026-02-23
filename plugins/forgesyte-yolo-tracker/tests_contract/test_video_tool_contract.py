"""Contract tests for video_player_detection tool manifest.

Tests that the manifest declares a video-capable tool with correct schema.
These tests validate the manifest structure without loading YOLO models.
"""

import json

import pytest

from forgesyte_yolo_tracker.plugin import Plugin


class TestVideoToolManifest:
    """Tests for video_player_detection tool in manifest.json."""

    @pytest.fixture
    def manifest(self) -> dict:
        """Load manifest.json as dict."""
        import forgesyte_yolo_tracker

        manifest_path = (
            forgesyte_yolo_tracker.__file__.rsplit("/", 1)[0] + "/manifest.json"
        )
        with open(manifest_path) as f:
            return json.load(f)

    def test_manifest_has_video_tool(self, manifest: dict) -> None:
        """Verify manifest has a tool with input_types containing 'video'."""
        tools = manifest.get("tools", [])
        video_tools = [t for t in tools if "video" in t.get("input_types", [])]

        assert len(video_tools) > 0, "Manifest should have at least one video tool"

    def test_video_tool_id_is_video_player_detection(self, manifest: dict) -> None:
        """Verify video tool has correct id."""
        tools = manifest.get("tools", [])
        video_tools = [t for t in tools if "video" in t.get("input_types", [])]

        tool_ids = [t["id"] for t in video_tools]
        assert "video_player_detection" in tool_ids, (
            f"Expected video_player_detection tool, found: {tool_ids}"
        )

    def test_video_tool_input_schema(self, manifest: dict) -> None:
        """Verify video_player_detection has correct input schema."""
        tools = manifest.get("tools", [])
        video_tool = next(
            (t for t in tools if t.get("id") == "video_player_detection"), None
        )

        assert video_tool is not None, "video_player_detection tool not found"

        inputs = video_tool.get("inputs", {})

        # Required: video_path
        assert "video_path" in inputs, "video_player_detection must accept video_path"

        # Optional: device (should default to 'cpu')
        assert "device" in inputs, "video_player_detection should accept device parameter"

    def test_video_tool_output_schema(self, manifest: dict) -> None:
        """Verify video_player_detection has correct output schema."""
        tools = manifest.get("tools", [])
        video_tool = next(
            (t for t in tools if t.get("id") == "video_player_detection"), None
        )

        assert video_tool is not None, "video_player_detection tool not found"

        outputs = video_tool.get("outputs", {})

        # Must return frames array and summary object
        assert "frames" in outputs, "video_player_detection must output frames"
        assert "summary" in outputs, "video_player_detection must output summary"

    def test_video_tool_input_types(self, manifest: dict) -> None:
        """Verify video_player_detection has input_types: ['video']."""
        tools = manifest.get("tools", [])
        video_tool = next(
            (t for t in tools if t.get("id") == "video_player_detection"), None
        )

        assert video_tool is not None, "video_player_detection tool not found"

        input_types = video_tool.get("input_types", [])
        assert "video" in input_types, (
            f"video_player_detection input_types must contain 'video', got: {input_types}"
        )


class TestVideoToolPluginRegistration:
    """Tests for video_player_detection tool registration in Plugin class."""

    def test_plugin_has_video_player_detection_tool(self) -> None:
        """Verify Plugin class has video_player_detection registered."""
        plugin = Plugin()

        assert "video_player_detection" in plugin.tools, (
            "video_player_detection not found in plugin.tools"
        )

    def test_video_tool_has_handler(self) -> None:
        """Verify video_player_detection tool has callable handler."""
        plugin = Plugin()
        tool = plugin.tools.get("video_player_detection", {})

        assert "handler" in tool, "video_player_detection missing handler"
        assert callable(tool["handler"]), "video_player_detection handler must be callable"

    def test_video_tool_input_schema_in_plugin(self) -> None:
        """Verify video_player_detection has input_schema in Plugin.tools."""
        plugin = Plugin()
        tool = plugin.tools.get("video_player_detection", {})

        assert "input_schema" in tool, "video_player_detection missing input_schema"

        input_schema = tool["input_schema"]
        assert "video_path" in input_schema, "input_schema must have video_path"
        assert "device" in input_schema, "input_schema should have device"

    def test_video_tool_output_schema_in_plugin(self) -> None:
        """Verify video_player_detection has output_schema in Plugin.tools."""
        plugin = Plugin()
        tool = plugin.tools.get("video_player_detection", {})

        assert "output_schema" in tool, "video_player_detection missing output_schema"

        output_schema = tool["output_schema"]
        assert "frames" in output_schema, "output_schema must have frames"
        assert "summary" in output_schema, "output_schema must have summary"
