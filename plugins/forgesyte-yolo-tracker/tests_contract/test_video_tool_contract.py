"""Contract tests for v0.9.7 video tools manifest.

Tests that the manifest declares video-capable tools with correct schema.
These tests validate the manifest structure without loading YOLO models.
"""

import json

import pytest

from forgesyte_yolo_tracker.plugin import Plugin

# v0.9.7 video tools
V097_VIDEO_TOOLS = [
    "video_ball_detection",
    "video_pitch_detection",
    "video_radar",
    "video_player_tracking",
]


class TestVideoToolManifest:
    """Tests for v0.9.7 video tools in manifest.json."""

    @pytest.fixture
    def manifest(self) -> dict:
        """Load manifest.json as dict."""
        import forgesyte_yolo_tracker

        manifest_path = (
            forgesyte_yolo_tracker.__file__.rsplit("/", 1)[0] + "/manifest.json"
        )
        with open(manifest_path) as f:
            return json.load(f)

    def test_manifest_has_video_tools(self, manifest: dict) -> None:
        """Verify manifest has tools with input_types containing 'video'."""
        tools = manifest.get("tools", [])
        video_tools = [t for t in tools if "video" in t.get("input_types", [])]

        assert len(video_tools) >= 4, "Manifest should have at least 4 video tools"

    def test_video_tools_have_correct_ids(self, manifest: dict) -> None:
        """Verify v0.9.7 video tools have correct ids."""
        tools = manifest.get("tools", [])
        video_tools = [t for t in tools if "video" in t.get("input_types", [])]

        tool_ids = [t["id"] for t in video_tools]
        for expected_tool in V097_VIDEO_TOOLS:
            assert expected_tool in tool_ids, (
                f"Expected {expected_tool} tool, found: {tool_ids}"
            )

    def test_video_tools_input_schema(self, manifest: dict) -> None:
        """Verify v0.9.7 video tools have correct input schema."""
        tools = manifest.get("tools", [])

        for tool_id in V097_VIDEO_TOOLS:
            video_tool = next(
                (t for t in tools if t.get("id") == tool_id), None
            )
            assert video_tool is not None, f"{tool_id} tool not found"

            inputs = video_tool.get("inputs", {})

            # Required: video_path
            assert "video_path" in inputs, f"{tool_id} must accept video_path"

            # Optional: device
            assert "device" in inputs, f"{tool_id} should accept device parameter"

    def test_video_tools_output_schema(self, manifest: dict) -> None:
        """Verify v0.9.7 video tools have correct output schema."""
        tools = manifest.get("tools", [])

        for tool_id in V097_VIDEO_TOOLS:
            video_tool = next(
                (t for t in tools if t.get("id") == tool_id), None
            )
            assert video_tool is not None, f"{tool_id} tool not found"

            outputs = video_tool.get("outputs", {})

            # Must return frames array and total_frames
            assert "frames" in outputs, f"{tool_id} must output frames"
            assert "total_frames" in outputs, f"{tool_id} must output total_frames"

    def test_video_tools_input_types(self, manifest: dict) -> None:
        """Verify v0.9.7 video tools have input_types: ['video']."""
        tools = manifest.get("tools", [])

        for tool_id in V097_VIDEO_TOOLS:
            video_tool = next(
                (t for t in tools if t.get("id") == tool_id), None
            )
            assert video_tool is not None, f"{tool_id} tool not found"

            input_types = video_tool.get("input_types", [])
            assert "video" in input_types, (
                f"{tool_id} input_types must contain 'video', got: {input_types}"
            )


class TestVideoToolPluginRegistration:
    """Tests for v0.9.7 video tools registration in Plugin class."""

    def test_plugin_has_video_tools(self) -> None:
        """Verify Plugin class has v0.9.7 video tools registered."""
        plugin = Plugin()

        for tool_id in V097_VIDEO_TOOLS:
            assert tool_id in plugin.tools, (
                f"{tool_id} not found in plugin.tools"
            )

    def test_video_tools_have_handlers(self) -> None:
        """Verify v0.9.7 video tools have callable handlers."""
        plugin = Plugin()

        for tool_id in V097_VIDEO_TOOLS:
            tool = plugin.tools.get(tool_id, {})

            assert "handler" in tool, f"{tool_id} missing handler"
            assert callable(tool["handler"]), f"{tool_id} handler must be callable"

    def test_video_tools_input_schema_in_plugin(self) -> None:
        """Verify v0.9.7 video tools have input_schema in Plugin.tools."""
        plugin = Plugin()

        for tool_id in V097_VIDEO_TOOLS:
            tool = plugin.tools.get(tool_id, {})

            assert "input_schema" in tool, f"{tool_id} missing input_schema"

            input_schema = tool["input_schema"]
            assert "video_path" in input_schema, f"{tool_id} input_schema must have video_path"
            assert "device" in input_schema, f"{tool_id} input_schema should have device"

    def test_video_tools_output_schema_in_plugin(self) -> None:
        """Verify v0.9.7 video tools have output_schema in Plugin.tools."""
        plugin = Plugin()

        for tool_id in V097_VIDEO_TOOLS:
            tool = plugin.tools.get(tool_id, {})

            assert "output_schema" in tool, f"{tool_id} missing output_schema"

            output_schema = tool["output_schema"]
            assert "frames" in output_schema, f"{tool_id} output_schema must have frames"
            assert "total_frames" in output_schema, f"{tool_id} output_schema must have total_frames"
