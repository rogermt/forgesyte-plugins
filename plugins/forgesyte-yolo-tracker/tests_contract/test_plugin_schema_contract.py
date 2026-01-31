"""Validate input/output schemas for each tool."""

from forgesyte_yolo_tracker.plugin import Plugin


def test_tool_schemas_are_well_formed() -> None:
    """Verify tool schemas have required structure."""
    plugin = Plugin()

    for tool_name, tool in plugin.tools.items():
        schema = tool["input_schema"]
        is_video_tool = "video" in tool_name

        if is_video_tool:
            # Video tools have different input schema
            assert "video_path" in schema, f"Tool {tool_name} missing video_path"
            assert "output_path" in schema, f"Tool {tool_name} missing output_path"
        else:
            # Frame tools require frame_base64
            assert "frame_base64" in schema, f"Tool {tool_name} missing frame_base64"
            assert schema["frame_base64"]["type"] == "string"

            # Optional fields for frame tools
            assert "annotated" in schema, f"Tool {tool_name} missing annotated"
            assert schema["annotated"]["type"] == "boolean"

        # Required for all tools
        assert "device" in schema, f"Tool {tool_name} missing device"
        assert schema["device"]["type"] == "string"

        # Output schema
        assert "output_schema" in tool, f"Tool {tool_name} missing output_schema"
