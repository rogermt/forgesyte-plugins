"""Validate input/output schemas for each tool."""

from forgesyte_yolo_tracker.plugin import Plugin


def test_tool_schemas_are_well_formed():
    """Verify tool schemas have required structure."""
    plugin = Plugin()

    for tool_name, tool in plugin.tools.items():
        schema = tool["input_schema"]

        # Required fields
        assert "frame_base64" in schema, f"Tool {tool_name} missing frame_base64"
        assert schema["frame_base64"]["type"] == "string"

        # Optional fields
        assert "device" in schema, f"Tool {tool_name} missing device"
        assert schema["device"]["type"] == "string"

        assert "annotated" in schema, f"Tool {tool_name} missing annotated"
        assert schema["annotated"]["type"] == "boolean"

        # Output schema
        assert "output_schema" in tool, f"Tool {tool_name} missing output_schema"
        assert "result" in tool["output_schema"], f"Tool {tool_name} output missing result"
        assert tool["output_schema"]["result"]["type"] == "object"
