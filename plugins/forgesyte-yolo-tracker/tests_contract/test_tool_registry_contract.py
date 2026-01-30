"""Validate tool registry structure and well-formedness."""

from forgesyte_yolo_tracker.plugin import Plugin


def test_tool_registry_structure():
    """Verify tool registry is well-formed."""
    plugin = Plugin()

    assert isinstance(plugin.tools, dict)
    assert len(plugin.tools) > 0

    for tool_name, tool in plugin.tools.items():
        # Required fields
        assert "description" in tool, f"Tool {tool_name} missing description"
        assert "input_schema" in tool, f"Tool {tool_name} missing input_schema"
        assert "output_schema" in tool, f"Tool {tool_name} missing output_schema"
        assert "handler" in tool, f"Tool {tool_name} missing handler"

        # Handler must be callable
        assert callable(tool["handler"]), f"Tool {tool_name} handler not callable"

        # Input schema must define required keys
        schema = tool["input_schema"]
        assert "frame_base64" in schema, f"Tool {tool_name} input missing frame_base64"
        assert schema["frame_base64"]["type"] == "string"

        assert "device" in schema, f"Tool {tool_name} input missing device"
        assert schema["device"]["type"] == "string"

        assert "annotated" in schema, f"Tool {tool_name} input missing annotated"
        assert schema["annotated"]["type"] == "boolean"

        # Output schema must define result object
        assert "result" in tool["output_schema"], f"Tool {tool_name} output missing result"
        assert tool["output_schema"]["result"]["type"] == "object"
