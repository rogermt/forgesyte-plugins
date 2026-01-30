"""Validate manifest structure without importing inference."""

from forgesyte_yolo_tracker.plugin import Plugin


def test_manifest_contract_structure() -> None:
    """Verify manifest has required fields."""
    plugin = Plugin()

    # Basic required fields
    assert hasattr(plugin, "name")
    assert hasattr(plugin, "version")
    assert hasattr(plugin, "description")
    assert isinstance(plugin.tools, dict)

    # Each tool must have required fields
    for tool_name, tool in plugin.tools.items():
        assert "description" in tool
        assert "input_schema" in tool
        assert "output_schema" in tool
        assert "handler" in tool
        assert callable(tool["handler"])
