"""Tests for plugin schema validation and loading.

Validates:
- Plugin instantiation
- Tools dict structure matches ToolSchema
- Handler callables
- JSON serialization of tool schemas
"""

import json

import pytest

from forgesyte_yolo_tracker.plugin import Plugin


class TestPluginInstantiation:
    """Test plugin loads and initializes correctly."""

    def test_plugin_class_level_tools(self) -> None:
        """Verify plugin defines tools as class attribute (ForgeSyte loader contract)."""
        assert hasattr(Plugin, "tools"), "Plugin must define class-level `tools`"
        assert isinstance(Plugin.tools, dict), "Class-level tools must be a dict"
        assert len(Plugin.tools) > 0, "Class-level tools must have at least one tool"

    def test_plugin_instantiates(self) -> None:
        """Verify plugin can be instantiated."""
        plugin = Plugin()
        assert plugin is not None
        assert isinstance(plugin, Plugin)

    def test_plugin_has_name(self) -> None:
        """Verify plugin defines a name."""
        plugin = Plugin()
        assert plugin.name == "yolo-tracker"

    def test_plugin_has_version(self) -> None:
        """Verify plugin defines a version."""
        plugin = Plugin()
        assert plugin.version == "0.2.0"

    def test_plugin_has_tools_dict(self) -> None:
        """Verify plugin defines tools as a dict."""
        plugin = Plugin()
        assert isinstance(plugin.tools, dict)
        assert len(plugin.tools) > 0


class TestToolsSchema:
    """Test tools dict matches ToolSchema requirements."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Fixture for plugin instance."""
        return Plugin()

    def test_tools_dict_structure(self, plugin: Plugin) -> None:
        """Verify each tool has required schema fields."""
        for tool_name, tool_meta in plugin.tools.items():
            assert isinstance(tool_meta, dict), f"Tool '{tool_name}' must be a dict"
            actual_fields = set(tool_meta.keys())
            # handler is optional, but description + input/output_schema are required
            assert "description" in actual_fields, f"Tool '{tool_name}' missing 'description'"
            assert "input_schema" in actual_fields, f"Tool '{tool_name}' missing 'input_schema'"
            assert "output_schema" in actual_fields, f"Tool '{tool_name}' missing 'output_schema'"

    def test_tools_have_handler_callables(self, plugin: Plugin) -> None:
        """Verify all tools have callable handlers (Option A: direct callables)."""
        for tool_name, tool_meta in plugin.tools.items():
            assert "handler" in tool_meta, f"Tool '{tool_name}' missing handler"
            handler = tool_meta["handler"]
            # Handler must be callable (function reference)
            assert callable(handler), (
                f"Tool '{tool_name}' handler must be callable, " f"got {type(handler)}"
            )

    def test_tools_schema_json_serializable(self, plugin: Plugin) -> None:
        """Verify tool schemas (without handler) are JSON-serializable."""
        for tool_name, tool_meta in plugin.tools.items():
            # Exclude handler from serialization (it's a callable)
            serializable = {k: v for k, v in tool_meta.items() if k != "handler"}
            try:
                json.dumps(serializable)
            except Exception as e:
                pytest.fail(f"Tool '{tool_name}' schema not JSON-serializable: {e}")

    def test_no_invalid_fields_in_tools(self, plugin: Plugin) -> None:
        """Verify tools don't use old field names (inputs/outputs)."""
        for tool_name, tool_meta in plugin.tools.items():
            assert "inputs" not in tool_meta, (
                f"Tool '{tool_name}' uses deprecated 'inputs' field. " "Use 'input_schema' instead."
            )
            assert "outputs" not in tool_meta, (
                f"Tool '{tool_name}' uses deprecated 'outputs' field. "
                "Use 'output_schema' instead."
            )


class TestPluginTools:
    """Test individual tool definitions."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Fixture for plugin instance."""
        return Plugin()

    @pytest.mark.parametrize(
        "tool_name",
        [
            "player_detection",
            "player_tracking",
            "ball_detection",
            "pitch_detection",
            "radar",
        ],
    )
    def test_tool_exists(self, plugin: Plugin, tool_name: str) -> None:
        """Verify expected tools are defined."""
        assert tool_name in plugin.tools, f"Tool '{tool_name}' not defined"

    @pytest.mark.parametrize(
        "tool_name",
        [
            "player_detection",
            "player_tracking",
            "ball_detection",
            "pitch_detection",
            "radar",
        ],
    )
    def test_tool_has_description(self, plugin: Plugin, tool_name: str) -> None:
        """Verify each tool has a description."""
        tool_meta = plugin.tools[tool_name]
        assert "description" in tool_meta
        assert isinstance(tool_meta["description"], str)
        assert len(tool_meta["description"]) > 0

    @pytest.mark.parametrize(
        "tool_name",
        [
            "player_detection",
            "player_tracking",
            "ball_detection",
            "pitch_detection",
            "radar",
        ],
    )
    def test_tool_input_schema_valid(self, plugin: Plugin, tool_name: str) -> None:
        """Verify each tool has valid input_schema."""
        tool_meta = plugin.tools[tool_name]
        assert "input_schema" in tool_meta
        assert isinstance(tool_meta["input_schema"], dict)
        # Should have frame_base64 at minimum
        assert "frame_base64" in tool_meta["input_schema"]

    @pytest.mark.parametrize(
        "tool_name",
        [
            "player_detection",
            "player_tracking",
            "ball_detection",
            "pitch_detection",
            "radar",
        ],
    )
    def test_tool_output_schema_valid(self, plugin: Plugin, tool_name: str) -> None:
        """Verify each tool has valid output_schema."""
        tool_meta = plugin.tools[tool_name]
        assert "output_schema" in tool_meta
        assert isinstance(tool_meta["output_schema"], dict)


class TestRunTool:
    """Test run_tool dispatcher."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Fixture for plugin instance."""
        return Plugin()

    def test_run_tool_exists(self, plugin: Plugin) -> None:
        """Verify plugin implements run_tool."""
        assert hasattr(plugin, "run_tool")
        assert callable(plugin.run_tool)

    def test_run_tool_rejects_unknown_tool(self, plugin: Plugin) -> None:
        """Verify run_tool rejects unknown tools."""
        with pytest.raises(ValueError, match="Unknown tool"):
            plugin.run_tool("unknown_tool", {})
