"""TEST-CHANGE: Test that YOLO-tracker accepts 'default' as tool alias.

Issue #164: Plugin should accept 'default' as alias for first available tool
for backward compatibility.
"""

import pytest

from forgesyte_yolo_tracker.plugin import Plugin


class TestDefaultToolAlias:
    """Tests for 'default' tool alias fallback."""

    def test_run_tool_accepts_default_alias_maps_to_first_tool(self) -> None:
        """Verify run_tool accepts 'default' and maps to first available tool.
        
        This test will FAIL before the fix is applied because plugin.run_tool()
        will raise ValueError("Unknown tool: default").
        
        After fix: 'default' should map to first tool in plugin.tools.
        """
        plugin = Plugin()
        
        # Get first tool name
        first_tool_name = next(iter(plugin.tools.keys()))
        
        # Create minimal args for any tool
        test_args = {
            "frame_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
            "device": "cpu",
            "annotated": False,
        }
        
        # This should NOT raise - "default" should work as alias
        try:
            # Try calling with "default" - should use first tool
            result = plugin.run_tool("default", test_args)
            # If we get here, the fix works
            assert result is not None
        except ValueError as e:
            if "Unknown tool: default" in str(e):
                pytest.fail(
                    f"Plugin.run_tool() should accept 'default' as alias for '{first_tool_name}' "
                    f"but got error: {e}. Fix needed in plugin.py run_tool()."
                )
            raise

    def test_plugin_has_explicit_tools_not_default(self) -> None:
        """Verify plugin doesn't have literal 'default' tool - explains why alias needed."""
        plugin = Plugin()
        
        # YOLO-tracker uses explicit tool names
        assert "default" not in plugin.tools
        
        # Should have real tools
        assert len(plugin.tools) > 0
        first_tool = next(iter(plugin.tools.keys()))
        assert first_tool in {
            "player_detection",
            "player_tracking",
            "ball_detection", 
            "pitch_detection",
            "radar",
        }
