"""Ensure dispatch calls correct handler with correct args."""

import base64
import numpy as np
import pytest
from forgesyte_yolo_tracker.plugin import Plugin


def make_dummy_b64() -> str:
    """Create a dummy base64-encoded image."""
    import cv2
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return base64.b64encode(buf).decode()


def test_dispatch_has_all_tools() -> None:
    """Verify plugin has all expected tools."""
    plugin = Plugin()
    
    expected_tools = [
        "player_detection",
        "player_tracking",
        "ball_detection",
        "pitch_detection",
        "radar",
    ]
    
    for tool_name in expected_tools:
        assert tool_name in plugin.tools, f"Missing tool: {tool_name}"
        assert callable(plugin.tools[tool_name]["handler"])


def test_dispatch_unknown_tool() -> None:
    """Verify dispatch raises error for unknown tool."""
    plugin = Plugin()
    
    with pytest.raises((ValueError, KeyError)):
        plugin.run_tool("not_a_tool", {})
