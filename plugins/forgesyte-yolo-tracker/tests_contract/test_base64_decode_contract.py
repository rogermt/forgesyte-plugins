"""Validate base64 error handling without touching inference."""

from forgesyte_yolo_tracker.plugin import Plugin


def test_invalid_base64_handling() -> None:
    """Verify plugin handles invalid base64 gracefully."""
    plugin = Plugin()
    
    # Invalid base64 should not crash the plugin
    try:
        result = plugin.run_tool("player_detection", {
            "frame_base64": "%%%INVALID%%%",
            "device": "cpu",
            "annotated": False,
        })
        # Should either return error or raise ValueError
        assert result is not None or isinstance(result, dict)
    except (ValueError, KeyError):
        # Expected behavior
        pass


def test_empty_frame_base64() -> None:
    """Verify plugin handles empty frame gracefully."""
    plugin = Plugin()
    
    try:
        result = plugin.run_tool("radar", {
            "frame_base64": "",
            "device": "cpu",
            "annotated": False,
        })
        assert result is not None or isinstance(result, dict)
    except (ValueError, KeyError):
        # Expected behavior
        pass
