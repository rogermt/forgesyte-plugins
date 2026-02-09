"""Test YOLO plugin Phase 12 input contract (image_bytes, not frame_base64).

This contract test suite ensures YOLO plugin accepts image_bytes (bytes)
and rejects the old frame_base64 (base64 string) format.
"""

import io

import pytest
from PIL import Image

from forgesyte_yolo_tracker.plugin import Plugin


@pytest.fixture
def plugin():
    """Create plugin instance."""
    return Plugin()


@pytest.fixture
def valid_image_bytes():
    """Create valid PNG image bytes."""
    img = Image.new("RGB", (10, 10), color="blue")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestPhase12InputContract:
    """Verify all tools accept image_bytes (Phase 12 unified contract)."""

    def test_player_detection_accepts_image_bytes(
        self, plugin: Plugin, valid_image_bytes: bytes
    ) -> None:
        """Verify player_detection accepts image_bytes."""
        result = plugin.run_tool(
            "player_detection",
            {"image_bytes": valid_image_bytes},
        )
        assert isinstance(result, dict)
        assert "error" not in result

    def test_player_tracking_accepts_image_bytes(
        self, plugin: Plugin, valid_image_bytes: bytes
    ) -> None:
        """Verify player_tracking accepts image_bytes."""
        result = plugin.run_tool(
            "player_tracking",
            {"image_bytes": valid_image_bytes},
        )
        assert isinstance(result, dict)
        assert "error" not in result

    def test_ball_detection_accepts_image_bytes(
        self, plugin: Plugin, valid_image_bytes: bytes
    ) -> None:
        """Verify ball_detection accepts image_bytes."""
        result = plugin.run_tool(
            "ball_detection",
            {"image_bytes": valid_image_bytes},
        )
        assert isinstance(result, dict)
        assert "error" not in result

    def test_pitch_detection_accepts_image_bytes(
        self, plugin: Plugin, valid_image_bytes: bytes
    ) -> None:
        """Verify pitch_detection accepts image_bytes."""
        result = plugin.run_tool(
            "pitch_detection",
            {"image_bytes": valid_image_bytes},
        )
        assert isinstance(result, dict)
        assert "error" not in result

    def test_radar_accepts_image_bytes(
        self, plugin: Plugin, valid_image_bytes: bytes
    ) -> None:
        """Verify radar accepts image_bytes."""
        result = plugin.run_tool(
            "radar",
            {"image_bytes": valid_image_bytes},
        )
        assert isinstance(result, dict)
        assert "error" not in result

    def test_rejects_none_image_bytes(self, plugin: Plugin) -> None:
        """Verify plugin rejects None as image_bytes."""
        result = plugin.run_tool(
            "player_detection",
            {"image_bytes": None},
        )
        assert isinstance(result, dict)
        assert "error" in result
        assert "invalid_image_bytes" in result["error"]

    def test_rejects_string_image_bytes(self, plugin: Plugin) -> None:
        """Verify plugin rejects string (old base64) as image_bytes."""
        result = plugin.run_tool(
            "player_detection",
            {"image_bytes": "data:image/png;base64,AAAA"},
        )
        assert isinstance(result, dict)
        assert "error" in result
        assert "invalid_image_bytes" in result["error"]

    def test_schema_declares_image_bytes_not_frame_base64(
        self, plugin: Plugin
    ) -> None:
        """Verify tool schema declares image_bytes, not frame_base64."""
        for tool_name, tool_def in plugin.tools.items():
            if "video" not in tool_name:
                schema = tool_def["input_schema"]
                assert "image_bytes" in schema, (
                    f"Tool {tool_name} must have 'image_bytes' in schema"
                )
                assert "frame_base64" not in schema, (
                    f"Tool {tool_name} must NOT have 'frame_base64' in schema"
                )

    def test_no_default_tool_alias(self, plugin: Plugin) -> None:
        """Verify 'default' tool name is rejected (Phase 12 forbids fallback)."""
        with pytest.raises(ValueError, match="Unknown tool"):
            plugin.run_tool("default", {"image_bytes": b"test"})

    def test_unknown_tool_raises_error(self, plugin: Plugin) -> None:
        """Verify unknown tool names raise ValueError."""
        with pytest.raises(ValueError, match="Unknown tool"):
            plugin.run_tool("nonexistent_tool", {"image_bytes": b"test"})
