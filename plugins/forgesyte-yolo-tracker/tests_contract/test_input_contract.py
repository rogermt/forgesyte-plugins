"""Test YOLO plugin Phase 12 input contract.

Verifies that YOLO plugin accepts image_bytes (bytes) and not base64.
This is a contract test that does NOT require YOLO models.
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
def fake_image_bytes():
    """Create minimal valid PNG bytes for testing."""
    # Create a 1x1 RGB image
    img = Image.new("RGB", (1, 1), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return buffer.getvalue()


class TestYOLOInputContract:
    """Tests for YOLO plugin Phase 12 input contract."""

    def test_player_detection_accepts_image_bytes(self, plugin, fake_image_bytes):
        """Verify player_detection accepts image_bytes."""
        result = plugin.run_tool(
            "player_detection",
            {"image_bytes": fake_image_bytes},
        )

        # Contract: must return dict
        assert isinstance(result, dict)
        # Contract: must not crash and must not have error
        assert "error" not in result

    def test_player_tracking_accepts_image_bytes(self, plugin, fake_image_bytes):
        """Verify player_tracking accepts image_bytes."""
        result = plugin.run_tool(
            "player_tracking",
            {"image_bytes": fake_image_bytes},
        )

        assert isinstance(result, dict)
        assert "error" not in result

    def test_ball_detection_accepts_image_bytes(self, plugin, fake_image_bytes):
        """Verify ball_detection accepts image_bytes."""
        result = plugin.run_tool(
            "ball_detection",
            {"image_bytes": fake_image_bytes},
        )

        assert isinstance(result, dict)
        assert "error" not in result

    def test_pitch_detection_accepts_image_bytes(self, plugin, fake_image_bytes):
        """Verify pitch_detection accepts image_bytes."""
        result = plugin.run_tool(
            "pitch_detection",
            {"image_bytes": fake_image_bytes},
        )

        assert isinstance(result, dict)
        assert "error" not in result

    def test_radar_accepts_image_bytes(self, plugin, fake_image_bytes):
        """Verify radar accepts image_bytes."""
        result = plugin.run_tool(
            "radar",
            {"image_bytes": fake_image_bytes},
        )

        assert isinstance(result, dict)
        assert "error" not in result

    def test_invalid_image_bytes_returns_error(self, plugin):
        """Verify plugin returns error for invalid image_bytes."""
        # Pass None instead of bytes
        result = plugin.run_tool(
            "player_detection",
            {"image_bytes": None},
        )

        assert isinstance(result, dict)
        assert "error" in result

    def test_invalid_image_bytes_string_returns_error(self, plugin):
        """Verify plugin returns error if given base64 string."""
        # Pass base64 string instead of bytes (old broken way)
        result = plugin.run_tool(
            "player_detection",
            {"image_bytes": "data:image/png;base64,AAAA"},
        )

        assert isinstance(result, dict)
        assert "error" in result

    def test_tool_schema_declares_image_bytes(self, plugin):
        """Verify tool schema declares image_bytes, not frame_base64."""
        for tool_name, tool_def in plugin.tools.items():
            if "video" not in tool_name:
                # Frame tools must have image_bytes
                assert "image_bytes" in tool_def["input_schema"]
                # Frame tools must NOT have frame_base64
                assert "frame_base64" not in tool_def["input_schema"]

    def test_no_default_fallback(self, plugin):
        """Verify plugin does not accept 'default' tool name (Phase 12)."""
        with pytest.raises(ValueError, match="Unknown tool"):
            plugin.run_tool(
                "default",
                {"image_bytes": b"test"},
            )
