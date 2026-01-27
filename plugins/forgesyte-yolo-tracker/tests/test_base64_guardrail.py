"""Tests for base64 decoding guardrail in YOLO plugin tools."""

import base64
import pytest


class TestBase64GuardrailP0:
    """P0 Critical Tests - Must handle malformed base64 gracefully."""

    @pytest.fixture
    def plugin(self):
        from forgesyte_yolo_tracker.plugin import Plugin
        return Plugin()

    def test_invalid_characters_in_base64(self, plugin):
        """Test that invalid base64 characters return 400, not 500."""
        result = plugin.player_detection(frame_b64="%%%NOTBASE64%%%", device="cpu")
        assert isinstance(result, dict)
        # Should return error dict, not raise exception
        assert result.get("error") == "invalid_base64"
        assert result.get("plugin") == "yolo-tracker"
        assert result.get("tool") == "player_detection"

    def test_truncated_base64(self, plugin):
        """Test that truncated base64 returns 400, not 500."""
        raw = base64.b64encode(b"hello world").decode()
        truncated = raw[:-4]  # Remove last 4 chars to corrupt it
        result = plugin.player_detection(frame_b64=truncated, device="cpu")
        assert isinstance(result, dict)
        assert result.get("error") == "invalid_base64"
        assert result.get("plugin") == "yolo-tracker"
        assert result.get("tool") == "player_detection"

    def test_empty_string(self, plugin):
        """Test that empty base64 string returns 400, not 500."""
        result = plugin.player_detection(frame_b64="", device="cpu")
        assert isinstance(result, dict)
        assert result.get("error") == "invalid_base64"
        assert result.get("plugin") == "yolo-tracker"
        assert result.get("tool") == "player_detection"

    def test_non_base64_string(self, plugin):
        """Test that non-base64 string returns 400, not 500."""
        result = plugin.player_detection(frame_b64="this-is-not-base64-at-all!!!", device="cpu")
        assert isinstance(result, dict)
        assert result.get("error") == "invalid_base64"
        assert result.get("plugin") == "yolo-tracker"
        assert result.get("tool") == "player_detection"


class TestBase64GuardrailAllTools:
    """Test all tool methods have guardrail protection."""

    @pytest.fixture
    def plugin(self):
        from forgesyte_yolo_tracker.plugin import Plugin
        return Plugin()

    @pytest.mark.parametrize("tool_method", [
        "player_detection", "player_tracking", "ball_detection", "pitch_detection", "radar"
    ])
    def test_all_tools_handle_invalid_base64(self, plugin, tool_method):
        """Verify all tools return structured error for invalid base64."""
        tool = getattr(plugin, tool_method)
        result = tool(frame_b64="%%%NOTBASE64%%%", device="cpu")
        assert isinstance(result, dict)
        assert result.get("error") == "invalid_base64"
        assert result.get("plugin") == "yolo-tracker"
        assert result.get("tool") == tool_method


class TestBase64GuardrailDataUrl:
    """Test that data URL prefix is handled correctly."""

    @pytest.fixture
    def plugin(self):
        from forgesyte_yolo_tracker.plugin import Plugin
        return Plugin()

    def test_data_url_prefix_stripped(self, plugin):
        """Test that data:image/jpeg;base64, prefix is stripped by _validate_base64."""
        from forgesyte_yolo_tracker.plugin import _validate_base64
        
        # Create a valid base64 string
        valid_b64 = "SGVsbG8gV29ybGQ="  # "Hello World" in base64
        
        # Wrap in data URL
        data_url = f"data:image/jpeg;base64,{valid_b64}"
        
        # Should strip prefix and return just the base64
        result = _validate_base64(data_url)
        assert result == valid_b64
        
    def test_data_url_prefix_handled(self, plugin, mocker):
        """Test that data:image/jpeg;base64, prefix is stripped before decoding."""
        import numpy as np
        import cv2
        
        # Create a valid small image and encode it
        small_img = np.zeros((10, 10, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', small_img)
        valid_b64_bytes = base64.b64encode(buffer.tobytes()).decode()
        valid_b64 = valid_b64_bytes
        
        # Wrap in data URL
        data_url = f"data:image/jpeg;base64,{valid_b64}"
        
        # Mock detect_players_json to avoid model loading
        from forgesyte_yolo_tracker import plugin as plugin_module
        mocker.patch.object(plugin_module, 'detect_players_json', return_value={"detections": [], "count": 0, "classes": []})
        
        # Should not raise, should process correctly
        result = plugin.player_detection(frame_b64=data_url, device="cpu")
        # Should return detections, not an error
        assert isinstance(result, dict)
        assert "detections" in result or result.get("error") is None

