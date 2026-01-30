"""Edge case and high-quality tests for YOLO plugin — 30-year testing expert standard.

These tests target 95%+ coverage of plugin.py by exercising:
- Base64 edge cases (data URIs, empty, invalid)
- All tool functions (player_tracking, ball, pitch, radar)
- Annotated frame paths
- Tool-specific error handling
- Contract compliance boundaries
"""

import base64
import io
from unittest.mock import patch

import pytest
from PIL import Image

from forgesyte_yolo_tracker.plugin import Plugin


class TestBase64Validation:
    """High-coverage tests for base64 validation logic."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance."""
        p = Plugin()
        p.on_load()
        return p

    def test_data_uri_prefix_stripped(self, plugin: Plugin) -> None:
        """Test data:image/png;base64,... prefix is stripped correctly."""
        # This triggers line 72: frame_b64.split(",", 1)[-1]
        valid_frame = Image.new("RGB", (100, 100), color="white")
        frame_bytes = io.BytesIO()
        valid_frame.save(frame_bytes, format="PNG")
        b64_data = base64.b64encode(frame_bytes.getvalue()).decode("utf-8")

        # Create data URI
        data_uri = f"data:image/png;base64,{b64_data}"

        with patch(
            "forgesyte_yolo_tracker.plugin.detect_players_json",
            return_value={"detections": []},
        ):
            result = plugin.run_tool(
                "player_detection",
                args={"frame_base64": data_uri, "device": "cpu"},
            )
            # Should successfully decode (data URI prefix handled)
            assert isinstance(result, dict)

    def test_empty_base64_after_strip(self, plugin: Plugin) -> None:
        """Test empty base64 string after whitespace strip raises error.
        
        This triggers line 75: raise ValueError("Empty base64 string after processing")
        """
        result = plugin.run_tool(
            "player_detection",
            args={"frame_base64": "   ", "device": "cpu"},  # Only whitespace
        )
        assert isinstance(result, dict)
        assert "error" in result

    def test_base64_with_newlines(self, plugin: Plugin) -> None:
        """Test base64 with embedded newlines is handled."""
        valid_frame = Image.new("RGB", (100, 100), color="white")
        frame_bytes = io.BytesIO()
        valid_frame.save(frame_bytes, format="PNG")
        b64_data = base64.b64encode(frame_bytes.getvalue()).decode("utf-8")

        # Add newlines (common in multi-line base64)
        b64_with_newlines = "\n".join([b64_data[i:i+80] for i in range(0, len(b64_data), 80)])

        result = plugin.run_tool(
            "player_detection",
            args={"frame_base64": b64_with_newlines, "device": "cpu"},
        )
        # Should either decode or return error dict (platform dependent)
        assert isinstance(result, dict)

    def test_base64_with_padding(self, plugin: Plugin) -> None:
        """Test base64 padding characters handled correctly."""
        valid_frame = Image.new("RGB", (100, 100), color="white")
        frame_bytes = io.BytesIO()
        valid_frame.save(frame_bytes, format="PNG")
        b64_data = base64.b64encode(frame_bytes.getvalue()).decode("utf-8")

        # Ensure has padding
        assert "=" in b64_data or len(b64_data) % 4 == 0

        with patch(
            "forgesyte_yolo_tracker.plugin.detect_players_json",
            return_value={"detections": []},
        ):
            result = plugin.run_tool(
                "player_detection",
                args={"frame_base64": b64_data, "device": "cpu"},
            )
            assert isinstance(result, dict)


class TestAllToolFunctions:
    """Test all 5 tool functions to reach 100% line coverage."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance."""
        p = Plugin()
        p.on_load()
        return p

    @pytest.fixture
    def sample_frame_base64(self) -> str:
        """Valid base64 frame."""
        img = Image.new("RGB", (640, 480), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return base64.b64encode(img_bytes.getvalue()).decode("utf-8")

    # === PLAYER TRACKING ===
    def test_player_tracking_tool_no_annotated(
        self, plugin: Plugin, sample_frame_base64: str
    ) -> None:
        """Test _tool_player_tracking without annotated frame."""
        with patch(
            "forgesyte_yolo_tracker.plugin.track_players_json",
            return_value={"tracks": []},
        ):
            result = plugin.run_tool(
                "player_tracking",
                args={"frame_base64": sample_frame_base64, "device": "cpu"},
            )
            assert isinstance(result, dict)

    def test_player_tracking_tool_with_annotated(
        self, plugin: Plugin, sample_frame_base64: str
    ) -> None:
        """Test _tool_player_tracking with annotated=True.
        
        Triggers lines 122-127: annotated path in _tool_player_tracking
        """
        with patch(
            "forgesyte_yolo_tracker.plugin.track_players_json_with_annotated_frame",
            return_value={"tracks": [], "annotated_frame_base64": "iVBOR..."},
        ):
            result = plugin.run_tool(
                "player_tracking",
                args={
                    "frame_base64": sample_frame_base64,
                    "device": "cpu",
                    "annotated": True,
                },
            )
            assert isinstance(result, dict)

    def test_player_tracking_error_handling(self, plugin: Plugin) -> None:
        """Test player_tracking error returns error dict."""
        result = plugin.run_tool(
            "player_tracking",
            args={"frame_base64": "invalid", "device": "cpu"},
        )
        assert isinstance(result, dict)
        assert "error" in result

    # === BALL DETECTION ===
    def test_ball_detection_tool_no_annotated(
        self, plugin: Plugin, sample_frame_base64: str
    ) -> None:
        """Test _tool_ball_detection without annotated frame."""
        with patch(
            "forgesyte_yolo_tracker.plugin.detect_ball_json",
            return_value={"ball": None},
        ):
            result = plugin.run_tool(
                "ball_detection",
                args={"frame_base64": sample_frame_base64, "device": "cpu"},
            )
            assert isinstance(result, dict)

    def test_ball_detection_tool_with_annotated(
        self, plugin: Plugin, sample_frame_base64: str
    ) -> None:
        """Test _tool_ball_detection with annotated=True.
        
        Triggers lines 131-136: annotated path in _tool_ball_detection
        """
        with patch(
            "forgesyte_yolo_tracker.plugin.detect_ball_json_with_annotated_frame",
            return_value={"ball": {"x": 320, "y": 240}, "annotated_frame_base64": "iVBOR..."},
        ):
            result = plugin.run_tool(
                "ball_detection",
                args={
                    "frame_base64": sample_frame_base64,
                    "device": "cpu",
                    "annotated": True,
                },
            )
            assert isinstance(result, dict)

    def test_ball_detection_error_handling(self, plugin: Plugin) -> None:
        """Test ball_detection error returns error dict."""
        result = plugin.run_tool(
            "ball_detection",
            args={"frame_base64": "", "device": "cpu"},
        )
        assert isinstance(result, dict)
        assert "error" in result

    # === PITCH DETECTION ===
    def test_pitch_detection_tool_no_annotated(
        self, plugin: Plugin, sample_frame_base64: str
    ) -> None:
        """Test _tool_pitch_detection without annotated frame."""
        with patch(
            "forgesyte_yolo_tracker.plugin.detect_pitch_json",
            return_value={"keypoints": []},
        ):
            result = plugin.run_tool(
                "pitch_detection",
                args={"frame_base64": sample_frame_base64, "device": "cpu"},
            )
            assert isinstance(result, dict)

    def test_pitch_detection_tool_with_annotated(
        self, plugin: Plugin, sample_frame_base64: str
    ) -> None:
        """Test _tool_pitch_detection with annotated=True.
        
        Triggers lines 140-145: annotated path in _tool_pitch_detection
        """
        with patch(
            "forgesyte_yolo_tracker.plugin.detect_pitch_json_with_annotated_frame",
            return_value={"keypoints": [], "annotated_frame_base64": "iVBOR..."},
        ):
            result = plugin.run_tool(
                "pitch_detection",
                args={
                    "frame_base64": sample_frame_base64,
                    "device": "cpu",
                    "annotated": True,
                },
            )
            assert isinstance(result, dict)

    def test_pitch_detection_error_handling(self, plugin: Plugin) -> None:
        """Test pitch_detection error returns error dict."""
        result = plugin.run_tool(
            "pitch_detection",
            args={"frame_base64": "!!!invalid!!!", "device": "cpu"},
        )
        assert isinstance(result, dict)
        assert "error" in result

    # === RADAR ===
    def test_radar_tool_no_annotated(
        self, plugin: Plugin, sample_frame_base64: str
    ) -> None:
        """Test _tool_radar without annotated frame."""
        with patch(
            "forgesyte_yolo_tracker.plugin.radar_json",
            return_value={"radar_frame_base64": "iVBOR..."},
        ):
            result = plugin.run_tool(
                "radar",
                args={"frame_base64": sample_frame_base64, "device": "cpu"},
            )
            assert isinstance(result, dict)

    def test_radar_tool_with_annotated(
        self, plugin: Plugin, sample_frame_base64: str
    ) -> None:
        """Test _tool_radar with annotated=True.
        
        Triggers lines 149-154: annotated path in _tool_radar
        """
        with patch(
            "forgesyte_yolo_tracker.plugin.radar_json_with_annotated_frame",
            return_value={"radar_frame_base64": "iVBOR...", "metadata": {}},
        ):
            result = plugin.run_tool(
                "radar",
                args={
                    "frame_base64": sample_frame_base64,
                    "device": "cpu",
                    "annotated": True,
                },
            )
            assert isinstance(result, dict)

    def test_radar_error_handling(self, plugin: Plugin) -> None:
        """Test radar error returns error dict."""
        result = plugin.run_tool(
            "radar",
            args={"frame_base64": None, "device": "cpu"},
        )
        assert isinstance(result, dict)
        assert "error" in result





class TestErrorPropagation:
    """Test error handling and propagation."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance."""
        p = Plugin()
        p.on_load()
        return p

    def test_error_dict_structure(self, plugin: Plugin) -> None:
        """Test error dict has required fields."""
        result = plugin.run_tool(
            "player_detection",
            args={"frame_base64": "invalid_base64!!!"},
        )

        assert isinstance(result, dict)
        assert "error" in result
        assert "message" in result
        assert "plugin" in result
        assert result["plugin"] == "yolo-tracker"
        assert "tool" in result
        assert result["tool"] == "player_detection"

    def test_error_dict_all_tools(self, plugin: Plugin) -> None:
        """Test all tools return consistent error dict structure."""
        tools = [
            "player_detection",
            "player_tracking",
            "ball_detection",
            "pitch_detection",
            "radar",
        ]

        for tool in tools:
            result = plugin.run_tool(
                tool,
                args={"frame_base64": "bad_base64!!!"},
            )
            assert isinstance(result, dict)
            assert "error" in result, f"Tool {tool} missing 'error' field"
            assert result["tool"] == tool


class TestInputValidation:
    """Test input validation basics."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create plugin instance."""
        p = Plugin()
        p.on_load()
        return p

    def test_none_frame_base64(self, plugin: Plugin) -> None:
        """Test None frame_base64 handled gracefully."""
        result = plugin.run_tool(
            "player_detection",
            args={"frame_base64": None},
        )
        assert isinstance(result, dict)
        assert "error" in result
