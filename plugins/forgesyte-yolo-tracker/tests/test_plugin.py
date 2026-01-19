"""Tests for the main Plugin class."""

import base64
from typing import Any, Dict

import cv2
import numpy as np
import pytest

from forgesyte_yolo_tracker.plugin import Plugin, decode_image, encode_frame


class TestImageConversion:
    """Tests for image encoding/decoding utilities."""

    def create_test_frame(self, width: int = 100, height: int = 100) -> np.ndarray:
        """Create a test BGR frame."""
        return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    def test_encode_frame_returns_string(self) -> None:
        """Test that encode_frame returns a base64 string."""
        frame = self.create_test_frame()
        result = encode_frame(frame)
        assert isinstance(result, str)

    def test_encode_frame_is_valid_base64(self) -> None:
        """Test that encode_frame output is valid base64."""
        frame = self.create_test_frame()
        encoded = encode_frame(frame)
        # Should not raise
        decoded = base64.b64decode(encoded)
        assert isinstance(decoded, bytes)
        assert len(decoded) > 0

    def test_decode_image_from_encoded_frame(self) -> None:
        """Test round-trip encoding and decoding."""
        original_frame = self.create_test_frame()
        encoded = encode_frame(original_frame)
        decoded = decode_image(encoded)

        assert isinstance(decoded, np.ndarray)
        assert decoded.shape == original_frame.shape
        assert decoded.dtype == original_frame.dtype

    def test_decode_image_returns_ndarray(self) -> None:
        """Test that decode_image returns numpy array."""
        frame = self.create_test_frame()
        encoded = encode_frame(frame)
        result = decode_image(encoded)
        assert isinstance(result, np.ndarray)

    def test_decode_image_bgr_color_space(self) -> None:
        """Test that decoded image is in BGR color space."""
        frame = self.create_test_frame()
        assert frame.shape[2] == 3  # 3 channels
        encoded = encode_frame(frame)
        decoded = decode_image(encoded)
        assert decoded.shape[2] == 3

    def test_encode_frame_with_different_sizes(self) -> None:
        """Test encoding frames of different sizes."""
        for width, height in [(100, 100), (640, 480), (1920, 1080)]:
            frame = self.create_test_frame(width, height)
            encoded = encode_frame(frame)
            decoded = decode_image(encoded)
            # After JPEG compression, exact match not guaranteed, but shape should match
            assert decoded.shape[:2] == (height, width)

    def test_decode_image_invalid_base64_raises(self) -> None:
        """Test that decode_image raises on invalid base64."""
        with pytest.raises(Exception):
            decode_image("not-valid-base64!!!")

    def test_decode_image_invalid_image_data_raises(self) -> None:
        """Test that decode_image raises on invalid image data."""
        invalid_b64 = base64.b64encode(b"not-an-image").decode("utf-8")
        with pytest.raises(ValueError, match="Failed to decode image"):
            decode_image(invalid_b64)

    def test_encode_frame_preserves_dimensions(self) -> None:
        """Test that encoding preserves frame dimensions."""
        width, height = 320, 240
        frame = self.create_test_frame(width, height)
        encoded = encode_frame(frame)
        decoded = decode_image(encoded)
        assert decoded.shape == (height, width, 3)


class TestPluginInstantiation:
    """Tests for Plugin class instantiation."""

    def test_plugin_can_be_instantiated(self) -> None:
        """Test that Plugin can be created."""
        plugin = Plugin()
        assert plugin is not None

    def test_plugin_has_all_detection_methods(self) -> None:
        """Test that Plugin has all required detection methods."""
        plugin = Plugin()
        methods = [
            "yolo_player_detection",
            "yolo_player_tracking",
            "yolo_ball_detection",
            "yolo_team_classification",
            "yolo_pitch_detection",
            "yolo_radar",
        ]
        for method in methods:
            assert hasattr(plugin, method)
            assert callable(getattr(plugin, method))


class TestPlayerDetection:
    """Tests for yolo_player_detection method."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create a plugin instance."""
        return Plugin()

    @pytest.fixture
    def test_image(self) -> str:
        """Create a test image."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return encode_frame(frame)

    def test_player_detection_returns_dict(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test that yolo_player_detection returns a dictionary."""
        result = plugin.yolo_player_detection(test_image)
        assert isinstance(result, dict)

    def test_player_detection_includes_frame(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test that result includes frame."""
        result = plugin.yolo_player_detection(test_image)
        assert "frame" in result
        assert isinstance(result["frame"], str)

    def test_player_detection_with_config(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test player detection with custom config."""
        config = {"device": "cpu", "confidence": 0.5}
        result = plugin.yolo_player_detection(test_image, config)
        assert isinstance(result, dict)

    def test_player_detection_with_none_config(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test player detection with None config."""
        result = plugin.yolo_player_detection(test_image, None)
        assert isinstance(result, dict)


class TestPlayerTracking:
    """Tests for yolo_player_tracking method."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create a plugin instance."""
        return Plugin()

    @pytest.fixture
    def test_image(self) -> str:
        """Create a test image."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return encode_frame(frame)

    def test_player_tracking_returns_dict(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test that yolo_player_tracking returns a dictionary."""
        result = plugin.yolo_player_tracking(test_image)
        assert isinstance(result, dict)

    def test_player_tracking_includes_frame(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test that result includes frame."""
        result = plugin.yolo_player_tracking(test_image)
        assert "frame" in result

    def test_player_tracking_with_config(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test player tracking with custom config."""
        config = {"device": "cpu"}
        result = plugin.yolo_player_tracking(test_image, config)
        assert isinstance(result, dict)


class TestBallDetection:
    """Tests for yolo_ball_detection method."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create a plugin instance."""
        return Plugin()

    @pytest.fixture
    def test_image(self) -> str:
        """Create a test image."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return encode_frame(frame)

    def test_ball_detection_returns_dict(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test that yolo_ball_detection returns a dictionary."""
        result = plugin.yolo_ball_detection(test_image)
        assert isinstance(result, dict)

    def test_ball_detection_includes_frame(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test that result includes frame."""
        result = plugin.yolo_ball_detection(test_image)
        assert "frame" in result

    def test_ball_detection_with_config(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test ball detection with custom config."""
        config: Dict[str, Any] = {}
        result = plugin.yolo_ball_detection(test_image, config)
        assert isinstance(result, dict)


class TestTeamClassification:
    """Tests for yolo_team_classification method."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create a plugin instance."""
        return Plugin()

    @pytest.fixture
    def test_image(self) -> str:
        """Create a test image."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return encode_frame(frame)

    def test_team_classification_returns_dict(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test that yolo_team_classification returns a dictionary."""
        result = plugin.yolo_team_classification(test_image)
        assert isinstance(result, dict)

    def test_team_classification_includes_frame(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test that result includes frame."""
        result = plugin.yolo_team_classification(test_image)
        assert "frame" in result

    def test_team_classification_with_config(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test team classification with custom config."""
        config: Dict[str, Any] = {}
        result = plugin.yolo_team_classification(test_image, config)
        assert isinstance(result, dict)


class TestPitchDetection:
    """Tests for yolo_pitch_detection method."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create a plugin instance."""
        return Plugin()

    @pytest.fixture
    def test_image(self) -> str:
        """Create a test image."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return encode_frame(frame)

    def test_pitch_detection_returns_dict(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test that yolo_pitch_detection returns a dictionary."""
        result = plugin.yolo_pitch_detection(test_image)
        assert isinstance(result, dict)

    def test_pitch_detection_includes_frame(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test that result includes frame."""
        result = plugin.yolo_pitch_detection(test_image)
        assert "frame" in result

    def test_pitch_detection_with_config(
        self, plugin: Plugin, test_image: str
    ) -> None:
        """Test pitch detection with custom config."""
        config: Dict[str, Any] = {}
        result = plugin.yolo_pitch_detection(test_image, config)
        assert isinstance(result, dict)


class TestRadar:
    """Tests for yolo_radar method."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create a plugin instance."""
        return Plugin()

    @pytest.fixture
    def test_image(self) -> str:
        """Create a test image."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        return encode_frame(frame)

    def test_radar_returns_dict(self, plugin: Plugin, test_image: str) -> None:
        """Test that yolo_radar returns a dictionary."""
        result = plugin.yolo_radar(test_image)
        assert isinstance(result, dict)

    def test_radar_includes_frame(self, plugin: Plugin, test_image: str) -> None:
        """Test that result includes frame."""
        result = plugin.yolo_radar(test_image)
        assert "frame" in result

    def test_radar_with_config(self, plugin: Plugin, test_image: str) -> None:
        """Test radar with custom config."""
        config: Dict[str, Any] = {}
        result = plugin.yolo_radar(test_image, config)
        assert isinstance(result, dict)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    @pytest.fixture
    def plugin(self) -> Plugin:
        """Create a plugin instance."""
        return Plugin()

    def test_empty_config_dict(self, plugin: Plugin) -> None:
        """Test methods with empty config dict."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_image = encode_frame(frame)

        result = plugin.yolo_player_detection(test_image, {})
        assert isinstance(result, dict)

    def test_all_methods_with_same_image(self, plugin: Plugin) -> None:
        """Test that all methods can process the same image."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_image = encode_frame(frame)

        methods = [
            plugin.yolo_player_detection,
            plugin.yolo_player_tracking,
            plugin.yolo_ball_detection,
            plugin.yolo_team_classification,
            plugin.yolo_pitch_detection,
            plugin.yolo_radar,
        ]

        for method in methods:
            result = method(test_image)
            assert isinstance(result, dict)
            assert "frame" in result
