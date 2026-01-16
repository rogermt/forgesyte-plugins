"""Unit tests for Motion Detector plugin.

Tests cover critical functionality:
- Motion detection and scoring
- Configuration options
- Error handling
- Return type validation (AnalysisResult)
- Lifecycle hooks
"""

import io
from typing import Any
from unittest.mock import patch

import pytest
from forgesyte_motion.plugin import BoundingBox, MotionRegion, Plugin
from PIL import Image


class TestMotionDetectorPlugin:
    """Test suite for Motion Detector Plugin."""

    @pytest.fixture  # type: ignore[untyped-decorator]
    def plugin(self) -> Plugin:
        """Create plugin instance for testing."""
        return Plugin()

    @pytest.fixture  # type: ignore[untyped-decorator]
    def sample_static_image_bytes(self) -> bytes:
        """Generate a static (no motion) image."""
        img = Image.new("RGB", (640, 480), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    @pytest.fixture  # type: ignore[untyped-decorator]
    def sample_high_contrast_motion_image(self) -> bytes:
        """Generate image with significant change for motion detection."""
        img = Image.new("RGB", (640, 480), color="black")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    # Metadata and configuration
    @patch("forgesyte_motion.plugin.PluginMetadata")
    def test_metadata_returns_valid_plugin_metadata(
        self, mock_metadata_cls: Any, plugin: Plugin
    ) -> None:
        """Test metadata endpoint returns valid PluginMetadata."""
        mock_instance = mock_metadata_cls.return_value
        mock_instance.name = "motion_detector"
        mock_instance.version = "1.1.0"
        mock_instance.inputs = ["image"]
        mock_instance.outputs = ["text", "blocks", "confidence"]

        metadata = plugin.metadata()
        assert metadata.name == "motion_detector"
        assert metadata.version == "1.1.0"
        assert "image" in metadata.inputs
        assert "text" in metadata.outputs
        assert "blocks" in metadata.outputs
        assert "confidence" in metadata.outputs

    @patch("forgesyte_motion.plugin.PluginMetadata")
    def test_metadata_includes_config_schema(
        self, mock_metadata_cls: Any, plugin: Plugin
    ) -> None:
        """Test metadata includes threshold, min_area, blur_size, reset_baseline."""
        mock_instance = mock_metadata_cls.return_value
        mock_instance.config_schema = {
            "threshold": {"default": 25.0},
            "min_area": {"default": 0.01},
            "blur_size": {"default": 5},
            "reset_baseline": {"default": False},
        }

        metadata = plugin.metadata()
        config = metadata.config_schema
        assert "threshold" in config
        assert "min_area" in config
        assert "blur_size" in config
        assert "reset_baseline" in config

    # Motion detection - core functionality
    @patch("forgesyte_motion.plugin.AnalysisResult")
    def test_baseline_established_on_first_frame(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_static_image_bytes: bytes
    ) -> None:
        """Test first frame establishes baseline without detecting motion."""
        expected_instance = mock_analysis_cls.return_value
        expected_instance.text = "Baseline established"

        result = plugin.analyze(sample_static_image_bytes)

        assert result == expected_instance
        mock_analysis_cls.assert_called_once()
        call_kwargs = mock_analysis_cls.call_args[1]
        assert call_kwargs["text"] == "Baseline established"
        assert call_kwargs["confidence"] == 0.0

    @patch("forgesyte_motion.plugin.AnalysisResult")
    def test_motion_detected_on_significant_change(
        self,
        mock_analysis_cls: Any,
        plugin: Plugin,
        sample_static_image_bytes: bytes,
        sample_high_contrast_motion_image: bytes,
    ) -> None:
        """Test motion is detected when frame changes significantly."""
        # Establish baseline
        plugin.analyze(sample_static_image_bytes)

        # Detect motion
        plugin.analyze(sample_high_contrast_motion_image)

        # The second call should have confidence > 0 and text "motion detected"
        call_args_list = mock_analysis_cls.call_args_list
        assert len(call_args_list) == 2

        success_call_kwargs = call_args_list[1][1]
        assert success_call_kwargs["text"] == "motion detected"
        assert success_call_kwargs["confidence"] > 0.0
        assert len(success_call_kwargs["blocks"]) > 0

    @patch("forgesyte_motion.plugin.AnalysisResult")
    def test_no_motion_with_identical_frames(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_static_image_bytes: bytes
    ) -> None:
        """Test no motion detected when frames are identical."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_static_image_bytes)

        call_args_list = mock_analysis_cls.call_args_list
        success_call_kwargs = call_args_list[1][1]
        assert success_call_kwargs["text"] == ""
        assert success_call_kwargs["confidence"] == 0.0

    # Configuration options
    @patch("forgesyte_motion.plugin.AnalysisResult")
    def test_custom_threshold_option(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_static_image_bytes: bytes
    ) -> None:
        """Test custom threshold configuration."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_static_image_bytes, options={"threshold": 100.0})

        call_args_list = mock_analysis_cls.call_args_list
        success_call_kwargs = call_args_list[1][1]
        assert success_call_kwargs["confidence"] == 0.0

    @patch("forgesyte_motion.plugin.AnalysisResult")
    def test_reset_baseline_option(
        self,
        mock_analysis_cls: Any,
        plugin: Plugin,
        sample_static_image_bytes: bytes,
        sample_high_contrast_motion_image: bytes,
    ) -> None:
        """Test reset_baseline configuration forces new baseline."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(
            sample_high_contrast_motion_image, options={"reset_baseline": True}
        )

        call_args_list = mock_analysis_cls.call_args_list
        # Second call should be a baseline establishment again
        success_call_kwargs = call_args_list[1][1]
        assert success_call_kwargs["text"] == "Baseline established"
        assert success_call_kwargs["confidence"] == 0.0

    # Region detection
    @patch("forgesyte_motion.plugin.AnalysisResult")
    def test_regions_empty_when_no_motion(
        self, mock_analysis_cls: Any, plugin: Plugin, sample_static_image_bytes: bytes
    ) -> None:
        """Test regions list is empty when no motion detected."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_static_image_bytes)

        call_args_list = mock_analysis_cls.call_args_list
        success_call_kwargs = call_args_list[1][1]
        assert success_call_kwargs["blocks"] == []

    @patch("forgesyte_motion.plugin.AnalysisResult")
    def test_regions_populated_when_motion_detected(
        self,
        mock_analysis_cls: Any,
        plugin: Plugin,
        sample_static_image_bytes: bytes,
        sample_high_contrast_motion_image: bytes,
    ) -> None:
        """Test regions are detected when motion found."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_high_contrast_motion_image)

        call_args_list = mock_analysis_cls.call_args_list
        success_call_kwargs = call_args_list[1][1]

        assert len(success_call_kwargs["blocks"]) > 0
        block = success_call_kwargs["blocks"][0]
        assert "bbox" in block
        assert "area" in block
        assert "center" in block

    # Error handling
    @patch("forgesyte_motion.plugin.AnalysisResult")
    def test_handles_invalid_image_data(
        self, mock_analysis_cls: Any, plugin: Plugin
    ) -> None:
        """Test error handling for invalid image bytes."""
        invalid_bytes = b"not an image"
        plugin.analyze(invalid_bytes)

        call_kwargs = mock_analysis_cls.call_args[1]
        assert call_kwargs["error"] is not None
        assert call_kwargs["confidence"] == 0.0

    # Pydantic model validation
    def test_bounding_box_model_validation(self) -> None:
        """Test BoundingBox Pydantic model."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        assert bbox.x == 10
        assert bbox.width == 100

    def test_motion_region_model_validation(self) -> None:
        """Test MotionRegion Pydantic model."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        region = MotionRegion(bbox=bbox, area=8000, center={"x": 60, "y": 60})
        assert region.area == 8000

    # Lifecycle hooks
    def test_on_load_lifecycle_hook(self, plugin: Plugin) -> None:
        """Test on_load lifecycle hook executes without error."""
        plugin.on_load()
        assert plugin._frame_count == 0

    def test_on_unload_resets_state(
        self, plugin: Plugin, sample_static_image_bytes: bytes
    ) -> None:
        """Test on_unload lifecycle hook resets state."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 2

        plugin.on_unload()

        assert plugin._frame_count == 0
        assert plugin._previous_frame is None

    def test_reset_clears_state(
        self, plugin: Plugin, sample_static_image_bytes: bytes
    ) -> None:
        """Test reset() clears all internal state."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 2

        plugin.reset()

        assert plugin._frame_count == 0
        assert plugin._previous_frame is None
        assert plugin._last_motion_time == 0
        assert plugin._motion_history == []

    # Motion tracking
    def test_frame_counting(
        self, plugin: Plugin, sample_static_image_bytes: bytes
    ) -> None:
        """Test frame counter increments."""
        assert plugin._frame_count == 0
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 1
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 2

    @patch("forgesyte_motion.plugin.AnalysisResult")
    def test_shape_mismatch_resets_baseline(
        self, mock_analysis_cls: Any, plugin: Plugin
    ) -> None:
        """Test shape mismatch between frames resets baseline."""
        img1 = Image.new("RGB", (640, 480), color=128)
        img2 = Image.new("RGB", (800, 600), color=128)
        bytes1 = io.BytesIO()
        bytes2 = io.BytesIO()
        img1.save(bytes1, format="PNG")
        img2.save(bytes2, format="PNG")

        plugin.analyze(bytes1.getvalue())
        plugin.analyze(bytes2.getvalue())

        call_args_list = mock_analysis_cls.call_args_list
        assert call_args_list[1][1]["text"] == "Baseline established"
