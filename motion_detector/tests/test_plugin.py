"""Unit tests for Motion Detector plugin.

Tests cover critical functionality:
- Motion detection and scoring
- Configuration options
- Error handling
- Return type validation (server compatibility)
- Lifecycle hooks
"""

import io
import time

import numpy as np
import pytest
from PIL import Image

from forgesyte_motion.plugin import (
    Plugin,
    MotionAnalysisResult,
    MotionRegion,
    BoundingBox,
)


class TestMotionDetectorPlugin:
    """Test suite for Motion Detector Plugin."""

    @pytest.fixture
    def plugin(self):
        """Create plugin instance for testing."""
        return Plugin()

    @pytest.fixture
    def sample_static_image_bytes(self):
        """Generate a static (no motion) image."""
        img = Image.new("RGB", (640, 480), color="white")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    @pytest.fixture
    def sample_high_contrast_motion_image(self):
        """Generate image with significant change for motion detection."""
        img = Image.new("RGB", (640, 480), color="black")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        return img_bytes.getvalue()

    # Metadata and configuration
    def test_metadata_returns_valid_plugin_metadata(self, plugin):
        """Test metadata endpoint returns valid PluginMetadata."""
        metadata = plugin.metadata()
        assert metadata.name == "motion_detector"
        assert metadata.version == "1.1.0"
        assert "image" in metadata.inputs
        assert "motion_detected" in metadata.outputs
        assert "motion_score" in metadata.outputs

    def test_metadata_includes_config_schema(self, plugin):
        """Test metadata includes threshold, min_area, blur_size, reset_baseline."""
        metadata = plugin.metadata()
        config = metadata.config_schema
        assert "threshold" in config
        assert "min_area" in config
        assert "blur_size" in config
        assert "reset_baseline" in config

    # Motion detection - core functionality
    def test_baseline_established_on_first_frame(self, plugin, sample_static_image_bytes):
        """Test first frame establishes baseline without detecting motion."""
        result = plugin.analyze(sample_static_image_bytes)
        # First frame returns a dict with status
        assert isinstance(result, dict)
        assert result["motion_detected"] is False
        assert result["motion_score"] == 0.0

    def test_motion_detected_on_significant_change(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test motion is detected when frame changes significantly."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_high_contrast_motion_image)
        
        # Subsequent frames return MotionAnalysisResult
        assert isinstance(result, MotionAnalysisResult)
        assert result.motion_detected is True
        assert result.motion_score > 0.0

    def test_no_motion_with_identical_frames(self, plugin, sample_static_image_bytes):
        """Test no motion detected when frames are identical."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_static_image_bytes)
        
        assert isinstance(result, MotionAnalysisResult)
        assert result.motion_detected is False
        assert result.motion_score == 0.0

    def test_motion_score_is_percentage(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test motion_score is reported as percentage (0-100)."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_high_contrast_motion_image)
        
        assert isinstance(result, MotionAnalysisResult)
        assert 0 <= result.motion_score <= 100

    # Configuration options
    def test_custom_threshold_option(self, plugin, sample_static_image_bytes):
        """Test custom threshold configuration."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(
            sample_static_image_bytes, options={"threshold": 100.0}
        )
        # High threshold = no motion
        assert isinstance(result, MotionAnalysisResult)
        assert result.motion_detected is False

    def test_reset_baseline_option(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test reset_baseline configuration forces new baseline."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(
            sample_high_contrast_motion_image, options={"reset_baseline": True}
        )
        # Resets baseline -> treats as first frame -> returns dict
        assert isinstance(result, dict)
        assert result["motion_detected"] is False  # New baseline

    # Region detection
    def test_regions_empty_when_no_motion(self, plugin, sample_static_image_bytes):
        """Test regions list is empty when no motion detected."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_static_image_bytes)
        
        assert isinstance(result, MotionAnalysisResult)
        assert result.regions == []

    def test_regions_populated_when_motion_detected(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test regions are detected when motion found."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_high_contrast_motion_image)
        
        assert isinstance(result, MotionAnalysisResult)
        if result.motion_detected:
            assert len(result.regions) > 0
            region = result.regions[0]
            # Region is a MotionRegion object (Pydantic)
            assert isinstance(region, MotionRegion)
            assert region.bbox.width > 0
            assert region.area > 0
            assert "x" in region.center

    # Error handling
    def test_handles_invalid_image_data(self, plugin):
        """Test error handling for invalid image bytes."""
        invalid_bytes = b"not an image"
        result = plugin.analyze(invalid_bytes)
        
        # Error case returns dict
        assert isinstance(result, dict)
        assert result["motion_detected"] is False
        assert result["motion_score"] == 0

    def test_handles_empty_image_bytes(self, plugin):
        """Test error handling for empty bytes."""
        empty_bytes = b""
        result = plugin.analyze(empty_bytes)
        
        # Error case returns dict
        assert isinstance(result, dict)
        assert result["motion_detected"] is False

    # Pydantic model validation
    def test_bounding_box_model_validation(self):
        """Test BoundingBox Pydantic model."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        assert bbox.x == 10
        assert bbox.width == 100

    def test_motion_region_model_validation(self):
        """Test MotionRegion Pydantic model."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        region = MotionRegion(bbox=bbox, area=8000, center={"x": 60, "y": 60})
        assert region.area == 8000

    def test_motion_analysis_result_serialization(self):
        """Test MotionAnalysisResult can be serialized to dict."""
        result = MotionAnalysisResult(
            motion_detected=False,
            motion_score=0.0,
            regions=[],
            frame_number=1,
            image_size={"width": 640, "height": 480},
            recent_motion_events_count=0,
        )
        data = result.model_dump()
        assert data["motion_detected"] is False
        assert data["image_size"]["width"] == 640

    # Server compatibility - CRITICAL
    def test_analyze_returns_pydantic_model_or_dict(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test analyze() returns MotionAnalysisResult or dict (for error/init cases).
        """
        # First call: init baseline (returns dict)
        result_init = plugin.analyze(sample_static_image_bytes)
        assert isinstance(result_init, dict)
        assert result_init["message"] == "Baseline established"

        # Second call: motion detected (returns MotionAnalysisResult)
        result = plugin.analyze(sample_high_contrast_motion_image)

        # Must be MotionAnalysisResult
        assert isinstance(result, MotionAnalysisResult), (
            f"Expected MotionAnalysisResult, got {type(result).__name__}. "
        )

        # Must have required fields
        assert result.motion_detected is True
        assert result.motion_score > 0
        assert len(result.regions) > 0
        assert result.image_size is not None

    # Lifecycle hooks
    def test_on_load_lifecycle_hook(self, plugin):
        """Test on_load lifecycle hook executes without error."""
        plugin.on_load()
        assert plugin._frame_count == 0

    def test_on_unload_resets_state(
        self, plugin, sample_static_image_bytes
    ):
        """Test on_unload lifecycle hook resets state."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 2

        plugin.on_unload()

        assert plugin._frame_count == 0
        assert plugin._previous_frame is None

    def test_reset_clears_state(self, plugin, sample_static_image_bytes):
        """Test reset() clears all internal state."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 2

        plugin.reset()

        assert plugin._frame_count == 0
        assert plugin._previous_frame is None
        assert plugin._last_motion_time == 0
        assert plugin._motion_history == []

    # Image handling
    def test_image_size_captured_in_result(self, plugin, sample_static_image_bytes):
        """Test image dimensions are captured in result."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_static_image_bytes)
        
        assert isinstance(result, MotionAnalysisResult)
        assert result.image_size["width"] == 640
        assert result.image_size["height"] == 480

    def test_handles_grayscale_and_rgba_images(self, plugin):
        """Test grayscale and RGBA images are handled correctly."""
        # Grayscale
        gray_img = Image.new("L", (200, 150), color=128)
        gray_bytes = io.BytesIO()
        gray_img.save(gray_bytes, format="PNG")
        plugin.analyze(gray_bytes.getvalue())
        result = plugin.analyze(gray_bytes.getvalue())
        
        assert isinstance(result, MotionAnalysisResult)
        assert result.image_size["width"] == 200

        # RGBA
        plugin.reset()
        rgba_img = Image.new("RGBA", (300, 200), color=(255, 255, 255, 255))
        rgba_bytes = io.BytesIO()
        rgba_img.save(rgba_bytes, format="PNG")
        plugin.analyze(rgba_bytes.getvalue())
        result = plugin.analyze(rgba_bytes.getvalue())
        
        assert isinstance(result, MotionAnalysisResult)
        assert result.image_size["width"] == 300

    # Motion tracking
    def test_time_since_last_motion_none_initially(self, plugin, sample_static_image_bytes):
        """Test time_since_last_motion is None if no motion detected."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_static_image_bytes)
        
        assert isinstance(result, MotionAnalysisResult)
        assert result.time_since_last_motion is None

    def test_frame_counting(self, plugin, sample_static_image_bytes):
        """Test frame counter increments."""
        assert plugin._frame_count == 0
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 1
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 2

    def test_shape_mismatch_resets_baseline(self, plugin):
        """Test shape mismatch between frames resets baseline."""
        img1 = Image.new("RGB", (640, 480), color=128)
        img2 = Image.new("RGB", (800, 600), color=128)
        bytes1 = io.BytesIO()
        bytes2 = io.BytesIO()
        img1.save(bytes1, format="PNG")
        img2.save(bytes2, format="PNG")

        plugin.analyze(bytes1.getvalue())
        result = plugin.analyze(bytes2.getvalue())
        # Resets baseline -> returns dict
        assert isinstance(result, dict)
        assert result["motion_detected"] is False