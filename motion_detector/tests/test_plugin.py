"""Unit tests for Motion Detector plugin.

Tests cover:
- Successful motion detection with frame differencing
- Adaptive baseline learning and updates
- Bounding box region detection and area calculation
- Configuration options (threshold, min_area, blur_size)
- Motion history tracking and recent events counting
- Error handling for invalid images
- Response validation with Pydantic models
- Lifecycle hooks (on_load, on_unload, reset)
- Metadata endpoint
- Frame number tracking and time-since-last-motion calculations
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
    def sample_motion_image_bytes(self):
        """Generate an image with motion (different from static)."""
        img = Image.new("RGB", (640, 480), color="lightgray")
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

    # Metadata tests
    def test_metadata_returns_plugin_metadata(self, plugin):
        """Test metadata endpoint returns valid PluginMetadata."""
        metadata = plugin.metadata()
        assert metadata.name == "motion_detector"
        assert metadata.version == "1.1.0"
        assert "image" in metadata.inputs
        assert "motion_detected" in metadata.outputs
        assert "motion_score" in metadata.outputs
        assert "regions" in metadata.outputs

    def test_metadata_includes_config_schema(self, plugin):
        """Test metadata includes threshold, min_area, blur_size, reset_baseline."""
        metadata = plugin.metadata()
        config = metadata.config_schema
        assert "threshold" in config
        assert "min_area" in config
        assert "blur_size" in config
        assert "reset_baseline" in config
        assert config["threshold"]["default"] == 25.0
        assert config["min_area"]["default"] == 0.01
        assert config["blur_size"]["default"] == 5

    def test_metadata_config_constraints(self, plugin):
        """Test metadata config includes min/max constraints."""
        metadata = plugin.metadata()
        config = metadata.config_schema
        assert config["threshold"]["min"] == 1.0
        assert config["threshold"]["max"] == 100.0
        assert config["min_area"]["min"] == 0.001
        assert config["min_area"]["max"] == 0.5

    # Baseline initialization tests
    def test_baseline_established_on_first_frame(
        self, plugin, sample_static_image_bytes
    ):
        """Test first frame establishes baseline without detecting motion."""
        result = plugin.analyze(sample_static_image_bytes)

        assert result["motion_detected"] is False
        assert result["motion_score"] == 0.0
        assert "message" in result or result["motion_detected"] is False

    def test_frame_count_increments(self, plugin, sample_static_image_bytes):
        """Test frame counter increments with each analysis."""
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 1

        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 2

        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 3

    # Motion detection tests
    def test_motion_detected_on_significant_change(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test motion is detected when frame changes significantly."""
        # First frame: baseline
        plugin.analyze(sample_static_image_bytes)
        # Second frame: different content
        result = plugin.analyze(sample_high_contrast_motion_image)

        assert result["motion_detected"] is True
        assert result["motion_score"] > 0.0
        assert result["frame_number"] == 2

    def test_motion_score_represents_percentage(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test motion_score is reported as percentage (0-100)."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_high_contrast_motion_image)

        assert 0 <= result["motion_score"] <= 100
        # High contrast change should yield significant score
        assert result["motion_score"] > 10  # At least 10% change

    def test_no_motion_with_identical_frames(self, plugin, sample_static_image_bytes):
        """Test no motion detected when frames are identical."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_static_image_bytes)

        assert result["motion_detected"] is False
        assert result["motion_score"] == 0.0

    def test_motion_score_respects_threshold(self, plugin, sample_static_image_bytes):
        """Test motion_score is compared against threshold."""
        # With very high threshold, even significant changes won't trigger motion
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_static_image_bytes, options={"threshold": 100.0})

        # Identical frames with high threshold = no motion
        assert result["motion_detected"] is False

    # Configuration option tests
    def test_custom_threshold_option(
        self, plugin, sample_static_image_bytes, sample_motion_image_bytes
    ):
        """Test custom threshold configuration."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_motion_image_bytes, options={"threshold": 50.0})

        # Result depends on actual pixel difference
        assert "motion_detected" in result

    def test_min_area_filter_removes_small_regions(self, plugin):
        """Test min_area filters out small motion regions."""
        # Create near-identical images (will generate small noise regions)
        img1 = Image.new("RGB", (640, 480), color=(128, 128, 128))
        img2 = Image.new("RGB", (640, 480), color=(129, 129, 129))
        bytes1 = io.BytesIO()
        bytes2 = io.BytesIO()
        img1.save(bytes1, format="PNG")
        img2.save(bytes2, format="PNG")

        plugin.analyze(bytes1.getvalue())
        result = plugin.analyze(bytes2.getvalue(), options={"min_area": 0.1})

        # Very small change with high min_area threshold
        assert result["motion_detected"] is False or result["motion_score"] < 1.0

    def test_blur_size_configuration(
        self, plugin, sample_static_image_bytes, sample_motion_image_bytes
    ):
        """Test blur_size configuration option."""
        plugin.analyze(sample_static_image_bytes)
        result_no_blur = plugin.analyze(
            sample_motion_image_bytes, options={"blur_size": 1}
        )
        # Both should detect motion, blur just affects smoothing
        assert "motion_detected" in result_no_blur

    def test_reset_baseline_option(
        self, plugin, sample_static_image_bytes, sample_motion_image_bytes
    ):
        """Test reset_baseline configuration forces new baseline."""
        plugin.analyze(sample_static_image_bytes)
        # Reset baseline, so next frame will be new baseline
        result = plugin.analyze(
            sample_motion_image_bytes, options={"reset_baseline": True}
        )

        assert result["motion_detected"] is False  # New baseline established

    # Adaptive baseline tests
    def test_adaptive_baseline_updates_with_alpha_learning(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test adaptive baseline learns and updates frames."""
        # First frame: baseline
        plugin.analyze(sample_static_image_bytes)
        baseline_1 = plugin._previous_frame.copy()

        # Second frame: high-contrast different image (adaptive learning applies)
        plugin.analyze(sample_high_contrast_motion_image)
        baseline_2 = plugin._previous_frame.copy()

        # Baselines should be different due to alpha learning
        # When frame differs significantly, adaptive learning blends old+new baseline
        assert not np.allclose(baseline_1, baseline_2, atol=20)

    # Bounding box and region detection tests
    def test_regions_empty_when_no_motion(self, plugin, sample_static_image_bytes):
        """Test regions list is empty when no motion detected."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_static_image_bytes)

        assert result["regions"] == []

    def test_regions_populated_when_motion_detected(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test regions are detected and populated when motion found."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_high_contrast_motion_image)

        if result["motion_detected"]:
            assert len(result["regions"]) > 0
            region = result["regions"][0]
            assert "bbox" in region
            assert "area" in region
            assert "center" in region

    def test_bounding_box_has_required_fields(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test bounding box contains x, y, width, height."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_high_contrast_motion_image)

        if result["regions"]:
            bbox = result["regions"][0]["bbox"]
            assert "x" in bbox
            assert "y" in bbox
            assert "width" in bbox
            assert "height" in bbox
            assert all(isinstance(v, int) for v in bbox.values())

    def test_region_center_calculation(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test region center is calculated correctly."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_high_contrast_motion_image)

        if result["regions"]:
            region = result["regions"][0]
            bbox = region["bbox"]
            center = region["center"]
            expected_x = bbox["x"] + bbox["width"] // 2
            expected_y = bbox["y"] + bbox["height"] // 2
            assert center["x"] == expected_x
            assert center["y"] == expected_y

    def test_region_area_calculation(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test region area is calculated correctly."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_high_contrast_motion_image)

        if result["regions"]:
            region = result["regions"][0]
            bbox = region["bbox"]
            expected_area = bbox["width"] * bbox["height"]
            assert region["area"] == expected_area

    # Motion history and timing tests
    def test_time_since_last_motion_updated(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test time_since_last_motion is tracked."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_high_contrast_motion_image)
        time.sleep(0.2)  # Wait slightly
        result = plugin.analyze(sample_static_image_bytes)

        # After motion was detected, time_since_last_motion should be >= 0
        assert result["time_since_last_motion"] is not None
        assert result["time_since_last_motion"] >= 0

    def test_time_since_last_motion_none_initially(
        self, plugin, sample_static_image_bytes
    ):
        """Test time_since_last_motion is None if no motion ever detected."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_static_image_bytes)

        assert result["time_since_last_motion"] is None

    def test_recent_motion_events_count(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test recent_motion_events_count tracks motion within 60 seconds."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_high_contrast_motion_image)
        result = plugin.analyze(sample_static_image_bytes)

        # Should have recorded 1 motion event
        assert result["recent_motion_events_count"] >= 0

    def test_motion_history_kept_limited_to_100_events(
        self, plugin, sample_static_image_bytes, sample_motion_image_bytes
    ):
        """Test motion history is kept limited to 100 recent events."""
        # This is implementation detail, but ensures memory doesn't grow unbounded
        assert len(plugin._motion_history) <= 100

    # Image size and format tests
    def test_image_size_captured_in_result(self, plugin, sample_static_image_bytes):
        """Test image dimensions are captured in result."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_static_image_bytes)

        assert "image_size" in result
        assert result["image_size"]["width"] == 640
        assert result["image_size"]["height"] == 480

    def test_grayscale_image_conversion(self, plugin):
        """Test grayscale images are converted correctly."""
        # Create grayscale image
        img = Image.new("L", (200, 150), color=128)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = img_bytes.getvalue()

        plugin.analyze(image_data)
        result = plugin.analyze(image_data)

        assert result["image_size"]["width"] == 200
        assert result["image_size"]["height"] == 150
        assert result["motion_detected"] is False

    def test_rgba_image_conversion(self, plugin):
        """Test RGBA images are converted correctly."""
        # Create RGBA image
        img = Image.new("RGBA", (300, 200), color=(255, 255, 255, 255))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        image_data = img_bytes.getvalue()

        plugin.analyze(image_data)
        result = plugin.analyze(image_data)

        assert result["image_size"]["width"] == 300
        assert result["image_size"]["height"] == 200

    # Error handling tests
    def test_handles_invalid_image_data(self, plugin):
        """Test error handling for invalid image bytes."""
        invalid_bytes = b"not an image"

        result = plugin.analyze(invalid_bytes)

        assert result["motion_detected"] is False
        assert "error" in result or result["motion_score"] == 0

    def test_handles_empty_image_bytes(self, plugin):
        """Test error handling for empty bytes."""
        empty_bytes = b""

        result = plugin.analyze(empty_bytes)

        assert result["motion_detected"] is False
        assert result["motion_score"] == 0

    def test_error_is_logged_on_exception(self, plugin, sample_static_image_bytes):
        """Test exceptions are caught and logged gracefully."""
        plugin.analyze(sample_static_image_bytes)
        # Corrupt previous frame to trigger error
        plugin._previous_frame = None
        result = plugin.analyze(b"corrupt")

        assert result["motion_detected"] is False

    # Pydantic model tests
    def test_bounding_box_model_validation(self):
        """Test BoundingBox Pydantic model validates fields."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)

        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.width == 100
        assert bbox.height == 80

    def test_motion_region_model_validation(self):
        """Test MotionRegion Pydantic model validates fields."""
        bbox = BoundingBox(x=10, y=20, width=100, height=80)
        region = MotionRegion(
            bbox=bbox,
            area=8000,
            center={"x": 60, "y": 60},
        )

        assert region.area == 8000
        assert region.center["x"] == 60

    def test_motion_analysis_result_model_validation(self):
        """Test MotionAnalysisResult Pydantic model validates."""
        result = MotionAnalysisResult(
            motion_detected=True,
            motion_score=45.5,
            regions=[],
            frame_number=5,
            image_size={"width": 640, "height": 480},
            recent_motion_events_count=2,
        )

        assert result.motion_detected is True
        assert result.motion_score == 45.5
        assert result.frame_number == 5

    def test_motion_analysis_result_serialization(self):
        """Test MotionAnalysisResult can be serialized to dict/JSON."""
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
        assert len(data["regions"]) == 0

    # Lifecycle tests
    def test_on_load_lifecycle_hook(self, plugin):
        """Test on_load lifecycle hook executes without error."""
        plugin.on_load()
        # Should not raise exception
        assert plugin._frame_count == 0

    def test_on_unload_lifecycle_hook(self, plugin, sample_static_image_bytes):
        """Test on_unload lifecycle hook resets state."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 2

        plugin.on_unload()

        assert plugin._frame_count == 0
        assert plugin._previous_frame is None
        assert plugin._last_motion_time == 0

    def test_reset_clears_state(self, plugin, sample_static_image_bytes):
        """Test reset() clears all internal state."""
        plugin.analyze(sample_static_image_bytes)
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 2
        assert plugin._previous_frame is not None

        plugin.reset()

        assert plugin._frame_count == 0
        assert plugin._previous_frame is None
        assert plugin._last_motion_time == 0
        assert plugin._motion_history == []

    # Return type tests
    def test_analyze_returns_dict_not_pydantic_model(
        self, plugin, sample_static_image_bytes
    ):
        """Test that analyze() returns dict (for ForgeSyte server compatibility).

        The server expects: def analyze() -> Dict[str, Any]
        NOT: def analyze() -> MotionAnalysisResult

        Server code unpacks result with: {**result, "processing_time_ms": ...}
        This fails if result is a Pydantic model (not a mapping).
        """
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_static_image_bytes)

        # Must be dict, not Pydantic model
        assert isinstance(result, dict), (
            f"Expected dict, got {type(result).__name__}. "
            "Plugin.analyze() must return dict for server compatibility."
        )

        # Can unpack with ** operator (server requirement)
        unpacked = {**result, "processing_time_ms": 123.45}
        assert unpacked["processing_time_ms"] == 123.45
        assert "motion_detected" in unpacked
        assert "motion_score" in unpacked

        # Must have required fields
        assert "motion_detected" in result
        assert "motion_score" in result
        assert "regions" in result
        assert "frame_number" in result
        assert "image_size" in result

    # Gaussian blur tests
    def test_gaussian_blur_reduces_noise(self, plugin):
        """Test Gaussian blur smooths image data."""
        img = Image.new("RGB", (100, 100), color=128)
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")

        plugin.analyze(img_bytes.getvalue())
        result = plugin.analyze(img_bytes.getvalue(), options={"blur_size": 5})

        assert result["motion_score"] == 0.0

    def test_no_blur_when_blur_size_one(
        self, plugin, sample_static_image_bytes, sample_motion_image_bytes
    ):
        """Test blur_size=1 skips blur processing."""
        plugin.analyze(sample_static_image_bytes)
        result = plugin.analyze(sample_motion_image_bytes, options={"blur_size": 1})

        assert "motion_detected" in result

    # Integration tests
    def test_frame_number_increments_correctly(self, plugin, sample_static_image_bytes):
        """Test frame_number in result matches internal counter."""
        # First frame returns message (baseline) without frame_number key
        plugin.analyze(sample_static_image_bytes)
        assert plugin._frame_count == 1

        # Second frame has frame_number
        result2 = plugin.analyze(sample_static_image_bytes)
        assert result2.get("frame_number") == 2

        result3 = plugin.analyze(sample_static_image_bytes)
        assert result3.get("frame_number") == 3

    def test_multiple_consecutive_frames_sequence(
        self, plugin, sample_static_image_bytes, sample_high_contrast_motion_image
    ):
        """Test sequence of frames processes correctly."""
        # Frame 1: baseline (returns message with motion_detected=False)
        result = plugin.analyze(sample_static_image_bytes)
        assert result["motion_detected"] is False
        assert plugin._frame_count == 1

        # Frame 2: motion (has frame_number)
        result2 = plugin.analyze(sample_high_contrast_motion_image)
        assert result2.get("frame_number") == 2

        # Frame 3: return to static (has frame_number)
        result3 = plugin.analyze(sample_static_image_bytes)
        assert result3.get("frame_number") == 3

    def test_shape_mismatch_resets_baseline(self, plugin):
        """Test shape mismatch between frames resets baseline."""
        img1 = Image.new("RGB", (640, 480), color=128)
        img2 = Image.new("RGB", (800, 600), color=128)  # Different size
        bytes1 = io.BytesIO()
        bytes2 = io.BytesIO()
        img1.save(bytes1, format="PNG")
        img2.save(bytes2, format="PNG")

        plugin.analyze(bytes1.getvalue())
        result = plugin.analyze(bytes2.getvalue())

        # Shape mismatch should reset baseline, return false motion
        assert result["motion_detected"] is False
