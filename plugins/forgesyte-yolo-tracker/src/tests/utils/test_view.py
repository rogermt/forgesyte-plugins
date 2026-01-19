"""Tests for view transformation utilities."""

import numpy as np
import pytest

from forgesyte_yolo_tracker.utils.view import ViewTransformer


class TestViewTransformer:
    """Tests for ViewTransformer class."""

    def test_initialization_valid(self) -> None:
        """Test ViewTransformer initialization with valid inputs."""
        source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)

        target = np.array(
            [[10.0, 10.0], [110.0, 10.0], [10.0, 110.0], [110.0, 110.0]], dtype=np.float32
        )

        transformer = ViewTransformer(source, target)
        assert transformer.m is not None

    def test_initialization_mismatched_shapes(self) -> None:
        """Test initialization fails with mismatched shapes."""
        source = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        target = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="same shape"):
            ViewTransformer(source, target)

    def test_initialization_non_2d_coordinates(self) -> None:
        """Test initialization fails with non-2D coordinates."""
        source = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        target = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="2D coordinates"):
            ViewTransformer(source, target)

    def test_transform_points_basic(self) -> None:
        """Test basic point transformation."""
        source = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )

        target = np.array(
            [[0.0, 0.0], [200.0, 0.0], [0.0, 200.0], [200.0, 200.0]], dtype=np.float32
        )

        transformer = ViewTransformer(source, target)

        # Transform a point at (50, 50) - should scale to (100, 100)
        points = np.array([[50.0, 50.0]], dtype=np.float32)
        result = transformer.transform_points(points)

        assert result.shape == (1, 2)
        np.testing.assert_array_almost_equal(result[0], [100.0, 100.0], decimal=1)

    def test_transform_points_empty(self) -> None:
        """Test transform_points with empty array."""
        source = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )
        target = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )

        transformer = ViewTransformer(source, target)
        points = np.array([], dtype=np.float32).reshape(0, 2)

        result = transformer.transform_points(points)

        assert result.shape == (0, 2)

    def test_transform_points_multiple(self) -> None:
        """Test transforming multiple points."""
        source = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )

        target = np.array(
            [[0.0, 0.0], [200.0, 0.0], [0.0, 200.0], [200.0, 200.0]], dtype=np.float32
        )

        transformer = ViewTransformer(source, target)

        points = np.array([[25.0, 25.0], [50.0, 50.0], [75.0, 75.0]], dtype=np.float32)

        result = transformer.transform_points(points)

        assert result.shape == (3, 2)

    def test_transform_points_invalid_shape(self) -> None:
        """Test transform_points with invalid shape."""
        source = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )
        target = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )

        transformer = ViewTransformer(source, target)

        # 3D coordinates
        points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)

        with pytest.raises(ValueError, match="2D coordinates"):
            transformer.transform_points(points)

    def test_transform_image_basic(self) -> None:
        """Test basic image transformation."""
        source = np.array(
            [[0.0, 0.0], [640.0, 0.0], [0.0, 480.0], [640.0, 480.0]], dtype=np.float32
        )

        target = np.array(
            [[0.0, 0.0], [1280.0, 0.0], [0.0, 960.0], [1280.0, 960.0]], dtype=np.float32
        )

        transformer = ViewTransformer(source, target)

        # Create a test image
        image = np.ones((480, 640, 3), dtype=np.uint8) * 128

        result = transformer.transform_image(image, (1280, 960))

        assert result.shape == (960, 1280, 3)
        assert result.dtype == np.uint8

    def test_transform_image_grayscale(self) -> None:
        """Test image transformation with grayscale image."""
        source = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )
        target = np.array(
            [[0.0, 0.0], [200.0, 0.0], [0.0, 200.0], [200.0, 200.0]], dtype=np.float32
        )

        transformer = ViewTransformer(source, target)

        # Grayscale image
        image = np.ones((100, 100), dtype=np.uint8) * 128

        result = transformer.transform_image(image, (200, 200))

        assert result.shape == (200, 200)
        assert result.dtype == np.uint8

    def test_transform_image_invalid_shape(self) -> None:
        """Test transform_image with invalid image shape."""
        source = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )
        target = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )

        transformer = ViewTransformer(source, target)

        # 4D array (invalid)
        image = np.ones((10, 10, 3, 1), dtype=np.uint8)

        with pytest.raises(ValueError, match="grayscale or color"):
            transformer.transform_image(image, (10, 10))

    def test_transform_identity(self) -> None:
        """Test identity transformation."""
        # Identity: source == target
        source = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )

        target = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )

        transformer = ViewTransformer(source, target)

        points = np.array([[50.0, 50.0]], dtype=np.float32)
        result = transformer.transform_points(points)

        np.testing.assert_array_almost_equal(result[0], [50.0, 50.0], decimal=1)

    def test_transform_preserves_dtype(self) -> None:
        """Test that transform preserves image dtype."""
        source = np.array(
            [[0.0, 0.0], [100.0, 0.0], [0.0, 100.0], [100.0, 100.0]], dtype=np.float32
        )
        target = np.array(
            [[0.0, 0.0], [200.0, 0.0], [0.0, 200.0], [200.0, 200.0]], dtype=np.float32
        )

        transformer = ViewTransformer(source, target)

        # Create image with uint8
        image = np.ones((100, 100, 3), dtype=np.uint8) * 200

        result = transformer.transform_image(image, (200, 200))

        assert result.dtype == np.uint8
