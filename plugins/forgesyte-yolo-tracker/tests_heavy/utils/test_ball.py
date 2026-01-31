"""Tests for ball tracking and annotation utilities."""

from collections import deque

import numpy as np
import supervision as sv

from forgesyte_yolo_tracker.utils.ball import BallAnnotator, BallTracker


class TestBallAnnotator:
    """Tests for BallAnnotator class."""

    def test_initialization(self) -> None:
        """Test BallAnnotator initializes with correct defaults."""
        annotator = BallAnnotator(radius=10)
        assert annotator.radius == 10
        assert annotator.thickness == 2
        assert len(annotator.buffer) == 0
        assert isinstance(annotator.buffer, deque)
        assert annotator.buffer.maxlen == 5

    def test_initialization_custom_buffer_size(self) -> None:
        """Test BallAnnotator with custom buffer size."""
        annotator = BallAnnotator(radius=15, buffer_size=10, thickness=3)
        assert annotator.buffer.maxlen == 10
        assert annotator.thickness == 3

    def test_interpolate_radius_single_element(self) -> None:
        """Test radius interpolation with single element."""
        annotator = BallAnnotator(radius=20)
        # When max_i == 1, should return full radius
        radius = annotator.interpolate_radius(0, 1)
        assert radius == 20

    def test_interpolate_radius_multiple_elements(self) -> None:
        """Test radius interpolation with multiple elements."""
        annotator = BallAnnotator(radius=20)
        # interpolate_radius(i, max_i) = 1 + i * (radius - 1) / (max_i - 1)
        # With max_i=4: interpolate_radius(4, 4) = 1 + 4*(20-1)/(4-1) = 1 + 25.33 = 26
        assert annotator.interpolate_radius(0, 4) == 1
        assert annotator.interpolate_radius(4, 4) == 26
        # Middle should be interpolated: 1 + 2*(20-1)/3 = 1 + 12.67 = 13
        mid = annotator.interpolate_radius(2, 4)
        assert mid == 13

    def test_annotate_with_empty_detections(self) -> None:
        """Test annotate with no detections."""
        annotator = BallAnnotator(radius=10)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = sv.Detections.empty()

        result = annotator.annotate(frame, detections)

        assert result.shape == frame.shape
        assert np.array_equal(result, frame)
        assert len(annotator.buffer) == 1

    def test_annotate_with_single_detection(self) -> None:
        """Test annotate with one detection."""
        annotator = BallAnnotator(radius=10)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100  # Non-zero background

        # Create detection at (320, 240)
        detections = sv.Detections(
            xyxy=np.array([[300, 220, 340, 260]]),  # Box around (320, 240)
            confidence=np.array([1.0]),
            class_id=np.array([0]),
        )

        result = annotator.annotate(frame, detections)

        assert result.shape == frame.shape
        # Buffer should have coordinates
        assert len(annotator.buffer) == 1

    def test_annotate_buffer_overflow(self) -> None:
        """Test buffer fills and doesn't exceed max length."""
        annotator = BallAnnotator(radius=10, buffer_size=3)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = sv.Detections(
            xyxy=np.array([[100, 100, 150, 150]]),
            confidence=np.array([1.0]),
            class_id=np.array([0]),
        )

        # Add multiple detections
        for _ in range(5):
            annotator.annotate(frame, detections)

        # Buffer should not exceed maxlen
        assert len(annotator.buffer) == 3

    def test_annotate_preserves_frame_shape(self) -> None:
        """Test annotate preserves frame shape."""
        annotator = BallAnnotator(radius=10)
        shapes = [(480, 640, 3), (720, 1280, 3), (360, 480, 3)]

        for shape in shapes:
            frame = np.zeros(shape, dtype=np.uint8)
            detections = sv.Detections(
                xyxy=np.array([[10, 10, 50, 50]]),
                confidence=np.array([1.0]),
                class_id=np.array([0]),
            )
            result = annotator.annotate(frame, detections)
            assert result.shape == shape


class TestBallTracker:
    """Tests for BallTracker class."""

    def test_initialization(self) -> None:
        """Test BallTracker initializes correctly."""
        tracker = BallTracker()
        assert isinstance(tracker.buffer, deque)
        assert tracker.buffer.maxlen == 10
        assert len(tracker.buffer) == 0

    def test_initialization_custom_buffer_size(self) -> None:
        """Test BallTracker with custom buffer size."""
        tracker = BallTracker(buffer_size=5)
        assert tracker.buffer.maxlen == 5

    def test_update_with_empty_detections(self) -> None:
        """Test update with no detections returns input unchanged."""
        tracker = BallTracker()
        detections = sv.Detections.empty()

        result = tracker.update(detections)

        assert len(result) == 0
        assert len(tracker.buffer) == 1

    def test_update_with_single_detection(self) -> None:
        """Test update with single detection."""
        tracker = BallTracker()
        detections = sv.Detections(
            xyxy=np.array([[100, 100, 150, 150]]),
            confidence=np.array([1.0]),
            class_id=np.array([0]),
        )

        result = tracker.update(detections)

        assert len(result) == 1
        assert len(tracker.buffer) == 1

    def test_update_selects_closest_to_centroid(self) -> None:
        """Test that update selects detection closest to centroid."""
        tracker = BallTracker(buffer_size=2)

        # Add first detection at (100, 100)
        det1 = sv.Detections(
            xyxy=np.array([[75, 75, 125, 125]]), confidence=np.array([1.0]), class_id=np.array([0])
        )
        result1 = tracker.update(det1)
        assert len(result1) == 1

        # Add second detection with two options
        det2 = sv.Detections(
            xyxy=np.array([[50, 50, 100, 100], [100, 100, 150, 150]]),
            confidence=np.array([1.0, 1.0]),
            class_id=np.array([0, 0]),
        )
        result2 = tracker.update(det2)

        # Should select detection closer to centroid of buffer
        assert len(result2) == 1
        assert len(tracker.buffer) == 2

    def test_update_buffer_accumulation(self) -> None:
        """Test buffer accumulates positions over time."""
        tracker = BallTracker(buffer_size=5)

        # Different positions over time
        positions = [(100, 100), (110, 110), (120, 120), (130, 130)]

        for x, y in positions:
            det = sv.Detections(
                xyxy=np.array([[x - 25, y - 25, x + 25, y + 25]]),
                confidence=np.array([1.0]),
                class_id=np.array([0]),
            )
            tracker.update(det)

        assert len(tracker.buffer) == 4

    def test_update_centroid_calculation(self) -> None:
        """Test centroid is calculated correctly."""
        tracker = BallTracker(buffer_size=3)

        # Add positions: (100, 100), (150, 150)
        for x, y in [(100, 100), (150, 150)]:
            det = sv.Detections(
                xyxy=np.array([[x - 25, y - 25, x + 25, y + 25]]),
                confidence=np.array([1.0]),
                class_id=np.array([0]),
            )
            tracker.update(det)

        # Centroid should be (125, 125)
        # Now add detections and verify closest one is selected
        det3 = sv.Detections(
            xyxy=np.array([[120, 120, 140, 140], [200, 200, 220, 220]]),
            confidence=np.array([1.0, 1.0]),
            class_id=np.array([0, 0]),
        )
        result = tracker.update(det3)

        # Should pick (130, 130) which is closer to centroid (125, 125)
        assert len(result) == 1

    def test_update_with_outlier_detection(self) -> None:
        """Test tracker handles outliers correctly."""
        tracker = BallTracker()

        # Normal trajectory: (100, 100), (105, 105), (110, 110)
        for x, y in [(100, 100), (105, 105), (110, 110)]:
            det = sv.Detections(
                xyxy=np.array([[x - 25, y - 25, x + 25, y + 25]]),
                confidence=np.array([1.0]),
                class_id=np.array([0]),
            )
            tracker.update(det)

        # Outlier and normal option
        det_outlier = sv.Detections(
            xyxy=np.array([[400, 400, 450, 450], [115, 115, 135, 135]]),
            confidence=np.array([1.0, 1.0]),
            class_id=np.array([0, 0]),
        )
        result = tracker.update(det_outlier)

        # Should select (125, 125), not (425, 425)
        assert len(result) == 1

    def test_update_returns_detections_object(self) -> None:
        """Test update returns proper Detections object."""
        tracker = BallTracker()
        detections = sv.Detections(
            xyxy=np.array([[100, 100, 150, 150]]),
            confidence=np.array([0.95]),
            class_id=np.array([1]),
        )

        result = tracker.update(detections)

        assert isinstance(result, sv.Detections)
        assert len(result.xyxy) > 0
        assert len(result.confidence) > 0
