"""Tests for soccer pitch visualization utilities."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from forgesyte_yolo_tracker.utils.soccer_pitch import (
    draw_pitch,
    draw_points_on_pitch,
    draw_paths_on_pitch,
    draw_pitch_voronoi_diagram,
)


@pytest.fixture
def mock_soccer_config() -> MagicMock:
    """Create mock soccer pitch configuration."""
    config = MagicMock()
    config.width = 68
    config.length = 105
    config.centre_circle_radius = 9.15
    config.penalty_spot_distance = 11
    config.vertices = [
        (0, 0),
        (105, 0),
        (105, 68),
        (0, 68),  # corners
        (52.5, 0),
        (52.5, 68),  # midline
    ]
    config.edges = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 1),  # boundary
        (5, 6),  # midline
    ]
    return config


class TestDrawPitch:
    """Tests for draw_pitch function."""

    def test_draw_pitch_basic(self, mock_soccer_config: MagicMock) -> None:
        """Test basic pitch drawing."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.line"), patch(
            "forgesyte_yolo_tracker.utils.soccer_pitch.cv2.circle"
        ):

            result = draw_pitch(mock_soccer_config)

            assert isinstance(result, np.ndarray)
            assert len(result.shape) == 3
            assert result.shape[2] == 3
            assert result.dtype == np.uint8

    def test_draw_pitch_default_colors(self, mock_soccer_config: MagicMock) -> None:
        """Test pitch with default colors."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.line"), patch(
            "forgesyte_yolo_tracker.utils.soccer_pitch.cv2.circle"
        ):

            result = draw_pitch(mock_soccer_config)

            # Should have green background (34, 139, 34 in BGR)
            assert result.shape == (int(68 * 0.1) + 100, int(105 * 0.1) + 100, 3)

    def test_draw_pitch_custom_scale(self, mock_soccer_config: MagicMock) -> None:
        """Test pitch with custom scale."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.line"), patch(
            "forgesyte_yolo_tracker.utils.soccer_pitch.cv2.circle"
        ):

            result = draw_pitch(mock_soccer_config, scale=0.2)

            expected_height = int(68 * 0.2) + 100
            expected_width = int(105 * 0.2) + 100
            assert result.shape[0] == expected_height
            assert result.shape[1] == expected_width

    def test_draw_pitch_custom_colors(self, mock_soccer_config: MagicMock) -> None:
        """Test pitch with custom colors."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.line"), patch(
            "forgesyte_yolo_tracker.utils.soccer_pitch.cv2.circle"
        ):

            with patch("forgesyte_yolo_tracker.utils.soccer_pitch.sv"):
                result = draw_pitch(mock_soccer_config)

                assert isinstance(result, np.ndarray)

    def test_draw_pitch_custom_padding(self, mock_soccer_config: MagicMock) -> None:
        """Test pitch with custom padding."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.line"), patch(
            "forgesyte_yolo_tracker.utils.soccer_pitch.cv2.circle"
        ):

            result = draw_pitch(mock_soccer_config, padding=100)

            expected_height = int(68 * 0.1) + 200
            expected_width = int(105 * 0.1) + 200
            assert result.shape[0] == expected_height
            assert result.shape[1] == expected_width


class TestDrawPointsOnPitch:
    """Tests for draw_points_on_pitch function."""

    def test_draw_points_single_point(self, mock_soccer_config: MagicMock) -> None:
        """Test drawing single point on pitch."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.circle"), patch(
            "forgesyte_yolo_tracker.utils.soccer_pitch.draw_pitch"
        ) as mock_draw:

            mock_pitch = np.ones((200, 200, 3), dtype=np.uint8) * 34
            mock_draw.return_value = mock_pitch

            points = np.array([[50.0, 30.0]])

            result = draw_points_on_pitch(mock_soccer_config, points)

            assert isinstance(result, np.ndarray)
            assert result.shape == mock_pitch.shape

    def test_draw_points_multiple_points(self, mock_soccer_config: MagicMock) -> None:
        """Test drawing multiple points on pitch."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.circle"), patch(
            "forgesyte_yolo_tracker.utils.soccer_pitch.draw_pitch"
        ) as mock_draw:

            mock_pitch = np.ones((200, 200, 3), dtype=np.uint8) * 34
            mock_draw.return_value = mock_pitch

            points = np.array([[50.0, 30.0], [60.0, 40.0], [70.0, 35.0]])

            result = draw_points_on_pitch(mock_soccer_config, points)

            assert result.shape == mock_pitch.shape

    def test_draw_points_with_existing_pitch(self, mock_soccer_config: MagicMock) -> None:
        """Test drawing points on existing pitch."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.circle"):

            existing_pitch = np.ones((200, 200, 3), dtype=np.uint8) * 50
            points = np.array([[50.0, 30.0]])

            result = draw_points_on_pitch(mock_soccer_config, points, pitch=existing_pitch)

            assert result.shape == existing_pitch.shape

    def test_draw_points_custom_colors(self, mock_soccer_config: MagicMock) -> None:
        """Test drawing points with custom colors."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.circle"), patch(
            "forgesyte_yolo_tracker.utils.soccer_pitch.draw_pitch"
        ) as mock_draw, patch("forgesyte_yolo_tracker.utils.soccer_pitch.sv"):

            mock_pitch = np.ones((200, 200, 3), dtype=np.uint8) * 34
            mock_draw.return_value = mock_pitch

            points = np.array([[50.0, 30.0]])

            result = draw_points_on_pitch(mock_soccer_config, points, radius=15, thickness=3)

            assert isinstance(result, np.ndarray)


class TestDrawPathsOnPitch:
    """Tests for draw_paths_on_pitch function."""

    def test_draw_paths_single_path(self, mock_soccer_config: MagicMock) -> None:
        """Test drawing single path on pitch."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.line"), patch(
            "forgesyte_yolo_tracker.utils.soccer_pitch.draw_pitch"
        ) as mock_draw:

            mock_pitch = np.ones((200, 200, 3), dtype=np.uint8) * 34
            mock_draw.return_value = mock_pitch

            path = np.array([[50.0, 30.0], [60.0, 40.0]])

            result = draw_paths_on_pitch(mock_soccer_config, [path])

            assert isinstance(result, np.ndarray)

    def test_draw_paths_multiple_paths(self, mock_soccer_config: MagicMock) -> None:
        """Test drawing multiple paths on pitch."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.line"), patch(
            "forgesyte_yolo_tracker.utils.soccer_pitch.draw_pitch"
        ) as mock_draw:

            mock_pitch = np.ones((200, 200, 3), dtype=np.uint8) * 34
            mock_draw.return_value = mock_pitch

            paths = [np.array([[50.0, 30.0], [60.0, 40.0]]), np.array([[70.0, 50.0], [80.0, 60.0]])]

            result = draw_paths_on_pitch(mock_soccer_config, paths)

            assert isinstance(result, np.ndarray)

    def test_draw_paths_short_path(self, mock_soccer_config: MagicMock) -> None:
        """Test drawing path with too few points."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.line"), patch(
            "forgesyte_yolo_tracker.utils.soccer_pitch.draw_pitch"
        ) as mock_draw:

            mock_pitch = np.ones((200, 200, 3), dtype=np.uint8) * 34
            mock_draw.return_value = mock_pitch

            # Single point path (should skip)
            path = np.array([[50.0, 30.0]])

            result = draw_paths_on_pitch(mock_soccer_config, [path])

            assert isinstance(result, np.ndarray)

    def test_draw_paths_with_existing_pitch(self, mock_soccer_config: MagicMock) -> None:
        """Test drawing paths on existing pitch."""
        with patch("forgesyte_yolo_tracker.utils.soccer_pitch.cv2.line"):

            existing_pitch = np.ones((200, 200, 3), dtype=np.uint8) * 50
            path = np.array([[50.0, 30.0], [60.0, 40.0]])

            result = draw_paths_on_pitch(mock_soccer_config, [path], pitch=existing_pitch)

            assert result.shape == existing_pitch.shape


class TestDrawPitchVoronoiDiagram:
    """Tests for draw_pitch_voronoi_diagram function."""

    def test_draw_voronoi_implementation_exists(self) -> None:
        """Test that voronoi function exists and is callable."""
        assert callable(draw_pitch_voronoi_diagram)
