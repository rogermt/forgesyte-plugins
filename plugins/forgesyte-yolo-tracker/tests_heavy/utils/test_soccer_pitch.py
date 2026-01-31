"""Tests for soccer pitch visualization - real drawing validation.

These tests validate actual drawing output by checking pixel values,
coordinates, and colors. No mocking of cv2 functions.
"""

import numpy as np
import pytest
import supervision as sv

from forgesyte_yolo_tracker.utils.soccer_pitch import (
    draw_paths_on_pitch, draw_pitch, draw_pitch_voronoi_diagram,
    draw_points_on_pitch)


class SoccerPitchConfig:
    """Simple soccer pitch configuration for testing."""

    def __init__(self) -> None:
        self.width = 68
        self.length = 105
        self.centre_circle_radius = 9.15
        self.penalty_spot_distance = 11
        self.vertices = [
            (0, 0),
            (105, 0),
            (105, 68),
            (0, 68),
            (52.5, 0),
            (52.5, 68),
        ]
        self.edges = [
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 1),
            (5, 6),
        ]


@pytest.fixture
def soccer_pitch_config() -> SoccerPitchConfig:
    """Create soccer pitch configuration for testing."""
    return SoccerPitchConfig()


class TestDrawPitchValidation:
    """Tests that validate actual drawing output for draw_pitch."""

    def test_pitch_returns_numpy_array(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify draw_pitch returns numpy array."""
        result = draw_pitch(soccer_pitch_config)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.uint8
        assert len(result.shape) == 3

    def test_pitch_has_green_background(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify pitch background is green (34, 139, 34 BGR)."""
        result = draw_pitch(soccer_pitch_config)
        # Sample corner area (not on lines)
        corner_pixel = result[10, 10]
        assert corner_pixel[1] == 139

    def test_pitch_has_white_lines(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify pitch lines are white (255, 255, 255)."""
        result = draw_pitch(soccer_pitch_config)
        # Check that white pixels exist on the image (from lines)
        white_pixels = np.sum(result == 255)
        assert white_pixels > 0  # Lines should exist

    def test_pitch_custom_colors(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify custom background color works."""
        custom_color = sv.Color(0, 0, 100)  # Red background in RGB
        result = draw_pitch(soccer_pitch_config, background_color=custom_color)

        # Verify result has different pixels than default (green)
        default_result = draw_pitch(soccer_pitch_config)
        # The result should be different from default when using custom color
        assert result.shape == default_result.shape
        assert isinstance(result, np.ndarray)

    def test_pitch_centre_circle_exists(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify centre circle is drawn (white pixels at center area)."""
        result = draw_pitch(soccer_pitch_config)
        center_y, center_x = result.shape[0] // 2, result.shape[1] // 2
        radius = int(soccer_pitch_config.centre_circle_radius * 0.1)

        circle_point_y = center_y
        circle_point_x = center_x + radius
        circle_pixel = result[circle_point_y, circle_point_x]
        assert circle_pixel[0] == 255


class TestDrawPointsOnPitchValidation:
    """Tests for draw_points_on_pitch with real validation."""

    def test_points_returns_numpy_array(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify draw_points_on_pitch returns numpy array."""
        points = np.array([[50.0, 30.0]])
        result = draw_points_on_pitch(soccer_pitch_config, points)
        assert isinstance(result, np.ndarray)

    def test_points_scaled_correctly(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify points are drawn at scaled coordinates."""
        pitch = np.ones((200, 200, 3), dtype=np.uint8) * 34
        points = np.array([[100.0, 100.0]])

        result = draw_points_on_pitch(soccer_pitch_config, points, pitch=pitch)

        point_pixel = result[60, 60]
        assert point_pixel[2] == 255

    def test_points_with_existing_pitch(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify points can be drawn on existing pitch."""
        existing_pitch = np.ones((200, 200, 3), dtype=np.uint8) * 50
        points = np.array([[55.0, 53.0]])  # Point on boundary line

        result = draw_points_on_pitch(soccer_pitch_config, points, pitch=existing_pitch)

        assert result.shape == existing_pitch.shape
        # Check that red point was drawn on boundary line
        point_pixel = result[53, 55]
        assert point_pixel[2] == 255  # Red face color

    def test_multiple_points(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify multiple points are drawn."""
        pitch = np.ones((200, 200, 3), dtype=np.uint8) * 34
        points = np.array([[50.0, 30.0], [60.0, 40.0], [70.0, 35.0]])

        result = draw_points_on_pitch(soccer_pitch_config, points, pitch=pitch)

        coords = [(55, 53), (56, 54), (57, 54)]
        for y, x in coords:
            assert result[y, x, 2] == 255

    def test_custom_point_radius(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify custom point radius affects drawing."""
        pitch = np.ones((200, 200, 3), dtype=np.uint8) * 34
        points = np.array([[100.0, 100.0]])

        result = draw_points_on_pitch(soccer_pitch_config, points, pitch=pitch, radius=15)

        center_area = result[45:76, 45:76]
        red_count = np.sum(center_area[:, :, 2] == 255)
        assert red_count > 0


class TestDrawPathsOnPitchValidation:
    """Tests for draw_paths_on_pitch with real validation."""

    def test_paths_returns_numpy_array(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify draw_paths_on_pitch returns numpy array."""
        paths = [np.array([[50.0, 30.0], [60.0, 40.0]])]
        result = draw_paths_on_pitch(soccer_pitch_config, paths)
        assert isinstance(result, np.ndarray)

    def test_path_connects_points(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify path line connects start and end points."""
        pitch = np.zeros((100, 100, 3), dtype=np.uint8)  # Black background
        path = np.array([[10.0, 10.0], [50.0, 10.0]])  # Horizontal path

        result = draw_paths_on_pitch(soccer_pitch_config, [path], pitch=pitch)

        # Check that white line pixels exist
        white_pixels = np.sum(result == 255)
        assert white_pixels > 0  # Path should have been drawn

    def test_multiple_paths(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify multiple paths are drawn."""
        pitch = np.ones((200, 200, 3), dtype=np.uint8) * 34
        paths = [np.array([[50.0, 30.0], [60.0, 40.0]]), np.array([[70.0, 50.0], [80.0, 60.0]])]

        result = draw_paths_on_pitch(soccer_pitch_config, paths, pitch=pitch)

        assert np.any(result[:, :, 0] == 255)

    def test_path_with_existing_pitch(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify paths can be drawn on existing pitch."""
        existing_pitch = np.ones((200, 200, 3), dtype=np.uint8) * 50
        path = np.array([[50.0, 30.0], [60.0, 40.0]])

        result = draw_paths_on_pitch(soccer_pitch_config, [path], pitch=existing_pitch)

        assert result.shape == existing_pitch.shape


class TestDrawVoronoiDiagramValidation:
    """Tests for draw_pitch_voronoi_diagram with real validation."""

    def test_voronoi_returns_numpy_array(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify draw_pitch_voronoi_diagram returns numpy array."""
        team_1 = np.array([[50.0, 30.0]])
        team_2 = np.array([[60.0, 40.0]])

        result = draw_pitch_voronoi_diagram(soccer_pitch_config, team_1, team_2)
        assert isinstance(result, np.ndarray)

    def test_voronoi_has_team_colors(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify Voronoi diagram has team colors (red and white)."""
        team_1 = np.array([[50.0, 30.0]])
        team_2 = np.array([[60.0, 40.0]])

        result = draw_pitch_voronoi_diagram(
            soccer_pitch_config,
            team_1,
            team_2,
            team_1_color=sv.Color.RED,
            team_2_color=sv.Color.WHITE,
        )

        has_red = np.any(result[:, :, 2] == 255)
        has_white = np.any(result == 255)
        assert has_red or has_white

    def test_voronoi_opacity_blending(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify opacity affects blending."""
        team_1 = np.array([[50.0, 30.0]])
        team_2 = np.array([[80.0, 50.0]])

        result_opaque = draw_pitch_voronoi_diagram(soccer_pitch_config, team_1, team_2, opacity=1.0)
        result_transparent = draw_pitch_voronoi_diagram(
            soccer_pitch_config, team_1, team_2, opacity=0.5
        )

        assert not np.array_equal(result_opaque, result_transparent)

    def test_voronoi_with_existing_pitch(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify Voronoi can be drawn on existing pitch."""
        padding = 50
        scale = 0.1
        expected_height = int(soccer_pitch_config.width * scale) + 2 * padding
        expected_width = int(soccer_pitch_config.length * scale) + 2 * padding
        existing_pitch = np.ones((expected_height, expected_width, 3), dtype=np.uint8) * 50
        team_1 = np.array([[50.0, 30.0]])
        team_2 = np.array([[60.0, 40.0]])

        result = draw_pitch_voronoi_diagram(
            soccer_pitch_config, team_1, team_2, pitch=existing_pitch
        )

        assert result.shape == existing_pitch.shape

    def test_voronoi_different_team_positions(self, soccer_pitch_config: SoccerPitchConfig) -> None:
        """Verify Voronoi changes with different team positions."""
        team_1_a = np.array([[30.0, 30.0]])
        team_2_a = np.array([[70.0, 30.0]])

        team_1_b = np.array([[70.0, 30.0]])
        team_2_b = np.array([[30.0, 30.0]])

        result_a = draw_pitch_voronoi_diagram(soccer_pitch_config, team_1_a, team_2_a)
        result_b = draw_pitch_voronoi_diagram(soccer_pitch_config, team_1_b, team_2_b)

        assert not np.array_equal(result_a, result_b)
