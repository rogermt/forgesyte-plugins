"""Tests for soccer pitch configuration."""

import pytest
import numpy as np


class TestSoccerPitchConfiguration:
    """Tests for SoccerPitchConfiguration."""

    def test_config_has_vertices(self) -> None:
        """Verify config has vertices attribute."""
        from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

        config = SoccerPitchConfiguration()
        assert hasattr(config, "vertices")

    def test_config_has_edges(self) -> None:
        """Verify config has edges attribute."""
        from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

        config = SoccerPitchConfiguration()
        assert hasattr(config, "edges")

    def test_config_has_dimensions(self) -> None:
        """Verify config has width and length."""
        from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

        config = SoccerPitchConfiguration()
        assert hasattr(config, "width")
        assert hasattr(config, "length")
        assert config.width == 7000  # cm
        assert config.length == 12000  # cm

    def test_vertices_count(self) -> None:
        """Verify we have expected number of keypoints."""
        from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

        config = SoccerPitchConfiguration()
        assert len(config.vertices) >= 14  # Standard pitch has 14+ keypoints

    def test_edges_form_complete_graph(self) -> None:
        """Verify edges connect vertices properly."""
        from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

        config = SoccerPitchConfiguration()
        assert len(config.edges) > 0
        # Each edge should be a tuple of two indices
        for edge in config.edges:
            assert len(edge) == 2
            assert isinstance(edge[0], int)
            assert isinstance(edge[1], int)

    def test_vertices_are_tuples(self) -> None:
        """Verify vertices are (x, y) coordinate tuples."""
        from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

        config = SoccerPitchConfiguration()
        for vertex in config.vertices:
            assert isinstance(vertex, tuple)
            assert len(vertex) == 2
            assert isinstance(vertex[0], (int, float))
            assert isinstance(vertex[1], (int, float))

    def test_pitch_aspect_ratio(self) -> None:
        """Verify pitch has correct 2:1 aspect ratio."""
        from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

        config = SoccerPitchConfiguration()
        # length:width should be approximately 2:1
        assert config.length / config.width == pytest.approx(2.0, rel=0.01)
