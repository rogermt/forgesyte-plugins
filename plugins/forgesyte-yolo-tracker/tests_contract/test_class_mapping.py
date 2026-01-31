"""Test player detection class mapping."""


class TestClassMapping:
    """Tests for CLASS_NAMES mapping in player detection."""

    def test_class_names_match_trained_model(self) -> None:
        """Verify CLASS_NAMES matches trained model's 4-class structure."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            CLASS_NAMES

        expected = {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
        assert CLASS_NAMES == expected

    def test_class_names_has_all_classes(self) -> None:
        """Verify CLASS_NAMES includes all 4 class IDs."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            CLASS_NAMES

        assert 0 in CLASS_NAMES  # ball
        assert 1 in CLASS_NAMES  # goalkeeper
        assert 2 in CLASS_NAMES  # player
        assert 3 in CLASS_NAMES  # referee

    def test_team_colors_has_all_classes(self) -> None:
        """Verify TEAM_COLORS includes all 4 class IDs."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            TEAM_COLORS

        assert 0 in TEAM_COLORS
        assert 1 in TEAM_COLORS
        assert 2 in TEAM_COLORS
        assert 3 in TEAM_COLORS

    def test_team_colors_are_valid_hex(self) -> None:
        """Verify TEAM_COLORS are valid hex color strings."""
        from forgesyte_yolo_tracker.inference.player_detection import \
            TEAM_COLORS

        for class_id, color in TEAM_COLORS.items():
            assert isinstance(color, str)
            assert color.startswith("#")
            assert len(color) == 7  # #RRGGBB
