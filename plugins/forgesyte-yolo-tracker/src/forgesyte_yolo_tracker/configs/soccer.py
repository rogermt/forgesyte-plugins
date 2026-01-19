"""Soccer pitch configuration for YOLO Tracker.

Provides pitch geometry, keypoint names, and coordinate transforms
for radar mapping and pitch detection.
"""

from typing import Dict, List, Tuple


class SoccerPitchConfiguration:
    """Configuration for soccer pitch geometry.

    Based on standard FIFA pitch dimensions scaled to centimeters.
    Provides vertices for keypoints and edges connecting them.
    """

    def __init__(self) -> None:
        """Initialize pitch configuration.

        Pitch dimensions:
        - Length: 12000 cm (120 meters)
        - Width: 7000 cm (70 meters)
        """
        self._width_cm = 7000
        self._length_cm = 12000

    @property
    def width(self) -> int:
        """Pitch width in centimeters."""
        return self._width_cm

    @property
    def length(self) -> int:
        """Pitch length in centimeters."""
        return self._length_cm

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        """List of pitch keypoint vertices in (x, y) centimeters.

        Order matches Roboflow pitch detection model output.
        Keypoints include: goal posts, penalty spots, corner arcs, etc.
        """
        w = self._width_cm
        l = self._length_cm

        return [
            (0, w / 2),  # 0: left goal center
            (0, 0),  # 1: bottom left corner
            (0, w),  # 2: top left corner
            (l / 2, 0),  # 3: bottom center line
            (l / 2, w),  # 4: top center line
            (l, 0),  # 5: bottom right corner
            (l, w),  # 6: top right corner
            (l, w / 2),  # 7: right goal center
            (l * 0.11, 0),  # 8: left penalty area bottom
            (l * 0.11, w),  # 9: left penalty area top
            (l * 0.39, 0),  # 10: left penalty spot
            (l * 0.61, 0),  # 11: right penalty spot
            (l * 0.89, 0),  # 12: right penalty area bottom
            (l * 0.89, w),  # 13: right penalty area top
        ]

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """List of edges connecting keypoints for drawing pitch outline.

        Each edge is a tuple of (from_index, to_index).
        """
        return [
            (1, 2),  # left touchline
            (0, 2),  # left goal line (partial)
            (3, 4),  # center line
            (5, 6),  # right touchline
            (0, 7),  # right goal line (partial)
            (1, 3),  # bottom left penalty area
            (3, 5),  # bottom penalty area line
            (2, 4),  # top left penalty area
            (4, 6),  # top penalty area line
            (10, 8),  # left penalty box
            (10, 9),  # left penalty area
            (11, 12),  # right penalty box
            (11, 13),  # right penalty area
        ]

    @property
    def keypoint_names(self) -> Dict[int, str]:
        """Map keypoint indices to human-readable names."""
        return {
            0: "left_goal_center",
            1: "bottom_left_corner",
            2: "top_left_corner",
            3: "bottom_center",
            4: "top_center",
            5: "bottom_right_corner",
            6: "top_right_corner",
            7: "right_goal_center",
            8: "left_penalty_area_bottom",
            9: "left_penalty_area_top",
            10: "left_penalty_spot",
            11: "right_penalty_spot",
            12: "right_penalty_area_bottom",
            13: "right_penalty_area_top",
        }

    @property
    def radar_resolution(self) -> Tuple[int, int]:
        """Radar view resolution in pixels (width, height)."""
        return (600, 300)

    def world_to_radar(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world coordinates (cm) to radar pixel coordinates.

        Args:
            x: X coordinate in centimeters (0 to length)
            y: Y coordinate in centimeters (0 to width)

        Returns:
            Tuple of (radar_x, radar_y) in pixels
        """
        radar_w, radar_h = self.radar_resolution

        # Normalize to [0, 1]
        norm_x = x / self._length_cm
        norm_y = y / self._width_cm

        # Map to radar pixels (flip Y because pitch coordinate system is different)
        radar_x = int(norm_x * radar_w)
        radar_y = int((1 - norm_y) * radar_h)

        return (radar_x, radar_y)

    def get_keypoint_by_name(self, name: str) -> Tuple[float, float]:
        """Get keypoint coordinates by name.

        Args:
            name: Keypoint name from keypoint_names

        Returns:
            Tuple of (x, y) in centimeters

        Raises:
            ValueError: If name not found
        """
        for idx, keypoint_name in self.keypoint_names.items():
            if keypoint_name == name:
                return self.vertices[idx]
        raise ValueError(f"Unknown keypoint name: {name}")
