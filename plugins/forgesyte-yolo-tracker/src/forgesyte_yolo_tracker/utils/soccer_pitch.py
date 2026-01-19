"""Soccer pitch visualization utilities."""

from typing import List, Optional

import cv2
import numpy as np
import supervision as sv


def draw_pitch(
    config: object,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1,
) -> np.ndarray:
    """
    Draws a soccer pitch with specified dimensions, colors, and scale.

    Args:
        config: Configuration object containing pitch dimensions and layout.
        background_color (sv.Color, optional): Color of the pitch background.
            Defaults to green.
        line_color (sv.Color, optional): Color of the pitch lines.
            Defaults to white.
        padding (int, optional): Padding around the pitch in pixels.
            Defaults to 50.
        line_thickness (int, optional): Thickness of the pitch lines.
            Defaults to 4.
        point_radius (int, optional): Radius of penalty spot points.
            Defaults to 8.
        scale (float, optional): Scaling factor for pitch dimensions.
            Defaults to 0.1.

    Returns:
        np.ndarray: Image of the soccer pitch.
    """
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    pitch_image = np.ones(
        (scaled_width + 2 * padding, scaled_length + 2 * padding, 3),
        dtype=np.uint8,
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    for start, end in config.edges:
        point1 = (
            int(config.vertices[start - 1][0] * scale) + padding,
            int(config.vertices[start - 1][1] * scale) + padding,
        )
        point2 = (
            int(config.vertices[end - 1][0] * scale) + padding,
            int(config.vertices[end - 1][1] * scale) + padding,
        )
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness,
        )

    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding,
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness,
    )

    penalty_spots = [
        (
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding,
        ),
        (
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding,
        ),
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1,
        )

    return pitch_image


def draw_points_on_pitch(
    config: object,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draws points on a soccer pitch.

    Args:
        config: Configuration object containing pitch dimensions.
        xy (np.ndarray): Array of points with (x, y) coordinates.
        face_color (sv.Color, optional): Color of point faces.
            Defaults to red.
        edge_color (sv.Color, optional): Color of point edges.
            Defaults to black.
        radius (int, optional): Radius of points in pixels. Defaults to 10.
        thickness (int, optional): Thickness of point edges. Defaults to 2.
        padding (int, optional): Padding around pitch. Defaults to 50.
        scale (float, optional): Scaling factor. Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image.
            Defaults to None.

    Returns:
        np.ndarray: Image of soccer pitch with points drawn.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    for point in xy:
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding,
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1,
        )
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness,
        )

    return pitch


def draw_paths_on_pitch(
    config: object,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draws paths on a soccer pitch.

    Args:
        config: Configuration object containing pitch dimensions.
        paths (List[np.ndarray]): List of paths as arrays of (x, y)
            coordinates.
        color (sv.Color, optional): Color of paths. Defaults to white.
        thickness (int, optional): Thickness of paths. Defaults to 2.
        padding (int, optional): Padding around pitch. Defaults to 50.
        scale (float, optional): Scaling factor. Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image.
            Defaults to None.

    Returns:
        np.ndarray: Image of soccer pitch with paths drawn.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    for path in paths:
        scaled_path = [
            (int(point[0] * scale) + padding, int(point[1] * scale) + padding)
            for point in path
            if point.size > 0
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=pitch,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness,
            )

        return pitch

    return pitch


def draw_pitch_voronoi_diagram(
    config: object,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Draws a Voronoi diagram on a soccer pitch representing control areas.

    Args:
        config: Configuration object containing pitch dimensions.
        team_1_xy (np.ndarray): Positions of team 1 players.
        team_2_xy (np.ndarray): Positions of team 2 players.
        team_1_color (sv.Color, optional): Color for team 1 control area.
            Defaults to red.
        team_2_color (sv.Color, optional): Color for team 2 control area.
            Defaults to white.
        opacity (float, optional): Opacity of Voronoi overlay. Defaults to 0.5.
        padding (int, optional): Padding around pitch. Defaults to 50.
        scale (float, optional): Scaling factor. Defaults to 0.1.
        pitch (Optional[np.ndarray], optional): Existing pitch image.
            Defaults to None.

    Returns:
        np.ndarray: Image of soccer pitch with Voronoi diagram overlay.
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices(
        (scaled_width + 2 * padding, scaled_length + 2 * padding)
    )

    y_coordinates = y_coordinates.astype(np.float32) - padding
    x_coordinates = x_coordinates.astype(np.float32) - padding

    def calculate_distances(xy: np.ndarray) -> np.ndarray:
        """Calculate distances from points to all grid coordinates."""
        return np.sqrt(
            (xy[:, 0][:, None, None] * scale - x_coordinates) ** 2
            + (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2
        )

    distances_team_1 = calculate_distances(team_1_xy)
    distances_team_2 = calculate_distances(team_2_xy)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    control_mask = min_distances_team_1 < min_distances_team_2

    voronoi[control_mask] = team_1_color_bgr
    voronoi[~control_mask] = team_2_color_bgr

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay
