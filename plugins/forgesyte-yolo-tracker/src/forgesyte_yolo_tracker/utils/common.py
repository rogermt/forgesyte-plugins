"""Common utility functions."""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


def load_model(model_path: str, device: str = "cpu") -> Any:
    """Load a YOLO model from file.

    Args:
        model_path: Path to model file
        device: Device to load model on ('cpu', 'cuda', etc.)

    Returns:
        Loaded model instance
    """
    # Placeholder implementation
    return None


def get_model_path(model_name: str) -> Path:
    """Get path to bundled model file.

    Args:
        model_name: Name of model (e.g., 'football-player-detection.pt')

    Returns:
        Path to model file
    """
    models_dir = Path(__file__).parent.parent / "models"
    return models_dir / model_name


def crop_image(image: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """Crop image using bounding box.

    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]

    Returns:
        Cropped image
    """
    x1, y1, x2, y2 = bbox.astype(int)
    return image[y1:y2, x1:x2]


def resize_image(
    image: np.ndarray,
    size: Tuple[int, int],
    keep_aspect_ratio: bool = True,
) -> np.ndarray:
    """Resize image to target size.

    Args:
        image: Input image
        size: Target size (width, height)
        keep_aspect_ratio: Whether to maintain aspect ratio

    Returns:
        Resized image
    """
    # Placeholder implementation
    return image


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    # Placeholder implementation
    return {}


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color to RGB tuple.

    Args:
        hex_color: Color in hex format (e.g., '#FF1493')

    Returns:
        RGB tuple (0-255 range)
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (r, g, b)
