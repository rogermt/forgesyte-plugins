"""Configs module for YOLO Tracker."""

from pathlib import Path
from typing import Any, Dict

import yaml

__all__ = [
    "SoccerPitchConfiguration",
    "load_model_config",
    "get_model_path",
    "get_confidence",
    "get_default_detections",
    "MODEL_CONFIG_PATH",
]

# Path to the models configuration file
MODEL_CONFIG_PATH = Path(__file__).parent / "models.yaml"

# Default configuration (fallback if YAML file is missing)
DEFAULT_MODEL_CONFIG: Dict[str, Any] = {
    "models": {
        "player_detection": "football-player-detection-v3.pt",
        "ball_detection": "football-ball-detection-v2.pt",
        "pitch_detection": "football-pitch-detection-v1.pt",
    },
    "confidence": {
        "player": 0.25,
        "ball": 0.20,
        "pitch": 0.25,
    },
    "device": "cpu",
    "default_detections": {
        "players": True,
        "ball": True,
        "pitch": True,
    },
}


def load_model_config(config_path: Path = MODEL_CONFIG_PATH) -> Dict[str, Any]:
    """Load model configuration from YAML file.

    Args:
        config_path: Path to the models.yaml file.

    Returns:
        Dictionary containing model paths and confidence thresholds.

    Raises:
        FileNotFoundError: If config file doesn't exist and no default.
        yaml.YAMLError: If config file is invalid YAML.
    """
    if not config_path.exists():
        return DEFAULT_MODEL_CONFIG.copy()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        return DEFAULT_MODEL_CONFIG.copy()

    # Merge with defaults to ensure all keys exist
    merged = DEFAULT_MODEL_CONFIG.copy()
    if "models" in config:
        merged["models"].update(config["models"])
    if "confidence" in config:
        merged["confidence"].update(config["confidence"])
    if "device" in config:
        merged["device"] = config["device"]
    if "default_detections" in config:
        merged["default_detections"] = config["default_detections"]

    return merged


def get_model_path(model_key: str) -> str:
    """Get model file name for the specified task.

    Args:
        model_key: One of 'player_detection', 'ball_detection', 'pitch_detection'.

    Returns:
        Model file name (e.g., 'football-player-detection-v3.pt').

    Raises:
        KeyError: If model_key is not found in config.
    """
    config = load_model_config()
    return config["models"][model_key]


def get_confidence(task: str) -> float:
    """Get confidence threshold for the specified task.

    Args:
        task: One of 'player', 'ball', 'pitch'.

    Returns:
        Confidence threshold as float between 0 and 1.

    Raises:
        KeyError: If task is not found in config.
    """
    config = load_model_config()
    return config["confidence"][task]


def get_default_detections() -> list:
    """Get list of detections enabled in config.

    Returns:
        List of detection types to run (e.g., ['players', 'pitch']).
    """
    config = load_model_config()
    detections_config = config.get(
        "default_detections",
        {"players": True, "ball": True, "pitch": True},
    )
    # Convert dict {name: bool} to list of enabled names
    return [name for name, enabled in detections_config.items() if enabled]
