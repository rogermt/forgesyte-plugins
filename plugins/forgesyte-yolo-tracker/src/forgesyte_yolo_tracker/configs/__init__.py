"""Configs module for YOLO Tracker.

Phase 12: Strict device governance.
- YAML is the single source of truth for models, confidence, device.
- No in-code fallbacks.
- Device is required in YAML; if missing, raise ConfigError.
"""

from pathlib import Path
from typing import Any, Dict

import yaml

__all__ = [
    "ConfigError",
    "load_model_config",
    "get_model_path",
    "get_confidence",
    "get_device",
    "get_default_detections",
    "MODEL_CONFIG_PATH",
    "_reset_config_cache_for_tests",
]

# Path to the models configuration file
MODEL_CONFIG_PATH = Path(__file__).parent / "models.yaml"

# Global cache for config (loaded once, never reloaded in production)
_CONFIG_CACHE: Dict[str, Any] | None = None


class ConfigError(RuntimeError):
    """Configuration error for YOLO tracker models."""


def _reset_config_cache_for_tests() -> None:
    """Reset config cache for test isolation.
    
    Tests that modify models.yaml at runtime must call this to reload.
    Not needed if tests use fixed config or mocks.
    """
    global _CONFIG_CACHE
    _CONFIG_CACHE = None


def load_model_config(config_path: Path = MODEL_CONFIG_PATH) -> Dict[str, Any]:
    """Load model configuration from YAML file.
    
    Phase 12 strict governance:
    - YAML is required to exist
    - All required keys (models, confidence, device) must be present
    - No in-code fallbacks or defaults
    - Missing or invalid config raises ConfigError immediately

    Args:
        config_path: Path to the models.yaml file.

    Returns:
        Dictionary containing model paths, confidence thresholds, device, etc.

    Raises:
        ConfigError: If YAML file missing, invalid, or missing required keys.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    if not config_path.exists():
        raise ConfigError(f"models.yaml not found at: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as exc:
        raise ConfigError(f"Failed to parse models.yaml: {exc}") from exc

    if config is None or not isinstance(config, dict):
        raise ConfigError("models.yaml is empty or not a valid YAML dict")

    # Validate required top-level keys
    required_keys = {"models", "confidence", "device"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ConfigError(f"models.yaml missing required keys: {missing}")

    # Validate device value
    device = config.get("device")
    if device not in ("cpu", "cuda"):
        raise ConfigError(f"Invalid device in models.yaml: {device!r} (expected 'cpu' or 'cuda')")

    _CONFIG_CACHE = config
    return config


def get_model_path(model_key: str) -> str:
    """Get model file name for the specified task.

    Args:
        model_key: One of 'player_detection', 'ball_detection', 'pitch_detection'.

    Returns:
        Model file name (e.g., 'football-player-detection-v3.pt').

    Raises:
        ConfigError: If model_key is not found in config.
    """
    config = load_model_config()
    models = config.get("models", {})
    if model_key not in models:
        raise ConfigError(f"models.yaml missing model entry for: {model_key}")
    return models[model_key]


def get_confidence(task: str) -> float:
    """Get confidence threshold for the specified task.

    Args:
        task: One of 'player', 'ball', 'pitch'.

    Returns:
        Confidence threshold as float between 0 and 1.

    Raises:
        ConfigError: If task is not found in config or value is invalid.
    """
    config = load_model_config()
    confidence = config.get("confidence", {})
    if task not in confidence:
        raise ConfigError(f"models.yaml missing confidence entry for: {task}")
    value = float(confidence[task])
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"Invalid confidence for {task}: {value} (expected 0.0-1.0)")
    return value


def get_device() -> str:
    """Get configured device for inference.
    
    Phase 12: Strict governance.
    - Device must be present in models.yaml
    - Valid values: 'cpu' or 'cuda'
    - No fallback; if missing, raises ConfigError
    
    Returns:
        Device string: either 'cpu' or 'cuda'
        
    Raises:
        ConfigError: If device is missing or invalid in models.yaml
    """
    config = load_model_config()
    device = config.get("device")
    if device not in ("cpu", "cuda"):
        raise ConfigError(f"Invalid device in models.yaml: {device!r}")
    return device


def get_default_detections() -> list:
    """Get list of detections enabled in config.

    Returns:
        List of detection types to run (e.g., ['players', 'pitch']).
        
    Raises:
        ConfigError: If default_detections is missing or invalid.
    """
    config = load_model_config()
    detections_config = config.get("default_detections")
    if detections_config is None:
        raise ConfigError("models.yaml missing key: default_detections")
    # Convert dict {name: bool} to list of enabled names
    return [name for name, enabled in detections_config.items() if enabled]
