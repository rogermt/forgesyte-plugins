"""Test constants for model paths.

Provides a single source of truth for model file names used in tests.
Reads from configs/models.yaml to avoid hardcoding.
"""

import os
from pathlib import Path

import yaml

# Base path to models directory
MODELS_DIR = Path(__file__).parents[2] / "forgesyte_yolo_tracker" / "models"

# Path to config file
CONFIG_PATH = Path(__file__).parents[2] / "forgesyte_yolo_tracker" / "configs" / "models.yaml"


# Load model names from config
def _load_model_names() -> dict:
    """Load model file names from configs/models.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            config = yaml.safe_load(f)
            return config.get("models", {})
    return {
        "player_detection": "football-player-detection-v3.pt",
        "ball_detection": "football-ball-detection-v2.pt",
        "pitch_detection": "football-pitch-detection-v1.pt",
    }


MODEL_NAMES = _load_model_names()

# Convenience exports matching config keys
PLAYER_MODEL = MODEL_NAMES.get("player_detection", "football-player-detection-v3.pt")
BALL_MODEL = MODEL_NAMES.get("ball_detection", "football-ball-detection-v2.pt")
PITCH_MODEL = MODEL_NAMES.get("pitch_detection", "football-pitch-detection-v1.pt")

# Full paths for checking existence
PLAYER_MODEL_PATH = MODELS_DIR / PLAYER_MODEL
BALL_MODEL_PATH = MODELS_DIR / BALL_MODEL
PITCH_MODEL_PATH = MODELS_DIR / PITCH_MODEL

# Check if any model exists
MODELS_EXIST = PLAYER_MODEL_PATH.exists() or BALL_MODEL_PATH.exists() or PITCH_MODEL_PATH.exists()

# Environment flag
RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"
