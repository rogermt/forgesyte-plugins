"""Utility modules for YOLO Tracker.

This package provides utilities for sports tracking:
- create_batches - Batch generation utility
- TeamClassifier - Team classification with umap fallback for Python 3.13
- ViewTransformer - View transformation with 4-point validation
- ball.py - Ball tracking (not annotating)
- soccer_pitch.py - Soccer pitch drawing utilities
- annotators.py - Custom annotators
- tracking.py - Tracking utilities
- transforms.py - Image transforms
- common.py - Common utilities

Note: roboflow/sports is not used directly due to Python 3.13 incompatibility
(umap-learn requires Python < 3.10). Local implementations mirror sports.common
with necessary modifications.
"""

from .team import create_batches, TeamClassifier
from .view import ViewTransformer

from . import (
    transforms,
    annotators,
    tracking,
    common,
    ball,
    soccer_pitch,
)

__all__ = [
    # From team.py (local implementation with umap fallback)
    "create_batches",
    "TeamClassifier",
    # From view.py (local implementation with 4-point validation)
    "ViewTransformer",
    # Custom forgeSYTE modules
    "transforms",
    "annotators",
    "tracking",
    "common",
    "ball",
    "soccer_pitch",
]
