"""Utility modules for YOLO Tracker.

This package provides utilities for sports tracking:

Core Classes:
- create_batches - Batch generation utility
- TeamClassifier - Team classification (based on sports.common with umap fallback)
- ViewTransformer - View transformation (based on sports.common with 4-point validation)

Custom Modules:
- ball.py - Ball tracking (not annotating)
- soccer_pitch.py - Soccer pitch drawing utilities
- annotators.py - Custom annotators
- tracking.py - Tracking utilities
- transforms.py - Image transforms
- common.py - Common utilities

Note on sports.common:
The roboflow/sports package has a Python version check bug that prevents
installation on Python 3.10+ despite setup.py stating python_requires>=3.8.
Local implementations are used instead, which mirror sports.common (MIT License)
with forgeSYTE-specific improvements:
- umap fallback for Python 3.13 compatibility
- 4-point validation for homography calculation
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
