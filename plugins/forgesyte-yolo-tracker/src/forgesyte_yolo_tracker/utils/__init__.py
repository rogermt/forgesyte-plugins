"""Utility modules for YOLO Tracker.

This package provides utilities for sports tracking:

Core Classes:
- create_batches - Batch generation utility
- TeamClassifier - Team classification (local implementation with umap fallback)
- ViewTransformer - View transformation (local implementation with 4-point validation)

Custom forgeSYTE modules:
- ball.py - Ball tracking (not annotating)
- soccer_pitch.py - Soccer pitch drawing utilities
"""

from . import ball, soccer_pitch
from .team import TeamClassifier, create_batches
from .view import ViewTransformer

__all__ = [
    # From local implementations
    "create_batches",
    "TeamClassifier",
    "ViewTransformer",
    # Custom forgeSYTE modules
    "ball",
    "soccer_pitch",
]
