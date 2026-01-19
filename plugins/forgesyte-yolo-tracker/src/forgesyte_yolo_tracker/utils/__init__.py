"""Utility modules for YOLO Tracker.

This package uses code from roboflow/sports (MIT License):
- sports.common.team.TeamClassifier - Team classification
- sports.common.view.ViewTransformer - View transformation

Custom forgeSYTE modules:
- ball.py - Ball tracking (not annotating)
- soccer_pitch.py - Soccer pitch drawing utilities
- annotators.py - Custom annotators
- tracking.py - Tracking utilities
- transforms.py - Image transforms
- common.py - Common utilities
"""

from sports.common import (
    create_batches,
    TeamClassifier,
    ViewTransformer,
)

from . import (
    transforms,
    annotators,
    tracking,
    common,
    ball,
    soccer_pitch,
)

__all__ = [
    # From sports.common
    "create_batches",
    "TeamClassifier",
    "ViewTransformer",
    # Custom forgeSYTE modules
    "transforms",
    "annotators",
    "tracking",
    "common",
    "ball",
    "soccer_pitch",
]
