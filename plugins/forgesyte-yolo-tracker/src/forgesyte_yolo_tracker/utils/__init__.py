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

# Lazy import to avoid torch/transformers at module load time
def __getattr__(name: str):
    """Lazy load heavy dependencies to avoid import errors in tests."""
    if name == "TeamClassifier":
        from .team import TeamClassifier
        return TeamClassifier
    elif name == "create_batches":
        from .team import create_batches
        return create_batches
    elif name == "ViewTransformer":
        from .view import ViewTransformer
        return ViewTransformer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # From local implementations
    "create_batches",
    "TeamClassifier",
    "ViewTransformer",
    # Custom forgeSYTE modules
    "ball",
    "soccer_pitch",
]
