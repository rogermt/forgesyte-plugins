"""ForgeSyte YOLO Tracker Plugin.

A comprehensive sports detection and tracking plugin providing:
- Player detection and tracking
- Ball detection with temporal tracking
- Team classification
- Pitch detection
- Radar visualization
"""

__version__ = "0.1.0"
__author__ = "ForgeSyte Contributors"

from .plugin import Plugin

__all__ = ["Plugin"]
