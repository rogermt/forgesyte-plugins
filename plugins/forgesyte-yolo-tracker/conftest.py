"""Pytest configuration and fixtures for YOLO Tracker plugin tests.

Sets up PYTHONPATH to include forgesyte server for importing app.models and app.plugins.base
"""

import sys
from pathlib import Path

# Add forgesyte server to path so we can import app.models and app.plugins.base
forgesyte_server = Path(__file__).parent.parent.parent.parent / "forgesyte" / "server"
if str(forgesyte_server) not in sys.path:
    sys.path.insert(0, str(forgesyte_server))
