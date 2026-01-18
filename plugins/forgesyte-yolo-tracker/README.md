# ForgeSyte YOLO Tracker Plugin

YOLO-based sports detection and tracking plugin for ForgeSyte, providing advanced computer vision capabilities for:

- **Player Detection**: Detect and locate players on the field
- **Ball Detection**: Locate ball position with sub-frame tracking
- **Team Classification**: Classify players by team using color clustering
- **Player Tracking**: Track individual players across frames using ByteTrack
- **Pitch Detection**: Detect and annotate field/pitch lines and vertices
- **Radar View**: Generate bird's-eye view radar visualization of player positions

## Installation

```bash
pip install -e .
```

## Usage

As a ForgeSyte plugin, this module is automatically discovered and available through the plugin interface.

## Features

### Detection Modes
- Pitch detection with keypoint annotation
- Player detection with bounding boxes
- Ball detection with temporal tracking
- Multi-model inference with NMS filtering

### Tracking
- ByteTrack integration for persistent player IDs
- Configurable tracking parameters
- Support for goalkeeper and referee classification

### Team Classification
- Unsupervised team clustering based on jersey colors
- Goalkeeper and referee handling
- Temporal consistency

### Visualization
- Radar pitch view with team-colored player positions
- Multiple annotation styles (boxes, ellipses, keypoints)
- Configurable color schemes

## Project Structure

```
src/forgesyte_yolo_tracker/
├── __init__.py              # Package initialization
├── plugin.py                # Main plugin class
├── manifest.json            # Plugin metadata
├── inference/               # Detection and tracking modules
│   ├── player_detection.py
│   ├── player_tracking.py
│   ├── ball_detection.py
│   ├── team_classification.py
│   ├── pitch_detection.py
│   └── radar.py
├── utils/                   # Utility modules
│   ├── transforms.py        # View transformations
│   ├── annotators.py        # Custom annotators
│   ├── tracking.py          # Tracking utilities
│   └── common.py            # Common utilities
├── models/                  # Model weights
└── data/                    # Configuration files
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type checking
mypy src/

# Linting
ruff check src/ --fix
black src/
```

## License

MIT
