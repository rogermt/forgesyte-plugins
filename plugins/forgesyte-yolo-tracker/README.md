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
├── plugin.py                # ForgeSyte-native plugin functions
├── manifest.json            # Plugin metadata
├── inference/               # Detection and tracking modules
│   ├── player_detection.py  # JSON + JSON+Base64 modes
│   ├── player_tracking.py   # ByteTrack integration
│   ├── ball_detection.py    # Ball localization
│   ├── team_classification.py  # Team clustering
│   ├── pitch_detection.py   # Pitch keypoints
│   └── radar.py             # Bird's-eye view
├── video/                   # Video processing modules
│   ├── player_detection_video.py
│   ├── player_tracking_video.py
│   ├── ball_detection_video.py
│   ├── team_classification_video.py
│   ├── pitch_detection_video.py
│   └── radar_video.py
├── utils/                   # Utility modules
│   ├── ball.py              # BallTracker
│   ├── soccer_pitch.py      # Soccer pitch utilities
│   └── (imports from sports.common)
├── configs/
│   └── soccer.py            # SoccerPitchConfiguration
└── models/                  # Model weights (placeholders)
    ├── football-player-detection-v3.pt
    ├── football-ball-detection-v2.pt
    └── football-pitch-detection-v1.pt
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
