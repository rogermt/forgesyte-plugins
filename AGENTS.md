# AGENTS.md

Guidelines for AI agents working on the ForgeSyte Plugins repository.

## Project Overview

This repository contains independent pip-installable plugins for the ForgeSyte platform. Each plugin is located in `plugins/<plugin-name>` and follows a standardized structure.

## Key Directories

```
forgesyte-plugins/
├── plugins/
│   ├── forgesyte-yolo-tracker/   # YOLO sports analysis plugin
│   ├── ocr/
│   ├── block_mapper/
│   ├── moderation/
│   ├── motion_detector/
│   └── plugin_template/          # Template for new plugins
├── scripts/                      # Utility scripts
├── docs/                         # Documentation
└── README.md                     # Main readme
```

## Development Commands

### Setting Up a Plugin

```bash
cd plugins/<plugin-name>

# Create virtual environment
uv venv --python 3.9
source .venv/bin/activate

# Install plugin in development mode
uv pip install -e .

# Install with all dependencies
uv pip install -e ".[dev]"
```

### Running Tests

**Fast tests only (no model/GPU):**
```bash
uv run pytest src/tests/ -v
```

**With model tests (requires GPU/network):**
```bash
RUN_MODEL_TESTS=1 uv run pytest src/tests/ -v
```

**Integration tests (requires GPU):**
```bash
RUN_INTEGRATION_TESTS=1 uv run pytest src/tests/integration/ -v
```

**Specific test file:**
```bash
uv run pytest src/tests/utils/test_team.py -v
```

### Code Quality

```bash
# Linting
uv run ruff check src/ --fix

# Type checking
uv run mypy src/

# Formatting
uv run black src/
uv run isort src/
```

## TDD Workflow for this Project

### 1. Test Structure

All tests follow this pattern:
- **Fast tests**: No model loading, mock dependencies, run on CPU
- **Model tests**: Require YOLO model loading, skipped by default
- **Integration tests**: Require GPU, skipped by default

### 2. Test Configuration

Each test file with model dependencies MUST include:

```python
import os
import pytest

RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="Set RUN_MODEL_TESTS=1 to run (requires YOLO model)"
)
```

### 3. TDD Pattern for New Features

When adding a new inference module:

**Step 1: Write failing tests first**
```bash
# Create test file
touch src/tests/test_inference_new_feature.py

# Write tests with proper skip markers
# Run to verify they fail (module doesn't exist yet)
uv run pytest src/tests/test_inference_new_feature.py -v
```

**Step 2: Implement the module**
```bash
# Create the inference module
touch src/forgesyte_yolo_tracker/inference/new_feature.py

# Implement functions to pass tests
# Use sports.common when available (MIT License)
```

**Step 3: Verify tests pass**
```bash
uv run pytest src/tests/test_inference_new_feature.py -v
```

### 4. Test File Naming Convention

| Type | Pattern | Example |
|------|---------|---------|
| Unit tests | `test_*.py` | `test_player_detection.py` |
| Integration tests | `src/tests/integration/test_*.py` | `test_team_integration.py` |
| Utils tests | `src/tests/utils/test_*.py` | `test_soccer_pitch.py` |

### 5. Model-Dependent Test Pattern

For tests requiring YOLO models (always skip on CPU):

```python
"""Tests for new feature inference module."""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="Set RUN_MODEL_TESTS=1 to run (requires YOLO model)"
)


class TestNewFeatureJSON:
    """Tests for detect_new_feature_json function."""

    def test_returns_dict(self) -> None:
        """Verify returns dictionary."""
        from forgesyte_yolo_tracker.inference.new_feature import detect_new_feature_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_new_feature_json(frame, device="cpu")

        assert isinstance(result, dict)

    def test_returns_expected_keys(self) -> None:
        """Verify returns expected keys."""
        from forgesyte_yolo_tracker.inference.new_feature import detect_new_feature_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_new_feature_json(frame, device="cpu")

        assert "detections" in result
```

## Plugin Structure for forgesyte-yolo-tracker

```
plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/
├── plugin.py              # ForgeSyte-native functions (no class)
├── manifest.json          # Tool schema
├── inference/             # Frame-based JSON modes
│   ├── player_detection.py    # detect_players_json(), *_with_annotated_frame()
│   ├── player_tracking.py     # track_players_json(), *_with_annotated_frame()
│   ├── ball_detection.py      # detect_ball_json(), *_with_annotated_frame()
│   ├── team_classification.py # classify_teams_json(), *_with_annotated_frame()
│   ├── pitch_detection.py     # detect_pitch_json(), *_with_annotated_frame()
│   └── radar.py               # radar_json(), radar_json_with_annotated_frame()
├── video/                 # Video processing modes
│   └── *_video.py         # run_*_video_frames(), run_*_video()
├── utils/                 # Utilities
│   ├── ball.py            # BallTracker
│   ├── soccer_pitch.py    # Soccer pitch drawing
│   └── (imports from sports.common)
├── configs/
│   └── soccer.py          # SoccerPitchConfiguration
└── models/                # Model files
    ├── football-player-detection-v3.pt
    ├── football-ball-detection-v2.pt
    └── football-pitch-detection-v1.pt
```

## Design Decisions (from Lead Designer)

| Decision | Value |
|----------|-------|
| Team colors | Team A: #00BFFF, Team B: #FF1493, GK: #FFD700, Ref: #FF6347 |
| Confidence defaults | Player: 0.25, Ball: 0.20, Pitch: 0.25 |
| Radar resolution | 600×300 px, pitch: 12000×7000 cm |
| Team classification | On-the-fly (collect → UMAP → KMeans → predict) |
| Model versions | player-v3, ball-v2, pitch-v1 |

## Useful Commands

```bash
# Check Python version
python --version

# List installed packages
uv pip list

# Freeze requirements
uv pip freeze > requirements.txt

# Create requirements.txt for plugin
cd plugins/forgesyte-yolo-tracker
uv pip freeze > requirements.txt
```

## Common Patterns

### Import Pattern for Inference Modules

```python
from forgesyte_yolo_tracker.inference.player_detection import (
    detect_players_json,
    detect_players_json_with_annotated_frame,
)
```

### Video Processing Pattern

```python
from forgesyte_yolo_tracker.video.player_detection_video import (
    run_player_detection_video_frames,
    run_player_detection_video,
)

# Generator pattern for frames
for annotated_frame in run_player_detection_video_frames(source, device="cpu"):
    process(annotated_frame)

# Full video pipeline
run_player_detection_video(source, target, device="cpu")
```

### Using sports.common

```python
from sports.common import TeamClassifier, ViewTransformer, create_batches
```

## Testing on GPU

Tests requiring GPU should be run on Colab or Kaggle:

```bash
git clone https://github.com/rogermt/forgesyte-plugins.git
cd forgesyte-plugins/plugins/forgesyte-yolo-tracker

# Install with GPU support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
uv pip install -e .

# Download model files to models/
# football-player-detection-v3.pt
# football-ball-detection-v2.pt
# football-pitch-detection-v1.pt

# Run all tests
RUN_MODEL_TESTS=1 RUN_INTEGRATION_TESTS=1 uv run pytest src/tests/ -v
```

## Code Style

- Follow PEP 8
- Use type hints
- Run `ruff check src/ --fix` before committing
- Add docstrings to all public functions
- Keep functions small and focused
- Use mocking for CPU tests (no model loading)

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

footer
```

Types: feat, fix, refactor, docs, chore, test

Example:
```
feat(yolo-tracker): Add player detection inference module

Implement detect_players_json() and detect_players_json_with_annotated_frame()

Closes #32
```

## Getting Help

- See `README.md` for plugin documentation
- See `docs/development/` for general guidelines
- Check existing tests in `src/tests/` for patterns
