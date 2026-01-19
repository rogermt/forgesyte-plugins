# YOLO Tracker Plugin - Implementation Plan v2

## Overview

Build the forgeSYTE YOLO Tracker plugin with dual-mode support based on Lead Designer specifications:
1. **JSON mode** - ForgeSyte-native, returns structured data for UI
2. **JSON+Base64 mode** - Debug/export with annotated frame
3. **Video mode** - Roboflow-style full video processing

## Design Decisions (from Lead Designer)

| Decision | Value |
|----------|-------|
| **Model versions** | Versioned .pt files: `football-player-detection-v3.pt`, `football-ball-detection-v2.pt`, `football-pitch-detection-v1.pt` |
| **Team colors** | Team A: `#00BFFF` (DeepSkyBlue), Team B: `#FF1493` (DeepPink), Goalkeeper: `#FFD700` (Gold), Referee: `#FF6347` |
| **Radar config (Tomato)** | 600×300 px, pitch: 12000×7000 cm, 2:1 aspect ratio |
| **Confidence defaults** | Player: 0.25, Ball: 0.20, Pitch: 0.25 |
| **Team classification** | On-the-fly (default), optional precomputed mode (v2) |

## Current State

```
src/forgesyte_yolo_tracker/
├── inference/
│   ├── __init__.py
│   ├── player_detection.py      # Placeholder
│   ├── player_tracking.py       # Placeholder
│   ├── ball_detection.py        # Placeholder
│   ├── team_classification.py   # Placeholder
│   ├── pitch_detection.py       # Placeholder
│   └── radar.py                 # Placeholder
├── utils/
│   ├── __init__.py              # From sports.common
│   ├── annotators.py
│   ├── ball.py
│   ├── common.py
│   ├── soccer_pitch.py
│   ├── tracking.py
│   └── transforms.py
├── plugin.py                    # Existing, needs updates
└── __init__.py
```

## Target Architecture

```
src/forgesyte_yolo_tracker/
├── inference/
│   ├── __init__.py
│   ├── player_detection.py      # JSON + JSON+Base64
│   ├── player_tracking.py       # JSON + JSON+Base64
│   ├── ball_detection.py        # JSON + JSON+Base64
│   ├── team_classification.py   # JSON + JSON+Base64
│   ├── pitch_detection.py       # JSON + JSON+Base64
│   └── radar.py                 # JSON + JSON+Base64
├── video/
│   ├── __init__.py
│   ├── player_detection_video.py
│   ├── player_tracking_video.py
│   ├── ball_detection_video.py
│   ├── team_classification_video.py
│   ├── pitch_detection_video.py
│   └── radar_video.py
├── utils/
│   ├── __init__.py              # From sports.common
│   ├── annotators.py
│   ├── ball.py
│   ├── common.py
│   ├── soccer_pitch.py
│   ├── tracking.py
│   └── transforms.py
├── configs/
│   ├── __init__.py
│   └── soccer.py                # SoccerPitchConfiguration
├── models/
│   ├── football-player-detection-v3.pt
│   ├── football-ball-detection-v2.pt
│   └── football-pitch-detection-v1.pt
├── plugin.py
└── __init__.py
```

## Output Specifications

### JSON Mode
```python
{
    "detections": [{"xyxy": [x1,y1,x2,y2], "confidence": 0.95, "class_id": 0, "tracking_id": 5}],
    "count": 11,
    "classes": {"player": 9, "goalkeeper": 1, "referee": 1}
}
```

### JSON+Base64 Mode
```python
{
    "detections": [...],
    "annotated_frame_base64": "<base64 JPEG>",
    "count": 11
}
```

### Video Mode
```python
def run_detection_video(source: str, target: str, device: str = "cpu") -> None:
    # Writes annotated video to target
```

## Implementation Phases

### Phase 1: Infrastructure (Day 1)
1. Create `configs/soccer.py` with `SoccerPitchConfiguration`
2. Add model files to `models/` (versioned .pt files)
3. Update `utils/__init__.py` if needed

### Phase 2: Inference Modules (Days 2-4)
| Day | Module | Functions |
|-----|--------|-----------|
| 2 | player_detection.py | detect_players_json(), detect_players_json_with_annotated() |
| 2 | player_tracking.py | track_players_json(), track_players_json_with_annotated() |
| 3 | ball_detection.py | detect_ball_json(), detect_ball_json_with_annotated() |
| 3 | team_classification.py | classify_teams_json(), classify_teams_json_with_annotated() |
| 4 | pitch_detection.py | detect_pitch_json(), detect_pitch_json_with_annotated() |
| 4 | radar.py | generate_radar_json(), generate_radar_json_with_annotated() |

### Phase 3: Video Modules (Days 5-6)
| Day | Module |
|-----|--------|
| 5 | player_detection_video.py, player_tracking_video.py, ball_detection_video.py |
| 6 | team_classification_video.py, pitch_detection_video.py, radar_video.py |

### Phase 4: Integration (Day 7)
1. Update `plugin.py` with new inference functions
2. Update `inference/__init__.py` exports
3. Create `video/__init__.py`
4. Update `manifest.json` if needed

### Phase 5: Testing (Day 8+)
1. Unit tests for each inference module (mocked YOLO)
2. Video processing tests (mocked frames)
3. Integration tests (GPU/network)

## Module Specifications

### 1. Player Detection
**File:** `inference/player_detection.py`
**Model:** `models/football-player-detection-v3.pt`
**Classes:** 0=player, 1=goalkeeper, 2=referee
**Confidence:** 0.25, NMS: 0.45
**Colors:** Team A: #00BFFF, Team B: #FF1493, GK: #FFD700, Ref: #FF6347

### 2. Player Tracking
**File:** `inference/player_tracking.py`
**Model:** `models/football-player-detection-v3.pt`
**Tracker:** ByteTrack
**Output:** Adds tracking_id to each detection

### 3. Ball Detection
**File:** `inference/ball_detection.py`
**Model:** `models/football-ball-detection-v2.pt`
**Confidence:** 0.20, NMS: 0.10

### 4. Team Classification
**File:** `inference/team_classification.py`
**Method:** On-the-fly (collect crops → UMAP → KMeans → predict)
**Output:** Adds team_id (0=Team A, 1=Team B) to each player detection

### 5. Pitch Detection
**File:** `inference/pitch_detection.py`
**Model:** `models/football-pitch-detection-v1.pt`
**Confidence:** 0.25, NMS: 0.45
**Output:** Keypoints with names, pitch polygon

### 6. Radar
**File:** `inference/radar.py`
**Resolution:** 600×300 px
**Pitch:** 12000×7000 cm (2:1 aspect ratio)
**Output:** Mapped player positions in bird's-eye view

## Configuration

### SoccerPitchConfiguration
```python
from forgesyte_yolo_tracker.configs.soccer import SoccerPitchConfiguration

CONFIG = SoccerPitchConfiguration()
# Properties:
# - vertices: list of (x, y) pitch keypoint coordinates
# - edges: connections between keypoints
# - width: 7000 cm
# - length: 12000 cm
```

### Plugin Config Schema
```python
{
    "confidence_threshold": {"type": "number", "default": 0.25},
    "ball_confidence": {"type": "number", "default": 0.20},
    "pitch_confidence": {"type": "number", "default": 0.25},
    "max_detections": {"type": "integer", "default": 100},
    "team_colors": {
        "type": "object",
        "default": {
            "team_a": "#00BFFF",
            "team_b": "#FF1493",
            "goalkeeper": "#FFD700",
            "referee": "#FF6347"
        }
    },
    "radar_resolution": {"type": "array", "default": [600, 300]}
}
```

## Testing Strategy

### Unit Tests (Mocked, Fast)
- `tests/test_inference_player_detection.py` - Mock YOLO, test JSON output
- `tests/test_inference_player_tracking.py` - Mock YOLO+ByteTrack
- `tests/test_inference_ball_detection.py` - Mock YOLO
- `tests/test_inference_team_classification.py` - Mock TeamClassifier
- `tests/test_inference_pitch_detection.py` - Mock YOLO
- `tests/test_inference_radar.py` - Mock ViewTransformer

### Integration Tests (GPU/Network)
- `tests/inference_video/` - Full video processing
- Existing `tests/integration/test_team_integration.py`

## Git Workflow

1. Create issue for implementation tracking
2. Branch per phase or per module
3. PR and merge for each completed phase
4. Update `CHANGELOG.md`

## Questions for Lead Designer

None - all decisions are locked:
- ✅ Model versions: v3 (player), v2 (ball), v1 (pitch)
- ✅ Team colors: Blue/Pink/Gold/Tomato
- ✅ Radar: 600×300 px, 12000×7000 cm
- ✅ Confidence: 0.25/0.20/0.25
- ✅ Team classification: On-the-fly

## Ready for Implementation

Plan approved → Create issue → Begin Phase 1

---
*Plan v2 - Updated with Lead Designer answers*
*Created: 2026-01-19*
