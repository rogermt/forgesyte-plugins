# Radar Test Plan (Phase 3)

## Overview
Expand radar tests from **3 → 25+ tests** to cover visualization, encoding, and multi-object scenarios.

## File to Expand
`src/tests/test_inference_radar.py`

## Test Breakdown (25+ tests)

### 1. Radar Frame Generation (4 tests)

#### test_radar_frame_dimensions
- **Setup:** Generate radar frame
- **Expected:** 600×300 px output
- **Assert:** `result.shape == (300, 600, 3)` (HxWxC)

#### test_radar_frame_is_ndarray
- **Setup:** Generate radar frame
- **Expected:** Returns numpy array
- **Assert:** `isinstance(result, np.ndarray)`

#### test_radar_frame_dtype_uint8
- **Setup:** Generate radar frame
- **Expected:** Data type is uint8
- **Assert:** `result.dtype == np.uint8`

#### test_radar_frame_color_channels
- **Setup:** Generate radar frame
- **Expected:** 3 color channels (RGB or BGR)
- **Assert:** `result.shape[2] == 3`

---

### 2. Pitch Mapping & Scale (4 tests)

#### test_pitch_mapping_dimensions
- **Setup:** Pitch detection with standard field
- **Expected:** Mapped to 600×300 px correctly
- **Assert:** Pitch occupies correct region

#### test_pitch_scale_12000x7000
- **Setup:** Detect pitch, validate mapping
- **Expected:** 12000×7000 cm scale applied
- **Assert:** Player positions mapped to physical coordinates

#### test_pitch_corner_mapping
- **Setup:** Players at pitch corners
- **Expected:** Rendered at radar corners
- **Assert:** Position mapping accurate

#### test_pitch_aspect_ratio
- **Setup:** Generate radar with standard pitch
- **Expected:** Aspect ratio ≈ 12000:7000 ≈ 1.71:1
- **Assert:** Rendered correctly on 600×300 (2:1 aspect)

---

### 3. Player Position Visualization (5 tests)

#### test_single_player_position
- **Setup:** 1 player at center of frame
- **Expected:** Rendered at center of radar
- **Assert:** Player dot visible at ~(300, 150) in 600×300 frame

#### test_multiple_player_positions
- **Setup:** 5 players at different positions
- **Expected:** All rendered with correct relative positions
- **Assert:** Distance between players preserved on radar

#### test_player_movement_trajectory
- **Setup:** Player moves 100px across frames
- **Expected:** Position updated on radar
- **Assert:** New position reflects movement direction/distance

#### test_player_off_pitch
- **Setup:** Player outside pitch boundaries
- **Expected:** Rendered outside pitch area or not shown
- **Assert:** Handles boundary cases

#### test_player_size_consistency
- **Setup:** Radar with multiple players
- **Expected:** All players same size (dot radius)
- **Assert:** Player markers consistent

---

### 4. Ball Position Visualization (4 tests)

#### test_ball_position_center
- **Setup:** Ball at pitch center
- **Expected:** Rendered at radar center
- **Assert:** Ball position matches expected coordinates

#### test_ball_position_corner
- **Setup:** Ball in corner of pitch
- **Expected:** Rendered in corresponding radar corner
- **Assert:** Position mapping accurate

#### test_ball_size_distinct
- **Setup:** Radar with ball and players
- **Expected:** Ball marker distinct from player markers
- **Assert:** Ball size/color different from player dots

#### test_ball_missing_renders_empty
- **Setup:** No ball detection
- **Expected:** Radar renders without ball
- **Assert:** No ball marker on radar

---

### 5. Team Colors (5 tests)

#### test_team_a_color_0fbfff
- **Setup:** Team A player on radar
- **Expected:** Rendered with #00BFFF
- **Assert:** RGB values match (0, 191, 255)

#### test_team_b_color_ff1493
- **Setup:** Team B player on radar
- **Expected:** Rendered with #FF1493
- **Assert:** RGB values match (255, 20, 147)

#### test_goalkeeper_color_ffd700
- **Setup:** Goalkeeper on radar
- **Expected:** Rendered with #FFD700 (gold)
- **Assert:** RGB values match (255, 215, 0)

#### test_referee_color_ff6347
- **Setup:** Referee on radar
- **Expected:** Rendered with #FF6347 (red)
- **Assert:** RGB values match (255, 99, 71)

#### test_color_contrast_on_pitch
- **Setup:** Radar with all 4 player types
- **Expected:** Colors clearly distinguishable
- **Assert:** Brightness/contrast sufficient

---

### 6. Confidence Threshold Filtering (2 tests)

#### test_low_confidence_player_excluded
- **Setup:** Player with confidence=0.15 (below 0.25)
- **Expected:** Not rendered on radar
- **Assert:** Confidence-filtered correctly

#### test_high_confidence_player_included
- **Setup:** Player with confidence=0.75
- **Expected:** Rendered on radar
- **Assert:** High-confidence players shown

---

### 7. Base64 Encoding (3 tests)

#### test_radar_json_base64_encoded
- **Setup:** Run `radar_json()`, get response
- **Expected:** Base64 string in `radar` field
- **Assert:** String is valid base64

#### test_base64_decode_to_image
- **Setup:** Decode base64 from `radar_json()`
- **Expected:** Decodes to valid PNG/JPEG
- **Assert:** Image dimensions match (600×300)

#### test_base64_image_valid_format
- **Setup:** Radar JSON output
- **Expected:** Base64 string is decodable and creates valid image
- **Assert:** `len(base64) > 1000` (not empty, not tiny)

---

### 8. Multi-Object Scenarios (4 tests)

#### test_11_players_radar
- **Setup:** 11 players (full team) + ball
- **Expected:** All rendered on radar
- **Assert:** 11 dots + 1 ball visible

#### test_2_team_colors_mixed
- **Setup:** Team A (6) vs Team B (5) players
- **Expected:** Team colors clearly distinguished
- **Assert:** Color separation visible

#### test_goalkeeper_among_field_players
- **Setup:** 10 field players + 1 goalkeeper
- **Expected:** Goalkeeper gold, field players colored
- **Assert:** Gold color distinct from team colors

#### test_crowded_center_radar
- **Setup:** Multiple players clustered at center
- **Expected:** All rendered, possibly overlapping
- **Assert:** No crashes, valid output

---

### 9. JSON Output Validation (1 test)

#### test_radar_json_schema_valid
- **Setup:** Run `radar_json()` with detections
- **Expected:** JSON matches ForgeSyte schema
- **Assert:** Has `radar` field (base64), optional `pitchCorners`, etc.

---

### 10. Device Parameter (2 tests)

#### test_device_cpu
- **Setup:** Generate radar with device="cpu"
- **Expected:** Works on CPU
- **Assert:** Valid output returned

#### test_device_cuda
- **Setup:** Generate radar with device="cuda"
- **Expected:** Works on CUDA if available
- **Assert:** Valid output returned

---

## Test Pattern Template

```python
"""Tests for radar visualization module."""

import os
import pytest
import numpy as np
import base64
from unittest.mock import MagicMock, patch
from PIL import Image
from io import BytesIO

RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="Set RUN_MODEL_TESTS=1 to run (requires YOLO model)"
)


class TestRadarJSON:
    """Tests for radar_json function."""

    def test_radar_frame_dimensions(self) -> None:
        """Verify radar frame is 600x300 px."""
        from forgesyte_yolo_tracker.inference.radar import radar_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = [
            {"bbox": [100, 100, 50, 100], "confidence": 0.8, "class": "player"}
        ]
        result = radar_json(frame, detections, device="cpu")

        assert isinstance(result, dict)
        assert "radar" in result
        # assertions here

    def test_base64_decode_to_image(self) -> None:
        """Verify base64 can be decoded to valid image."""
        from forgesyte_yolo_tracker.inference.radar import radar_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = []
        result = radar_json(frame, detections, device="cpu")

        # Decode base64
        radar_b64 = result["radar"]
        radar_bytes = base64.b64decode(radar_b64)
        radar_img = Image.open(BytesIO(radar_bytes))

        # Verify dimensions
        assert radar_img.size == (600, 300)  # (width, height)

    # ... other tests
```

## Fixtures

```python
@pytest.fixture
def sample_frame():
    """Create a sample frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_detections():
    """Create sample detections."""
    return [
        {"bbox": [100, 100, 50, 100], "confidence": 0.8, "class": "player", "trackID": 1},
        {"bbox": [200, 150, 50, 100], "confidence": 0.75, "class": "player", "trackID": 2},
        {"bbox": [300, 120, 30, 30], "confidence": 0.9, "class": "ball"},
    ]
```

## Success Criteria

- [ ] All 25+ tests defined
- [ ] All tests have docstrings
- [ ] Tests cover 10 categories
- [ ] Tests pass on CPU (mocked if needed)
- [ ] Tests pass on GPU (RUN_MODEL_TESTS=1) **KAGGLE ONLY - DO NOT RUN ON LAPTOP**
- [ ] Coverage > 95% for radar.py
- [ ] Base64 encoding validates
- [ ] All colors match design spec

## ⚠️ IMPORTANT: GPU Tests on Kaggle Only

**RUN_MODEL_TESTS=1 LOCKS YOUR LAPTOP — DO NOT RUN LOCALLY**

```bash
# ❌ DO NOT RUN THIS ON YOUR LAPTOP:
RUN_MODEL_TESTS=1 pytest src/tests/ -v

# ✅ ONLY RUN ON KAGGLE GPU with timeout:
RUN_MODEL_TESTS=1 pytest src/tests/ -v --timeout=300
```

**Reason:** Loading YOLO models requires GPU memory. Without GPU, process hangs indefinitely.

**Agent Instructions:** Never invoke `RUN_MODEL_TESTS=1` tests. User will run on Kaggle only.

## Expected Coverage Impact

```
Before:  radar.py ≈ 35% coverage
After:   radar.py ≈ 95%+ coverage
```

## Notes
- Use fixtures for common frame/detection creation
- Mock YOLO model for CPU tests
- Test base64 encoding/decoding thoroughly
- Validate colors match design spec exactly
- Test boundary conditions (players at edges, off-pitch)
- Run GPU tests on Kaggle only
