# Test Migration & Coverage Gap Checklist

**Target:** 80% overall coverage by Week 3 (Kaggle GPU)

---

## Phase 1: Validate CPU Tests (Week 2, Local)

- [ ] Run CPU tests without GPU:
  ```bash
  cd plugins/forgesyte-yolo-tracker
  uv run pytest src/tests/ \
    --ignore=src/tests/integration/ \
    -k "not base_detector and not (inference and not refactored)" -v
  ```

- [ ] Verify **132 CPU tests** pass
  - [ ] Plugin tests: 60 ✅
  - [ ] Config tests: 36 ✅
  - [ ] Utils tests: 36 ✅

- [ ] Check that test_plugin.py covers plugin.py (99% target)
  ```bash
  uv run pytest src/tests/test_plugin*.py \
    --cov=src/forgesyte_yolo_tracker/plugin \
    --cov-report=term-missing
  ```

---

## Phase 2: Assess GPU Test Quality (Week 3, Kaggle)

### 2.1 Run All GPU Tests
```bash
RUN_MODEL_TESTS=1 uv run pytest src/tests/ \
  --ignore=src/tests/integration/ -v --tb=short
```

- [ ] Record pass/fail for each test file
- [ ] Note any model loading errors
- [ ] Identify hanging tests (these are GPU-heavy)

### 2.2 Identify Dead Code

**Check: Is `_BaseDetector` used?**
```bash
grep -r "_BaseDetector" src/forgesyte_yolo_tracker/inference/ --include="*.py"
```

- [ ] If NOT used: Mark test_base_detector.py (38 tests) for deletion
- [ ] If used: Keep and validate tests

**Check: What does test_class_mapping.py test?**
```bash
head -30 src/tests/test_class_mapping.py
```

- [ ] If unclear: Review + consolidate with other config tests

### 2.3 Consolidate Duplicate GPU Tests

#### Player Detection
- [ ] Review test_inference_player_detection.py (10 tests)
- [ ] Compare with test_inference_player_detection_refactored.py (35 tests)
- [ ] Copy any unique tests to refactored version
- [ ] Delete old version

#### Ball Detection
- [ ] Review test_inference_ball_detection.py (11 tests)
- [ ] Compare with test_inference_ball_detection_refactored.py (32 tests)
- [ ] Copy any unique tests to refactored version
- [ ] Delete old version

#### Pitch Detection
- [ ] Review test_inference_pitch_detection.py (4 tests)
- [ ] Compare with test_inference_pitch_detection_refactored.py (33 tests)
- [ ] Copy any unique tests to refactored version
- [ ] Delete old version

**Expected Result:** ~59 test functions deleted, consolidation complete

---

## Phase 3: Expand Critical Coverage Gaps (Week 3, Kaggle)

### 3.1 Expand player_tracking.py Tests

**Current:** 6 tests  
**Target:** 30 tests  
**File:** Create or expand `src/tests/test_inference_player_tracking_refactored.py`

**Missing test coverage:**
- [ ] Track ID assignment (unique per player)
- [ ] Track persistence across frames (same player = same ID)
- [ ] Track confidence and sorting
- [ ] Multi-player scenarios (5+ players)
- [ ] Occlusion handling (players going behind obstacles)
- [ ] Track termination (confidence drop below threshold)
- [ ] Track resumption (player re-appears)
- [ ] Edge cases (frame boundaries, extreme distances)
- [ ] JSON output validation
- [ ] Device parameter handling (cpu, cuda)

**Test Template:**
```python
import os
import pytest
import numpy as np
from unittest.mock import patch

from tests.constants import RUN_MODEL_TESTS, MODELS_EXIST

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS or not MODELS_EXIST,
    reason="Set RUN_MODEL_TESTS=1 AND download models to run",
)

class TestPlayerTrackingJSON:
    """Tests for track_players_json function."""
    
    def test_returns_dict_with_tracks(self) -> None:
        """Test returns dictionary with tracks key."""
        from forgesyte_yolo_tracker.inference.player_tracking import track_players_json
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = track_players_json(frame, device="cpu")
        assert isinstance(result, dict)
        assert "tracks" in result
    
    # Add 29+ more tests...
```

### 3.2 Expand radar.py Tests

**Current:** 3 tests  
**Target:** 25 tests  
**File:** Expand `src/tests/test_inference_radar.py`

**Missing test coverage:**
- [ ] Radar frame generation (600×300 pixels)
- [ ] Pitch mapping (12000×7000 cm scale)
- [ ] Player position visualization
- [ ] Ball position visualization
- [ ] Team color coding (Team A: #00BFFF, Team B: #FF1493)
- [ ] Goalkeeper color (#FFD700)
- [ ] Referee color (#FF6347)
- [ ] Confidence threshold filtering
- [ ] Radar with/without annotated output
- [ ] Multi-object scenarios
- [ ] Edge cases (empty frame, all players, no ball)
- [ ] JSON output validation
- [ ] Base64 encoding of radar frame

**Test Template:**
```python
class TestRadarJSON:
    """Tests for radar_json function."""
    
    def test_radar_frame_resolution(self) -> None:
        """Test radar frame is 600×300 pixels."""
        from forgesyte_yolo_tracker.inference.radar import radar_json
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = radar_json(frame, device="cpu")
        # result should have radar_frame_base64
        # decoded frame should be 600×300
    
    # Add 24+ more tests...
```

### 3.3 (Optional) Expand video module tests

**Current:** Minimal (0-3 tests per module)  
**If needed for 80%:** Create comprehensive video tests

---

## Phase 4: Integration Tests (Week 3, Kaggle)

### 4.1 Validate Existing Integration Tests
```bash
RUN_INTEGRATION_TESTS=1 uv run pytest src/tests/integration/ -v
```

- [ ] test_team_integration.py (6 tests) passes
- [ ] Real SiglipVisionModel loads correctly
- [ ] TeamClassifier fitting works

### 4.2 (Optional) Add Integration Tests for Inference

If targeting >85% coverage, add:
- [ ] Real player detection with actual video frame
- [ ] Real ball detection with actual video frame
- [ ] Real pitch detection with actual video frame

---

## Phase 5: Coverage Measurement (Week 3, Kaggle)

### 5.1 Generate Coverage Report
```bash
RUN_MODEL_TESTS=1 RUN_INTEGRATION_TESTS=1 uv run pytest src/tests/ \
  --cov=src/forgesyte_yolo_tracker \
  --cov-report=html \
  --cov-report=term-missing:skip-covered
```

- [ ] Generate HTML report
- [ ] Identify remaining gaps
- [ ] Verify 80%+ overall coverage
- [ ] Verify 99%+ plugin.py coverage
- [ ] Verify 85%+ inference/ coverage

### 5.2 Coverage by Module
Expected after all phases:
```
plugin.py                   99%  ✅
configs/                    95%  ✅
utils/                      85%  ✅
inference/player_detection  90%  ✅ (after consolidation)
inference/ball_detection    90%  ✅ (after consolidation)
inference/pitch_detection   90%  ✅ (after consolidation)
inference/player_tracking   75%  ✅ (after expansion)
inference/radar             80%  ✅ (after expansion)
video/                      70%  ⚠️ (defer to later)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OVERALL                      80%  ✅ TARGET
```

---

## Test Count Timeline

| Phase | CPU | GPU | Int | Dead | Total | Coverage |
|-------|-----|-----|-----|------|-------|----------|
| **Start** | 132 | 238 | 6 | 0 | 376 | ~75% |
| **After consolidation** | 132 | 179 | 6 | 59 | 317 | ~76% |
| **After expansion** | 132 | 234 | 12 | 59 | 378 | 80%+ |

**Note:** Net test count increases slightly, but quality improves significantly.

---

## File Status Tracker

### Plugin Tests
- [x] test_plugin.py (20 tests) — ✅ READY
- [x] test_plugin_edge_cases.py (19 tests) — ✅ READY
- [x] test_plugin_schema.py (15 tests) — ✅ READY
- [x] test_plugin_tool_methods.py (6 tests) — ✅ READY

### Config Tests
- [x] test_manifest.py (10 tests) — ✅ READY
- [x] test_config_models.py (19 tests) — ✅ READY
- [x] test_config_soccer.py (7 tests) — ✅ READY

### Utils Tests
- [x] test_soccer_pitch.py (19 tests) — ✅ READY
- [x] test_ball.py (17 tests) — ✅ READY
- [x] test_view.py (15 tests) — ✅ READY
- [x] test_team.py (8 tests) — ✅ READY
- [x] test_team_model.py (13 tests) — ✅ READY
- [x] test_team_prediction.py (14 tests) — ✅ READY
- [ ] test_class_mapping.py (4 tests) — ❓ REVIEW

### Inference Tests (GPU)
- [x] test_inference_player_detection_refactored.py (35) — ✅ KEEP
- [x] test_inference_ball_detection_refactored.py (32) — ✅ KEEP
- [x] test_inference_pitch_detection_refactored.py (33) — ✅ KEEP
- [ ] test_inference_player_detection.py (10) — ❌ RETIRE (duplicate)
- [ ] test_inference_ball_detection.py (11) — ❌ RETIRE (duplicate)
- [ ] test_inference_pitch_detection.py (4) — ❌ RETIRE (sparse)
- [ ] test_inference_player_tracking.py (6) — ⚠️ EXPAND (need 30)
- [ ] test_inference_radar.py (3) — ⚠️ EXPAND (need 25)

### Base/Meta Tests (GPU)
- [ ] test_base_detector.py (38) — ❓ REVIEW (is _BaseDetector used?)
- [ ] test_model_files.py (6) — ❓ REVIEW (necessary?)
- [ ] test_models_directory_structure.py (4) — ❓ REVIEW (necessary?)
- [ ] test_video_model_paths.py (7) — ❓ REVIEW (necessary?)

### Integration Tests
- [x] test_team_integration.py (6) — ✅ KEEP

---

## Commands for Each Phase

### Phase 1: Validate CPU
```bash
cd plugins/forgesyte-yolo-tracker
uv run pytest src/tests/ --ignore=src/tests/integration/ \
  -k "not base_detector and not (inference and not refactored)" -v
```

### Phase 2: Run GPU Tests on Kaggle
```bash
RUN_MODEL_TESTS=1 uv run pytest src/tests/ --ignore=src/tests/integration/ \
  -v --tb=short
```

### Phase 3: Check Coverage
```bash
RUN_MODEL_TESTS=1 uv run pytest src/tests/ \
  --cov=src/forgesyte_yolo_tracker \
  --cov-report=term-missing
```

### Phase 4: Full Report
```bash
RUN_MODEL_TESTS=1 RUN_INTEGRATION_TESTS=1 uv run pytest src/tests/ \
  --cov=src/forgesyte_yolo_tracker \
  --cov-report=html \
  --cov-report=term-missing:skip-covered
```

---

## Questions Before Proceeding

1. **Is _BaseDetector class still in use?**
   - If no → delete test_base_detector.py
   - If yes → keep but ensure comprehensive

2. **What inference modules are video-specific?**
   - Separate unit tests from video-only integration tests

3. **Should we test video frame processing?**
   - If not for Week 3 → focus on inference JSON contracts

4. **What's the priority?**
   - A) Consolidate + expand gaps (80% target) ✅ RECOMMENDED
   - B) Deep coverage of video modules (85%+ target)
   - C) Add stress tests for large videos

**Recommendation:** Focus on A) for Week 3, plan B) for future

---

## Success Criteria

- [ ] **All CPU tests pass locally** (132 tests)
- [ ] **All GPU tests pass on Kaggle** (238 tests)
- [ ] **Coverage ≥ 80%** overall
- [ ] **plugin.py coverage ≥ 99%**
- [ ] **No more than 380 total tests** (consolidation)
- [ ] **Zero duplicate tests** (consolidation complete)
- [ ] **player_tracking has 30+ tests**
- [ ] **radar has 25+ tests**
- [ ] **All tests have docstrings** explaining what they test

---

**Timeline:** Week 3 (Kaggle GPU)  
**Commit Message Format:** `test(yolo-tracker): Consolidate/expand tests for 80% coverage`  
**Target Merge:** After GPU tests pass + coverage verified

