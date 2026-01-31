# Test Consolidation Plan (Phase 2)

## Overview
Remove ~59 duplicate tests by consolidating old test files into refactored versions.

## Tests to DELETE (Old Duplicates)

### 1. src/tests/test_inference_player_detection.py
**Status:** ❌ DELETE (replaced by refactored version)  
**Tests:** 10 tests  
**Reason:** Redundant with refactored version; less comprehensive

```
test_returns_dict()
test_returns_expected_keys()
test_detections_list()
test_detections_have_required_fields()
test_detections_bounding_boxes()
test_detections_confidence()
test_returns_annotated_frame()
test_annotated_frame_is_ndarray()
test_annotated_frame_shape()
test_device_parameter()
```

### 2. src/tests/test_inference_ball_detection.py
**Status:** ❌ DELETE (replaced by refactored version)  
**Tests:** 11 tests  
**Reason:** Redundant with refactored version; less comprehensive

```
test_returns_dict()
test_returns_expected_keys()
test_detections_list()
test_detections_have_required_fields()
test_ball_confidence()
test_ball_radius()
test_ball_position()
test_returns_annotated_frame()
test_annotated_frame_is_ndarray()
test_device_parameter()
test_confidence_filtering()
```

### 3. src/tests/test_inference_pitch_detection.py
**Status:** ❌ DELETE (replaced by refactored version)  
**Tests:** 4 tests  
**Reason:** Redundant with refactored version

```
test_returns_dict()
test_returns_expected_keys()
test_detections_list()
test_detections_have_required_fields()
```

### 4. Duplicate test_inference_team_classification.py (if exists)
**Status:** ❌ DELETE  
**Tests:** ~34 tests  
**Reason:** Identified as duplicates in analysis

## Tests to KEEP (Refactored Versions)

### 1. src/tests/test_inference_player_detection_refactored.py
**Status:** ✅ KEEP  
**Tests:** 35 tests  
**Improvements:**
- Comprehensive edge case coverage
- Multi-player scenarios
- Confidence threshold validation
- JSON schema validation
- Batch processing support

### 2. src/tests/test_inference_ball_detection_refactored.py
**Status:** ✅ KEEP  
**Tests:** 32 tests  
**Improvements:**
- Ball tracking across frames
- Multiple ball detection
- Confidence threshold edge cases
- JSON schema validation
- Device parameter variations

### 3. src/tests/test_inference_pitch_detection_refactored.py
**Status:** ✅ KEEP  
**Tests:** 33 tests  
**Improvements:**
- Pitch corner detection
- Line detection
- Perspective transform validation
- JSON schema validation
- Edge cases (partial pitch, occluded lines)

### 4. src/tests/test_inference_team_classification_refactored.py
**Status:** ✅ KEEP  
**Tests:** 28 tests  
**Improvements:**
- Team color classification
- On-the-fly UMAP + KMeans
- Multi-team scenarios
- Goalkeeper detection
- JSON schema validation

## Migration Checklist

### Per Old File:
- [ ] Review old test file
- [ ] Check for unique tests NOT in refactored version
- [ ] Migrate unique tests to refactored version
- [ ] Verify all unique tests are now covered
- [ ] Delete old file
- [ ] Run tests: `uv run pytest src/tests/test_inference_*.py -v`

### Validation:
- [ ] All refactored test files pass
- [ ] No test loss (migrate unique tests)
- [ ] Old files deleted
- [ ] Coverage maintained or improved

## Expected Results

```
Before Consolidation:
  - 10 + 11 + 4 + 34 = 59 duplicate tests
  - ~180 tests in inference modules

After Consolidation:
  - 0 duplicate tests
  - ~128 tests in inference modules (refactored + consolidated)
  - Same coverage, less duplication
```

## Commands

```bash
cd plugins/forgesyte-yolo-tracker

# ✅ CPU TESTS ONLY (safe to run on laptop):
uv run pytest src/tests/test_inference_*.py -v --timeout=60

# Step 2: Compare old vs refactored
diff -u src/tests/test_inference_player_detection.py \
        src/tests/test_inference_player_detection_refactored.py

# Step 3: Delete old files
rm src/tests/test_inference_player_detection.py
rm src/tests/test_inference_ball_detection.py
rm src/tests/test_inference_pitch_detection.py

# Step 4: Verify refactored tests still pass (CPU only)
uv run pytest src/tests/test_inference_*_refactored.py -v --timeout=60

# Step 5: Run full suite (CPU only on laptop)
uv run pytest src/tests/ -v --timeout=60 --ignore=src/tests/integration/
```

## ⚠️ GPU Tests on Kaggle Only

**DO NOT RUN WITH RUN_MODEL_TESTS=1 ON YOUR LAPTOP**

```bash
# ❌ DO NOT RUN:
RUN_MODEL_TESTS=1 uv run pytest src/tests/ -v

# ✅ ONLY ON KAGGLE GPU with timeout:
RUN_MODEL_TESTS=1 uv run pytest src/tests/ -v --timeout=300
```

## Timeout Configuration

Add to `pyproject.toml`:
```toml
[tool.pytest.ini_options]
timeout = 60  # CPU tests: 60 seconds
```

## Notes
- Keep all unique tests from old files
- Verify no functionality is lost
- Refactored versions are comprehensive baseline
- This is mechanical consolidation, not optimization
- **AGENT:** Never run `RUN_MODEL_TESTS=1` — user will run on Kaggle GPU
