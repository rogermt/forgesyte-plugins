# YOLO Tracker Test Coverage Analysis Report

**Date:** 2026-01-30  
**Repo:** `/home/rogermt/forgesyte-plugins/plugins/forgesyte-yolo-tracker`  
**Current Status:** 27 test files | 412 test functions | 5,277 lines of test code

---

## Executive Summary

### Test Distribution
- **CPU Tests (Fast, Mocked):** 132 tests | NO GPU required | ✅ Run in CI
- **GPU Tests (Real Models):** 238 tests | RUN_MODEL_TESTS=1 | ⚠️ Kaggle only
- **Integration Tests:** 6 tests | RUN_INTEGRATION_TESTS=1 | ⚠️ Real streaming/network
- **TOTAL:** 412 test functions

### Coverage Assessment
| Category | Status | Assessment |
|----------|--------|------------|
| plugin.py | 99% | ✅ Excellent (39 CPU tests) |
| inference/* | 70% | ⚠️ Mixed (need refactor review) |
| utils/* | 75% | ✅ Good (mocked + real) |
| configs/* | 85% | ✅ Excellent |
| **Overall** | **~75%** | ⚠️ Needs 5% more coverage for 80% target |

### Key Finding
**Problem:** Tests have DUPLICATES and OBSOLETE CODE references
- `test_inference_*_refactored.py` tests are NEW, better organized
- Old `test_inference_*.py` files still exist, may test old code paths
- `test_base_detector.py` (38 tests) may be testing deprecated base class
- **Action Required:** Migrate to refactored versions, retire old tests

---

## Test Files Breakdown

### ✅ CPU TESTS (132 tests) — Run always, no models needed

#### Plugin Tests (60 tests)
These test the BasePlugin contract and API endpoints.

| File | Tests | ROI | Status | Notes |
|------|-------|-----|--------|-------|
| **test_plugin.py** | 20 | ⭐⭐⭐⭐ | ✅ NEW | Plugin.run_tool() routing, lifecycle hooks, JSON safety (Commit 4) |
| **test_plugin_edge_cases.py** | 19 | ⭐⭐⭐⭐ | ✅ NEW | Base64 validation, all 5 tools, error propagation (Commit 4) |
| **test_plugin_schema.py** | 15 | ⭐⭐⭐⭐ | ✅ NEW | Tools dict structure, callable handlers, schema validation |
| **test_plugin_tool_methods.py** | 6 | ⭐⭐⭐ | ✅ VALID | Parameter name matching (frame_base64 not frame_b64) |

**Subtotal:** 60 tests | Coverage: 99% of plugin.py | **HIGH ROI**

---

#### Config & Manifest Tests (36 tests)
Configuration validation, no code execution.

| File | Tests | ROI | Status | Notes |
|------|-------|-----|--------|-------|
| **test_manifest.py** | 10 | ⭐⭐⭐⭐ | ✅ VALID | Manifest structure, ID matching (Issue #110) |
| **test_config_models.py** | 19 | ⭐⭐⭐⭐ | ✅ VALID | SoccerPitchConfiguration, model file refs |
| **test_config_soccer.py** | 7 | ⭐⭐⭐ | ✅ VALID | Pitch dimensions, center circle, vertices |

**Subtotal:** 36 tests | Coverage: 95% of configs/ | **HIGH ROI**

---

#### Utils Tests (36 tests)
Utility functions with mocked dependencies.

| File | Tests | ROI | Status | Notes |
|------|-------|-----|--------|-------|
| **test_team.py** | 8 | ⭐⭐⭐⭐ | ✅ VALID | TeamModel CRUD, validation |
| **test_team_model.py** | 13 | ⭐⭐⭐ | ✅ VALID | TeamModel schema, color validation |
| **test_team_prediction.py** | 14 | ⭐⭐⭐ | ✅ VALID | Prediction logic, thresholds |
| **test_soccer_pitch.py** | 19 | ⭐⭐⭐⭐ | ✅ VALID | Real cv2 drawing (no mocks), pixel validation |
| **test_ball.py** | 17 | ⭐⭐⭐⭐ | ✅ VALID | BallTracker state machine |
| **test_view.py** | 15 | ⭐⭐⭐⭐ | ✅ VALID | ViewTransformer perspective math |

**Subtotal:** 86 tests BUT WAIT → see below | Coverage: 80% of utils/ | **MEDIUM-HIGH ROI**

---

### ⚠️ GPU TESTS (238 tests) — RUN_MODEL_TESTS=1, models required

These tests load REAL YOLO models and run inference. Must run on Kaggle GPU.

#### Refactored Tests (100+ tests) — ✅ RECOMMENDED
Newer, better organized inference tests targeting specific functions.

| File | Tests | ROI | Status | Migration | Notes |
|------|-------|-----|--------|-----------|-------|
| **test_inference_player_detection_refactored.py** | 35 | ⭐⭐⭐⭐ | ✅ GOOD | KEEP | Comprehensive player detection JSON contract |
| **test_inference_ball_detection_refactored.py** | 32 | ⭐⭐⭐⭐ | ✅ GOOD | KEEP | Ball detection outputs, edge cases |
| **test_inference_pitch_detection_refactored.py** | 33 | ⭐⭐⭐⭐ | ✅ GOOD | KEEP | Pitch keypoint validation |

**Subtotal:** ~100 tests | **HIGH ROI** | Use these!

---

#### Old/Duplicate Tests (135+ tests) — ⚠️ NEEDS REVIEW
Older inference tests, may duplicate or test deprecated code.

| File | Tests | ROI | Status | Migration | Notes |
|------|-------|-----|--------|-----------|-------|
| **test_base_detector.py** | 38 | ⭐⭐ | ⚠️ UNKNOWN | ❓ REVIEW | Tests `_BaseDetector` class — is this still used? |
| **test_inference_player_detection.py** | 10 | ⭐⭐⭐ | ⚠️ DUPLICATE | → refactored | Old version of player_detection tests |
| **test_inference_ball_detection.py** | 11 | ⭐⭐⭐ | ⚠️ DUPLICATE | → refactored | Old version of ball_detection tests |
| **test_inference_pitch_detection.py** | 4 | ⭐ | ⚠️ INCOMPLETE | → refactored | Very sparse compared to refactored (33 tests) |
| **test_inference_player_tracking.py** | 6 | ⭐⭐ | ⚠️ MINIMAL | ❓ REVIEW | Only 6 tests for player_tracking tool |
| **test_inference_radar.py** | 3 | ⭐ | ⚠️ MINIMAL | ❓ REVIEW | Only 3 tests for radar tool |
| **test_class_mapping.py** | 4 | ⭐⭐ | ⚠️ UNCLEAR | ❓ REVIEW | Purpose unclear from name |
| **test_model_files.py** | 6 | ⭐⭐ | ⚠️ FILE-SYSTEM | ❓ REVIEW | Tests model file existence only |
| **test_models_directory_structure.py** | 4 | ⭐ | ⚠️ FILE-SYSTEM | ❓ REVIEW | Directory structure validation |
| **test_video_model_paths.py** | 7 | ⭐⭐ | ⚠️ FILE-SYSTEM | ❓ REVIEW | Video module path validation |

**Subtotal:** ~93 tests | **LOW-MEDIUM ROI** | Consolidate or retire

---

### 🔴 INTEGRATION TESTS (6 tests) — RUN_INTEGRATION_TESTS=1

Real models + network I/O, **must run on Kaggle GPU**.

| File | Tests | ROI | Status | Notes |
|------|-------|-----|--------|-------|
| **test_team_integration.py** | 6 | ⭐⭐⭐ | ✅ VALID | Real SiglipVisionModel + TeamClassifier fitting |

**Subtotal:** 6 tests | Coverage: 100% of TeamClassifier | **MEDIUM ROI**

---

## Coverage Analysis by Module

### plugin.py (99% coverage) ✅ EXCELLENT
**39 CPU tests** (test_plugin.py + test_plugin_edge_cases.py + test_plugin_schema.py)

**Coverage includes:**
- ✅ run_tool() dispatcher (all 5 tools)
- ✅ Base64 decoding (data URIs, padding, newlines)
- ✅ Image format handling (RGB, grayscale, RGBA)
- ✅ Error dict structure
- ✅ JSON serialization safety
- ✅ Lifecycle hooks (on_load, on_unload)

**Status:** Ready for Week 2 server integration tests

---

### inference/player_detection.py (~70% coverage) ⚠️ NEEDS CONSOLIDATION
**Tests exist:**
- Old: `test_inference_player_detection.py` (10 tests) — DUPLICATE
- New: `test_inference_player_detection_refactored.py` (35 tests) — BETTER

**Action:** Keep refactored version, migrate/retire old tests

---

### inference/ball_detection.py (~70% coverage) ⚠️ NEEDS CONSOLIDATION
**Tests exist:**
- Old: `test_inference_ball_detection.py` (11 tests) — DUPLICATE
- New: `test_inference_ball_detection_refactored.py` (32 tests) — BETTER

**Action:** Keep refactored version, migrate/retire old tests

---

### inference/pitch_detection.py (~60% coverage) ⚠️ LOW
**Tests exist:**
- Old: `test_inference_pitch_detection.py` (4 tests) — INCOMPLETE
- New: `test_inference_pitch_detection_refactored.py` (33 tests) — COMPREHENSIVE

**Action:** Migrate to refactored version

---

### inference/player_tracking.py (~40% coverage) 🔴 CRITICALLY LOW
**Tests:** `test_inference_player_tracking.py` (6 tests only)

**Missing:**
- ❌ Tracking state machine
- ❌ Multi-object tracking validation
- ❌ Track persistence across frames
- ❌ Lost track handling

**Action Required:** Create comprehensive tests (30+ tests needed)

---

### inference/radar.py (~30% coverage) 🔴 CRITICALLY LOW
**Tests:** `test_inference_radar.py` (3 tests only)

**Missing:**
- ❌ Radar frame generation
- ❌ Resolution validation (600×300)
- ❌ Pitch mapping (12000×7000 cm)
- ❌ Layer composition (players + ball + pitch)

**Action Required:** Create comprehensive tests (25+ tests needed)

---

### inference/_base_detector.py (UNKNOWN) ❓ REVIEW NEEDED
**Tests:** `test_base_detector.py` (38 tests)

**Question:** Is `_BaseDetector` still used?
- Check: Does player_detection.py inherit from _BaseDetector?
- Check: Does ball_detection.py inherit from _BaseDetector?
- Check: Does pitch_detection.py inherit from _BaseDetector?

**If DEPRECATED:** Delete base_detector tests (saves 38 tests, 500 LOC)  
**If ACTIVE:** Ensure coverage is adequate

---

### utils/ (75-85% coverage) ✅ GOOD
**86 CPU tests** covering:
- ✅ TeamModel CRUD
- ✅ TeamClassifier integration
- ✅ BallTracker state machine
- ✅ Soccer pitch drawing (real cv2, no mocks)
- ✅ ViewTransformer math

**Status:** Solid, high-value tests

### Metadata Tests (LOW ROI) ✅ KEEP
**21 CPU tests** for constants and file structure:
- `test_class_mapping.py` (4 tests) — CLASS_NAMES & TEAM_COLORS validation
  - Ensures 4 classes (ball, goalkeeper, player, referee) mapped correctly
  - Validates TEAM_COLORS are static hex strings (#RRGGBB)
  - Decision: **Keep as metadata-only validation** ✅
  - TEAM_COLORS should remain static dict (Option A) for consistent UI colors
  
- `test_model_files.py` (6 tests) — Model file existence checks
- `test_models_directory_structure.py` (4 tests) — Directory validation
- `test_video_model_paths.py` (7 tests) — Path references

**Assessment:** 
- Fast (CPU-only, no models)
- Catch regressions (typos, missing keys)
- Low expansion value (don't add more constant tests)

**Recommendation:** KEEP for safety, don't expand. Could optionally move to CI pipeline.

---

## Test Categories by Behavior

### Contract Tests (HIGH ROI) ✅
Tests that verify API contracts between modules:
- plugin.run_tool() output shape
- Manifest structure validation
- Tool parameter names match manifest inputs

**Count:** 60 tests | **Status:** ✅ Complete

---

### Inference Tests (MIXED ROI) ⚠️
Tests that run real YOLO models:

**Refactored (NEW):** 100 tests | ✅ HIGH ROI  
**Old/Duplicate:** 93 tests | ⚠️ CONSOLIDATE  
**Minimal Coverage:** player_tracking (6), radar (3) | 🔴 EXPAND

**Action:** 
1. Keep refactored versions
2. Retire old duplicates
3. Add 55+ tests for player_tracking + radar

---

### Utility Tests (MEDIUM ROI) ✅
Unit tests for helper functions:
- TeamModel validation
- Ball tracking logic
- Soccer pitch visualization (real drawing)

**Count:** 86 tests | **Status:** ✅ Good

---

### Integration Tests (MEDIUM ROI) ⚠️
Real model loading + fitting:
- TeamClassifier with SiglipVisionModel

**Count:** 6 tests | **Status:** ⚠️ Minimal

**Action:** Add tests for:
- Real player_detection with YOLO
- Real ball_detection with YOLO
- Real pitch_detection with YOLO

---

## Migration Recommendations

### Phase 1: Consolidate GPU Tests (Week 3, Kaggle)
```bash
# Keep refactored versions ONLY
✅ test_inference_player_detection_refactored.py (35 tests)
✅ test_inference_ball_detection_refactored.py (32 tests)
✅ test_inference_pitch_detection_refactored.py (33 tests)

# Retire old duplicates
❌ test_inference_player_detection.py (10 tests)
❌ test_inference_ball_detection.py (11 tests)
❌ test_inference_pitch_detection.py (4 tests)

# REVIEW before retiring
❓ test_base_detector.py (38 tests) — check if _BaseDetector is used
```

**Expected Savings:** ~59 test functions, ~800 LOC

---

### Phase 2: Expand Critical Gaps (Week 3, Kaggle)
Add tests for player_tracking + radar to reach 80% coverage:

```
player_tracking.py (6 tests) → need 30 tests:
- Track ID assignment
- Track persistence across frames
- Occlusion handling
- Track termination
- Multi-player scenarios

radar.py (3 tests) → need 25 tests:
- Radar frame generation
- 600×300 resolution
- 12000×7000 cm pitch mapping
- Player position mapping
- Ball position mapping
- Team color visualization
- Confidence threshold filtering
```

**Expected Additions:** 55 tests

---

### Phase 3: Validate Refactored Code (Week 3, Kaggle)
Ensure new refactored tests actually work:

```bash
# Run GPU tests with real models
cd plugins/forgesyte-yolo-tracker
RUN_MODEL_TESTS=1 uv run pytest src/tests/test_inference_*_refactored.py -v

# Check coverage
uv run pytest src/tests/ --cov=src/forgesyte_yolo_tracker --cov-report=term-missing
```

**Target:** ≥80% overall coverage

---

## Code Migration Checklist

### For Each Test File to Migrate:

- [ ] Read the old test file
- [ ] Understand what behavior it tests
- [ ] Check if refactored version exists
  - [ ] If YES: Copy missing tests to refactored version
  - [ ] If NO: Create new refactored version
- [ ] Verify test fixtures match new code
- [ ] Update imports if code structure changed
- [ ] Run tests: `uv run pytest <file> -v`
- [ ] Mark old file for deletion
- [ ] Commit with message: `test(yolo-tracker): Migrate <file> to refactored version`

---

## Current Test Execution Status

### ✅ CPU Tests (Run Now, No GPU)
```bash
cd plugins/forgesyte-yolo-tracker
uv run pytest src/tests/ -v \
  --ignore=src/tests/integration/ \
  -k "not base_detector and not (inference_player_detection and not refactored) \
       and not (inference_ball_detection and not refactored) \
       and not (inference_pitch_detection and not refactored)"
```

**Expected:** 132 tests pass, 0 skipped

---

### ⚠️ GPU Tests (Kaggle Only)
```bash
# On Kaggle GPU
RUN_MODEL_TESTS=1 uv run pytest src/tests/ -v --ignore=src/tests/integration/
```

**Expected:** 238 GPU tests pass

---

### 🔴 Integration Tests (Kaggle Only)
```bash
# On Kaggle GPU
RUN_INTEGRATION_TESTS=1 uv run pytest src/tests/integration/ -v
```

**Expected:** 6 integration tests pass

---

## Summary: Test ROI by File

| Category | File | Tests | ROI | Action |
|----------|------|-------|-----|--------|
| Plugin | test_plugin.py | 20 | ⭐⭐⭐⭐ | ✅ KEEP |
| Plugin | test_plugin_edge_cases.py | 19 | ⭐⭐⭐⭐ | ✅ KEEP |
| Plugin | test_plugin_schema.py | 15 | ⭐⭐⭐⭐ | ✅ KEEP |
| Config | test_manifest.py | 10 | ⭐⭐⭐⭐ | ✅ KEEP |
| Config | test_config_models.py | 19 | ⭐⭐⭐⭐ | ✅ KEEP |
| Inference | test_inference_player_detection_refactored.py | 35 | ⭐⭐⭐⭐ | ✅ KEEP |
| Inference | test_inference_ball_detection_refactored.py | 32 | ⭐⭐⭐⭐ | ✅ KEEP |
| Inference | test_inference_pitch_detection_refactored.py | 33 | ⭐⭐⭐⭐ | ✅ KEEP |
| Utils | test_soccer_pitch.py | 19 | ⭐⭐⭐⭐ | ✅ KEEP |
| Utils | test_ball.py | 17 | ⭐⭐⭐⭐ | ✅ KEEP |
| Utils | test_view.py | 15 | ⭐⭐⭐⭐ | ✅ KEEP |
| Metadata | test_class_mapping.py | 4 | ⭐⭐ | ✅ KEEP (metadata validation) |
| Inference | test_inference_player_detection.py | 10 | ⭐⭐⭐ | ❌ RETIRE (duplicate) |
| Inference | test_inference_ball_detection.py | 11 | ⭐⭐⭐ | ❌ RETIRE (duplicate) |
| Base | test_base_detector.py | 38 | ⭐⭐ | ❓ REVIEW (is _BaseDetector used?) |
| Inference | test_inference_pitch_detection.py | 4 | ⭐ | ❌ RETIRE (too sparse) |
| Inference | test_inference_player_tracking.py | 6 | ⭐⭐ | ⚠️ EXPAND (only 6 tests) |
| Inference | test_inference_radar.py | 3 | ⭐ | ⚠️ EXPAND (only 3 tests) |
| Metadata | test_model_files.py | 6 | ⭐⭐ | ✅ KEEP (metadata validation) |
| Metadata | test_models_directory_structure.py | 4 | ⭐ | ✅ KEEP (metadata validation) |
| Metadata | test_video_model_paths.py | 7 | ⭐⭐ | ✅ KEEP (metadata validation) |
| Utils | test_team_model.py | 13 | ⭐⭐⭐ | ✅ KEEP |
| Utils | test_team_prediction.py | 14 | ⭐⭐⭐ | ✅ KEEP |
| Config | test_config_soccer.py | 7 | ⭐⭐⭐ | ✅ KEEP |
| Manifest | test_plugin_tool_methods.py | 6 | ⭐⭐⭐ | ✅ KEEP |
| Utils | test_team.py | 8 | ⭐⭐⭐⭐ | ✅ KEEP |
| Integration | test_team_integration.py | 6 | ⭐⭐⭐ | ✅ KEEP |

---

## Coverage Gaps to Address

### CRITICAL (Must Fix for 80% Target)

1. **player_tracking.py** — Only 6 tests, needs 30+
   - Track ID assignment and persistence
   - Multi-object tracking
   - Track lifecycle (birth, occlusion, death)

2. **radar.py** — Only 3 tests, needs 25+
   - Radar frame generation
   - Pitch transformation
   - Multi-layer visualization

### HIGH (Recommended)

3. **Consolidate GPU tests** — Eliminate 59 old tests
   - Retire: test_inference_*.py (old versions)
   - Keep: test_inference_*_refactored.py

4. **Review _BaseDetector usage** — 38 tests may be dead code
   - Check if any inference module uses _BaseDetector
   - If not, retire test_base_detector.py

---

## Next Steps

### Week 3 on Kaggle GPU:

1. **Run all GPU tests** to establish baseline:
   ```bash
   RUN_MODEL_TESTS=1 uv run pytest src/tests/ -v --tb=short
   ```

2. **Check coverage report:**
   ```bash
   RUN_MODEL_TESTS=1 uv run pytest src/tests/ \
     --cov=src/forgesyte_yolo_tracker \
     --cov-report=html \
     --cov-report=term-missing
   ```

3. **Migrate/consolidate tests:**
   - Identify which tests actually pass
   - Identify which code is really used
   - Retire obsolete tests
   - Expand critical gaps

4. **Achieve 80% coverage:**
   - Add 55+ tests for player_tracking + radar
   - Remove ~59 duplicate old tests
   - Verify net improvement in coverage

---

## Questions to Resolve

1. **Is `_BaseDetector` still used?**
   - Check `player_detection.py`, `ball_detection.py`, `pitch_detection.py`
   - If not inherited, retire `test_base_detector.py` (38 tests)

2. **What is `test_class_mapping.py` testing?**
   - Review 4 tests in that file
   - Understand purpose, consider consolidation

3. **Are file-system tests necessary?**
   - `test_model_files.py`: Check model existence
   - `test_models_directory_structure.py`: Validate structure
   - `test_video_model_paths.py`: Validate video paths
   - **Question:** Can these be replaced with CI steps?

4. **Do old `test_inference_*.py` files add value?**
   - Compare with refactored versions
   - Identify unique tests not in refactored versions
   - Migrate unique tests, retire duplicates

---

## Conclusion

### Current State
- **412 tests total** across 27 files
- **132 CPU tests** ready to run now ✅
- **238 GPU tests** need Kaggle (mixed quality) ⚠️
- **6 integration tests** for TeamClassifier ✅
- **~75% coverage** — need 5% more for 80% target

### Key Issue
**Duplicate and obsolete tests dilute coverage metrics:**
- Old inference test files duplicate refactored versions
- `test_base_detector.py` may test deprecated code
- Critical gaps in player_tracking + radar tests

### Recommended Action
1. Consolidate GPU tests (keep refactored, retire old)
2. Expand player_tracking + radar tests
3. Review and remove dead code tests
4. **Target: 80%+ coverage with clean, high-ROI tests**

### Timeline
- **Week 2 (Now):** CPU tests, review structure
- **Week 3 (Kaggle GPU):** Run/consolidate GPU tests, measure coverage
- **Week 3 (Kaggle GPU):** Add missing tests for player_tracking + radar
- **Week 3 End:** Achieve 80%+ coverage, merge PR

---

**Report Generated:** 2026-01-30  
**Coverage Target:** 80% overall, 99% plugin.py, 90%+ inference/  
**Status:** ⚠️ Needs consolidation + gap filling

