# Week 3 Implementation Notes — Quick Reference

**Task:** Consolidate tests + expand gaps → reach 80%+ coverage  
**Timeline:** Week 3 on Kaggle GPU  
**Status:** Ready to execute

---

## 🎯 Three Main Actions

### 1. Consolidate Duplicate Tests (Phase 2)
**Remove:** ~59 old tests, keep refactored versions

```bash
# Tests to DELETE (old duplicates)
❌ src/tests/test_inference_player_detection.py (10 tests)
❌ src/tests/test_inference_ball_detection.py (11 tests)
❌ src/tests/test_inference_pitch_detection.py (4 tests)

# Tests to KEEP (refactored, better)
✅ src/tests/test_inference_player_detection_refactored.py (35 tests)
✅ src/tests/test_inference_ball_detection_refactored.py (32 tests)
✅ src/tests/test_inference_pitch_detection_refactored.py (33 tests)
```

**Action:** Compare old vs refactored, migrate unique tests, delete old files.

---

### 2. Expand Coverage Gaps (Phase 3)
**Add:** 55 new tests for player_tracking + radar

#### player_tracking: 6 → 30 tests
```bash
# CREATE: src/tests/test_inference_player_tracking_refactored.py (30+ tests)

Test areas:
  • Track ID assignment & uniqueness
  • Track persistence across frames
  • Occlusion handling (player disappears, reappears)
  • Track lifecycle (birth, death, resumption)
  • Multi-player scenarios (5+ players)
  • Confidence threshold filtering
  • JSON output validation
  • Device parameter (cpu, cuda)
```

#### radar: 3 → 25 tests
```bash
# EXPAND: src/tests/test_inference_radar.py (25+ tests)

Test areas:
  • Radar frame generation (600×300 px)
  • Pitch mapping (12000×7000 cm scale)
  • Player position visualization
  • Ball position visualization
  • Team colors (Team A: #00BFFF, Team B: #FF1493)
  • Goalkeeper: #FFD700, Referee: #FF6347
  • Confidence threshold filtering
  • Base64 encoding of output
  • Multi-object scenarios
```

---

### 3. Verify Coverage (Phase 5)
```bash
# Run GPU tests
RUN_MODEL_TESTS=1 pytest src/tests/ -v --ignore=src/tests/integration/

# Generate coverage report
RUN_MODEL_TESTS=1 pytest src/tests/ \
  --cov=src/forgesyte_yolo_tracker \
  --cov-report=html \
  --cov-report=term-missing:skip-covered

# Verify targets
✅ Overall coverage ≥ 80%
✅ plugin.py coverage ≥ 99%
✅ All tests passing
```

---

## 🚀 Quick Commands

```bash
# Phase 1: Run all GPU tests
cd plugins/forgesyte-yolo-tracker
RUN_MODEL_TESTS=1 pytest src/tests/ -v --tb=short

# Phase 2: Consolidate (manual process)
# Compare old vs refactored, migrate unique tests, delete old

# Phase 3: Add new tests
# Create/expand test files per above

# Phase 5: Verify coverage
RUN_MODEL_TESTS=1 pytest src/tests/ \
  --cov=src/forgesyte_yolo_tracker \
  --cov-report=term-missing

# Commit
git add .
git commit -m "test(yolo-tracker): Consolidate + expand tests for 80% coverage"
```

---

## ✅ Success Criteria

- [ ] All CPU tests pass (132)
- [ ] All GPU tests pass (238)
- [ ] Coverage ≥ 80% overall
- [ ] Coverage ≥ 99% for plugin.py
- [ ] Zero duplicate test files
- [ ] player_tracking ≥ 30 tests
- [ ] radar ≥ 25 tests
- [ ] All tests have docstrings
- [ ] Merged to main

---

## 📊 Expected Results

```
Before:
  412 tests, 75% coverage, 93 duplicates

After:
  408 tests, 80%+ coverage, 0 duplicates
```

---

## 📁 Reference Docs

Location: `/home/rogermt/forgesyte-plugins/.amp_code/`

- `TEST_MIGRATION_CHECKLIST.md` — 5-phase detailed plan
- `TEST_COVERAGE_ANALYSIS.md` — Full technical analysis
- `METADATA_TESTS_DECISION.md` — test_class_mapping.py decision
- `TEST_SUMMARY.txt` — Quick overview

---

## 🎯 Focus Areas

**HIGH PRIORITY:**
1. Consolidate duplicates (easy win, -59 tests)
2. Expand player_tracking (30 tests)
3. Expand radar (25 tests)

**MEDIUM PRIORITY:**
4. Review _BaseDetector usage (keep or delete 38 tests?)
5. Validate integration tests

**LOW PRIORITY:**
- Metadata tests (21 tests) — keep as-is

---

**Start:** Phase 1 (validate GPU tests)  
**Timeline:** 2-3 days on Kaggle  
**Target:** 80%+ coverage ✅

