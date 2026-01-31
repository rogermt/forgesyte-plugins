# Analysis Updates — 2026-01-30 Final

**Update:** Reviewed test_class_mapping.py with user

---

## ✅ Final Decisions Made

### 1. test_class_mapping.py Status
**Decision:** ✅ KEEP

**Rationale:**
- Tests critical constants (CLASS_NAMES, TEAM_COLORS)
- Fast CPU-only validation
- Catches regressions (typos, missing keys)
- Low maintenance burden (only 4 tests)

**Don't expand:** No need for 20+ constant tests

---

### 2. TEAM_COLORS Design
**Decision:** Option A — Static Dict ✅

**Implementation:**
```python
TEAM_COLORS = {
    0: "#FF0000",   # ball
    1: "#FFD700",   # goalkeeper
    2: "#00BFFF",   # player Team A
    3: "#FF1493",   # player Team B
}
```

**Why:**
- Consistent UI across radar, bounding boxes, overlays
- Simple & stable (no runtime changes)
- Deterministic & performance-friendly
- Validated by test_class_mapping.py

---

### 3. Metadata Tests Classification
**Decision:** ✅ KEEP ALL (21 tests)

| File | Tests | Status |
|------|-------|--------|
| test_class_mapping.py | 4 | ✅ KEEP |
| test_model_files.py | 6 | ✅ KEEP |
| test_models_directory_structure.py | 4 | ✅ KEEP |
| test_video_model_paths.py | 7 | ✅ KEEP |

**Reasoning:**
- Fast sanity checks
- Prevent common mistakes
- Low overhead (don't expand)
- Consider moving to CI pipeline (optional)

---

## 📊 Updated Test Distribution

| Category | Tests | Status | Action |
|----------|-------|--------|--------|
| Plugin (High ROI) | 60 | ✅ KEEP | Core functionality |
| Config (High ROI) | 36 | ✅ KEEP | Schema validation |
| Utils (High ROI) | 86 | ✅ KEEP | Helper functions |
| Inference Refactored (High ROI) | 100+ | ✅ KEEP | GPU tests |
| Metadata (Low ROI) | 21 | ✅ KEEP | Sanity checks |
| Integration (Medium ROI) | 6 | ✅ KEEP | Real models |
| **Keep Subtotal** | **309+** | — | — |
| **Duplicate (To Retire)** | **59** | ❌ RETIRE | Consolidate |
| **Gaps (To Expand)** | **55** | ⚠️ ADD | Week 3 |

---

## 📈 Coverage Trajectory (Updated)

### Current State
- 412 tests total
- 75% coverage
- 93 duplicate tests diluting metrics

### After Consolidation (Phase 2)
- 353 tests (412 - 59 duplicates)
- 76% coverage
- Cleaner codebase

### After Gap Expansion (Phase 3)
- 408 tests (353 + 55 new)
- 80%+ coverage ✅
- Comprehensive & clean

### Metadata Tests Impact
- Metadata tests NOT counted in expansion
- They remain as "safety harness" around changes
- 21 tests provide validation without bloating core coverage

---

## ✅ Final Answer to Your Question

**Q: Keep test_class_mapping.py?**  
**A:** Yes, keep it. ✅

**Q: TEAM_COLORS as static dict (Option A)?**  
**A:** Yes, exactly right. ✅

**Q: Why not Option B (model-derived)?**
```
❌ Adds complexity
❌ Non-deterministic
❌ Breaks UI consistency
❌ Makes testing harder
```

**Q: Expand metadata tests further?**  
**A:** No, keep them minimal. Focus on player_tracking + radar gaps instead. ✅

---

## 📚 New Document

**METADATA_TESTS_DECISION.md** — Detailed decision rationale for metadata tests

Location: `/home/rogermt/forgesyte-plugins/.amp_code/METADATA_TESTS_DECISION.md`

---

## 🚀 Week 3 Plan (Unchanged)

**Phase 1:** Validate GPU tests  
**Phase 2:** Consolidate duplicates (-59 tests)  
**Phase 3:** Expand gaps (+55 tests for player_tracking + radar)  
**Phase 4:** Validate integration tests  
**Phase 5:** Verify 80%+ coverage  

Metadata tests continue to validate all phases.

---

**Status:** ✅ Analysis complete with final decisions documented  
**Ready for:** Week 3 Kaggle GPU implementation

