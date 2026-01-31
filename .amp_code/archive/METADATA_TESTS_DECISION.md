# Metadata Tests Decision — test_class_mapping.py & Friends

**Decision Date:** 2026-01-30  
**Status:** ✅ FINAL DECISION

---

## Question: What to Do with test_class_mapping.py?

### What It Tests
- **CLASS_NAMES** dict: {0: "ball", 1: "goalkeeper", 2: "player", 3: "referee"}
- **TEAM_COLORS** dict: hex color strings (#RRGGBB format)
- All 4 classes have proper mappings

### Why It Exists
Catch common mistakes:
- Typos in class names
- Duplicate class IDs
- Invalid hex color codes
- Missing color definitions

---

## Decision: ✅ KEEP as Metadata Validation Test

### Rationale

**Keep because:**
1. **Fast** — CPU-only, no models, instant execution
2. **Safety** — Catches regressions in constants
3. **Low burden** — Only 4 tests, no maintenance needed
4. **Prevents breakage** — Catches typos that would break visualization

**Don't expand because:**
1. **Low ROI** — Not testing core behavior
2. **Stable constants** — Unlikely to need many more tests
3. **Better uses of time** — Focus on player_tracking + radar gaps

---

## TEAM_COLORS Design Decision: Option A ✅

**TEAM_COLORS should remain STATIC DICT** (not model-derived)

```python
TEAM_COLORS = {
    0: "#FF0000",   # ball (red)
    1: "#FFD700",   # goalkeeper (gold)
    2: "#00BFFF",   # player Team A (blue)
    3: "#FF1493",   # player Team B (pink)
}
```

**Why Option A is correct:**

1. **Consistent UI** — Same colors across:
   - Radar view
   - Bounding box overlays
   - Pitch visualization
   - Frontend annotations

2. **Simple & stable** — No runtime changes
3. **Deterministic** — Same input → same visual output
4. **Performance** — O(1) lookup, no computation

**NOT Option B** (model-derived colors):
- ❌ Would require recomputing colors on each run
- ❌ Would introduce non-determinism
- ❌ Would complicate testing
- ❌ Would break UI consistency

---

## Related Metadata Tests: Also Keep ✅

Same reasoning applies to:

| File | Tests | Decision | Why |
|------|-------|----------|-----|
| test_class_mapping.py | 4 | ✅ KEEP | Constant validation |
| test_model_files.py | 6 | ✅ KEEP | File existence checks |
| test_models_directory_structure.py | 4 | ✅ KEEP | Path validation |
| test_video_model_paths.py | 7 | ✅ KEEP | Path references |

**Total metadata tests:** 21 (keep all)

---

## Classification

**Type:** Metadata/Structural Validation Tests  
**Category:** LOW-ROI but HIGH-VALUE (safety)  
**ROI Rating:** ⭐⭐ (low expansion, high stability)  
**Action:** ✅ KEEP, don't expand, don't delete

---

## Week 3 Action

### ✅ DO KEEP
```bash
uv run pytest src/tests/test_class_mapping.py -v
```

### ❌ DON'T DELETE
Removing would lose validation of critical constants.

### ❌ DON'T EXPAND
No need for 20 more constant-mapping tests.

### ✅ OPTIONAL
Could move to CI pipeline as pre-test checks instead.

---

## Summary

**test_class_mapping.py Status:**
- ✅ Keeps
- ✅ Useful for regression prevention
- ✅ Validates TEAM_COLORS are static hex strings
- ✅ Part of essential "sanity check" suite
- ⭐⭐ LOW expansion ROI (but HIGH stability value)

**TEAM_COLORS Design:**
- ✅ Static dict (Option A)
- ✅ Consistent across UI
- ✅ Simple & maintainable
- ✅ Validated by test_class_mapping.py

---

**Final Verdict:** Keep test_class_mapping.py as-is. Focus Week 3 effort on player_tracking & radar gap expansion instead.

