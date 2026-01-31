# Test Plan Review Checklist

**Status:** READY FOR REVIEW  
**Created:** Week 3 Pre-Implementation  
**Files:**
1. `TEST_PLAN_CONSOLIDATION.md` — Remove 59 duplicates
2. `TEST_PLAN_PLAYER_TRACKING.md` — 30+ new tests
3. `TEST_PLAN_RADAR.md` — 25+ new tests

---

## Review Feedback Form

### Phase 1: Consolidation Plan

**Question 1:** Are the old test files correctly identified for deletion?
- [x] Yes, approved
- [ ] No, modifications needed (explain)

**Question 2:** Are the refactored versions comprehensive enough to replace old tests?
- [x] Yes, they cover all important cases
- [ ] No, suggest what's missing

**Question 3:** Is the migration checklist clear and complete?
- [ x] Yes
- [ ] No (explain)

**Feedback:**
```
also inclue ruff and mypy keep watch on test coverage %
```

---

### Phase 2: Player Tracking Test Plan

**Question 1:** Are the 8 test categories comprehensive?
- [ ] Yes, covers all important scenarios
- [ ] No, suggest additional categories (e.g., ___________)

**Question 2:** Is 30+ tests reasonable for player_tracking coverage?
- [ ] Yes
- [ ] No, should be (higher/lower): ___

**Question 3:** Do the test descriptions provide clear implementation guidance?
- [ ] Yes, very clear
- [ ] Somewhat clear, need clarification on: ___________
- [ ] No, unclear

**Question 4:** Are there edge cases I missed?
- [ ] No, comprehensive
- [ ] Yes: ________________________________________

**Question 5:** Is the test pattern template appropriate?
- [ ] Yes
- [ ] No, should use different pattern

**Feedback:**
```
[Your feedback here]
```

---

### Phase 3: Radar Test Plan

**Question 1:** Are the 10 test categories sufficient?
- [ ] Yes, covers all important areas
- [ ] No, missing: __________

**Question 2:** Is 25+ tests reasonable for radar coverage?
- [ ] Yes
- [ ] No, should be (higher/lower): ___

**Question 3:** Do the color tests match the design spec correctly?
- [ ] Yes (Team A: #00BFFF, Team B: #FF1493, GK: #FFD700, Ref: #FF6347)
- [ ] No, should be: ________

**Question 4:** Is base64 encoding/decoding testing adequate?
- [ ] Yes
- [ ] No, should also test: ________

**Question 5:** Are the test fixtures helpful?
- [ ] Yes, clear and reusable
- [ ] Could be improved: ________

**Feedback:**
```
[Your feedback here]
```

---

## Overall Assessment

### Completeness
- [ ] All three plans complete
- [ ] Plans cover all identified gaps (59 consolidate + 30 tracking + 25 radar)
- [ ] Tests align with manifest requirements

### Quality
- [ ] Clear, specific test descriptions
- [ ] Appropriate level of detail
- [ ] Implementation guidance provided
- [ ] Edge cases considered

### Feasibility
- [ ] Plans are achievable in Week 3
- [ ] No unreasonable dependencies
- [ ] CPU/GPU test separation clear
- [ ] Fixtures and patterns established

### Test Coverage Expected
- [ ] Consolidation: Removes duplicates without losing coverage
- [ ] Player Tracking: 6 → 30+ tests, coverage 40% → 90%+
- [ ] Radar: 3 → 25+ tests, coverage 35% → 95%+
- [ ] Overall impact: ~75% → 80%+ coverage

---

## Reviewer Sign-Off

**Reviewer Name:** ___________________

**Date:** ___________________

**Approval:**
- [x] ✅ APPROVED — Proceed with implementation
- [ ] ✅ APPROVED WITH SUGGESTIONS — See feedback above
- [ ] ❌ REJECTED — Major revisions needed (explain)

**Summary Comment:**
```
✅ APPROVED - All three phases ready for Week 3 implementation.
- Phase 1 (Consolidation): Clear migration path, no test loss
- Phase 2 (Player Tracking): 30+ tests, all behaviors covered
- Phase 3 (Radar): 25+ tests, design spec validated
- Target: 75% → 80%+ coverage

Important: GPU tests (RUN_MODEL_TESTS=1) run ONLY on Kaggle with --timeout=300
CPU tests run on laptop with --timeout=60
Agent will NOT run GPU tests.
```

---

## Next Steps After Approval

### If APPROVED:

#### On Your Laptop (CPU Tests Only):

1. **Create test files:**
   ```bash
   cd plugins/forgesyte-yolo-tracker
   
   # Consolidation
   rm src/tests/test_inference_player_detection.py
   rm src/tests/test_inference_ball_detection.py
   rm src/tests/test_inference_pitch_detection.py
   
   # Player Tracking (create/expand)
   touch src/tests/test_inference_player_tracking_refactored.py
   
   # Radar (expand)
   # Edit src/tests/test_inference_radar.py to add 25+ tests
   ```

2. **Implement tests in order:**
   - Phase 1: Consolidation (1-2 hours)
   - Phase 2: Player Tracking (3-4 hours)
   - Phase 3: Radar (3-4 hours)

3. **Run quality checks:**
   ```bash
   # Lint
   uv run ruff check src/ --fix
   
   # Type check
   uv run mypy src/
   
   # Format
   uv run black src/
   ```

4. **Run CPU tests only (safe on laptop) with coverage:**
   ```bash
   pytest src/tests/ -v --timeout=60 --ignore=src/tests/integration/ \
     --cov=src/forgesyte_yolo_tracker \
     --cov-report=term-missing:skip-covered
   ```
   
   **Watch for:** Coverage ≥ 80% overall, ≥ 99% plugin.py

5. **Commit:**
   ```bash
   git add .
   git commit -m "test(yolo-tracker): Consolidate + expand tests for 80% coverage
   
   - Remove 59 duplicate tests
   - Add 30+ player_tracking tests
   - Add 25+ radar tests
   - Improve coverage: 75% → 80%+
   - Ruff lint clean
   - Mypy type check clean"
   ```

#### On Kaggle GPU (Run GPU Tests):

5. **Install pytest-timeout plugin:**
   ```bash
   uv pip install pytest-timeout
   ```

6. **Run all tests with timeout protection:**
   ```bash
   RUN_MODEL_TESTS=1 uv run pytest src/tests/ -v --timeout=300 --tb=short
   ```

7. **Quality checks on Kaggle:**
   ```bash
   # Final lint/format check
   uv run ruff check src/ --fix
   uv run mypy src/
   uv run black src/
   ```

8. **Verify coverage on Kaggle:**
   ```bash
   RUN_MODEL_TESTS=1 uv run pytest src/tests/ \
     --cov=src/forgesyte_yolo_tracker \
     --cov-report=term-missing:skip-covered \
     --cov-report=html \
     --timeout=300
   ```
   
   **Target:**
   - Overall: ≥ 80% ✅
   - plugin.py: ≥ 99% ✅
   - player_tracking.py: ≥ 90% ✅
   - radar.py: ≥ 95% ✅
   
   Check HTML report: `htmlcov/index.html`

#### ⚠️ CRITICAL: GPU Tests Kaggle-Only

**DO NOT RUN `RUN_MODEL_TESTS=1` ON YOUR LAPTOP — IT WILL HANG/LOCK**

- Agent will only run CPU tests (safe)
- You will run GPU tests on Kaggle (with timeout protection)
- Timeout = 300 seconds for GPU, 60 seconds for CPU

### If CHANGES REQUESTED:

1. **Update affected plans** based on feedback
2. **Re-submit for review**
3. **Once approved, proceed with implementation**

---

## Questions?

If unclear about any test, ask before implementing:
- Test descriptions can be clarified
- Implementation approach can be discussed
- Edge cases can be validated
- Fixtures can be refined

**Goal:** Avoid surprises during implementation → maximize speed during Week 3 on Kaggle.
