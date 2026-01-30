# Issue #79: Pytest Coverage Refactor — AWAITING APPROVAL

**Status:** Plan Ready for Review  
**Completed:** Commit 1 (pyproject.toml coverage config)  
**Pending:** Commits 2-3 (test reorganization + AGENTS.md)

---

## What's Done ✅

**Commit 1:** `chore(yolo-tracker): Add coverage configuration to pyproject.toml (fixes #79)`

- [x] Added `[tool.coverage.run]` with exclusions
- [x] Added `[tool.coverage.report]` configuration  
- [x] Tests verified: **203 passed, 218 skipped**
- [x] Coverage: **99%** (plugin.py contract only, internals excluded)

**Result:** Single source of truth for coverage (no more ad-hoc .coveragerc files)

---

## What's Pending ⏳

**Commit 2:** Reorganize test directory structure + Create pytest.ini & Makefile

```
Current:
plugins/forgesyte-yolo-tracker/
└── src/tests/
    ├── test_plugin.py
    ├── test_inference_*.py  (model-dependent)
    ├── test_video_*.py      (heavy)
    ├── test_*_refactored.py (legacy)
    └── utils/

Proposed:
plugins/forgesyte-yolo-tracker/
├── tests_contract/          ← Run in CI (fast, ~2 min)
│   ├── test_plugin.py
│   ├── test_manifest.py
│   ├── test_plugin_schema.py
│   └── ...
├── tests_heavy/             ← Skip in CI (slow, optional, ~15 min)
│   ├── inference/
│   ├── utils/
│   ├── video/
│   ├── legacy/
│   └── integration/
├── pytest.ini               ← testpaths = tests_contract (CI-safe)
├── Makefile                 ← make test-fast / make test-all
└── pyproject.toml           ← Coverage config (Commit 1 ✅)
```

**Commit 3:** Update pytest config + AGENTS.md

- Add `[tool.pytest.ini_options]` with `testpaths = ["src/tests/contract"]`
- Update AGENTS.md test runner commands
- Reference /docs/design/TEST_FLOW.md

---

## Rationale

### **Problem**
- Tests are scattered with no clear separation
- Model-dependent tests run in CI (slow, flaky)
- No single pytest config (uses env vars)

### **Solution**
- Directory separation: **contract/** vs **heavy/**
- Pytest defaults to contract tests only
- CI: ~2 min (contract)
- Manual full run: ~15 min (contract + heavy)

### **Benefits**
- Faster CI/CD
- Clear developer intent
- No regression (heavy tests preserved)
- Aligns with industry standards

---

## Request

**Please review and approve:**

1. ✅ Coverage config (already committed)
2. ⏳ Directory reorganization strategy
3. ⏳ Pytest config changes
4. ⏳ Documentation updates

See full plan: `/home/rogermt/forgesyte-plugins/.amp_code/ISSUE_79_PYTEST_COVERAGE_REFACTOR.md`

---

## Next Steps (if approved)

```bash
# Commit 2: Move tests
mkdir -p src/tests/contract src/tests/heavy
git mv src/tests/test_plugin*.py src/tests/contract/
git mv src/tests/test_manifest.py src/tests/contract/
# ... move other contract tests
git mv src/tests/test_inference*.py src/tests/heavy/
# ... move other heavy tests
git commit -m "refactor(yolo-tracker): Separate contract and heavy tests"

# Commit 3: Update pytest config
# Update pyproject.toml [tool.pytest.ini_options]
# Update AGENTS.md
git commit -m "chore(yolo-tracker): Update pytest config for contract-first testing"
```

---

**Awaiting your approval to proceed.**
