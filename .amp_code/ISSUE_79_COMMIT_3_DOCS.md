# Commit 3: Documentation Updates for Issue #79

**Status:** Plan Only (executes after Commit 2)  
**Depends On:** Commit 2 (test reorganization)  
**Type:** `docs` commit

---

## Commit Message

```
docs(yolo-tracker): Update docs for contract-first testing (issue #79)

Update documentation to reflect new test organization:
- tests_contract/ (CI-safe, fast, ~2 min)
- tests_heavy/ (optional, slow, ~15 min)
- pytest.ini (default to contract tests)
- Makefile (make test-fast, make test-all)

Changes:
- Update AGENTS.md testing section
- Update CONTRIBUTING.md with testing philosophy
- Update ARCHITECTURE.md with test flow reference
- Cleanup obsolete test planning documents

This reflects the final state of issue #79:
✅ Coverage config in pyproject.toml
✅ Tests organized into tests_contract/ and tests_heavy/
✅ pytest.ini defaults to contract tests
✅ Makefile provides convenient test commands
✅ Documentation complete

Closes #79
```

---

## Files to Stage and Commit

### New Files (Add)
- `.amp_code/ISSUE_79_PYTEST_COVERAGE_REFACTOR.md` - Full implementation plan
- `.amp_code/ISSUE_79_APPROVAL_SUMMARY.md` - Project summary
- `docs/design/TEST_FLOW.md` - Test architecture diagram (already exists)

### Files to Delete (from unstaged)
- `.amp_code/IMPLEMENT_WEEK3.md`
- `.amp_code/METADATA_TESTS_DECISION.md`
- `.amp_code/README_TEST_COVERAGE.md`
- `.amp_code/REVIEW_CHECKLIST.md`
- `.amp_code/STATUS.md`
- `.amp_code/TEST_COVERAGE_ANALYSIS.md`
- `.amp_code/TEST_MIGRATION_CHECKLIST.md`
- `.amp_code/TEST_PLAN_CONSOLIDATION.md`
- `.amp_code/TEST_PLAN_PLAYER_TRACKING.md`
- `.amp_code/TEST_PLAN_RADAR.md`
- `.amp_code/TEST_SUMMARY.txt`
- `.amp_code/UPDATES.md`
- `docs/design/TEST_RCHITECTURE.md` (typo: "RCHITECTURE" → already have TEST_FLOW.md)

### Files to Update (Edit)
- `AGENTS.md` - Testing section (from Commit 2)
- `CONTRIBUTING.md` - Add "Testing Philosophy" section
- `ARCHITECTURE.md` - Reference new test structure

---

## Changes to Make

### 1. AGENTS.md Updates

**Section: "Running Tests"**
```markdown
### Running Tests

**Contract tests only (CI-safe, fast):**
```bash
uv run pytest src/tests/contract/ -v
```

**Heavy tests only (requires GPU/network):**
```bash
uv run pytest src/tests/heavy/ -v
```

**All tests (contract + heavy):**
```bash
uv run pytest src/tests/ -v
```

**With coverage (contract only):**
```bash
uv run pytest src/tests/contract/ -v --cov --cov-report=term-missing
```
```

### 2. CONTRIBUTING.md - New Section

Add under "## Testing" or create new section:

```markdown
## Testing Philosophy

ForgeSyte plugins use **contract‑driven testing**:
- **Contract tests** (required, CI): Test plugin API, dispatch, schemas
- **Heavy tests** (optional): Test YOLO internals, inference, utilities

See [Test Architecture](docs/design/TEST_FLOW.md) for detailed breakdown.

### Running Tests

See [AGENTS.md#Running Tests](AGENTS.md#running-tests)
```

### 3. ARCHITECTURE.md - Add Reference

```markdown
## Testing Architecture

The YOLO Tracker test suite is organized into two categories:

1. **Contract Tests** (`src/tests/contract/`)
   - Plugin API, dispatch, schema validation
   - Run in CI, included in coverage

2. **Heavy Tests** (`src/tests/heavy/`)
   - YOLO inference, video processing, utilities
   - Optional, excluded from CI

See [Test Execution Flow](docs/design/TEST_FLOW.md) for details.
```

---

## Commit Checklist

- [ ] Commit 2 is merged and tests pass
- [ ] New plan docs are complete
- [ ] AGENTS.md is updated
- [ ] CONTRIBUTING.md is updated  
- [ ] ARCHITECTURE.md is updated
- [ ] Obsolete docs are deleted
- [ ] Quality checks pass:
  - `ruff check docs/ --fix`
  - No markdown syntax errors
- [ ] Commit created with message above
- [ ] Issue #79 is closed

---

## Commands to Run (after Commit 2)

```bash
cd /home/rogermt/forgesyte-plugins

# Stage new docs
git add .amp_code/ISSUE_79_PYTEST_COVERAGE_REFACTOR.md
git add .amp_code/ISSUE_79_APPROVAL_SUMMARY.md
git add docs/design/TEST_FLOW.md

# Delete obsolete docs
git rm .amp_code/IMPLEMENT_WEEK3.md
git rm .amp_code/METADATA_TESTS_DECISION.md
git rm .amp_code/README_TEST_COVERAGE.md
git rm .amp_code/REVIEW_CHECKLIST.md
git rm .amp_code/STATUS.md
git rm .amp_code/TEST_COVERAGE_ANALYSIS.md
git rm .amp_code/TEST_MIGRATION_CHECKLIST.md
git rm .amp_code/TEST_PLAN_CONSOLIDATION.md
git rm .amp_code/TEST_PLAN_PLAYER_TRACKING.md
git rm .amp_code/TEST_PLAN_RADAR.md
git rm .amp_code/TEST_SUMMARY.txt
git rm .amp_code/UPDATES.md
git rm docs/design/TEST_RCHITECTURE.md

# Update main docs (after editing above files)
git add AGENTS.md
git add CONTRIBUTING.md
git add ARCHITECTURE.md

# Quality check
uv run ruff check docs/ --fix

# Commit
git commit -m "docs(yolo-tracker): Update docs for contract-first testing (issue #79)

Update documentation to reflect new test organization:
- src/tests/contract/ (CI-safe, required)
- src/tests/heavy/ (optional, skipped in CI)

Changes:
- Update AGENTS.md testing section
- Update CONTRIBUTING.md with testing philosophy
- Update ARCHITECTURE.md with test flow reference
- Cleanup obsolete test planning documents

This reflects the final state of issue #79:
✅ Coverage config in pyproject.toml
✅ Tests organized into contract/heavy
✅ Pytest defaults to contract tests
✅ Documentation complete

Closes #79"
```

---

## Notes

- This commit happens **after** Commit 2 (test reorganization) passes
- All quality checks must pass before committing
- This is the final commit for issue #79
- After this, issue #79 is closed
