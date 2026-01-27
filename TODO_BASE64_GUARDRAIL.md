# Backend Guardrail Implementation - TODO Tracker

## PHASE 0 - Setup & Branch Creation
- [x] Navigate to plugins repo: cd /home/rogermt/forgesyte-plugins
- [x] git checkout main && git fetch origin main && git reset --hard origin/main
- [x] git checkout -b feature/base64-guardrail
- [x] Verify pytest works: cd plugins/forgesyte-yolo-tracker && pytest --version

## PHASE 1 - RED: Create Failing Tests
Create: plugins/forgesyte-yolo-tracker/tests/test_base64_guardrail.py

### P0 Tests (expect 400, not 500):
- [x] test_invalid_characters_in_base64
- [x] test_truncated_base64
- [x] test_empty_string
- [x] test_non_base64_string

## PHASE 2 - GREEN: Implementation Complete ✅
Modify: plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/plugin.py

- [x] Add logging import
- [x] Add _validate_base64() helper function
- [x] Add _decode_frame_base64_safe() function
- [x] Wrap decode in try/except
- [x] Return {"error": "invalid_base64", ...} with 400 status

### Tools Updated:
- [x] player_detection
- [x] player_tracking
- [x] ball_detection
- [x] pitch_detection
- [x] radar
- [x] Plugin.analyze()

## PHASE 3 - REFACTOR (Optional Future)
- [ ] Create utils/base64_utils.py with validation logic (deferred)
- [ ] Update imports in plugin.py (deferred)
- [ ] Add structured logging for validation failures (done)
- [ ] Verify tests still pass (pending manual verification)

## PHASE 4 - Final Verification (Manual)
- [x] All P0 tests pass: `pytest tests/test_base64_guardrail.py -v` (11 passed)
- [x] No 500 errors for malformed base64
- [x] Plugin returns 400 with structured error
- [x] Logs contain debugging info
- [x] git diff main shows only guardrail changes

## PHASE 5 - Test Fixes (Completed)
Fixed `src/tests/test_plugin.py`:
- Added missing mocks for `detect_ball_json` and `detect_pitch_json` in all analyze tests
- Previously only `detect_players_json` was mocked, causing unpatched detection functions to try loading corrupted model stubs
- All 142 tests now pass (197 skipped for model tests)

## Quick Start Commands
```bash
cd /home/rogermt/forgesyte-plugins
git checkout main && git fetch origin main && git reset --hard origin/main
git checkout -b feature/base64-guardrail
cd plugins/forgesyte-yolo-tracker
pytest --version
pytest tests/test_base64_guardrail.py -v  # Run tests
```

## File Locations
- Tests: plugins/forgesyte-yolo-tracker/tests/test_base64_guardrail.py
- Plugin: plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/plugin.py
- Utils: plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/utils/base64_utils.py (pending)

