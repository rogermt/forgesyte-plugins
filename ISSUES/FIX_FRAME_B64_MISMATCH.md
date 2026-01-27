# Fix Plan: frame_base64 vs frame_b64 Argument Mismatch

## Problem

The web-ui sends `frame_base64` but the plugin tool methods expect `frame_b64`, causing a `TypeError` and 500 error.

| Component | Argument Name |
|-----------|---------------|
| manifest.json | `frame_base64` ✅ |
| Web-UI useVideoProcessor | `frame_base64` ✅ |
| Plugin tool methods | `frame_b64` ❌ |
| Module-level functions | `frame_base64` ✅ |

## Files to Change

1. `plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/plugin.py`
   - Lines 359, 363, 367, 371, 375 (Plugin class tool methods)

## Fix

Change all Plugin class tool method signatures from `frame_b64` to `frame_base64`:

### Before
```python
def player_detection(self, frame_b64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
    return player_detection(frame_b64, device, annotated)
```

### After
```python
def player_detection(self, frame_base64: str, device: str = "cpu", annotated: bool = False) -> Dict[str, Any]:
    return player_detection(frame_base64, device, annotated)
```

## Methods to Update

| Line | Method | Current | Fix To |
|------|--------|---------|--------|
| 359 | `player_detection` | `frame_b64` | `frame_base64` |
| 363 | `player_tracking` | `frame_b64` | `frame_base64` |
| 367 | `ball_detection` | `frame_b64` | `frame_base64` |
| 371 | `pitch_detection` | `frame_b64` | `frame_base64` |
| 375 | `radar` | `frame_b64` | `frame_base64` |

## Verification

1. Run lint: `uv run ruff check src/ --fix`
2. Run tests: `uv run pytest src/tests/ -v` (skip GPU tests)
3. Push and test on Kaggle with real models

## Risk

Low - just renaming parameter, logic unchanged.
