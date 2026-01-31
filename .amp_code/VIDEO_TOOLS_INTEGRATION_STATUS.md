# Video Tools Integration Status

**Date:** 2026-01-31  
**Status:** Plugin registry updated, server tests need fixes  
**Impact:** GPU tests on Kaggle will fail until server integration tests are updated

---

## Summary

Plugin now supports 10 tools (5 frame + 5 video). Frame tools use `frame_base64` input, video tools use `video_path`/`output_path`. Tests must differentiate between tool types.

---

## Files Fixed ✅

### forgesyte-plugins

| File | Change | Status |
|------|--------|--------|
| `plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/plugin.py` | Added 5 video tool handlers with lazy-loaded imports | ✅ FIXED |
| `plugins/forgesyte-yolo-tracker/tests_contract/test_plugin_schema_contract.py` | Conditional validation: frame vs video tools | ✅ FIXED |
| `plugins/forgesyte-yolo-tracker/tests_contract/test_tool_registry_contract.py` | Conditional validation: frame vs video tools | ✅ FIXED |

**Test Results:** 83 passed locally  
**Quality:** ruff clean, mypy clean

---

## Files Still Failing ❌

### forgesyte/server

| File | Location | Issue | Impact | Fix Required |
|------|----------|-------|--------|--------------|
| `tests/integration/test_video_tracker.py` | Line 160-168 | `test_manifest_tool_has_frame_base64_input()` - Hardcoded assertion: `"frame_base64" in tool["inputs"]` for ALL tools | **Will FAIL on GPU** when video tools exist in manifest | Add conditional: if "video" in tool_name → check video_path/output_path, else → check frame_base64 |
| `app/models.py` | Line 249 | PluginToolRunRequest example shows frame_base64 only | API docs misleading (not critical) | Add note: video tools use different schema OR create separate example |
| `app/api.py` | Line 548 | POST /plugins/{id}/tools/{tool}/run example shows frame_base64 | OpenAPI docs misleading (not critical) | Add note in docstring |

---

## Test Analysis

### Currently Passing (Why?)

| Test | Reason |
|------|--------|
| `tests/contract/test_yolo_image_json_safe.py` | Filters by "image" in tool_name - video tools ignored |
| `tests/contract/test_yolo_video_json_safe.py` | Skips gracefully if no video tool found |
| `tests/api/test_plugins_run_yolo_tools_cpu.py` | Tests specific frame tools only, doesn't iterate all |

### Will Fail on GPU

| Test | Reason |
|------|--------|
| `test_manifest_tool_has_frame_base64_input` | Loops all tools, asserts frame_base64 exists (video tools have video_path) |

---

## Fix Instructions

### Priority 1: CRITICAL (Blocks GPU tests)

**File:** `/home/rogermt/forgesyte/server/tests/integration/test_video_tracker.py:160`

**Current:**
```python
async def test_manifest_tool_has_frame_base64_input(self, client_with_mock_yolo):
    """Each tool should accept frame_base64"""
    response = await client_with_mock_yolo.get("/v1/plugins/yolo-tracker/manifest")
    manifest = response.json()

    for tool_name, tool in manifest["tools"].items():
        assert (
            "frame_base64" in tool["inputs"]
        ), f"Tool {tool_name} missing frame_base64 input"
```

**Fix:**
```python
async def test_manifest_tool_has_frame_base64_input(self, client_with_mock_yolo):
    """Frame tools should accept frame_base64, video tools should have video_path"""
    response = await client_with_mock_yolo.get("/v1/plugins/yolo-tracker/manifest")
    manifest = response.json()

    for tool_name, tool in manifest["tools"].items():
        if "video" in tool_name:
            # Video tools need video_path and output_path
            assert "video_path" in tool["inputs"], f"Video tool {tool_name} missing video_path"
            assert "output_path" in tool["inputs"], f"Video tool {tool_name} missing output_path"
        else:
            # Frame tools need frame_base64
            assert (
                "frame_base64" in tool["inputs"]
            ), f"Frame tool {tool_name} missing frame_base64 input"
```

### Priority 2: NICE-TO-HAVE (Documentation)

**File:** `/home/rogermt/forgesyte/server/app/models.py:249`  
**Action:** Add comment explaining frame tools only, video tools differ

**File:** `/home/rogermt/forgesyte/server/app/api.py:548`  
**Action:** Add docstring note about tool-specific schemas

---

## Verification Steps

1. **Local CPU tests:**
   ```bash
   cd /home/rogermt/forgesyte/server
   uv run pytest tests/integration/test_video_tracker.py -v
   ```

2. **GPU tests on Kaggle:**
   ```bash
   cd /kaggle/working/forgesyte
   git pull
   RUN_MODEL_TESTS=1 uv run pytest tests/contract/test_yolo_video_json_safe.py -v
   RUN_MODEL_TESTS=1 uv run pytest tests/integration/test_video_tracker.py -v
   ```

3. **Quality checks:**
   ```bash
   uv run ruff check app/ tests/ --fix
   uv run mypy app/ --no-site-packages
   ```

---

## Schema Differences

### Frame Tools (player_detection, ball_detection, pitch_detection, player_tracking, radar)
```python
input_schema: {
    "frame_base64": {"type": "string"},  # Required: base64 encoded image
    "device": {"type": "string", "default": "cpu"},
    "annotated": {"type": "boolean", "default": False}
}
output_schema: {
    "result": {"type": "object"}  # JSON-safe result dict
}
```

### Video Tools (player_detection_video, ball_detection_video, etc.)
```python
input_schema: {
    "video_path": {"type": "string"},  # Required: file path to video
    "output_path": {"type": "string"},  # Required: output file path
    "device": {"type": "string", "default": "cpu"}
}
output_schema: {
    "status": {"type": "string"},  # "success" or error
    "output_path": {"type": "string"}  # Path to generated video
}
```

---

## Timeline

- **✅ Jan 31:** Plugin updated, contract tests fixed, PR #82 merged
- **⏳ Next:** Fix server integration test before GPU tests run
- **⏳ GPU Phase:** Run all tests with RUN_MODEL_TESTS=1 on Kaggle

