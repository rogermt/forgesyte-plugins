# Issue #164: Phase 12 Root Cause Analysis — Multiple Divergences

**Date:** 2026-02-09
**Main branches:** forgesyte-plugins `ac5b980`, forgesyte `b1c91a0`
**Kaggle error:** `ValueError: Unknown tool: default`

---

## Executive Summary

Three distinct issues prevent YOLO plugin from working:

1. **Tool name mismatch** (NEW) - Server sends `"default"`, plugin rejects it
2. **Image bytes key mismatch** - Server sends `image_bytes`, manifest declares `frame_base64`
3. **Device hardcoding** - Device config in models.yaml never used

---

## Issue 1: Tool Name Mismatch (BLOCKING)

### Current State (2026-02-09)

**Server (forgesyte/main/b1c91a0)** — [tasks.py L387](file:///home/rogermt/forgesyte/server/app/tasks.py#L387)
```python
tool_name = options.get("tool", "default")  # Always sends "default"
tool_args = {
    "image_bytes": image_bytes,
    "options": {k: v for k, v in options.items() if k != "tool"},
}
result = await loop.run_in_executor(
    self._executor, plugin.run_tool, tool_name, tool_args
)
```
→ Passes `tool_name="default"`

**Plugin (forgesyte-plugins/main/ac5b980)** — [plugin.py L341-342](file:///home/rogermt/forgesyte-plugins/plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/plugin.py#L341-L342)
```python
def run_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
    if tool_name not in self.tools:
        raise ValueError(f"Unknown tool: {tool_name}")  # ← REJECTS "default"
```
→ `self.tools = {"player_detection": {...}, "player_tracking": {...}, ...}`
→ `"default"` is NOT in tools dict → **ValueError**

### Root Cause

The plugin alias handling was **removed** (cleaned up) but the server still sends `"default"`.

**Old code (before ac5b980):**
```python
if tool_name == "default":
    tool_name = next(iter(self.tools.keys()))  # Maps "default" → "player_detection"
```

**Current code (ac5b980):**
- Alias logic deleted
- Direct tool name lookup only
- Server not updated

### Kaggle Error Log

```json
{
  "timestamp": "2026-02-09T18:30:47.176914+00:00",
  "name": "app.tasks",
  "message": "Job failed with exception",
  "exc_info": "...plugin.py\", line 342, in run_tool\n    raise ValueError(f\"Unknown tool: {tool_name}\")\nValueError: Unknown tool: default"
}
```

---

## Issue 2: Image Bytes Key Mismatch (SECONDARY)

### Current State

**Server sends:** `{"image_bytes": <bytes>, ...}` ✓
**Plugin expects:** Reads `args.get("image_bytes")` ✓
**Manifest declares:** `"frame_base64": "string"` ✗

**The manifest is outdated** — it declares `frame_base64` input but plugin reads `image_bytes`.

This won't cause a crash (plugin validates type correctly at L356), but the manifest is incorrect documentation.

---

## Issue 3: Device Hardcoding (TERTIARY)

**Problem:** Device defaults to `"cpu"` throughout, ignoring models.yaml config.

**Code path:**
1. Server defaults to "cpu" (api.py L124)
2. Server does NOT pass device in tool_args (tasks.py L388-391)
3. Plugin falls back to "cpu" (plugin.py L364)
4. models.yaml device: "cuda" is never read in request path

---

## Fix Priority

### Immediate (P0 - Blocking)

**Fix Issue 1:** Tool name mismatch

**Option A: Server sends proper tool name**
- Change tasks.py L387: `tool_name = options.get("tool", "player_detection")`
- Pros: Server decides default tool name
- Cons: Hard to change later

**Option B: Plugin accepts "default" alias**
- Add back: `if tool_name == "default": tool_name = next(iter(self.tools.keys()))`
- Pros: Backward compatible
- Cons: Keeps "legacy" code (see #99)

**Recommended:** Option A (Phase 12 removes aliases per #99)

### Secondary (P1 - Documentation)

**Fix Issue 2:** Update manifest to match code

**Change:** All tool inputs from `"frame_base64": "string"` to `"image_bytes": "bytes"`

### Tertiary (P2 - Enhancement)

**Fix Issue 3:** Pass device through pipeline

**See:** GitHub issue #100 (device from models.yaml)

---

## Files Requiring Changes

| File | Repo | Issue | Action |
|------|------|-------|--------|
| `server/app/tasks.py` L387 | forgesyte | #1 | Change default tool_name |
| `plugins/.../plugin.py` | forgesyte-plugins | #1 | DONE (already expects non-default) |
| `plugins/.../manifest.json` | forgesyte-plugins | #2 | Update input schema |
| `server/app/api.py` | forgesyte | #3 | Pass device in tool_args |
| `plugins/.../plugin.py` | forgesyte-plugins | #3 | Use device from args |

---

## Testing Strategy

1. **Local (CPU):** Run contract tests with corrected tool_name
2. **Kaggle (GPU):** Deploy and test with real YOLO models
3. **Web-UI:** Verify tool selector sends proper tool name (not "default")

---

## Timeline

- **Phase 12:** Fix #1 + #2 (tool name + manifest) — BLOCKING
- **Phase 13:** Fix #3 (device handling) — ENHANCEMENT

---

## Related Issues

- #99 - Remove backward compatibility aliases (already implemented)
- #100 - Use device from models.yaml
- #164 - Original YOLO crash report (multiple root causes)
