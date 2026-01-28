# Migration Required: BasePlugin Metadata-Based Tools

**Status:** NOT STARTED  
**Scope:** 3 plugins  
**Related Issue:** rogermt/forgesyte#123

---

## Summary

The ForgeSyte server architecture was refactored to enforce strict schema validation on all plugins. Plugins must now define tools as **metadata dictionaries** instead of raw callables.

This issue tracks migration of the remaining 3 plugins to the new architecture.

---

## Background: What Changed

**Old Architecture (Deprecated):**
```python
class Plugin:
    name = "plugin_name"
    tools = {
        "analyze": self.analyze  # Just a callable
    }
```

**New Architecture (Required):**
```python
from app.plugins.base import BasePlugin

class Plugin(BasePlugin):
    name = "plugin_name"
    
    def __init__(self):
        self.tools = {
            "analyze": {
                "handler": self.analyze,  # Callable
                "description": "Human-readable description",
                "input_schema": {
                    "frame_base64": {"type": "string"}
                },
                "output_schema": {
                    "result": {"type": "object"}
                }
            }
        }
        super().__init__()
    
    def run_tool(self, tool_name: str, args: dict) -> Any:
        meta = self.tools[tool_name]
        return meta["handler"](**args)
```

**See:** [rogermt/forgesyte#123](https://github.com/rogermt/forgesyte/issues/123)

---

## Plugins to Migrate

### 1. **moderation** ⚠️

**File:** `plugins/moderation/src/forgesyte_moderation/plugin.py`

**Current Structure:**
- Class: `Plugin` (not BasePlugin)
- Method: `metadata()` returns `PluginMetadata` 
- Method: `analyze(image_bytes, options)` 
- Lifecycle: `on_load()`, `on_unload()`

**Required Changes:**

1. **Import BasePlugin**
   ```python
   from app.plugins.base import BasePlugin
   ```

2. **Inherit from BasePlugin**
   ```python
   class Plugin(BasePlugin):
   ```

3. **Define tools in __init__**
   ```python
   def __init__(self) -> None:
       self.sensitivity: str = "medium"
       self.tools = {
           "analyze": {
               "handler": self.analyze,
               "description": "Analyze image for content safety",
               "input_schema": {
                   "image_bytes": {
                       "type": "bytes",
                       "description": "Raw image data"
                   },
                   "sensitivity": {
                       "type": "string",
                       "default": "medium",
                       "enum": ["low", "medium", "high"],
                       "description": "Detection sensitivity level"
                   },
                   "categories": {
                       "type": "array",
                       "default": ["nsfw", "violence", "hate"],
                       "description": "Categories to check"
                   }
               },
               "output_schema": {
                   "text": {"type": "string"},
                   "blocks": {"type": "array"},
                   "confidence": {"type": "number"},
                   "error": {"type": "string", "nullable": True}
               }
           }
       }
       super().__init__()
   ```

4. **Implement run_tool()**
   ```python
   def run_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
       if tool_name == "analyze":
           meta = self.tools[tool_name]
           return meta["handler"](**args)
       raise ValueError(f"Unknown tool: {tool_name}")
   ```

5. **Keep metadata() method (for backward compatibility)**
   - Optional but recommended for documentation
   - Can still return PluginMetadata

6. **Update analyze() signature**
   - May need to extract `sensitivity` and `categories` from `options` dict
   - Or handle as args passed to `run_tool()`

---

### 2. **motion_detector** ⚠️

**File:** `plugins/motion_detector/src/forgesyte_motion/plugin.py`

**Current Structure:**
- Class: `Plugin` (not BasePlugin)
- Method: `metadata()` returns `PluginMetadata`
- Method: `analyze(image_bytes, options)`
- State: `_previous_frame`, `_frame_count`, `_last_motion_time`, `_motion_history`

**Required Changes:**

1. **Import BasePlugin**
   ```python
   from app.plugins.base import BasePlugin
   ```

2. **Inherit from BasePlugin**
   ```python
   class Plugin(BasePlugin):
   ```

3. **Define tools in __init__** (preserve existing state)
   ```python
   def __init__(self) -> None:
       self._previous_frame: np.ndarray | None = None
       self._frame_count: int = 0
       self._last_motion_time: float = 0
       self._motion_history: list[dict[str, Any]] = []
       
       self.tools = {
           "detect_motion": {
               "handler": self.analyze,
               "description": "Detect motion between consecutive frames",
               "input_schema": {
                   "image": {
                       "type": "bytes",
                       "description": "Frame image bytes"
                   },
                   "threshold": {
                       "type": "float",
                       "default": 25.0,
                       "min": 1.0,
                       "max": 100.0
                   },
                   "min_area": {
                       "type": "float",
                       "default": 100.0
                   }
               },
               "output_schema": {
                   "text": {"type": "string"},
                   "blocks": {"type": "array"},
                   "confidence": {"type": "number"}
               }
           }
       }
       super().__init__()
   ```

4. **Implement run_tool()**
   ```python
   def run_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
       if tool_name == "detect_motion":
           meta = self.tools[tool_name]
           return meta["handler"](**args)
       raise ValueError(f"Unknown tool: {tool_name}")
   ```

5. **Update analyze() signature**
   - Extract `threshold` and `min_area` from options/args

---

### 3. **plugin_template** ⚠️

**File:** `plugins/plugin_template/src/forgesyte_plugin_template/plugin.py`

**Current Structure:**
- Class: `Plugin` (not BasePlugin)
- Method: `metadata()` returns `PluginMetadata`
- Method: `analyze(image_bytes, options)`
- Lifecycle: `on_load()`, `on_unload()`

**Required Changes:**

1. **Import BasePlugin**
   ```python
   from app.plugins.base import BasePlugin
   ```

2. **Inherit from BasePlugin**
   ```python
   class Plugin(BasePlugin):
   ```

3. **Define tools in __init__**
   ```python
   def __init__(self) -> None:
       self.supported_modes = ["default"]
       self.tools = {
           "analyze": {
               "handler": self.analyze,
               "description": "Template plugin — replace with your description",
               "input_schema": {
                   "image_bytes": {
                       "type": "bytes",
                       "description": "Raw image data"
                   },
                   "mode": {
                       "type": "string",
                       "default": "default",
                       "enum": self.supported_modes,
                       "description": "Processing mode"
                   }
               },
               "output_schema": {
                   "text": {"type": "string"},
                   "blocks": {"type": "array"},
                   "confidence": {"type": "number"},
                   "error": {"type": "string", "nullable": True}
               }
           }
       }
       super().__init__()
   ```

4. **Implement run_tool()**
   ```python
   def run_tool(self, tool_name: str, args: dict[str, Any]) -> Any:
       if tool_name == "analyze":
           meta = self.tools[tool_name]
           return meta["handler"](**args)
       raise ValueError(f"Unknown tool: {tool_name}")
   ```

5. **Update analyze() signature**
   - Extract mode from options/args

---

## Testing Requirements

After migration, each plugin must:

1. **Pass BasePlugin contract validation**
   ```bash
   cd plugins/<plugin>/
   uv run pytest src/tests/ -v
   ```

2. **Have valid tool schemas**
   - description: non-empty string
   - input_schema: valid dict
   - output_schema: valid dict
   - All schemas JSON-serializable

3. **Implement run_tool()**
   - Accept `tool_name: str` and `args: dict`
   - Return JSON-serializable result
   - Raise clear errors for invalid tools

4. **Maintain backward compatibility**
   - Keep `metadata()` method for documentation
   - Keep `on_load()` and `on_unload()` hooks
   - All existing tests should pass

---

## Validation Checklist

For each plugin, after migration:

- [ ] Inherits from `BasePlugin`
- [ ] Defines `tools` dict with metadata (not just callables)
- [ ] Each tool has: `handler`, `description`, `input_schema`, `output_schema`
- [ ] `run_tool()` implemented correctly
- [ ] All schemas are dicts (not None, not strings, etc.)
- [ ] All schemas are JSON-serializable
- [ ] Tests pass: `uv run pytest src/tests/ -v`
- [ ] Type check passes: `uv run mypy src/`
- [ ] Lint passes: `uv run ruff check src/`

---

## Migration Order

**Recommended order** (easiest to hardest):

1. ✅ **plugin_template** (template, simplest structure)
2. ⚠️ **moderation** (single analyze tool)
3. ⚠️ **motion_detector** (more complex state management)

---

## Reference Implementation

See: rogermt/forgesyte commits

- **10ebf67**: BasePlugin refactor + ToolSchema validation
- **a03ce2c**: AGENTS.md pre-commit hook docs

---

## Key Files to Update

| Plugin | File |
|--------|------|
| moderation | `src/forgesyte_moderation/plugin.py` |
| motion_detector | `src/forgesyte_motion/plugin.py` |
| plugin_template | `src/forgesyte_plugin_template/plugin.py` |

---

## Questions?

Refer to:
- **BasePlugin contract:** rogermt/forgesyte/server/app/plugins/base.py
- **PluginRegistry validation:** rogermt/forgesyte/server/app/plugin_loader.py
- **Regression tests:** rogermt/forgesyte/server/tests/plugins/

---

## Related Issues

- rogermt/forgesyte#122 (schema validation)
- rogermt/forgesyte#123 (reject invalid plugins)
- rogermt/forgesyte#119 (plugin loader milestone)
