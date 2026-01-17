# 🚀 ForgeSyte Plugin Template

This repository provides a **canonical, future‑proof template** for building ForgeSyte plugins.  
It includes:

- Backend entry point  
- Python `Plugin` class  
- Frontend manifest  
- Config form  
- Result renderer  
- Icon  
- Tests  
- Documentation  

It is designed so that **new plugins load automatically** via ForgeSyte’s dynamic discovery system.

---

# 📘 Manifest Formats (IMPORTANT)

ForgeSyte supports **two valid plugin manifest schemas**.  
This is the source of confusion when comparing the OCR plugin to this template.

## 1. Minimal Manifest (legacy, used by OCR)

```json
{
  "name": "ocr",
  "version": "1.0.0",
  "description": "ocr UI plugin",
  "ui": {
    "entry": "ResultRenderer.tsx"
  }
}
```

### When this format is used
- Older plugins (like OCR)  
- Plugins with **only one UI component**  
- No config form  
- No icon  
- No metadata  

### Why it still works  
ForgeSyte’s loader checks for:

- `"ui.entry"` → load a single renderer

This keeps older plugins compatible.

---

## 2. Full Manifest (recommended for all new plugins)

```json
{
  "id": "plugin_template",
  "displayName": "ForgeSyte Plugin Template",
  "description": "Template plugin with backend and UI manifest.",
  "version": "1.0.0",
  "icon": "plugin-icon.svg",
  "configForm": "ConfigForm.tsx",
  "resultRenderer": "ResultRenderer.tsx"
}
```

### When this format is used
- All new plugins  
- Plugins with config forms  
- Plugins that need icons  
- Plugins that want richer UI integration  

### Why this template uses it  
It is the **canonical**, future‑proof schema that supports:

- Config forms  
- Result renderers  
- Icons  
- Display names  
- Tags  
- Better UX in the plugin selector  

---

## Rule of Thumb

- **Use the full manifest** for every new plugin.  
- **Use the minimal manifest** only for legacy compatibility or extremely simple plugins.  
- Both formats are supported by the dynamic loader.

---

# ⚡ Quick Start Checklist

1. Copy the template  
2. Update `pyproject.toml`  
3. Implement your backend `Plugin` class  
4. Update `web-ui/plugin.json`  
5. Customize `ConfigForm.tsx` and `ResultRenderer.tsx`  
6. Add an icon (optional but recommended)  
7. Install in editable mode:  
   ```
   uv pip install -e .
   ```  
8. Restart ForgeSyte — your plugin appears automatically

---

# 🔧 Troubleshooting

### Plugin does not appear in UI  
- Folder must be named **`web-ui/`**  
- `plugin.json` must exist  
- Manifest filenames must match actual files  
- Backend entry point must be correct  
- Plugin must be installed with `uv pip install -e .`

### Plugin appears but UI errors  
- React components must `export default`  
- Case sensitivity matters  
- Manifest fields must match component filenames

### Backend loads but results don’t render  
- `ResultRenderer` must accept a `result` prop  
- Config form must call `onChange({ ... })`

---

# 🌟 Known Good Example (Fully Working Plugin)

### Folder Structure

```
my_plugin/
  pyproject.toml
  src/my_plugin/plugin.py
  web-ui/plugin.json
  web-ui/ConfigForm.tsx
  web-ui/ResultRenderer.tsx
  web-ui/plugin-icon.svg
```

### Backend Example

```python
from pydantic import BaseModel
from forgesyte_core.plugin import BasePlugin

class MyPluginInput(BaseModel):
    text: str

class MyPluginOutput(BaseModel):
    reversed: str

class Plugin(BasePlugin):
    id = "my_plugin"
    display_name = "My Plugin"

    def run(self, input: MyPluginInput) -> MyPluginOutput:
        return MyPluginOutput(reversed=input.text[::-1])
```

### Full Manifest Example

```json
{
  "id": "my_plugin",
  "displayName": "My Plugin",
  "description": "Reverses text using a simple backend example.",
  "version": "1.0.0",
  "icon": "plugin-icon.svg",
  "configForm": "ConfigForm.tsx",
  "resultRenderer": "ResultRenderer.tsx"
}
```

### Config Form Example

```tsx
const ConfigForm = ({ onChange }) => {
  const update = (value) => onChange({ text: value });
  return (
    <label>
      Text to reverse:
      <input onChange={(e) => update(e.target.value)} />
    </label>
  );
};
export default ConfigForm;
```

### Result Renderer Example

```tsx
const ResultRenderer = ({ result }) => (
  <pre>{result ? result.reversed : "No result yet."}</pre>
);
export default ResultRenderer;
```

---

