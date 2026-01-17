# 📘 ForgeSyte Plugin Development — Detailed How‑To Guide

This guide explains **every step** of creating a ForgeSyte plugin, including backend logic, frontend UI, manifest formats, dynamic discovery, and debugging. It is designed for engineers who want to understand not just *what* to do, but *why* ForgeSyte works the way it does.

- **What’s happening**  
- **Why it works that way**  
- **How ForgeSyte discovers plugins**  
- **How the backend and frontend connect**  
- **How manifests are parsed**  
- **How UI components are loaded dynamically**  
- **How to debug when something goes wrong**

---

# 1. Understanding the Plugin Architecture

ForgeSyte plugins have **two halves**:

## 🔹 Backend (Python)
- Defines the plugin’s logic  
- Exposes a `Plugin` class  
- Registers itself via Python entry points  
- Receives input from the UI  
- Returns structured output  

## 🔹 Frontend (React)
- Defines how users configure the plugin  
- Defines how results are displayed  
- Provides metadata (name, icon, description)  
- Is discovered dynamically via `web-ui/plugin.json`

These two halves are **loosely coupled** — they communicate through ForgeSyte Core, not directly.

---

# 2. How ForgeSyte Discovers Plugins

Understanding this is key.

## 🔹 Backend Discovery (Python Entry Points)
ForgeSyte scans:

```
[project.entry-points."forgesyte.plugins"]
```

in your `pyproject.toml`.

This tells ForgeSyte:

> “Load this Python class as a plugin.”

Example:

```toml
[project.entry-points."forgesyte.plugins"]
my_plugin = "forgesyte_my_plugin.plugin:Plugin"
```

ForgeSyte then:

- Imports the module  
- Instantiates the `Plugin` class  
- Exposes it via `/api/plugins`  

This is how the **backend** becomes visible.

---

## 🔹 Frontend Discovery (Dynamic Glob Imports)

ForgeSyte’s UI scans:

```
*/web-ui/plugin.json
```

using Vite/Webpack glob imports.

This means:

- No manual registration  
- No central index file  
- No hardcoded imports  

If your plugin has:

```
web-ui/plugin.json
```

it will be discovered automatically.

---

# 3. Creating a New Plugin (Step‑by‑Step)

## Step 1 — Copy the Template

```
cp -r plugin_template my_new_plugin
```

Rename the folder and update Python package names.

---

## Step 2 — Update `pyproject.toml`

This is the backend registration.

```toml
[project]
name = "forgesyte-my-new-plugin"
version = "1.0.0"

[project.entry-points."forgesyte.plugins"]
my_new_plugin = "forgesyte_my_new_plugin.plugin:Plugin"
```

If this is wrong, the backend will not load.

---

## Step 3 — Implement the Backend Plugin Class

Inside:

```
src/forgesyte_my_new_plugin/plugin.py
```

Define:

- `id`  
- `display_name`  
- `run()` method  
- Input/output models  

Example:

```python
class Plugin(BasePlugin):
    id = "my_new_plugin"
    display_name = "My New Plugin"

    def run(self, input: Input) -> Output:
        return Output(reversed=input.text[::-1])
```

---

# 4. Creating the UI (Frontend)

## Step 4 — Create the `web-ui/` Folder

This folder **must** be named:

```
web-ui/
```

ForgeSyte will not detect:

- `ui/`
- `frontend/`
- `web/`
- `react/`

Only:

```
web-ui/
```

---

## Step 5 — Choose a Manifest Format

ForgeSyte supports **two schemas**.

### A. Minimal Schema (used by OCR)

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

Use this if:

- Your plugin has no config form  
- You only need one UI component  
- You want the simplest possible setup  

### B. Full Schema (recommended)

```json
{
  "id": "my_new_plugin",
  "displayName": "My New Plugin",
  "description": "Reverses text",
  "version": "1.0.0",
  "icon": "plugin-icon.svg",
  "configForm": "ConfigForm.tsx",
  "resultRenderer": "ResultRenderer.tsx"
}
```

Use this if:

- Your plugin has configuration  
- You want icons  
- You want metadata  
- You want a richer UI experience  

---

# 5. Implementing the UI Components

## Step 6 — Config Form

This component collects user input.

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
```

ForgeSyte expects:

```
onChange({ fieldName: value })
```

---

## Step 7 — Result Renderer

This displays backend output.

```tsx
const ResultRenderer = ({ result }) => (
  <pre>{result ? result.reversed : "No result yet."}</pre>
);
```

ForgeSyte passes the backend output as the `result` prop.

---

# 6. Installing the Plugin

From the plugin root:

```
uv pip install -e .
```

This makes the backend importable.

---

# 7. Restart ForgeSyte

ForgeSyte will:

- Load your backend via entry points  
- Load your UI via glob imports  
- Register your manifest  
- Display your plugin in the selector  

---

# 8. Debugging (Deep Detail)

## Plugin not showing in UI?
Check:

- Folder name must be `web-ui/`
- `plugin.json` must exist
- Manifest fields must match filenames
- Plugin must be installed with `uv pip install -e .`

## Plugin showing but UI errors?
Check:

- Components must `export default`
- Case sensitivity
- Manifest paths must be correct

## Backend errors?
Check:

- Entry point path in `pyproject.toml`
- `Plugin.id` must match manifest `id`
- Input/output models must be valid Pydantic models

---

# 9. Understanding the Two Manifest Schemas (Deep Explanation)

This is the part you specifically wanted more detail on.

### Why OCR uses `"ui": { "entry": "ResultRenderer.tsx" }`

Because OCR was built **before** ForgeSyte had:

- Config forms  
- Icons  
- Display names  
- Tags  
- Rich metadata  

It only needed:

- A single UI renderer  
- No configuration  

So the minimal schema was enough.

### Why the template uses the full schema

Because ForgeSyte evolved.

The full schema supports:

- Config forms  
- Result renderers  
- Icons  
- Metadata  
- Better UX  
- Future features (categories, filtering, sorting)

### Why both still work

ForgeSyte’s loader is intentionally backward‑compatible.

It checks:

- If `"ui.entry"` exists → use minimal mode  
- If `"configForm"` exists → use full mode  

This is why OCR still loads correctly.

---

