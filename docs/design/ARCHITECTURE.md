# 🚀 **ForgeSyte Architecture Deep Dive**
*A complete mental model of how ForgeSyte works under the hood.*

ForgeSyte is a **modular plugin platform** built around three pillars:

1. **Backend Plugin Runtime (Python)**
2. **Frontend Plugin Runtime (React)**
3. **Core Orchestrator (API + UI Shell)**

The key design principle:  
**ForgeSyte core never hardcodes plugins. Everything is discovered.**

This deep dive explains exactly how that works.

---

# 1. 🔩 Core Architectural Philosophy

ForgeSyte is built on four non‑negotiable principles:

### **1. Plugins are first‑class citizens**
Plugins are not “extensions” — they *are* the product.  
Core exists to orchestrate them.

### **2. Discovery, not registration**
Plugins declare themselves.  
Core finds them.  
No central registry file.  
No manual imports.

### **3. Strict backend/frontend separation**
Backend = Python logic  
Frontend = React UI  
They communicate only through the core API.

### **4. Explicit contracts everywhere**
- Python entry points  
- Pydantic schemas  
- UI manifests  
- Component exports  

This makes ForgeSyte **auditable, predictable, and stable**.

---

# 2. 🧠 Backend Architecture (Python)

## 2.1. Plugin Discovery via Entry Points

ForgeSyte uses Python’s entry point system:

```toml
[project.entry-points."forgesyte.plugins"]
ocr = "forgesyte_ocr.plugin:Plugin"
my_plugin = "forgesyte_my_plugin.plugin:Plugin"
```

At startup, ForgeSyte:

1. Scans the `forgesyte.plugins` group  
2. Imports each module  
3. Instantiates the `Plugin` class  
4. Registers it in the backend registry

### **Backend Registry Structure**
Conceptually:

```python
{
  "ocr": <Plugin instance>,
  "my_plugin": <Plugin instance>,
  ...
}
```

This registry powers:

- `/api/plugins`
- `/api/plugins/{id}/run`

---

## 2.2. Plugin Contract

Every plugin must implement:

### **Identity**
```python
id = "my_plugin"
display_name = "My Plugin"
```

### **Input/Output Models**
```python
class Input(BaseModel): ...
class Output(BaseModel): ...
```

### **Execution**
```python
def run(self, input: Input) -> Output:
    ...
```

### Why Pydantic?
- Validation  
- Type safety  
- Schema introspection  
- Future UI auto‑generation  

---

## 2.3. Backend Execution Flow

When the UI calls:

```
POST /api/plugins/my_plugin/run
```

ForgeSyte:

1. Looks up plugin in registry  
2. Validates input against Pydantic model  
3. Calls `plugin.run()`  
4. Validates output  
5. Returns JSON to UI  

This is the **backend half** of the plugin lifecycle.

---

# 3. 🎨 Frontend Architecture (React)

## 3.1. Dynamic Manifest Discovery

ForgeSyte scans:

```
*/web-ui/plugin.json
```

using Vite/Webpack glob imports.

This is the frontend equivalent of Python entry points.

### Example minimal manifest (OCR)
```json
{
  "name": "ocr",
  "ui": { "entry": "ResultRenderer.tsx" }
}
```

### Example full manifest (template)
```json
{
  "id": "my_plugin",
  "displayName": "My Plugin",
  "configForm": "ConfigForm.tsx",
  "resultRenderer": "ResultRenderer.tsx"
}
```

---

## 3.2. Dynamic Component Loading

ForgeSyte also glob‑imports components:

```ts
const configForms = import.meta.glob("*/web-ui/ConfigForm.tsx", { eager: true });
const resultRenderers = import.meta.glob("*/web-ui/ResultRenderer.tsx", { eager: true });
```

Then it builds a **UI registry**:

```ts
{
  "my_plugin": {
    manifest,
    ConfigForm,
    ResultRenderer
  },
  "ocr": {
    manifest,
    ResultRenderer
  }
}
```

This mirrors the backend registry.

---

## 3.3. UI Composition

When a user selects a plugin:

1. UI fetches `/api/plugins`  
2. Matches backend plugin `id` with frontend manifest `id`  
3. Renders:
   - Icon  
   - Display name  
   - Config form (if present)  
   - Result renderer  

The UI is **data‑driven**, not hardcoded.

---

# 4. 🔗 Backend ↔ Frontend Integration

The glue between backend and frontend is:

- **Plugin ID**  
- **Input/Output JSON**  
- **Manifest metadata**

### The ID must match:
- `Plugin.id` in Python  
- `"id"` in `plugin.json`  

This is the only coupling.

Everything else is dynamic.

---

# 5. 🧬 Two Manifest Schemas (Why They Exist)

This is the part that caused confusion with OCR.

## 5.1. Minimal Schema (Legacy)
Used by OCR:

```json
{
  "name": "ocr",
  "ui": { "entry": "ResultRenderer.tsx" }
}
```

Designed for:

- Single renderer  
- No config form  
- No icon  
- No metadata  

## 5.2. Full Schema (Modern)
Used by the template:

```json
{
  "id": "my_plugin",
  "displayName": "My Plugin",
  "configForm": "ConfigForm.tsx",
  "resultRenderer": "ResultRenderer.tsx"
}
```

Designed for:

- Config forms  
- Icons  
- Metadata  
- Future UI features  

## 5.3. Why both work
The loader is backward‑compatible:

- If it sees `"ui.entry"` → minimal mode  
- If it sees `"configForm"` → full mode  

This is intentional.

---

# 6. 🏗️ Plugin Lifecycle (End‑to‑End)

### 1. Plugin repo exists on disk  
Backend + frontend.

### 2. Backend installed  
`uv pip install -e .`

### 3. Core starts  
Loads backend plugins via entry points.

### 4. Frontend starts  
Loads manifests + components via glob imports.

### 5. User selects plugin  
UI mounts config form.

### 6. User submits config  
UI sends JSON to backend.

### 7. Backend executes plugin  
Returns structured output.

### 8. UI renders results  
Using plugin’s `ResultRenderer`.

---

# 7. 🧱 Architectural Invariants (Your instincts were right)

ForgeSyte enforces:

- **Strict folder naming** (`web-ui/`)  
- **Strict entry point registration**  
- **Strict ID matching**  
- **Strict component exports**  
- **Strict schema validation**  

These invariants prevent:

- Architectural drift  
- Dead code  
- UI/backend mismatches  
- Plugin breakage  

This is why your insistence on clarity and structure is so important.

---

# 8. 🚀 Future Evolution (What this architecture enables)

Because the system is declarative and dynamic, you can add:

### **1. Plugin categories**
`"tags": ["ocr", "text", "vision"]`

### **2. Capability metadata**
`"requires": ["gpu"]`

### **3. Auto‑generated config forms**
From Pydantic schemas.

### **4. Remote plugin registries**
Install plugins from a marketplace.

### **5. Version negotiation**
Core ↔ plugin compatibility.

### **6. Plugin sandboxing**
Run plugins in isolated environments.

None of this requires changing the core architecture — it’s already extensible.

---

