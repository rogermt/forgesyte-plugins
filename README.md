# ForgeSyte Plugins

A collection of plugins for the ForgeSyte platform.

## Available Plugins

- **block_mapper** - Maps visual blocks in images
- **moderation** - Content moderation plugin
- **motion_detector** - Detects motion in video/images
- **ocr** - Optical character recognition

## Plugin Development

See [PLUGIN_DEVELOPMENT.md](../PLUGIN_DEVELOPMENT.md) in the main ForgeSyte repository for development guidelines.

## Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Contributing

Each plugin should follow the structure in `plugin_template/`.

Absolutely, Roger — here is a **clean, publication‑ready architecture diagram** you can drop straight into your ForgeSyte docs.  
It’s written in **GitHub‑flavored Markdown**, uses **ASCII architecture blocks**, and reflects the **new plugin ecosystem**, **document store**, **MCP adapter**, and **Web UI**.

No images, no external dependencies — just a crisp, readable diagram that communicates the whole platform at a glance.

---

# 🌌 **ForgeSyte Platform Architecture Diagram**

# ForgeSyte Platform Architecture

                     ┌──────────────────────────────────────────┐
                     │                Gemini‑CLI                 │
                     │        (MCP Client / Tool Caller)         │
                     │   /call ocr, /call block_mapper, ...      │
                     └───────────────────────────┬──────────────┘
                                                 │
                                          MCP Protocol
                                                 │
                                                 ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ForgeSyte Core                                  │
│──────────────────────────────────────────────────────────────────────────────│
│                                                                              │
│  server/                                                                     │
│    app/                                                                      │
│      ├─ main.py                 → FastAPI entrypoint                         │
│      ├─ api.py                  → REST API (Web UI, jobs, documents)         │
│      ├─ mcp_adapter.py          → MCP tool exposure + request translation    │
│      ├─ plugin_loader.py        → Discovers & loads external plugins         │
│      ├─ document_store/         → Unified storage abstraction                │
│      │     ├─ base.py           → Interface: save/load/delete                │
│      │     ├─ filesystem_store.py                                            │
│      │     ├─ minio_store.py    → S3-compatible backend (MinIO)              │
│      │     └─ s3_store.py       → AWS S3 backend                             │
│      ├─ models.py               → Shared schemas                              │
│      └─ tasks.py                → Job queue (async analysis)                 │
│                                                                              │
│  web-ui/                                                                     │
│    ├─ Upload image → doc_id                                                  │
│    ├─ Trigger analysis → job_id                                              │
│    └─ Poll job results                                                       │
│                                                                              │
│  docs/                                                                       │
│    ├─ ARCHITECTURE.md                                                        │
│    ├─ MCP_CONFIGURATION.md                                                   │
│    ├─ PLUGIN_API.md                                                          │
│    └─ DOCUMENT_STORE.md                                                      │
│                                                                              │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │ Plugin API (Python entry points)
                                ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                           forgesyte-plugins Repository                        │
│──────────────────────────────────────────────────────────────────────────────│
│                                                                              │
│  Each plugin is an independent pip package:                                  │
│                                                                              │
│    forgesyte-plugins/                                                        │
│      ├─ ocr/                                                                 │
│      │    ├─ forgesyte_ocr/plugin.py                                         │
│      │    └─ pyproject.toml (entry point: forgesyte.plugins)                 │
│      ├─ block_mapper/                                                        │
│      ├─ moderation/                                                          │
│      ├─ motion_detector/                                                     │
│      └─ plugin_template/                                                     │
│                                                                              │
│  Plugin contract:                                                            │
│      class Plugin:                                                           │
│          name                                                                │
│          metadata()                                                          │
│          analyze(image_bytes, options)                                       │
│                                                                              │
└───────────────────────────────┬──────────────────────────────────────────────┘
                                │
                                ▼

┌──────────────────────────────────────────────────────────────────────────────┐
│                               Document Store                                 │
│──────────────────────────────────────────────────────────────────────────────│
│                                                                              │
│  Backends:                                                                   │
│    • MinIO (S3-compatible OSS)                                               │
│    • AWS S3                                                                  │
│    • Local filesystem                                                        │
│    • Database (optional)                                                     │
│                                                                              │
│  Stores:                                                                     │
│    • Images                                                                  │
│    • PDFs                                                                    │
│    • Video frames                                                            │
│    • Metadata                                                                │
│                                                                              │
│  Web UI and MCP both reference documents by doc_id                           │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

# 🎯 What this diagram communicates

This diagram shows:

### **1. ForgeSyte Core**
- MCP server  
- REST API  
- Web UI  
- Document store  
- Plugin loader  
- Job system  

### **2. External Plugin Ecosystem**
- Independent repo  
- Pip‑installable plugins  
- Entry‑point based discovery  

### **3. Document Store**
- MinIO/S3/FS abstraction  
- Shared between Web UI and Gemini  

### **4. Gemini‑CLI Integration**
- Calls MCP tools  
- Uses plugins indirectly  
- Can operate on `doc_id`  

### **5. Clean separation of concerns**
- Core = runtime  
- Plugins = ecosystem  
- Storage = shared state  
- UI = client  
- Gemini = client  

---

