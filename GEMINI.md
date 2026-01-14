# ForgeSyte Plugins - Gemini Context

This repository is a monorepo containing plugins for the ForgeSyte platform. Each plugin is a self-contained Python package designed to perform specific analysis tasks (e.g., OCR, motion detection, moderation) on data processed by ForgeSyte.

## Platform Architecture Overview

The ForgeSyte platform consists of several components working together:

- **Gemini-CLI**: MCP Client/Tool Caller that interacts with the core via MCP Protocol.
- **ForgeSyte Core**: FastAPI entry point, REST API, MCP adapter, and Plugin Loader.
- **Document Store**: Unified storage abstraction (MinIO, S3, Filesystem) used by both the Web UI and plugins.
- **Plugins Ecosystem**: This repository, containing independent pip packages discovered via entry points.

## Project Structure

- **`plugin_template/`**: Canonical reference implementation for new plugins.
- **`ocr/`**: Optical Character Recognition plugin.
- **`motion_detector/`**: Motion detection analysis.
- **`moderation/`**: Content moderation.
- **`block_mapper/`**: Visual block mapping.
- **`docs/`**: Shared documentation and standards.

## Plugin Architecture

All plugins must adhere to the structure defined in `plugin_template`.

### File Structure
```
plugin_name/
├── pyproject.toml              # Build config & dependencies
├── README.md
├── forgesyte_plugin_name/      # Main package
│   ├── __init__.py
│   └── plugin.py               # Plugin implementation
└── tests/
    ├── __init__.py
    └── test_plugin.py          # Unit tests
```

### Key Components

1.  **`pyproject.toml`**: Defines the project metadata and entry point.
    -   **Entry Point:** Must define a `forgesyte.plugins` entry point.
        ```toml
        [project.entry-points."forgesyte.plugins"]
        plugin_name = "forgesyte_plugin_name.plugin:Plugin"
        ```
2.  **`plugin.py`**: Contains the `Plugin` class.
    -   **`metadata()`**: Returns `PluginMetadata` (inputs, outputs, config schema).
    -   **`analyze(image_bytes, options)`**: Core logic. Returns a dictionary based on `AnalysisResult`.
    -   **Lifecycle Hooks**: `on_load()` and `on_unload()`.

### Data Models
Plugins interact with the main application via models (expected at runtime):
-   `PluginMetadata`: Capabilities description.
-   `AnalysisResult`: Structured analysis results.

## Development Standards

Adhere strictly to `docs/development/PYTHON_STANDARDS.md`.

### Mandatory Tools
- **Formatting:** `black . && isort .`
- **Linting:** `ruff check --fix .`
- **Type Checking:** `mypy . --no-site-packages`
- **Testing:** `pytest`

### Coding Conventions
- **Type Hints:** Mandatory for all functions.
- **Path Handling:** Use `pathlib.Path`.
- **Logging:** Use `logging.getLogger(__name__)`.
- **Error Handling:** Return errors within the `AnalysisResult` dictionary.

## Creating a New Plugin

1.  Copy `plugin_template/` to `new_plugin/`.
2.  Rename `forgesyte_plugin_template/` package and update `pyproject.toml` entry points.
3.  Implement `Plugin.name`, `metadata()`, and `analyze()` in `plugin.py`.
4.  Update tests in `tests/test_plugin.py`.

## Building and Testing

```bash
# In a plugin directory
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest
```
