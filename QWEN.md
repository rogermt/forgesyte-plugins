# ForgeSyte Plugins Project Context

## Project Overview

The `forgesyte-plugins` repository contains a collection of plugins for the ForgeSyte platform. These plugins provide specialized functionality for processing and analyzing various types of media, particularly images and documents. The project follows a modular architecture where each plugin is an independent pip-installable package that implements a common plugin interface.

### Key Components

- **Plugin System**: A standardized plugin architecture that allows for extensible functionality
- **Core Plugins**: Pre-built plugins for OCR, block mapping, content moderation, and motion detection
- **Plugin Template**: A starter template for developing new plugins
- **Quality Assurance**: Comprehensive linting, type checking, and testing infrastructure

### Available Plugins

- **block_mapper** - Maps visual blocks in images
- **moderation** - Content moderation plugin  
- **motion_detector** - Detects motion in video/images
- **ocr** - Optical character recognition

## Architecture & Plugin Interface

Plugins follow a consistent interface defined by the `Plugin` class:

- `name`: Unique identifier for the plugin
- `version`: Version string
- `metadata()`: Returns plugin metadata including inputs, outputs, and configuration schema
- `analyze()`: Main processing function that takes input data and options, returns `AnalysisResult`
- `on_load()` / `on_unload()`: Lifecycle hooks

The plugin system integrates with the broader ForgeSyte platform through the MCP (Model Context Protocol) adapter, allowing AI models to access plugin functionality.

## Development Environment

### Prerequisites
- Python 3.10+
- uv package manager
- Node.js (for web UI components)

### Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Quality Assurance
The project uses multiple tools for maintaining code quality:
- **Black**: Code formatting
- **Ruff**: Linting and fixing
- **MyPy**: Type checking
- **PyTest**: Testing
- **Isort**: Import sorting

Run all quality checks with:
```bash
./check_all.sh
```

## Building and Running

### Individual Plugin Development
Each plugin is located in the `plugins/` directory and follows the structure:
```
plugins/
├── plugin_name/
│   ├── src/
│   │   └── forgesyte_plugin_name/
│   │       └── plugin.py
│   ├── tests/
│   ├── pyproject.toml
│   └── README.md
│   ├── web-ui/
│   │   └── plugin.json
│   │   └── *.tsx
```

e.g. structure
plugin_template/
  pyproject.toml
  README.md
  src/
    forgesyte_plugin_template/
      __init__.py
      plugin.py
    tests/
      test_plugin.py
  web-ui/
    plugin.json
    ConfigForm.tsx
    ConfigForm.test.tsx
    ResultRenderer.tsx
    ResultRenderer.test.tsx


### Creating New Plugins
Use the plugin template as a starting point:
```bash
cp -r plugins/plugin_template plugins/new_plugin_name
# Update the plugin name, functionality, and dependencies in pyproject.toml
```

### Testing
Run tests for individual plugins:
```bash
cd plugins/plugin_name
uv run pytest
```

Or run all plugin tests:
```bash
./check_all.sh
```

## Development Conventions

### Coding Standards
- Follow PEP 8 style guidelines (enforced by Black and Ruff)
- Use type hints (enforced by MyPy)
- Write docstrings for all public methods and classes
- Use Pydantic models for data validation

### Plugin Development Guidelines
- Each plugin should be self-contained with minimal dependencies
- Implement proper error handling and logging
- Define clear input/output contracts in the metadata
- Support configurable options through the config_schema
- Implement lifecycle hooks for initialization and cleanup

### Testing Practices
- Write unit tests for all plugin functionality
- Test edge cases and error conditions
- Use pytest for test organization and execution
- Maintain high test coverage

### Documentation
- Update plugin README files with usage examples
- Document configuration options clearly
- Include examples of expected inputs and outputs

## File Structure
```
forgesyte-plugins/
├── README.md                 # Project overview and architecture
├── requirements.txt          # Base dependencies
├── pyproject.toml            # Python configuration (black, ruff, mypy)
├── package.json              # JavaScript/TypeScript dependencies
├── check_all.sh              # Quality assurance script
├── mypy.ini                  # MyPy configuration
├── plugins/                  # Individual plugin implementations
│   ├── block_mapper/         # Visual block mapping plugin
│   ├── moderation/           # Content moderation plugin
│   ├── motion_detector/      # Motion detection plugin
│   ├── ocr/                  # OCR plugin
│   └── plugin_template/      # Template for new plugins
└── scripts/                  # Utility scripts
```

## Integration with ForgeSyte Platform

The plugins integrate with the main ForgeSyte platform through:
- MCP (Model Context Protocol) adapter for AI model access
- Document store for persistent storage
- Job queue system for async processing
- Web UI for user interaction

The architecture supports multiple document store backends (MinIO, AWS S3, local filesystem) and provides a unified API for plugin access.