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
