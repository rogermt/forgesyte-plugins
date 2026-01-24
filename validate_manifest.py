#!/usr/bin/env python3
"""
Validate plugin manifest files for required frame-based tool structure.

Supports three resolution modes:
1. CLI argument: python validate_manifest.py /path/to/manifest.json
2. Environment variable: MANIFEST_PATH=/path/to/manifest.json python validate_manifest.py
3. Default path: plugins/forgesyte-yolo-tracker/src/forgesyte_yolo_tracker/manifest.json

Usage:
    python validate_manifest.py                                # Uses default
    python validate_manifest.py /path/to/manifest.json        # Uses CLI arg
    MANIFEST_PATH=/path/to/manifest.json python validate_manifest.py  # Uses env var
"""

import json
import sys
import os
from pathlib import Path

# Default manifest path (YOLO tracker plugin layout)
DEFAULT_MANIFEST_PATH = (
    Path(__file__).parent
    / "plugins"
    / "forgesyte-yolo-tracker"
    / "src"
    / "forgesyte_yolo_tracker"
    / "manifest.json"
)


def resolve_manifest_path():
    """
    Resolve manifest path in this priority order:
    1. CLI argument
    2. MANIFEST_PATH environment variable
    3. Default path
    """
    # 1. CLI argument
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()

    # 2. Environment variable
    env_path = os.environ.get("MANIFEST_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    # 3. Default
    return DEFAULT_MANIFEST_PATH


def main():
    """Validate manifest structure and required fields."""
    manifest_path = resolve_manifest_path()

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        sys.exit(1)

    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in manifest: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load manifest: {e}")
        sys.exit(1)

    tools = manifest.get("tools", {})
    if not isinstance(tools, dict) or not tools:
        print("ERROR: Manifest has no tools section")
        sys.exit(1)

    for tool_name, tool in tools.items():
        if not isinstance(tool, dict):
            print(f"ERROR: Tool '{tool_name}' is not a dictionary")
            sys.exit(1)

        inputs = tool.get("inputs", {})
        if "frame_base64" not in inputs:
            print(
                f"ERROR: Tool '{tool_name}' missing required input 'frame_base64'"
            )
            sys.exit(1)

    print("OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
