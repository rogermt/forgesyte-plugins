#!/usr/bin/env python3
"""
Unified validator for ForgeSyte plugin manifests (Phase 12).

Validation rules:
1. Manifest must contain: id, name, version, tools, type
2. type must be one of: yolo, ocr, custom
3. Tools must be a list of dicts with 'id' field
4. Tool ids must be URL-safe
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, cast

ALLOWED_PLUGIN_TYPES = {"yolo", "ocr", "custom"}

DEFAULT_MANIFEST_PATH = (
    Path(__file__).parent
    / "plugins"
    / "forgesyte-yolo-tracker"
    / "src"
    / "forgesyte_yolo_tracker"
    / "manifest.json"
)


def resolve_manifest_path() -> Path:
    if len(sys.argv) > 1:
        return Path(sys.argv[1]).expanduser().resolve()

    env_path = os.environ.get("MANIFEST_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    return DEFAULT_MANIFEST_PATH


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        print(f"ERROR: Manifest not found at {path}")
        sys.exit(1)

    try:
        with open(path, encoding="utf-8") as f:
            return cast(dict[str, Any], json.load(f))
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in manifest: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to load manifest: {e}")
        sys.exit(1)


def is_url_safe(name: str) -> bool:
    return re.match(r"^[a-z0-9_-]+$", name) is not None


def validate_manifest(manifest: dict[str, Any]) -> None:
    required = ["id", "name", "version", "tools", "type"]
    for field in required:
        if field not in manifest:
            print(f"ERROR: Manifest missing required field '{field}'")
            sys.exit(1)

    plugin_type = manifest["type"]
    if plugin_type not in ALLOWED_PLUGIN_TYPES:
        print(
            f"ERROR: type must be one of {sorted(ALLOWED_PLUGIN_TYPES)}, "
            f"got '{plugin_type}'"
        )
        sys.exit(1)

    tools = manifest["tools"]
    if not isinstance(tools, list):
        print("ERROR: tools must be a list")
        sys.exit(1)

    for tool in tools:
        tool_id = tool.get("id")
        if not tool_id:
            print(f"ERROR: Tool missing required 'id' field: {tool}")
            sys.exit(1)
        if not is_url_safe(tool_id):
            print(f"ERROR: Tool id '{tool_id}' is not URL-safe")
            sys.exit(1)


def main() -> None:
    manifest_path = resolve_manifest_path()
    manifest = load_manifest(manifest_path)
    validate_manifest(manifest)
    print("OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
