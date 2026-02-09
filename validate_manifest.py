#!/usr/bin/env python3
"""
Unified validator for ForgeSyte plugin manifests.

Supports:
- Frame-based plugins (e.g., YOLO tracker)
- Image-based plugins (e.g., OCR)

Validation rules:
1. Manifest must contain: id, name, version, tools, mode
2. Tools may be a dict (frame-based) or list (OCR-style)
3. Frame-based tools must accept 'frame_base64'
4. OCR tools must accept 'image_base64'
5. Tool names must be URL-safe
6. mode must be consistent with tools:
   - mode == "image" → list tools + image_base64
   - mode == "frame" → dict tools + frame_base64
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any, cast

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


def validate_common_fields(manifest: dict[str, Any]) -> None:
    required = ["id", "name", "version", "tools", "mode"]
    for field in required:
        if field not in manifest:
            print(f"ERROR: Manifest missing required field '{field}'")
            sys.exit(1)

    mode = manifest["mode"]
    if mode not in ("image", "frame"):
        print(f"ERROR: mode must be 'image' or 'frame', got '{mode}'")
        sys.exit(1)


def is_url_safe(name: str) -> bool:
    return re.match(r"^[a-z0-9_-]+$", name) is not None


def detect_plugin_type(tools: dict[str, Any] | list[dict[str, Any]]) -> str:
    """
    Decide plugin type based on tool input fields.
    """
    # Dict-style tools (YOLO)
    if isinstance(tools, dict):
        for tool in tools.values():
            inputs = tool.get("inputs", {})
            if "frame_base64" in inputs:
                return "frame"
        return "unknown"

    # List-style tools (OCR)
    if isinstance(tools, list):
        for tool in tools:
            inputs = tool.get("inputs", {})
            if "image_base64" in inputs:
                return "ocr"
        return "unknown"

    return "unknown"


def validate_frame_plugin(tools: dict[str, Any]) -> None:
    if not isinstance(tools, dict):
        print("ERROR: Frame-based plugins must use dict-style tools")
        sys.exit(1)

    for tool_name, tool in tools.items():
        if not is_url_safe(tool_name):
            print(f"ERROR: Tool name '{tool_name}' is not URL-safe")
            sys.exit(1)

        inputs = tool.get("inputs", {})
        if "frame_base64" not in inputs:
            print(f"ERROR: Tool '{tool_name}' missing required input 'frame_base64'")
            sys.exit(1)


def validate_ocr_plugin(tools: list[dict[str, Any]]) -> None:
    if not isinstance(tools, list):
        print("ERROR: OCR plugins must use list-style tools")
        sys.exit(1)

    for tool in tools:
        name = tool.get("name")
        if not name or not is_url_safe(name):
            print(f"ERROR: Tool name '{name}' is invalid or not URL-safe")
            sys.exit(1)

        inputs = tool.get("inputs", {})
        if "image_base64" not in inputs:
            print(f"ERROR: OCR tool '{name}' missing required input 'image_base64'")
            sys.exit(1)


def main() -> None:
    manifest_path = resolve_manifest_path()
    manifest = load_manifest(manifest_path)

    validate_common_fields(manifest)

    tools = manifest["tools"]
    mode = manifest["mode"]
    plugin_type = detect_plugin_type(tools)

    # Enforce consistency between mode and inferred type
    if mode == "image":
        if plugin_type != "ocr":
            print(
                f"ERROR: mode='image' but tools look like '{plugin_type}' "
                "(expected list-style tools with image_base64)"
            )
            sys.exit(1)
        validate_ocr_plugin(tools)

    elif mode == "frame":
        if plugin_type != "frame":
            print(
                f"ERROR: mode='frame' but tools look like '{plugin_type}' "
                "(expected dict-style tools with frame_base64)"
            )
            sys.exit(1)
        validate_frame_plugin(tools)

    print("OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
