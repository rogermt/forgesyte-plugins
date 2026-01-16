#!/usr/bin/env bash
# Usage: ./create_plugin.sh my_plugin

PLUGIN_NAME=$1
PLUGIN_DIR="forgesyte-plugins/plugins/$PLUGIN_NAME"

# Create directory structure
mkdir -p $PLUGIN_DIR/src/$PLUGIN_NAME

# Add __init__.py
echo "def main():
    print('Hello from $PLUGIN_NAME')" > $PLUGIN_DIR/src/$PLUGIN_NAME/__init__.py

# Create pyproject.toml
cat > $PLUGIN_DIR/pyproject.toml <<EOF
[project]
name = "forgesyte-$PLUGIN_NAME"
version = "0.1.0"
description = "ForgeSyte plugin: $PLUGIN_NAME"
requires-python = ">=3.9"

[tool.setuptools]
package-dir = {"" = "src"}
packages = ["$PLUGIN_NAME"]

[project.entry-points."forgesyte.plugins"]
$PLUGIN_NAME = "$PLUGIN_NAME:main"
EOF

# Create plugin.json
cat > $PLUGIN_DIR/plugin.json <<EOF
{
  "name": "$PLUGIN_NAME",
  "version": "0.1.0",
  "description": "ForgeSyte plugin: $PLUGIN_NAME",
  "inputs": ["text"],
  "outputs": ["analysisResult"]
}
EOF

echo "✅ Plugin scaffold created at $PLUGIN_DIR"
echo "Next: uv pip install -e $PLUGIN_DIR"

