#!/bin/bash

# ForgeSyte Plugin Quality Check Script
# Runs black, isort, ruff, mypy, and pytest across all active plugins.

set -e

PLUGINS=("ocr" "forgesyte-yolo-tracker")

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Starting ForgeSyte Plugins Quality Checks..."
echo "-------------------------------------------"

for plugin in "${PLUGINS[@]}"; do
    echo -e "${GREEN}Checking plugin: $plugin${NC}"
    cd "plugins/$plugin"

    # Ensure dev dependencies are installed (including mypy and isort)
    echo "  Ensuring dev dependencies..."
    uv pip install -e ".[dev]" mypy isort > /dev/null 2>&1

    echo "  Running black..."
    uv run black .

    echo "  Running isort..."
    uv run isort .

    echo "  Running ruff..."
    uv run ruff check --fix .

    echo "  Running mypy..."
    # Using the root mypy.ini and MYPYPATH to handle the app.models mock correctly
    export MYPYPATH=../..
    uv run mypy . --no-site-packages --config-file ../../mypy.ini

    echo "  Running pytest..."
    uv run pytest

    cd - > /dev/null
    echo -e "${GREEN}✓ $plugin passed all checks${NC}"
    echo "-------------------------------------------"
done

echo -e "${GREEN}All plugins passed quality checks!${NC}"
