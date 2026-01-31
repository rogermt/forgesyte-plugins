"""Diagnostic script to verify pre-commit is installed."""

import shutil
import sys

print("Running pre-commit diagnostic...")

path = shutil.which("pre-commit")

if path is None:
    print("❌ pre-commit is NOT installed in this environment")
    sys.exit(1)

print(f"✓ pre-commit found at: {path}")
sys.exit(0)
