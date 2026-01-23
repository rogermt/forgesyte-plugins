# MyPy Type Checking Issues

## Overview

The following pre-existing mypy errors prevent full type checking. These are documented in GitHub Issue #48.

## Missing Type Stubs

### PyYAML (types-PyYAML)
- **File**: `src/forgesyte_yolo_tracker/configs/__init__.py:6`
- **Error**: Library stubs not installed for "yaml"
- **Fix**: `uv pip install types-PyYAML`

### tqdm (types-tqdm)
- **File**: `src/forgesyte_yolo_tracker/utils/team.py:9`
- **Error**: Library stubs not installed for "tqdm"
- **Fix**: `uv pip install types-tqdm`

### scikit-learn (sklearn)
- **File**: `src/forgesyte_yolo_tracker/utils/team.py:8`
- **Error**: Skipping analyzing "sklearn.cluster": missing library stubs or py.typed marker
- **Note**: Package is installed but lacks type hints
- **Workaround**: Suppress error in mypy.ini or wait for scikit-learn 1.3+ with py.typed

### UMAP (umap-learn)
- **File**: `src/forgesyte_yolo_tracker/utils/team.py:13`
- **Error**: Skipping analyzing "umap": missing library stubs or py.typed marker
- **Note**: Package is installed but lacks type hints
- **Workaround**: Suppress error in mypy.ini or upgrade umap-learn to version with py.typed

## FastAPI Context Issues

### app.models
- **Files**: 
  - `src/forgesyte_yolo_tracker/plugin.py:18`
  - `src/tests/test_plugin_all_detections.py:13`
- **Error**: Cannot find implementation or library stub for module named "app.models"
- **Note**: This is expected in CPU-only test environment without FastAPI context
- **Status**: Not critical for plugin functionality

## Recommended Actions

### For Development
1. Install types packages:
   ```bash
   uv pip install types-PyYAML types-tqdm
   ```

2. Add to `pyproject.toml`:
   ```toml
   [project.optional-dependencies]
   dev = [
       "types-PyYAML>=6.0",
       "types-tqdm>=4.65",
   ]
   ```

### For CI/CD
Consider adding mypy suppressions for known issues:

```ini
# mypy.ini
[mypy-sklearn.*]
ignore_missing_imports = True

[mypy-umap]
ignore_missing_imports = True

[mypy-app.models]
ignore_missing_imports = True
```

## References
- GitHub Issue: #48
- Related: DRY refactor PR #46, team_classification removal PR #47
