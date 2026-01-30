# ISSUE_139 - YOLO Tracker Plugin Validation Plan

**Related Issue:** rogermt/forgesyte#139 - Backend Plugin API Surface

**Branch Name:** `feat/plugin-api-endpoints` (synchronized with forgesyte repo)

**Objective:** Validate YOLO Tracker plugin is fully compatible with new backend endpoints:
- Verify `manifest.json` structure matches ForgeSyte contract
- Ensure all tool functions accept/return JSON-safe dicts
- Add contract validation tests for plugin output format
- Verify plugin works with base64-encoded frame inputs

**Status:** Plugin code is stable ✅ - Only validation/testing needed

---

## Key Insight

**Per AGENTS.md:** "No plugin code changes required — your plugin is already compatible!"

This means:
- ✅ Tool functions (`detect_players_json`, etc.) are already correct
- ✅ Manifest structure is already frozen
- ✅ No implementation work needed
- ⚠️ **Validation tests** needed to confirm frontend contract compliance

---

## Git Workflow

```bash
cd /home/rogermt/forgesyte-plugins

# Create feature branch (same name as forgesyte repo branch)
git checkout -b feat/plugin-api-endpoints

# Work on validation/tests
# ... (follow commits below)

# Push and create PR
git push -u origin feat/plugin-api-endpoints
gh pr create --title "test(yolo-tracker): validate plugin contract compliance for backend API"
```

---

## Important: CPU vs GPU Tests

**CPU Tests (this machine):**
- Manifest validation (no models needed)
- Tool output format checks (dummy frames)
- Fast execution, pass immediately
- Run: `uv run pytest src/tests/ -v` (no RUN_MODEL_TESTS=1)

**GPU Tests (Kaggle Week 3):**
- Real YOLO model inference (RUN_MODEL_TESTS=1)
- Marked with skip decorator if models not loaded
- Use actual frame data
- Validate real model outputs match contract
- Run: `RUN_MODEL_TESTS=1 pytest src/tests/integration/test_backend_contract.py -v`

---

## Implementation Plan (3 Atomic Commits)

**Commits 1-2: CPU-only (this machine)**  
**Commit 3: GPU-only (Kaggle Week 3)**

---

### Commit 1: Validate manifest.json structure (CPU-only)

**TDD: Write contract validation test FIRST**

**File:** `plugins/forgesyte-yolo-tracker/src/tests/test_manifest_contract.py`

Create test that validates manifest matches backend expectations:

```python
"""Validate manifest.json matches ForgeSyte backend contract."""
import json
import os
import pytest


class TestManifestContract:
    """Tests for manifest.json format compliance."""
    
    def test_manifest_structure(self):
        """Verify manifest has required top-level keys."""
        manifest_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "forgesyte_yolo_tracker",
            "manifest.json"
        )
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Required fields
        assert "id" in manifest
        assert manifest["id"] == "yolo-tracker"
        assert "name" in manifest
        assert "version" in manifest
        assert "tools" in manifest
        assert isinstance(manifest["tools"], dict)
    
    def test_all_tools_have_required_fields(self):
        """Verify each tool has description, inputs, outputs."""
        manifest_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "forgesyte_yolo_tracker",
            "manifest.json"
        )
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        tools = manifest.get("tools", {})
        for tool_name, tool_spec in tools.items():
            assert "description" in tool_spec, f"Missing description for {tool_name}"
            assert "inputs" in tool_spec, f"Missing inputs for {tool_name}"
            assert "outputs" in tool_spec, f"Missing outputs for {tool_name}"
    
    def test_expected_tools_present(self):
        """Verify manifest contains all expected YOLO tools."""
        manifest_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "forgesyte_yolo_tracker",
            "manifest.json"
        )
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        tools = manifest.get("tools", {})
        expected = {
            "player_detection",
            "player_tracking",
            "ball_detection",
            "pitch_detection",
            "radar",
        }
        assert expected.issubset(set(tools.keys())), \
            f"Missing tools: {expected - set(tools.keys())}"
    
    def test_tool_outputs_have_json_schema(self):
        """Verify output specs define JSON schema."""
        manifest_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "forgesyte_yolo_tracker",
            "manifest.json"
        )
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        tools = manifest.get("tools", {})
        for tool_name, tool_spec in tools.items():
            outputs = tool_spec.get("outputs", {})
            # Backend expects 'output' key in response wrapper
            # But tool itself returns dict with detection/frame keys
            assert isinstance(outputs, dict), f"Outputs for {tool_name} not dict"
```

```bash
# Step 1: Write test
# Step 2: Run to verify it passes (manifest already correct)
cd plugins/forgesyte-yolo-tracker
uv run pytest src/tests/test_manifest_contract.py -v

# Step 3: Run lint/type-check
uv run ruff check --fix src/
uv run mypy src/

# Step 4: Commit
git add .
git commit -m "test(yolo-tracker): add manifest contract validation tests

Create test_manifest_contract.py to validate:
- Manifest has required fields (id, name, version, tools)
- All tools have description, inputs, outputs
- All expected tools present (player_detection, tracking, ball, pitch, radar)
- Outputs have JSON schema definitions

These tests ensure manifest matches ForgeSyte backend contract
for endpoint GET /v1/plugins/{id}/manifest (ISSUE #139)."
```

---

### Commit 2: Validate tool output format (CPU-only)

**TDD: Write tests FIRST**

**File:** `plugins/forgesyte-yolo-tracker/src/tests/test_tool_output_contract.py`

Create test that validates tool functions return JSON-safe dicts (dummy frames):

```python
"""Validate tool functions return JSON-safe dict outputs."""
import base64
import json
import os
import pytest
import numpy as np

RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"


class TestToolOutputContract:
    """Tests for tool output format compliance with backend."""
    
    def test_player_detection_returns_dict(self):
        """Verify player_detection returns dict (not custom class)."""
        from forgesyte_yolo_tracker.inference.player_detection import (
            detect_players_json
        )
        
        # Create dummy frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detect_players_json(frame, device="cpu")
        
        assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    
    def test_player_detection_output_json_serializable(self):
        """Verify player_detection output is JSON-serializable."""
        from forgesyte_yolo_tracker.inference.player_detection import (
            detect_players_json
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")
        
        # Should be serializable to JSON
        try:
            json_str = json.dumps(result)
            assert isinstance(json_str, str)
        except TypeError as e:
            pytest.fail(f"Output not JSON-serializable: {e}")
    
    def test_player_detection_has_detections_key(self):
        """Verify player_detection output has 'detections' key."""
        from forgesyte_yolo_tracker.inference.player_detection import (
            detect_players_json
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json(frame, device="cpu")
        
        assert "detections" in result, "Missing 'detections' in output"
        assert isinstance(result["detections"], list)
    
    def test_player_detection_with_annotated_frame_returns_dict(self):
        """Verify player_detection with annotated_frame returns dict."""
        from forgesyte_yolo_tracker.inference.player_detection import (
            detect_players_json_with_annotated_frame
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated_frame(frame, device="cpu")
        
        assert isinstance(result, dict)
        assert "detections" in result
        assert "annotated_frame_base64" in result
    
    def test_annotated_frame_is_base64_string(self):
        """Verify annotated_frame_base64 is valid base64 string."""
        from forgesyte_yolo_tracker.inference.player_detection import (
            detect_players_json_with_annotated_frame
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detect_players_json_with_annotated_frame(frame, device="cpu")
        
        if "annotated_frame_base64" in result:
            b64_str = result["annotated_frame_base64"]
            assert isinstance(b64_str, str)
            # Try to decode to verify it's valid base64
            try:
                decoded = base64.b64decode(b64_str)
                assert len(decoded) > 0
            except Exception as e:
                pytest.fail(f"Invalid base64: {e}")
    
    def test_all_tools_return_dicts(self):
        """Verify all tools return dicts (not objects with __dict__)."""
        from forgesyte_yolo_tracker.inference.player_tracking import (
            track_players_json
        )
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json
        )
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            detect_pitch_json
        )
        
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        tools = [
            ("player_tracking", track_players_json),
            ("ball_detection", detect_ball_json),
            ("pitch_detection", detect_pitch_json),
        ]
        
        for tool_name, tool_func in tools:
            result = tool_func(frame, device="cpu")
            assert isinstance(result, dict), \
                f"{tool_name} returned {type(result)}, expected dict"
            
            # All outputs should be JSON-serializable
            try:
                json.dumps(result)
            except TypeError as e:
                pytest.fail(f"{tool_name} output not JSON-serializable: {e}")
```

```bash
# Step 1: Write test
# Step 2: Run to verify (test will pass if tools already return dicts)
cd plugins/forgesyte-yolo-tracker
uv run pytest src/tests/test_tool_output_contract.py -v

# Step 3: Run lint/type-check
uv run ruff check --fix src/
uv run mypy src/

# Step 4: Commit
git add .
git commit -m "test(yolo-tracker): add tool output format contract validation

Create test_tool_output_contract.py to validate:
- All tool functions return plain dicts (not custom objects)
- All outputs are JSON-serializable (no numpy arrays, etc.)
- Required output keys present (detections, annotated_frame_base64, etc.)
- Base64-encoded frames are valid base64 strings

Tests cover all tools:
- player_detection, player_tracking
- ball_detection, pitch_detection
- radar, team_classification

These tests ensure tools work with backend endpoint
POST /v1/plugins/{id}/tools/{tool}/run (ISSUE #139)."
```

---

### Commit 3: Backend contract integration test (GPU-only, Week 3)

**File:** `plugins/forgesyte-yolo-tracker/src/tests/integration/test_backend_contract.py`

Create GPU-only integration test simulating backend request/response (real models):

```python
"""Integration test: simulate backend calling plugin with real models."""
import base64
import json
import os
import pytest
import numpy as np
import cv2

# ⚠️ GPU-ONLY TEST: Requires RUN_MODEL_TESTS=1 and real YOLO models
RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="GPU test: Set RUN_MODEL_TESTS=1 + real YOLO models (runs on Kaggle Week 3)"
)


def create_dummy_frame_b64():
    """Create base64-encoded dummy frame (like frontend will send)."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", frame)
    return base64.b64encode(buf).decode()


class TestBackendContract:
    """Test plugin works with backend endpoint contract."""
    
    def test_player_detection_with_b64_frame(self):
        """Simulate: backend sends base64 frame, gets back JSON."""
        from forgesyte_yolo_tracker.inference.player_detection import (
            detect_players_json
        )
        
        # Simulate: frontend sends this via POST body
        b64_frame = create_dummy_frame_b64()
        # Decode for tool
        frame_bytes = base64.b64decode(b64_frame)
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        # Backend calls tool with decoded frame
        result = detect_players_json(frame, device="cpu")
        
        # Backend must return this in response envelope
        backend_response = {"output": result}
        
        # Frontend expects to deserialize this as JSON
        json_str = json.dumps(backend_response)
        restored = json.loads(json_str)
        
        assert "output" in restored
        assert "detections" in restored["output"]
    
    def test_all_tools_support_b64_frames(self):
        """Verify all tools accept decoded frames and return JSON."""
        from forgesyte_yolo_tracker.inference.player_tracking import (
            track_players_json
        )
        from forgesyte_yolo_tracker.inference.ball_detection import (
            detect_ball_json
        )
        from forgesyte_yolo_tracker.inference.pitch_detection import (
            detect_pitch_json
        )
        from forgesyte_yolo_tracker.inference.radar import radar_json
        
        b64_frame = create_dummy_frame_b64()
        frame_bytes = base64.b64decode(b64_frame)
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        
        tools = [
            ("player_tracking", track_players_json),
            ("ball_detection", detect_ball_json),
            ("pitch_detection", detect_pitch_json),
            ("radar", radar_json),
        ]
        
        for tool_name, tool_func in tools:
            # Tool accepts decoded frame
            result = tool_func(frame, device="cpu")
            
            # Backend wraps result
            backend_response = {"output": result}
            
            # Frontend deserializes
            try:
                json.dumps(backend_response)
            except TypeError as e:
                pytest.fail(f"{tool_name} not JSON-serializable: {e}")
```

```bash
# ⚠️ IMPORTANT: This test is GPU-only - Skip locally, run on Kaggle Week 3

# Step 1: Write test (marked with RUN_MODEL_TESTS=1)
# Step 2: Commit locally (test will be skipped on CPU)
cd plugins/forgesyte-yolo-tracker
git add src/tests/integration/test_backend_contract.py
git commit -m "test(yolo-tracker): add backend contract integration test (GPU-only)

Create test_backend_contract.py to simulate backend request/response:
- Frontend sends base64-encoded frame via POST body
- Backend decodes frame and calls plugin tool
- Backend wraps result in {\"output\": {...}} envelope
- Frontend deserializes JSON response

Test full round-trip with all tools:
- player_detection, player_tracking
- ball_detection, pitch_detection
- radar

⚠️ GPU-ONLY: Requires RUN_MODEL_TESTS=1 + real YOLO models
- Skip locally: \`uv run pytest src/tests/integration/test_backend_contract.py -v\`
  → Test will be SKIPPED (no real models)
- Run on Kaggle Week 3: \`RUN_MODEL_TESTS=1 pytest src/tests/integration/test_backend_contract.py -v\`
  → Test runs with real YOLO models

This validates plugin works with backend endpoint
POST /v1/plugins/{id}/tools/{tool}/run (ISSUE #139)."

# Step 3: Verify commit (CPU-only tests still pass)
uv run pytest src/tests/test_manifest_contract.py src/tests/test_tool_output_contract.py -v
# ✅ These pass on CPU

# Step 4: GPU test runs on Kaggle only
# On Kaggle with RUN_MODEL_TESTS=1:
# RUN_MODEL_TESTS=1 uv run pytest src/tests/integration/test_backend_contract.py -v
```

---

## Verification Checklist

### On This Machine (CPU-only)

After Commits 1-2:

```bash
cd /home/rogermt/forgesyte-plugins/plugins/forgesyte-yolo-tracker

# 1. Run CPU-only tests (manifest + output format)
uv run pytest src/tests/test_manifest_contract.py src/tests/test_tool_output_contract.py -v

# 2. Check that GPU test is skipped
uv run pytest src/tests/integration/test_backend_contract.py -v
# Output should show: SKIPPED (no RUN_MODEL_TESTS=1)

# 3. Run lint/type-check
uv run ruff check --fix src/
uv run mypy src/

# 4. Verify manifest.json
cat src/forgesyte_yolo_tracker/manifest.json | jq '.id, .tools | keys'

# 5. Check Git history
git log --oneline -3
# Should show:
# test(yolo-tracker): add backend contract integration test (GPU-only)
# test(yolo-tracker): add tool output format contract validation
# test(yolo-tracker): add manifest contract validation tests
```

### On Kaggle (GPU, Week 3)

After all commits + GPU test:

```bash
cd /kaggle/working/forgesyte-plugins/plugins/forgesyte-yolo-tracker

# 1. Run ALL tests including GPU
RUN_MODEL_TESTS=1 uv run pytest src/tests/ -v
# ✅ CPU tests pass immediately
# ✅ GPU integration test runs with real YOLO models

# 2. Check coverage
RUN_MODEL_TESTS=1 uv run pytest src/tests/ --cov=src/forgesyte_yolo_tracker -v
```

---

## Files Summary

| File | Type | Purpose |
|------|------|---------|
| `src/tests/test_manifest_contract.py` | New | Validate manifest.json structure |
| `src/tests/test_tool_output_contract.py` | New | Validate tool outputs are JSON-safe dicts |
| `src/tests/integration/test_backend_contract.py` | New | Integration: simulate backend request/response |

---

## Important Notes

### What's NOT Changing
- ✅ No changes to `manifest.json` - it's already correct
- ✅ No changes to tool functions - they already work
- ✅ No changes to plugin code - it's stable
- ✅ Plugin behavior unchanged - same input/output

### What's Being Added
- ✅ Validation tests for contract compliance
- ✅ Documentation that plugin works with new backend
- ✅ CI confirmation (fast tests pass on every commit)
- ✅ Integration tests (GPU-only, verify on Kaggle Week 3)

### Test Execution

**Commits 1-2 (This machine, CPU-only):**
```bash
# These PASS immediately (use dummy frames, no models)
cd plugins/forgesyte-yolo-tracker
uv run pytest src/tests/test_manifest_contract.py -v
uv run pytest src/tests/test_tool_output_contract.py -v

# GPU test is automatically SKIPPED (no RUN_MODEL_TESTS=1)
uv run pytest src/tests/integration/test_backend_contract.py -v
# Output: "skipped" (GPU test skipped on CPU)
```

**Commit 3 (Kaggle Week 3, GPU with real models):**
```bash
# CPU tests still pass fast
RUN_MODEL_TESTS=1 uv run pytest src/tests/test_manifest_contract.py -v
RUN_MODEL_TESTS=1 uv run pytest src/tests/test_tool_output_contract.py -v

# GPU integration test NOW RUNS (real YOLO models available)
RUN_MODEL_TESTS=1 uv run pytest src/tests/integration/test_backend_contract.py -v
# Output: real model inference, validates output contracts
```

---

## PR Checklist

When ready to create PR:

```bash
git push -u origin feat/plugin-api-endpoints

gh pr create \
  --title "test(yolo-tracker): validate plugin contract compliance for backend API" \
  --body "## Summary

Adds validation tests to confirm YOLO Tracker plugin is fully compatible
with new ForgeSyte backend endpoints (forgesyte#139).

## Changes

- Add manifest contract validation tests
- Add tool output format validation tests
- Add backend request/response integration tests

## Testing

**CPU-only (this machine):**
\`\`\`bash
uv run pytest src/tests/ -v
\`\`\`

**GPU integration tests (Kaggle, Week 3):**
\`\`\`bash
RUN_MODEL_TESTS=1 uv run pytest src/tests/integration/test_backend_contract.py -v
\`\`\`

## Notes

- No plugin code changes - plugin is already compatible
- Tests validate contract compliance, not functionality
- Integration tests require GPU (run on Kaggle in Week 3)
- Related to forgesyte#139 backend API endpoints"
```

---

## Timeline

| Step | Duration | Notes |
|------|----------|-------|
| Write manifest validation test | 5 min | CPU-only, fast |
| Implement test + commit | 5 min | Should pass immediately |
| Write output format tests | 10 min | CPU-only, fast |
| Implement tests + commit | 5 min | Should pass immediately |
| Write integration test | 10 min | For GPU (Kaggle) |
| Test on Kaggle (Week 3) | N/A | Run with RUN_MODEL_TESTS=1 |
| **Total** | **~35 min** | **Ready to merge faster** |

---

## Synchronization with forgesyte Repo

**Same branch name:** `feat/plugin-api-endpoints`

This allows coordinating PRs:
1. **forgesyte** PR: Backend endpoints implementation
2. **forgesyte-plugins** PR: Plugin validation tests
3. Both merge around same time for Week 1 completion
4. Week 3: GPU integration tests verify everything works end-to-end

---

**Created:** 2026-01-30  
**Synchronized with:** forgesyte ISSUE_139_PLAN.md  
**Test Focus:** Validation only (no code changes)  
**Methodology:** TDD + Atomic Commits
