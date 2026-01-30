# Player Tracking Test Plan (Phase 3)

## Overview
Expand player_tracking tests from **6 → 30+ tests** to cover track lifecycle, persistence, and multi-player scenarios.

## File to Create/Expand
`src/tests/test_inference_player_tracking_refactored.py`

## Test Breakdown (30+ tests)

### 1. Track ID Assignment & Uniqueness (4 tests)

#### test_track_id_assignment_single_player
- **Setup:** Single player in frame
- **Expected:** Track ID assigned uniquely
- **Assert:** `trackID` is int, > 0, consistent across frames

#### test_track_id_uniqueness_multi_player
- **Setup:** 5 players in same frame
- **Expected:** Each player gets unique track ID
- **Assert:** All track IDs unique, no duplicates

#### test_track_id_persistence_single_frame
- **Setup:** Frame with 3 players
- **Expected:** Same track IDs in subsequent frame
- **Assert:** Track IDs remain same (deterministic)

#### test_track_id_format_validation
- **Setup:** Detect players, get track response
- **Expected:** trackID field exists, is integer
- **Assert:** `isinstance(detection['trackID'], int)`

---

### 2. Track Persistence Across Frames (5 tests)

#### test_track_persistence_stationary_player
- **Setup:** Player stationary in frame 1, frame 2
- **Expected:** Same track ID assigned
- **Assert:** `frame2_track_id == frame1_track_id`

#### test_track_persistence_moving_player
- **Setup:** Player moves 50px between frames
- **Expected:** Same track ID persists
- **Assert:** Track ID unchanged, position updated

#### test_track_persistence_slow_motion
- **Setup:** 10 sequential frames, slow movement
- **Expected:** Track ID remains same throughout
- **Assert:** `all(tid == track_ids[0] for tid in track_ids)`

#### test_track_persistence_across_occlusion_brief
- **Setup:** Player visible → occluded (1 frame) → visible
- **Expected:** Same track ID when reappears
- **Assert:** Track ID resumes after 1-frame absence

#### test_track_persistence_speed_variations
- **Setup:** Player moves at varying speeds (slow, fast, slow)
- **Expected:** Track ID persistent despite speed changes
- **Assert:** Single consistent track ID across motion

---

### 3. Occlusion Handling (6 tests)

#### test_occlusion_brief_disappearance
- **Setup:** Player visible → occluded 1 frame → visible
- **Expected:** Same track ID resumes
- **Assert:** Track ID matches pre-occlusion ID

#### test_occlusion_medium_disappearance
- **Setup:** Player visible → occluded 3 frames → visible
- **Expected:** Track reestablished (may create new ID or resume)
- **Assert:** Track logic handles 3-frame absence

#### test_occlusion_partial_bounding_box
- **Setup:** Player partially occluded (50% visible)
- **Expected:** Still detected with reduced confidence
- **Assert:** Detection still in output, confidence < full visible

#### test_occlusion_multiple_players
- **Setup:** 3 players, 2 get occluded simultaneously
- **Expected:** Non-occluded player track persists
- **Assert:** Track ID unchanged for visible player

#### test_occlusion_reappearance_position_jump
- **Setup:** Player occluded, reappears in different position
- **Expected:** Either new track or position updated
- **Assert:** JSON validates, position reflects reappearance

#### test_occlusion_recovery_confidence
- **Setup:** Player reappears after occlusion
- **Expected:** Confidence possibly lower initially
- **Assert:** Confidence field exists, valid range [0,1]

---

### 4. Track Lifecycle (Birth, Death, Resumption) (5 tests)

#### test_track_birth_new_player_enters
- **Setup:** Frame 1 no player, Frame 2 player enters
- **Expected:** New track ID assigned
- **Assert:** trackID > previous max ID

#### test_track_death_player_leaves
- **Setup:** Player visible → leaves frame
- **Expected:** Track removed from output
- **Assert:** Player not in next frame detections

#### test_track_resumption_after_absence
- **Setup:** Track visible → absent 2 frames → visible again
- **Expected:** Either resume or new track (implementation decision)
- **Assert:** Track handling is consistent

#### test_multiple_births_simultaneous
- **Setup:** Frame 1: 2 players, Frame 2: 5 players enter
- **Expected:** 3 new tracks created
- **Assert:** New track IDs for new players

#### test_track_lifecycle_complete
- **Setup:** Birth → persistence → movement → exit
- **Expected:** All lifecycle stages tracked
- **Assert:** Track goes through all states cleanly

---

### 5. Multi-Player Scenarios (4 tests)

#### test_5_player_tracking
- **Setup:** 5 players in single frame
- **Expected:** All 5 tracked with unique IDs
- **Assert:** 5 detections, 5 unique track IDs

#### test_11_player_tracking
- **Setup:** 11 players (full soccer team) in frame
- **Expected:** All tracked with unique IDs
- **Assert:** 11 detections, 11 unique track IDs

#### test_crowded_scene_tracking
- **Setup:** 20+ overlapping/close players
- **Expected:** Each player tracked (even if partially visible)
- **Assert:** Count matches ground truth (or reasonable approximation)

#### test_player_switching_positions
- **Setup:** Players swap positions between frames
- **Expected:** Each player maintains unique track ID
- **Assert:** Track IDs switch only if players actually swap

---

### 6. Confidence Threshold Filtering (3 tests)

#### test_low_confidence_player_excluded
- **Setup:** Detect players with confidence=0.15 (below threshold)
- **Expected:** Player not in detections
- **Assert:** Detection count < total detected objects

#### test_high_confidence_player_included
- **Setup:** Detect players with confidence=0.75
- **Expected:** Player included in detections
- **Assert:** Detection count = expected

#### test_confidence_threshold_boundary
- **Setup:** Player at exact threshold (e.g., 0.25)
- **Expected:** Included (threshold is inclusive)
- **Assert:** Player in detections at boundary

---

### 7. JSON Output Validation (2 tests)

#### test_track_json_schema_valid
- **Setup:** Run `track_players_json()` on frame
- **Expected:** JSON matches ForgeSyte schema
- **Assert:** Validate against manifest schema

#### test_track_json_contains_all_fields
- **Setup:** Get track JSON output
- **Expected:** All required fields present
- **Assert:** Has `trackID`, `bbox`, `confidence`, `class`, etc.

---

### 8. Device Parameter (2 tests)

#### test_device_cpu
- **Setup:** Run tracking with device="cpu"
- **Expected:** Works on CPU
- **Assert:** Returns valid output (may be slower)

#### test_device_cuda
- **Setup:** Run tracking with device="cuda"
- **Expected:** Works on CUDA if available
- **Assert:** Returns valid output, faster than CPU

---

## Test Pattern Template

```python
"""Tests for player tracking inference module."""

import os
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

RUN_MODEL_TESTS = os.getenv("RUN_MODEL_TESTS", "0") == "1"

pytestmark = pytest.mark.skipif(
    not RUN_MODEL_TESTS,
    reason="Set RUN_MODEL_TESTS=1 to run (requires YOLO model)"
)


class TestPlayerTrackingJSON:
    """Tests for track_players_json function."""

    def test_track_id_assignment_single_player(self) -> None:
        """Verify track ID assigned to single player."""
        from forgesyte_yolo_tracker.inference.player_tracking import track_players_json

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = track_players_json(frame, device="cpu")

        assert isinstance(result, dict)
        assert "detections" in result
        # assertions here

    # ... other tests
```

## Success Criteria

- [ ] All 30+ tests defined
- [ ] All tests have docstrings
- [ ] Tests cover 8 categories
- [ ] Tests pass on CPU (mocked if needed)
- [ ] Tests pass on GPU (RUN_MODEL_TESTS=1) **KAGGLE ONLY - DO NOT RUN ON LAPTOP**
- [ ] Coverage > 90% for player_tracking.py

## ⚠️ IMPORTANT: GPU Tests on Kaggle Only

**RUN_MODEL_TESTS=1 LOCKS YOUR LAPTOP — DO NOT RUN LOCALLY**

```bash
# ❌ DO NOT RUN THIS ON YOUR LAPTOP:
RUN_MODEL_TESTS=1 pytest src/tests/ -v

# ✅ ONLY RUN ON KAGGLE GPU with timeout:
RUN_MODEL_TESTS=1 pytest src/tests/ -v --timeout=300
```

**Reason:** Loading YOLO models requires GPU memory. Without GPU, process hangs indefinitely.

**Agent Instructions:** Never invoke `RUN_MODEL_TESTS=1` tests. User will run on Kaggle only.

## Expected Coverage Impact

```
Before:  player_tracking.py ≈ 40% coverage
After:   player_tracking.py ≈ 90%+ coverage
```

## Notes
- Use fixtures for common frame creation
- Mock YOLO model for CPU tests
- Run GPU tests on Kaggle only
- Validate against manifest schema
