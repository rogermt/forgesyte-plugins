"""Unit test for NumPy float32 JSON serialization error.

This test reproduces the worker crash:
    Object of type float32 is not JSON serializable

It happens when plugin output contains NumPy float32 values,
such as the 'center' coordinates in video_player_tracking.
"""

import json

import numpy as np
import pytest


def test_float32_json_serialization_fails() -> None:
    """
    This test reproduces the REAL worker crash:

        Object of type float32 is not JSON serializable

    It happens when plugin output contains NumPy float32 values,
    such as the 'center' coordinates in video_player_tracking.
    """

    bad_output = {
        "track_id": 12,
        "class_id": 0,
        "xyxy": [10, 20, 30, 40],
        "center": [
            np.float32(15.5),   # ← EXACT crash source
            np.float32(25.5)
        ]
    }

    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(bad_output)   # worker does this internally


def test_float32_json_serialization_fixed() -> None:
    """
    This test verifies the FIX:

        float(np_value)

    After converting NumPy float32 → Python float,
    JSON serialization MUST succeed.
    """

    fixed_output = {
        "track_id": 12,
        "class_id": 0,
        "xyxy": [10, 20, 30, 40],
        "center": [
            float(np.float32(15.5)),   # ← FIXED
            float(np.float32(25.5))
        ]
    }

    try:
        json.dumps(fixed_output)
    except Exception as e:
        pytest.fail(f"Fix failed, unexpected exception: {e}")
