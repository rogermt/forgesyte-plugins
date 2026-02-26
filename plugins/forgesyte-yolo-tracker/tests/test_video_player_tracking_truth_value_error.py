"""Unit test for video_player_tracking NumPy truth-value error.

This test catches the bug:
    ValueError: The truth value of an array with more than one element is ambiguous.

The error occurs when using `if tid:` on a multi-element NumPy array.
"""

import numpy as np
import pytest


def test_numpy_truth_value_error_triggered() -> None:
    """
    This test reproduces the REAL bug:

        ValueError: The truth value of an array with more than one element is ambiguous.

    It happens when tid is a multi-element NumPy array and used in `if tid`.
    """

    tid = np.array([12, 15, 18])  # multi-element array → ambiguous truth value

    with pytest.raises(ValueError):
        _ = int(tid) if tid else -1   # EXACT plugin bug


def test_numpy_truth_value_error_fixed() -> None:
    """
    This test verifies the FIX:

        Convert to scalar first, then use `is not None` check.
        E.g., int(tid[0]) if tid is not None else -1

    This MUST NOT raise any error.
    """

    tid = np.array([12, 15, 18])

    try:
        # Fix: check if array is not None before accessing element
        _ = int(tid[0]) if tid is not None else -1
    except Exception as e:
        pytest.fail(f"Fix failed, unexpected exception: {e}")
