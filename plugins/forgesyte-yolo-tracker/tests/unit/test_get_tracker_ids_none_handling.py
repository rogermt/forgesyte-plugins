"""Unit test for get_tracker_ids None handling with NumPy arrays.

Demonstrates that:
1. get_tracker_ids(...) or [] raises ValueError on empty NumPy arrays
2. Explicit if tracker_ids is None: tracker_ids = [] avoids the error
"""

import numpy as np
import pytest

from forgesyte_yolo_tracker.tracking import get_tracker_ids


class DummyDetections:
    """Simulates sv.Detections with various tracker_id states."""

    def __init__(self, tracker_id: np.ndarray | None = None) -> None:
        self.tracker_id = tracker_id
        self.class_id = np.array([0, 0])
        self.xyxy = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])


def test_get_tracker_ids_or_empty_list_raises_on_empty_array() -> None:
    """
    Demonstrates the bug: using `or []` with empty NumPy array raises ValueError.

    The pattern `get_tracker_ids(...) or []` fails when tracker_id is an
    empty NumPy array, because the `or` operator evaluates the truth value
    of the array, which is ambiguous for multi-element arrays.
    """
    det = DummyDetections(tracker_id=np.array([]))  # Empty array
    tracker_ids = get_tracker_ids(det)

    # This SHOULD raise ValueError about ambiguous truth value
    with pytest.raises(ValueError, match="truth value"):
        _ = tracker_ids or []


def test_get_tracker_ids_explicit_none_check_handles_empty_array() -> None:
    """
    Demonstrates the fix: explicit None check avoids the error.

    This pattern avoids the truth-value error:
        if tracker_ids is None:
            tracker_ids = []

    We check specifically for None (not falsy), so empty arrays are handled safely.
    """
    det = DummyDetections(tracker_id=np.array([]))  # Empty array
    tracker_ids = get_tracker_ids(det)

    # This is the FIXED pattern - should NOT raise
    if tracker_ids is None:
        tracker_ids = []

    assert isinstance(tracker_ids, (list, np.ndarray))


def test_get_tracker_ids_explicit_none_check_handles_none() -> None:
    """Verifies the fix also handles None return value correctly."""
    det = DummyDetections(tracker_id=None)
    tracker_ids = get_tracker_ids(det)

    # Explicit None check should convert None to empty list
    if tracker_ids is None:
        tracker_ids = []

    assert isinstance(tracker_ids, list)
    assert tracker_ids == []


def test_get_tracker_ids_explicit_none_check_preserves_array() -> None:
    """Verifies the fix preserves non-empty arrays."""
    det = DummyDetections(tracker_id=np.array([1, 2, 3]))
    tracker_ids = get_tracker_ids(det)

    # Explicit None check doesn't affect non-None values
    if tracker_ids is None:
        tracker_ids = []

    assert isinstance(tracker_ids, np.ndarray)
    np.testing.assert_array_equal(tracker_ids, np.array([1, 2, 3]))
