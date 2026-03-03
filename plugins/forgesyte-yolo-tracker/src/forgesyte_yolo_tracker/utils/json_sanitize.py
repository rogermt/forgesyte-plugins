"""JSON Sanitizer for NumPy types.

v0.10.0: Ensures all plugin outputs are JSON-serializable.

Converts NumPy types to Python primitives:
- np.float16, np.float32, np.float64 → float
- np.int8, np.int16, np.int32, np.int64 → int
- np.uint8, np.uint16, np.uint32, np.uint64 → int
- np.ndarray → list
- np.bool_ → bool
"""

import numpy as np
from typing import Any


def sanitize_json(obj: Any) -> Any:
    """
    Recursively convert NumPy types into JSON-safe Python types.

    Ensures no np.float32, np.int64, np.ndarray, or other non-serializable
    objects appear in the final plugin output.

    Args:
        obj: Any Python object, potentially containing NumPy types

    Returns:
        Object with all NumPy types converted to Python primitives

    Example:
        >>> sanitize_json({
        ...     "track_id": np.int64(1),
        ...     "center": np.array([np.float32(100.5), np.float32(200.5)])
        ... })
        {'track_id': 1, 'center': [100.5, 200.5]}
    """
    # Dict → sanitize values
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}

    # List / tuple → sanitize each element
    if isinstance(obj, (list, tuple)):
        return [sanitize_json(v) for v in obj]

    # NumPy array → convert to list, then sanitize elements
    if isinstance(obj, np.ndarray):
        return sanitize_json(obj.tolist())

    # NumPy floats → Python float
    if isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)

    # NumPy ints → Python int
    if isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)

    # NumPy unsigned ints → Python int
    if isinstance(obj, (np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)

    # NumPy bool → Python bool
    if isinstance(obj, np.bool_):
        return bool(obj)

    # Pass through Python primitives and None
    return obj
