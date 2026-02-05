"""Utility for handling missing/invalid metric values."""

from __future__ import annotations

import math
from typing import Optional, Union

# Default value to return instead of NaN
DEFAULT_MISSING_VALUE = 0.0


def safe_score(value: float, default: float = DEFAULT_MISSING_VALUE) -> float:
    """
    Return a safe score value, replacing NaN/None with default.
    
    Args:
        value: The computed score (may be NaN or None)
        default: Value to return if value is invalid (default: 0.0)
        
    Returns:
        Original value if valid, otherwise default
    """
    if value is None:
        return default
    if isinstance(value, float) and math.isnan(value):
        return default
    return value


def validate_inputs(*args, default: float = DEFAULT_MISSING_VALUE) -> Optional[float]:
    """
    Validate that required inputs are present.
    
    Returns default (instead of NaN) if any input is None or empty.
    Returns None if all inputs are valid (caller should continue).
    
    Example:
        early_return = validate_inputs(answer, context)
        if early_return is not None:
            return early_return
    """
    for arg in args:
        if arg is None:
            return default
        if hasattr(arg, '__len__') and len(arg) == 0:
            return default
    return None  # All inputs valid
