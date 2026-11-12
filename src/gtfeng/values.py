# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Jiayu Feng
"""How two values are compared.

GT-FENG asks whether an answer A equals a documented rule's gold value G. Values can be numbers, Booleans,
strings, or sets, so each value is first turned into a comparison key, and two values are equal when their keys
are equal.
"""
from __future__ import annotations

import pandas as pd

COLLECTIONS = (list, tuple, set, frozenset)


def is_missing(value) -> bool:
    """True for an empty cell: None, NaN, pd.NA, or NaT. Collections are never missing as a whole."""
    if isinstance(value, COLLECTIONS):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def normalize(value, decimals: int | None = None):
    """Return the comparison key of a value, or None when the value is missing.

    - Booleans are compared exactly (True is not the number 1).
    - Numbers are compared as floats, rounded to `decimals` places when given (so 205 equals 205.0).
    - Lists, tuples, and sets are compared as sets (order and repeats do not matter). A collection with a
      missing member counts as missing.
    - Anything else is compared as text.
    """
    if is_missing(value):
        return None
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, (int, float)):
        number = float(value)
        return ("number", round(number, decimals) if decimals is not None else number)
    if isinstance(value, COLLECTIONS):
        members = [normalize(item, decimals) for item in value]
        return None if None in members else ("set", frozenset(members))
    return ("text", str(value))
