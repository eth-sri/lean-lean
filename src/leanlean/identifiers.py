"""Identifiers of artifacts published before the project's rename.

The leanlean_20260914 benchmark bundle (and its comparator contracts and
repository database) records kinds and schemas with the earlier
``leancompression_`` prefix. They are read as their current ``leanlean_``
equivalents; new artifacts are always written with the current names.
"""

from __future__ import annotations

from typing import Any

LEGACY_PREFIX = "leancompression_"
PREFIX = "leanlean_"


def same_identifier(value: Any, current: str) -> bool:
    """True for ``current`` or its pre-rename spelling."""
    return value == current or (
        isinstance(value, str)
        and value.startswith(LEGACY_PREFIX)
        and PREFIX + value.removeprefix(LEGACY_PREFIX) == current
    )
