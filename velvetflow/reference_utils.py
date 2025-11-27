"""Utilities for normalizing workflow reference strings."""

from __future__ import annotations

import re
from typing import Any


_REFERENCE_PATTERNS = [
    re.compile(r"^\{\{\s*(?P<path>[^{}]+?)\s*\}\}$"),
    re.compile(r"^\$\{\{\s*(?P<path>[^{}]+?)\s*\}\}$"),
]


def normalize_reference_path(path: Any) -> Any:
    """Strip supported template wrappers and return the inner reference path.

    The workflow DSL historically used raw dotted strings such as
    ``result_of.node.output`` for node/field references. Users now sometimes
    express the same intent with ``{{result_of.node.output}}`` or
    ``${{ result_of.node.output }}``. This helper normalizes those templated
    inputs back to the original raw dotted form so downstream validators and
    executors can handle all styles uniformly.

    Non-string inputs are returned as-is.
    """

    if not isinstance(path, str):
        return path

    for pattern in _REFERENCE_PATTERNS:
        match = pattern.match(path)
        if match:
            return match.group("path").strip()

    return path
