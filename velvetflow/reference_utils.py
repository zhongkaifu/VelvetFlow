# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Utilities for normalizing workflow reference strings."""

from __future__ import annotations

import re
from typing import Any, List, Mapping, Union


_REFERENCE_PATTERNS = [
    re.compile(r"^\{\{\s*(?P<path>[^{}]+?)\s*\}\}$"),
    re.compile(r"^\$\{\{\s*(?P<path>[^{}]+?)\s*\}\}$"),
    re.compile(r"^\$\{\s*(?P<path>[^{}]+?)\s*\}$"),
]

_TEMPLATE_CANONICALIZE_PATTERN = re.compile(
    r"\$\{\{\s*([^{}]+?)\s*\}\}|\$\{\s*([^{}]+?)\s*\}",
)

_TEMPLATE_REF_PATTERN = re.compile(
    r"\{\{\s*([^{}]+?)\s*\}\}|\$\{\{\s*([^{}]+?)\s*\}\}|\$\{\s*([^{}]+?)\s*\}",
)


def normalize_reference_path(path: Any) -> Any:
    """Strip supported template wrappers and return the inner reference path.

    The workflow DSL historically used raw dotted strings such as
    ``result_of.node.output`` for node/field references. Users now sometimes
    express the same intent with ``{{result_of.node.output}}`` or
    ``${ result_of.node.output }`` (including the historic ``${{ ... }}`` form).
    This helper normalizes those templated inputs back to the original raw
    dotted form so downstream validators and executors can handle all styles
    uniformly.

    Non-string inputs are returned as-is.
    """

    if not isinstance(path, str):
        return path

    for pattern in _REFERENCE_PATTERNS:
        match = pattern.match(path)
        if match:
            return match.group("path").strip()

    return path


def canonicalize_template_placeholders(text: Any) -> Any:
    """Return ``text`` with ``${...}``/``${{...}}`` placeholders rewritten to ``{{...}}``.

    Non-string inputs are returned unchanged to keep call sites ergonomic. Whitespace
    inside placeholders is stripped, matching ``normalize_reference_path`` behavior.
    """

    if not isinstance(text, str):
        return text

    def _replace(match: re.Match[str]) -> str:
        path = match.group(1) or match.group(2) or ""
        return f"{{{{{path.strip()}}}}}"

    return _TEMPLATE_CANONICALIZE_PATTERN.sub(_replace, text)


def _normalize_template_reference_path(path: str) -> str | None:
    """Best-effort cleanup for template reference paths.

    The planner sometimes produces placeholders with excessive whitespace
    around dots or list indices (e.g., ``{{result_of.foo[ 0 ].bar}}``) which
    would otherwise fail ``parse_field_path`` during validation. This helper
    trims those artifacts so downstream checks can operate on a consistent
    representation. ``None`` is returned when the path cannot be repaired.
    """

    if not isinstance(path, str):
        return None

    normalized = normalize_reference_path(path)
    if not isinstance(normalized, str):
        return None

    candidates = [normalized]
    compact_dots = re.sub(r"\s*\.\s*", ".", normalized).strip(" .")
    compact_indices = re.sub(r"\[\s*([^\]]+?)\s*\]", r"[\1]", compact_dots)
    candidates.append(compact_indices)

    for candidate in candidates:
        try:
            parse_field_path(candidate)
            return candidate
        except Exception:
            continue

    return None


def normalize_template_placeholders(obj: Any) -> Any:
    """Recursively canonicalize template placeholders across a workflow tree."""

    if isinstance(obj, str):
        normalized = canonicalize_template_placeholders(obj)

        def _repair(match: re.Match[str]) -> str:
            raw = match.group(1) or match.group(2) or match.group(3) or ""
            repaired = _normalize_template_reference_path(raw)
            if not repaired:
                return match.group(0)
            return f"{{{{{repaired}}}}}"

        return _TEMPLATE_REF_PATTERN.sub(_repair, normalized)

    if isinstance(obj, list):
        return [normalize_template_placeholders(item) for item in obj]

    if isinstance(obj, Mapping):
        return {k: normalize_template_placeholders(v) for k, v in obj.items()}

    return obj


def parse_field_path(path: str) -> List[Union[str, int]]:
    """Split dotted/bracket paths (e.g., ``a[0].b``) into tokens.

    Returns a list of strings and integers representing object keys and list
    indices. Invalid fragments (such as missing brackets or non-numeric
    indices) will raise ``ValueError``. ``*`` is accepted inside brackets to
    indicate a wildcard match for list elements.
    """

    if not isinstance(path, str) or not path:
        raise ValueError("path must be non-empty string")

    tokens: List[Union[str, int]] = []

    for segment in path.split("."):
        if segment == "":
            raise ValueError("empty path segment")

        cursor = 0
        while cursor < len(segment):
            bracket_pos = segment.find("[", cursor)
            if bracket_pos == -1:
                remainder = segment[cursor:]
                if remainder.isdigit():
                    tokens.append(int(remainder))
                else:
                    tokens.append(remainder)
                cursor = len(segment)
                continue

            if bracket_pos > cursor:
                tokens.append(segment[cursor:bracket_pos])

            end_bracket = segment.find("]", bracket_pos)
            if end_bracket == -1:
                raise ValueError(f"missing closing bracket in segment '{segment}'")

            index_literal = segment[bracket_pos + 1 : end_bracket]
            if index_literal == "*":
                tokens.append("*")
            elif index_literal.isdigit():
                tokens.append(int(index_literal))
            else:
                raise ValueError(
                    f"list index must be integer or '*' in segment '{segment}'"
                )
            cursor = end_bracket + 1

        if not tokens:
            raise ValueError("no tokens parsed from path")

    return tokens
