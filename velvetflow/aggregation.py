"""Helpers for the aggregation DSL backed by Jinja expressions."""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence

from velvetflow.jinja_utils import eval_jinja_expr, validate_jinja_expr


class JinjaExprValidationError(ValueError):
    """Raised when a Jinja expression is invalid."""


def _is_instance_of_type(value: Any, expected: str | None) -> bool:
    if not expected:
        return True
    normalized = expected.lower()
    if normalized.startswith("array") or normalized == "list":
        return isinstance(value, Sequence) and not isinstance(
            value, (str, bytes, bytearray)
        )
    if normalized in {"object", "dict", "mapping"}:
        return isinstance(value, MutableMapping)
    if normalized in {"string", "str"}:
        return isinstance(value, str)
    if normalized in {"integer", "int"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if normalized in {"number", "float"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if normalized in {"boolean", "bool"}:
        return isinstance(value, bool)
    return True


def validate_jinja_expression(expr: Any, *, path: str = "expression") -> None:
    """Compatibility wrapper for validating Jinja expressions in params."""

    try:
        validate_jinja_expr(expr, path=path)
    except ValueError as exc:
        raise JinjaExprValidationError(str(exc)) from exc


def eval_jinja_expression(expr: str, context: Mapping[str, Any]) -> Any:
    """Evaluate a Jinja expression and bubble up syntax/runtime errors."""

    return eval_jinja_expr(expr, context)


__all__ = [
    "JinjaExprValidationError",
    "eval_jinja_expression",
    "validate_jinja_expression",
    "_is_instance_of_type",
]
