"""Helpers for the aggregation DSL (Mini-Expression AST + typing)."""

from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Sequence


class MiniExprValidationError(ValueError):
    """Raised when a Mini-Expression AST is invalid."""


SUPPORTED_OPS = {"==", "!=", ">", ">=", "<", "<=", "and", "or", "not", "in", "exists"}


def _resolve_var(path: str, context: Mapping[str, Any]) -> Any:
    cur: Any = context
    for token in path.split("."):
        if isinstance(cur, Mapping) and token in cur:
            cur = cur[token]
        elif hasattr(cur, token):
            cur = getattr(cur, token)
        elif isinstance(cur, (list, tuple)):
            try:
                idx = int(token)
            except ValueError:
                return None
            if 0 <= idx < len(cur):
                cur = cur[idx]
            else:
                return None
        else:
            return None
    return cur


def validate_mini_expr(expr: Any, *, path: str = "condition") -> None:
    """Validate the shape of a Mini-Expression AST.

    Supported forms:
    - {"const": <any>} (literal)
    - {"var": "path.to.value"}
    - {"op": "<binary op>", "left": <expr>, "right": <expr>}
    - {"op": "and"|"or", "args": [<expr>, ...]}
    - {"op": "not", "arg": <expr>}
    - {"op": "exists", "arg": {"var": "..."}}
    """

    if isinstance(expr, (int, float, str, bool)) or expr is None:
        return

    if not isinstance(expr, Mapping):
        raise MiniExprValidationError(f"{path} 必须是对象或常量表达式")

    if "const" in expr:
        return

    if "var" in expr:
        if not isinstance(expr["var"], str) or not expr["var"].strip():
            raise MiniExprValidationError(f"{path}.var 需要非空字符串")
        return

    op = expr.get("op")
    if op not in SUPPORTED_OPS:
        raise MiniExprValidationError(
            f"{path}.op 不支持值 {op}，可选：{', '.join(sorted(SUPPORTED_OPS))}"
        )

    if op == "not":
        validate_mini_expr(expr.get("arg"), path=f"{path}.arg")
    elif op in {"and", "or"}:
        args = expr.get("args")
        if not isinstance(args, Sequence) or not args:
            raise MiniExprValidationError(f"{path}.args 需要非空数组用于 {op}")
        for idx, arg in enumerate(args):
            validate_mini_expr(arg, path=f"{path}.args[{idx}]")
    elif op == "exists":
        validate_mini_expr(expr.get("arg"), path=f"{path}.arg")
    else:
        validate_mini_expr(expr.get("left"), path=f"{path}.left")
        validate_mini_expr(expr.get("right"), path=f"{path}.right")


def eval_mini_expr(expr: Any, context: Mapping[str, Any]) -> Any:
    """Evaluate a Mini-Expression AST against a context mapping."""

    if isinstance(expr, Mapping):
        if "const" in expr:
            return expr.get("const")
        if "var" in expr:
            return _resolve_var(str(expr.get("var")), context)

        op = expr.get("op")
        if op == "not":
            return not bool(eval_mini_expr(expr.get("arg"), context))
        if op in {"and", "or"}:
            args = expr.get("args") or []
            if op == "and":
                return all(eval_mini_expr(arg, context) for arg in args)
            return any(eval_mini_expr(arg, context) for arg in args)
        if op == "exists":
            value = eval_mini_expr(expr.get("arg"), context)
            if value is None:
                return False
            if isinstance(value, (list, tuple, str, Mapping)):
                return len(value) > 0  # type: ignore[arg-type]
            return True

        left = eval_mini_expr(expr.get("left"), context)
        right = eval_mini_expr(expr.get("right"), context)
        try:
            if op == "==":
                return left == right
            if op == "!=":
                return left != right
            if op == ">":
                return left > right
            if op == ">=":
                return left >= right
            if op == "<":
                return left < right
            if op == "<=":
                return left <= right
            if op == "in":
                return left in right if right is not None else False
        except Exception:
            return False

        return False

    return expr


def _is_instance_of_type(value: Any, expected: str | None) -> bool:
    if not expected:
        return True
    normalized = expected.lower()
    if normalized.startswith("array") or normalized == "list":
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))
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


__all__ = [
    "MiniExprValidationError",
    "eval_mini_expr",
    "validate_mini_expr",
    "_is_instance_of_type",
]
