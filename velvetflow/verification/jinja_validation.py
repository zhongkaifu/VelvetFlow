"""Jinja expression validation helpers for workflow params."""
from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Tuple

from jinja2 import TemplateError

from velvetflow.jinja_utils import get_jinja_env
from velvetflow.models import ValidationError
from velvetflow.reference_utils import normalize_reference_path


_SIMPLE_PATH_RE = re.compile(r"^(result_of|loop)\.[A-Za-z_][\w.]*$")


def _normalize_jinja_expr(value: Any) -> Tuple[Any, bool]:
    """Normalize legacy binding/value formats into Jinja-friendly strings."""

    if isinstance(value, Mapping) and "__from__" in value and isinstance(value.get("__from__"), str):
        return f"{{{{ {normalize_reference_path(value['__from__'])} }}}}", True

    if isinstance(value, str):
        stripped = value.strip()
        if stripped and "{{" not in stripped and "{%" not in stripped:
            if _SIMPLE_PATH_RE.match(stripped):
                return f"{{{{ {stripped} }}}}", True

    return value, False


def _validate_template(expr: str) -> str | None:
    env = get_jinja_env()
    try:
        env.parse(expr)
    except TemplateError as exc:  # pragma: no cover - parser raises directly
        return str(exc)
    return None


def normalize_condition_params_to_jinja(
    workflow: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[ValidationError]]:
    """Ensure condition.params only carry Jinja-friendly expressions.

    The function performs two tasks:
    1) Convert legacy binding dicts或裸露的 result_of/loop 路径为标准的
       ``"{{ ... }}"`` Jinja 字符串；
    2) 使用 Jinja 解析器检查所有字符串字段是否语法合法，返回错误列表
       以便进入自动/LLM 修复流程。
    """

    normalized = deepcopy(workflow)
    errors: List[ValidationError] = []
    replacements: List[Dict[str, Any]] = []
    applied = False

    def _walk(container: Any, path_parts: List[str]) -> None:
        nonlocal applied

        if isinstance(container, Mapping):
            for key, val in list(container.items()):
                normalized_val, changed = _normalize_jinja_expr(val)
                if changed:
                    container[key] = normalized_val
                    applied = True
                    replacements.append(
                        {
                            "path": ".".join(path_parts + [str(key)]),
                            "from": val,
                            "to": normalized_val,
                        }
                    )
                _walk(container[key], path_parts + [str(key)])
            return

        if isinstance(container, list):
            for idx, item in enumerate(list(container)):
                _walk(item, path_parts + [f"[{idx}]"])
            return

        if isinstance(container, str):
            error = _validate_template(container)
            if error:
                errors.append(
                    ValidationError(
                        code="INVALID_JINJA_EXPRESSION",
                        node_id=None,
                        field=".".join(path_parts),
                        message=f"Jinja 表达式无法解析: {error}",
                    )
                )

    for node in normalized.get("nodes", []) or []:
        if not isinstance(node, Mapping) or node.get("type") != "condition":
            continue

        params = node.get("params")
        if not isinstance(params, Mapping):
            continue

        _walk(params, ["params"])
        for err in errors:
            if err.node_id is None:
                err.node_id = node.get("id")  # type: ignore[assignment]

    summary: Dict[str, Any] = {"applied": applied}
    if replacements:
        summary["replacements"] = replacements

    return normalized, summary, errors


__all__ = ["normalize_condition_params_to_jinja"]
