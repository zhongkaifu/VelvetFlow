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
    """Normalize binding/value formats into Jinja-friendly strings."""

    if (
        isinstance(value, Mapping)
        and "__from__" in value
        and "__agg__" not in value
        and isinstance(value.get("__from__"), str)
    ):
        return f"{{{{ {normalize_reference_path(value['__from__'])} }}}}", True

    if isinstance(value, str):
        stripped = value.strip()
        if stripped and "{{" not in stripped and "{%" not in stripped:
            if _SIMPLE_PATH_RE.match(stripped):
                return f"{{{{ {stripped} }}}}", True
            # Fallback: wrap raw literals as Jinja templates so every param
            # remains Jinja-compatible (e.g., condition.field="employee_id").
            return f"{{{{ {repr(value)} }}}}", True

    return value, False


def _validate_template(expr: str) -> str | None:
    env = get_jinja_env()
    try:
        env.parse(expr)
    except TemplateError as exc:  # pragma: no cover - parser raises directly
        return str(exc)
    return None


def _validate_string_as_jinja(
    value: str, *, path_parts: List[str], node_id: str | None, errors: List[ValidationError]
) -> None:
    error = _validate_template(value)
    if error:
        errors.append(
            ValidationError(
                code="INVALID_JINJA_EXPRESSION",
                node_id=node_id,
                field=".".join(path_parts),
                message=f"Failed to parse Jinja expression: {error}",
            )
        )


def normalize_condition_params_to_jinja(
    workflow: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[ValidationError]]:
    """Ensure condition.params only carry Jinja-friendly expressions."""

    normalized = deepcopy(workflow)
    errors: List[ValidationError] = []
    replacements: List[Dict[str, Any]] = []
    applied = False

    def _walk(container: Any, path_parts: List[str], node_id: str | None) -> None:
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
                _walk(container[key], path_parts + [str(key)], node_id)
            return

        if isinstance(container, list):
            for idx, item in enumerate(list(container)):
                _walk(item, path_parts + [f"[{idx}]"] , node_id)
            return

        if isinstance(container, str):
            _validate_string_as_jinja(container, path_parts=path_parts, node_id=node_id, errors=errors)

    for node in normalized.get("nodes", []) or []:
        if not isinstance(node, Mapping) or node.get("type") != "condition":
            continue

        params = node.get("params")
        if not isinstance(params, Mapping):
            continue

        _walk(params, ["params"], node.get("id"))

    summary: Dict[str, Any] = {"applied": applied}
    if replacements:
        summary["replacements"] = replacements

    return normalized, summary, errors


def normalize_params_to_jinja(
    workflow: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[ValidationError]]:
    """Normalize *all* node params to Jinja and validate syntax.

    - Converts ``__from__`` bindings or裸露的 ``result_of``/``loop`` 路径
      to标准的 ``"{{ ... }}"`` 模板字符串；
    - 标记任何 ``__agg__``/``__from__`` DSL 字段为 Jinja 违规，促使
      自动/LLM 修复；
    - 使用 Jinja 解析器校验所有字符串，确保最终结构对引擎可解析。
    """

    normalized = deepcopy(workflow)
    errors: List[ValidationError] = []
    replacements: List[Dict[str, Any]] = []
    forbidden_paths: List[str] = []
    applied = False

    def _walk_value(container: Any, path_parts: List[str], node_id: str | None) -> None:
        nonlocal applied

        if isinstance(container, Mapping):
            if "__agg__" in container:
                errors.append(
                    ValidationError(
                        code="INVALID_JINJA_EXPRESSION",
                        node_id=node_id,
                        field=".".join(path_parts),
                        message="params must be Jinja expressions; detected __agg__ aggregation DSL",
                    )
                )
                forbidden_paths.append(".".join(path_parts))

            # Avoid silently rewriting aggregator DSL; treat as violation and skip conversion
            if "__agg__" in container:
                for key, val in list(container.items()):
                    _walk_value(val, path_parts + [str(key)], node_id)
                return

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
                _walk_value(container[key], path_parts + [str(key)], node_id)
            return

        if isinstance(container, list):
            for idx, item in enumerate(list(container)):
                _walk_value(item, path_parts + [f"[{idx}]"] , node_id)
            return

        if isinstance(container, str):
            _validate_string_as_jinja(container, path_parts=path_parts, node_id=node_id, errors=errors)

    def _walk_nodes(nodes: List[Any]) -> None:
        for node in nodes or []:
            if not isinstance(node, Mapping):
                continue
            node_id = node.get("id") if isinstance(node.get("id"), str) else None
            params = node.get("params")
            if isinstance(params, Mapping):
                _walk_value(params, ["params"], node_id)

                if node.get("type") == "loop":
                    body = params.get("body_subgraph") if isinstance(params, Mapping) else None
                    if isinstance(body, Mapping) and isinstance(body.get("nodes"), list):
                        _walk_nodes(body.get("nodes") or [])

    _walk_nodes(normalized.get("nodes", []) or [])

    summary: Dict[str, Any] = {"applied": applied}
    if replacements:
        summary["replacements"] = replacements
    if forbidden_paths:
        summary["forbidden_paths"] = forbidden_paths

    return normalized, summary, errors


__all__ = ["normalize_condition_params_to_jinja", "normalize_params_to_jinja"]
