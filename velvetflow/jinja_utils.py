"""Shared Jinja environment helpers used across planning and execution."""
from __future__ import annotations

import json
import re
from datetime import datetime, date as date_cls
from functools import lru_cache
from typing import Any, Mapping

from collections.abc import Mapping as MappingABC

from jinja2 import Environment, StrictUndefined, TemplateError, nodes


@lru_cache(maxsize=1)
def get_jinja_env() -> Environment:
    """Return a preconfigured Jinja environment.

    The environment disables autoescaping, raises on undefined variables, and
    trims/lstrips blocks so both inline表达式 and多行模板 behave consistently.
    A small set of filters is pre-installed for convenience.
    """

    env = Environment(
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.globals.setdefault("now", datetime.now)
    env.filters.setdefault("tojson", lambda obj: json.dumps(obj, ensure_ascii=False))
    env.filters.setdefault("length", lambda obj: len(obj) if obj is not None else 0)
    env.filters.setdefault(
        "split",
        lambda value, sep=",": []
        if value is None or value == ""
        else str(value).split(sep),
    )

    def _format_date(value: Any, fmt: str = "yyyy-MM-dd") -> str:
        """A lightweight date filter compatible with the templated inputs we receive."""

        dt: datetime

        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, date_cls):
            dt = datetime.combine(value, datetime.min.time())
        elif isinstance(value, str) and value.strip().lower() in {"now", "today"}:
            dt = datetime.now()
        elif isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value)
            except Exception:
                dt = datetime.now()
        else:
            dt = datetime.now()

        # Support common "yyyy-MM-dd" style patterns by mapping to strftime tokens.
        format_str = (
            fmt.replace("yyyy", "%Y")
            .replace("MM", "%m")
            .replace("dd", "%d")
            .replace("HH", "%H")
            .replace("mm", "%M")
            .replace("ss", "%S")
        )
        return dt.strftime(format_str)

    env.filters.setdefault("date", _format_date)
    return env


_CONST_STR_TMPL = re.compile(r"^\s*\{\{\s*(['\"])(.*)\1\s*\}\}\s*$", re.DOTALL)
POTENTIAL_JINJA_EXPR = re.compile(
    r"\b[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z0-9_]+|\[[^\]]+\])+\b"
)


def looks_like_missing_jinja(value: str) -> bool:
    if "{{" in value or "{%" in value:
        return False
    return bool(POTENTIAL_JINJA_EXPR.search(value))


def has_unwrapped_variable(template: str) -> bool:
    if "{{" in template or "{%" in template:
        return False
    if not looks_like_missing_jinja(template):
        return False

    env = get_jinja_env()
    try:
        env.compile_expression(template)
    except TemplateError:
        return False
    return True


def render_jinja_string_constants(value: Any) -> Any:
    """Recursively render Jinja templates that are pure string literals.

    This collapses values like ``"{{ 'action' }}"`` into ``"action"`` so
    structural fields (e.g., ``type``) and params that are written as
    Jinja-wrapped literals are normalized before validation/execution.
    """

    def _render(val: Any) -> Any:
        if isinstance(val, str):
            match = _CONST_STR_TMPL.match(val)
            if match:
                return match.group(2)
            return val
        if isinstance(val, MappingABC):
            return {k: _render(v) for k, v in val.items()}
        if isinstance(val, list):
            return [_render(v) for v in val]
        return val

    return _render(value)


def validate_jinja_expr(expr: Any, *, path: str = "expression") -> None:
    """Validate that ``expr`` is a syntactically valid Jinja expression string."""

    if not isinstance(expr, str) or not expr.strip():
        raise ValueError(f"{path} 需要非空字符串表达式")

    env = get_jinja_env()
    try:
        parsed = env.parse(f"{{{{ {expr} }}}}")
    except TemplateError as exc:  # pragma: no cover - syntax errors are surfaced to callers
        raise ValueError(f"{path} 不是合法的 Jinja 表达式: {exc}") from exc

    def _iter_filter_nodes(node: nodes.Node) -> list[nodes.Filter]:
        found: list[nodes.Filter] = []
        if isinstance(node, nodes.Filter):
            found.append(node)
        for child in node.iter_child_nodes():
            found.extend(_iter_filter_nodes(child))
        return found

    for filter_node in _iter_filter_nodes(parsed):
        if filter_node.name not in env.filters:
            raise ValueError(f"{path} 使用了未注册的过滤器: {filter_node.name}")
        if filter_node.name == "map" and filter_node.args:
            first_arg = filter_node.args[0]
            if isinstance(first_arg, nodes.Const) and isinstance(first_arg.value, str):
                if first_arg.value not in env.filters:
                    raise ValueError(f"{path} 使用了未注册的过滤器: {first_arg.value}")


def eval_jinja_expr(expr: str, context: Mapping[str, Any]) -> Any:
    """Evaluate a Jinja expression string with the provided context."""

    env = get_jinja_env()
    compiled = env.compile_expression(expr)
    return compiled(**_prepare_context(context))


def render_jinja_template(template: str, context: Mapping[str, Any]) -> str:
    """Render an arbitrary Jinja template string with ``context``."""

    env = get_jinja_env()
    return env.from_string(template).render(_prepare_context(context))
class _AttrDict(dict):
    def __getattribute__(self, item: str) -> Any:  # pragma: no cover - small wrapper
        if item in ("__class__", "__iter__", "__len__", "__getitem__", "__setitem__"):
            return super().__getattribute__(item)
        if item in self:
            return self[item]
        return super().__getattribute__(item)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AttributeError(item) from exc


def _wrap_value(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return _AttrDict({k: _wrap_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap_value(v) for v in value]
    return value


def _prepare_context(context: Mapping[str, Any]) -> Mapping[str, Any]:
    prepared = dict(context)
    system_ctx = prepared.get("system") if isinstance(prepared.get("system"), Mapping) else {}
    if not system_ctx:
        system_ctx = {}
    if "date" not in system_ctx:
        system_ctx = dict(system_ctx)
        system_ctx["date"] = datetime.now().date().isoformat()
    prepared["system"] = system_ctx

    return {k: _wrap_value(v) for k, v in prepared.items()}
