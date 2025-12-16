"""Shared Jinja environment helpers used across planning and execution."""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Mapping

from collections.abc import Mapping as MappingABC

from jinja2 import Environment, StrictUndefined, TemplateError


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
    env.filters.setdefault("tojson", lambda obj: json.dumps(obj, ensure_ascii=False))
    return env


def validate_jinja_expr(expr: Any, *, path: str = "expression") -> None:
    """Validate that ``expr`` is a syntactically valid Jinja expression string."""

    if not isinstance(expr, str) or not expr.strip():
        raise ValueError(f"{path} 需要非空字符串表达式")

    env = get_jinja_env()
    try:
        env.parse(f"{{{{ {expr} }}}}")
    except TemplateError as exc:  # pragma: no cover - syntax errors are surfaced to callers
        raise ValueError(f"{path} 不是合法的 Jinja 表达式: {exc}") from exc


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
    return {k: _wrap_value(v) for k, v in context.items()}

