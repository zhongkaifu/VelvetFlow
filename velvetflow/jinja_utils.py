"""Shared Jinja environment helpers used across planning and execution."""
from __future__ import annotations

import json
import re
from datetime import datetime, date as date_cls
from functools import lru_cache
import operator
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
    env.filters.setdefault(
        "tojson",
        lambda obj: json.dumps(obj, ensure_ascii=False, default=str),
    )
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
    env.tests.setdefault("truthy", lambda value: bool(value))
    return env


_CONST_STR_TMPL = re.compile(r"^\s*\{\{\s*(['\"])(.*)\1\s*\}\}\s*$", re.DOTALL)
_NUMERIC_INT_RE = re.compile(r"^-?\d+$")
_NUMERIC_FLOAT_RE = re.compile(r"^-?\d+\.\d+$")


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
        if filter_node.name in {"select", "reject"} and filter_node.args:
            test_arg = filter_node.args[0]
            if isinstance(test_arg, nodes.Const) and isinstance(test_arg.value, str):
                if test_arg.value not in env.tests:
                    raise ValueError(f"{path} 使用了未注册的测试: {test_arg.value}")
        if filter_node.name in {"selectattr", "rejectattr"} and len(filter_node.args) >= 2:
            test_arg = filter_node.args[1]
            if isinstance(test_arg, nodes.Const) and isinstance(test_arg.value, str):
                if test_arg.value not in env.tests:
                    raise ValueError(f"{path} 使用了未注册的测试: {test_arg.value}")


def eval_jinja_expr(expr: str, context: Mapping[str, Any]) -> Any:
    """Evaluate a Jinja expression string with the provided context."""

    env = get_jinja_env()
    compiled = env.compile_expression(expr)
    return _unwrap_value(compiled(**_prepare_context(context)))


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


class _AutoValue:
    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def _unwrap(self, other: Any) -> Any:
        return other.value if isinstance(other, _AutoValue) else other

    def _coerce_numeric(self, value: Any) -> int | float | None:
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if _NUMERIC_INT_RE.match(stripped):
                return int(stripped)
            if _NUMERIC_FLOAT_RE.match(stripped):
                return float(stripped)
        return None

    def _compare(self, other: Any, op: Any) -> bool:
        left = self.value
        right = self._unwrap(other)
        left_num = self._coerce_numeric(left)
        right_num = self._coerce_numeric(right)
        if left_num is not None and right_num is not None:
            return op(left_num, right_num)
        try:
            return op(left, right)
        except TypeError:
            return op(str(left), str(right))

    def __eq__(self, other: Any) -> bool:
        return self._compare(other, operator.eq)

    def __ne__(self, other: Any) -> bool:
        return self._compare(other, operator.ne)

    def __lt__(self, other: Any) -> bool:
        return self._compare(other, operator.lt)

    def __le__(self, other: Any) -> bool:
        return self._compare(other, operator.le)

    def __gt__(self, other: Any) -> bool:
        return self._compare(other, operator.gt)

    def __ge__(self, other: Any) -> bool:
        return self._compare(other, operator.ge)

    def __bool__(self) -> bool:
        return bool(self.value)

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return repr(self.value)

    def __len__(self) -> int:
        return len(self.value)

    def __iter__(self) -> Any:
        return iter(self.value)

    def __getitem__(self, item: Any) -> Any:
        return self.value[item]

    def __getattr__(self, item: str) -> Any:
        return getattr(self.value, item)


def _wrap_value(value: Any) -> Any:
    if isinstance(value, MappingABC):
        return _AttrDict({k: _wrap_value(v) for k, v in value.items()})
    if isinstance(value, list):
        return [_wrap_value(v) for v in value]
    if isinstance(value, str):
        return _AutoValue(value)
    return value


def _unwrap_value(value: Any) -> Any:
    if isinstance(value, _AutoValue):
        return value.value
    if isinstance(value, list):
        return [_unwrap_value(v) for v in value]
    if isinstance(value, MappingABC):
        return {k: _unwrap_value(v) for k, v in value.items()}
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
