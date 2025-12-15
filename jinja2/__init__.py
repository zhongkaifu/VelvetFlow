"""Lightweight local Jinja2 subset for offline environments."""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Optional

__all__ = ["Environment", "StrictUndefined", "TemplateError"]


class TemplateError(Exception):
    pass


class StrictUndefined:
    def __init__(self, name: str = "") -> None:
        self.name = name

    def _raise(self) -> None:
        raise TemplateError(f"未定义的变量: {self.name}")

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - defensive
        self._raise()

    def __str__(self) -> str:  # pragma: no cover - defensive
        self._raise()


class Environment:
    def __init__(self, *, autoescape: bool = False, undefined: Any = StrictUndefined, **_: Any) -> None:
        self.autoescape = autoescape
        self.undefined = undefined
        self.filters: Dict[str, Callable[[Any], Any]] = {}

    def parse(self, template: str) -> str:
        self.from_string(template)
        return template

    def compile_expression(self, expr: str) -> Callable[..., Any]:
        def _compiled(**context: Any) -> Any:
            try:
                return eval(expr, {"__builtins__": {}}, context)
            except NameError as exc:  # pragma: no cover - rare
                raise TemplateError(f"表达式缺少变量: {exc}") from exc
            except Exception as exc:  # pragma: no cover - fallback
                raise TemplateError(f"表达式执行失败: {exc}") from exc

        return _compiled

    def from_string(self, template: str) -> "Template":
        return Template(template, self)

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - compatibility
        raise AttributeError(item)


class Template:
    def __init__(self, template: str, env: Environment) -> None:
        self.template = template
        self.env = env

    def render(self, *args: Any, **kwargs: Any) -> str:
        context: Dict[str, Any] = {}
        if args:
            maybe_mapping = args[0]
            if isinstance(maybe_mapping, dict):
                context.update(maybe_mapping)
        context.update(kwargs)

        def _replace(match: re.Match[str]) -> str:
            expr = match.group(1).strip()
            compiled = self.env.compile_expression(expr)
            result = compiled(**context)
            return "" if result is None else str(result)

        try:
            return re.sub(r"\{\{\s*(.*?)\s*\}\}", _replace, self.template)
        except TemplateError:
            raise
        except Exception as exc:  # pragma: no cover - fallback
            raise TemplateError(f"模板渲染失败: {exc}") from exc
