"""Lightweight Jinja2 shim that prefers the real library when available.

This module first attempts to load the installed ``jinja2`` package from
``site-packages``. If it exists, the real implementation (and its full filter
support) is re-exported so expressions like ``| length`` behave correctly. If
the package is unavailable, it falls back to a minimal offline subset.
"""
from __future__ import annotations

import os
import re
import sys
import importlib.util
import importlib.machinery
from typing import Any, Callable, Dict, Optional

# Try to reuse the real jinja2 implementation if it is installed (preferred).
_CURRENT_FILE = os.path.abspath(__file__)
_real_module = None
for _path in sys.path:
    candidate = os.path.join(_path or ".", "jinja2", "__init__.py")
    if not os.path.isfile(candidate):
        continue
    if os.path.abspath(candidate) == _CURRENT_FILE:
        continue
    spec = importlib.util.spec_from_file_location(
        "_velvetflow_real_jinja2",
        candidate,
        submodule_search_locations=[os.path.dirname(candidate)],
    )
    if spec and spec.loader:
        _real_module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = _real_module
        spec.loader.exec_module(_real_module)
        break

if _real_module:
    # Expose the real implementation as the active ``jinja2`` package so
    # its submodules (e.g., ``jinja2.runtime``) resolve correctly. We load the
    # module under a temporary name to avoid recursive imports, then alias it
    # back to ``jinja2`` and mirror its ``__path__`` so relative imports work
    # during template compilation.
    sys.modules[__name__] = _real_module
    sys.modules.setdefault("jinja2", _real_module)
    __path__ = getattr(_real_module, "__path__", [])

    Environment = _real_module.Environment
    StrictUndefined = _real_module.StrictUndefined
    TemplateError = _real_module.TemplateError
    Template = _real_module.Template
    __all__ = ["Environment", "StrictUndefined", "TemplateError", "Template"]
else:
    __all__ = ["Environment", "StrictUndefined", "TemplateError", "Template"]

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
                    translated = expr
                    # Translate ``| default(x)`` into a Python coalesce expression.
                    translated = re.sub(
                        r"(?P<target>[^|]+?)\|\s*default\((?P<arg>[^)]*)\)",
                        lambda m: f"({m.group('target').strip()} if {m.group('target').strip()} not in (None, '') else {m.group('arg').strip() or 'None'})",
                        translated,
                    )
                    # Translate the limited ``| length`` filter into a Python ``len`` call
                    # so common Jinja expressions keep working without the full parser.
                    translated = re.sub(
                        r"(?P<target>[^|]+?)\|\s*length\b",
                        lambda m: f"len({m.group('target').strip()})",
                        translated,
                    )
                    return eval(translated, {"__builtins__": {}, "len": len}, context)
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

            def _eval_if_blocks(text: str) -> str:
                pattern = re.compile(
                    r"\{%\s*if\s+(.*?)\s*%\}(.*?)((?:\{%\s*else\s*%\}(.*?))?)\{%\s*endif\s*%\}",
                    re.S,
                )

                def _replace(match: re.Match[str]) -> str:
                    expr = match.group(1).strip()
                    truthy_block = match.group(2) or ""
                    else_block = match.group(4) or ""
                    compiled = self.env.compile_expression(expr)
                    try:
                        cond_value = compiled(**context)
                    except Exception as exc:
                        raise TemplateError(f"条件表达式执行失败: {exc}") from exc

                    chosen = truthy_block if cond_value else else_block
                    # 支持嵌套 if：递归处理选中的文本
                    return _eval_if_blocks(chosen)

                prev = None
                cur = text
                while prev != cur:
                    prev = cur
                    cur = pattern.sub(_replace, cur)
                return cur

            def _replace(match: re.Match[str]) -> str:
                expr = match.group(1).strip()
                compiled = self.env.compile_expression(expr)
                result = compiled(**context)
                return "" if result is None else str(result)

            try:
                rendered = _eval_if_blocks(self.template)
                return re.sub(r"\{\{\s*(.*?)\s*\}\}", _replace, rendered)
            except TemplateError:
                raise
            except Exception as exc:  # pragma: no cover - fallback
                raise TemplateError(f"模板渲染失败: {exc}") from exc
