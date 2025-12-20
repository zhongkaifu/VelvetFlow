"""Template rendering helpers used by the executor."""
from __future__ import annotations

from typing import Any, Mapping
import re

from jinja2 import TemplateError

from velvetflow.jinja_utils import eval_jinja_expr, render_jinja_template
from velvetflow.reference_utils import canonicalize_template_placeholders


class TemplateRendererMixin:
    def _render_template(self, value: Any, params: Mapping[str, Any]) -> Any:
        if isinstance(value, str):
            normalized_value = canonicalize_template_placeholders(value)

            # If the template is a pure expression (no surrounding text), evaluate it to
            # preserve native types (list/dict/bool) instead of always coercing to str.
            expr_match = re.match(r"^\{\{\s*(.+?)\s*\}\}$", normalized_value, re.DOTALL)
            if expr_match:
                expr_body = expr_match.group(1)
                try:
                    return eval_jinja_expr(expr_body, params)
                except TemplateError:
                    # fall back to string rendering below
                    pass

            try:
                return render_jinja_template(normalized_value, params)
            except TemplateError:
                return value

        if isinstance(value, list):
            return [self._render_template(v, params) for v in value]

        if isinstance(value, dict):
            return {k: self._render_template(v, params) for k, v in value.items()}

        return value
