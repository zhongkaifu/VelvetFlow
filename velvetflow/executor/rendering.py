"""Template rendering helpers used by the executor."""
from __future__ import annotations

import re
from typing import Any, Mapping

from velvetflow.reference_utils import canonicalize_template_placeholders


class TemplateRendererMixin:
    def _render_template(self, value: Any, params: Mapping[str, Any]) -> Any:
        if isinstance(value, str):
            normalized_value = canonicalize_template_placeholders(value)
            pattern = r"\{\{\s*([^{}]+)\s*\}\}"

            def _replace(match: re.Match[str]) -> str:
                key = match.group(1)
                return str(params.get(key, match.group(0)))

            return re.sub(pattern, _replace, normalized_value)

        if isinstance(value, list):
            return [self._render_template(v, params) for v in value]

        if isinstance(value, dict):
            return {k: self._render_template(v, params) for k, v in value.items()}

        return value
