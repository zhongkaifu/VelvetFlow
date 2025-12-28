"""Utilities for stripping legacy binding objects in favor of pure Jinja strings."""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from velvetflow.bindings import normalize_reference_path


class LegacyBindingError(ValueError):
    """Raised when legacy binding objects are encountered after deprecation."""


def strip_legacy_bindings_to_jinja(obj: Any, path: str = "params") -> Any:
    """Recursively rewrite legacy ``__from__`` bindings into Jinja strings.

    The system now exclusively supports Jinja expressions. Any legacy binding
    object that uses ``__from__`` will be converted into a string expression of
    the form ``"{{ result_of.node.field }}"`` so downstream code never needs to
    handle mapping-based bindings. ``__agg__`` is no longer supported and will
    raise ``LegacyBindingError`` so callers can surface a clear migration error.
    """

    if isinstance(obj, Mapping):
        # Bail out early if the legacy aggregation syntax sneaks in.
        if "__agg__" in obj:
            raise LegacyBindingError(
                f"{path} 使用了已废弃的 __agg__ 聚合语法，请改写为 Jinja 表达式"
            )
        if "__from__" in obj:
            source = obj.get("__from__")
            if not isinstance(source, str):
                raise LegacyBindingError(
                    f"{path} 包含非法的 legacy 绑定对象，__from__ 必须是字符串"
                )
            normalized = normalize_reference_path(source)
            return f"{{{{ {normalized} }}}}"

        new_obj: MutableMapping[str, Any] = (
            obj if isinstance(obj, MutableMapping) else dict(obj)
        )
        for key, value in obj.items():
            child_path = f"{path}.{key}" if path else str(key)
            new_obj[key] = strip_legacy_bindings_to_jinja(value, child_path)
        return new_obj

    if isinstance(obj, list):
        return [strip_legacy_bindings_to_jinja(item, f"{path}[{idx}]") for idx, item in enumerate(obj)]

    return obj
