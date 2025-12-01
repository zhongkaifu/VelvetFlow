"""Error handling helpers for validation routines."""
from contextlib import contextmanager
from typing import Any, Mapping, Optional

from velvetflow.models import ValidationError
from velvetflow.reference_utils import parse_field_path


class _RepairingErrorList(list):
    """Record validation errors while stripping offending fields in-place."""

    def __init__(self, workflow: Mapping[str, Any]):
        super().__init__()
        self._context: Optional[Mapping[str, Any]] = None

    @contextmanager
    def contextualize(self, ctx: Mapping[str, Any]):
        prev = self._context
        self._context = ctx
        try:
            yield
        finally:
            self._context = prev

    def set_context(self, ctx: Optional[Mapping[str, Any]]):
        self._context = ctx

    def append(self, error: ValidationError):  # type: ignore[override]
        self._repair(error)
        return super().append(error)

    def extend(self, values):  # type: ignore[override]
        for val in values:
            self.append(val)

    def _repair(self, error: ValidationError) -> None:
        ctx = self._context or {}
        if not isinstance(ctx, Mapping):
            return

        field = getattr(error, "field", None)
        if not field:
            return

        target = ctx.get("node") or ctx.get("edge")
        if isinstance(target, Mapping):
            self._remove_field(target, field)

    def _remove_field(self, container: Mapping[str, Any], field: str) -> None:
        try:
            parts = parse_field_path(field)
        except Exception:
            return

        if not parts:
            return

        current: Any = container
        for part in parts[:-1]:
            if isinstance(part, int):
                if isinstance(current, list) and 0 <= part < len(current):
                    current = current[part]
                else:
                    return
            else:
                if isinstance(current, Mapping):
                    current = current.get(part)
                else:
                    return

        last = parts[-1]
        if isinstance(last, int):
            if isinstance(current, list) and 0 <= last < len(current):
                current.pop(last)
        elif isinstance(current, dict):
            current.pop(last, None)


__all__ = ["_RepairingErrorList"]
