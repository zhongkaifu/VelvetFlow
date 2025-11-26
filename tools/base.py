"""Base definitions for simple callable tools."""
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


@dataclass
class Tool:
    """Metadata wrapper around a callable tool."""

    name: str
    description: str
    function: Callable[..., Any]
    args_schema: Optional[Mapping[str, Any]] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - thin wrapper
        return self.function(*args, **kwargs)


__all__ = ["Tool"]
