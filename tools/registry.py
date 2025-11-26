"""Simple in-memory registry for callable tools."""
from typing import Any, Dict, Iterable, Optional

from tools.base import Tool


class ToolRegistry:
    """Register and retrieve business tools by name."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        key = tool.name
        self._tools[key] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> Iterable[str]:
        return self._tools.keys()

    def all(self) -> Iterable[Tool]:
        return self._tools.values()

    def call(self, name: str, **kwargs: Any) -> Any:
        tool = self.get(name)
        if not tool:
            raise KeyError(f"tool '{name}' is not registered")
        return tool(**kwargs)


GLOBAL_TOOL_REGISTRY = ToolRegistry()


def register_tool(tool: Tool) -> None:
    """Convenience wrapper to register a tool globally."""

    GLOBAL_TOOL_REGISTRY.register(tool)


def get_registered_tool(name: str) -> Optional[Tool]:
    return GLOBAL_TOOL_REGISTRY.get(name)


def list_registered_tools() -> Dict[str, Tool]:
    return dict(GLOBAL_TOOL_REGISTRY._tools)


def call_registered_tool(name: str, **kwargs: Any) -> Any:
    return GLOBAL_TOOL_REGISTRY.call(name, **kwargs)


__all__ = [
    "ToolRegistry",
    "GLOBAL_TOOL_REGISTRY",
    "register_tool",
    "get_registered_tool",
    "list_registered_tools",
    "call_registered_tool",
]
