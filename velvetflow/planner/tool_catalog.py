"""Utility helpers to share LLM tool definitions across planner stages."""

from typing import Any, Dict, Iterable, List, Mapping, Set


def _extract_tool_name(tool: Mapping[str, Any]) -> str | None:
    if not isinstance(tool, Mapping):
        return None
    func = tool.get("function")
    if not isinstance(func, Mapping):
        return None
    name = func.get("name")
    return str(name) if isinstance(name, str) else None


def _merge_tools(tool_lists: Iterable[Iterable[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for tools in tool_lists:
        for tool in tools or []:
            name = _extract_tool_name(tool)
            if name and name in seen:
                continue

            merged.append(dict(tool))
            if name:
                seen.add(name)

    return merged


def get_llm_toolbox(
    *, include_planner: bool = True, include_repair: bool = True
) -> List[Dict[str, Any]]:
    """Return the merged tool specs for planner + repair flows."""

    tool_lists = []
    if include_planner:
        from velvetflow.planner.tools import PLANNER_TOOLS

        tool_lists.append(PLANNER_TOOLS)
    if include_repair:
        from velvetflow.planner.repair_tools import REPAIR_TOOLS

        tool_lists.append(REPAIR_TOOLS)

    return _merge_tools(tool_lists)


def list_tool_names(
    *, include_planner: bool = True, include_repair: bool = True
) -> Set[str]:
    """Return the tool names available to the LLM."""

    tool_lists = []
    if include_planner:
        from velvetflow.planner.tools import PLANNER_TOOLS

        tool_lists.append(PLANNER_TOOLS)
    if include_repair:
        from velvetflow.planner.repair_tools import REPAIR_TOOLS

        tool_lists.append(REPAIR_TOOLS)

    names: Set[str] = set()
    for tool in tool_lists:
        for name in (_extract_tool_name(t) for t in tool):
            if name:
                names.add(name)
    return names


__all__ = ["get_llm_toolbox", "list_tool_names"]
