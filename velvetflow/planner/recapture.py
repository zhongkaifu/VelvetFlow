"""LLM-callable recapture utility to summarize workflow progress.

This helper is intentionally deterministic and heuristic: it highlights
apparent coverage/supporting nodes, parameter gaps, and dangling references
without replacing the stricter LLM coverage check that runs after
``finalize_workflow``. Treat it as a quick progress recap rather than a
definitive coverage verdict.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence

from velvetflow.loop_dsl import iter_workflow_and_loop_body_nodes
from velvetflow.reference_utils import normalize_reference_path


def _split_requirement(requirement: str) -> List[str]:
    if not isinstance(requirement, str):
        return []

    parts = re.split(r"[。！？!?；;\n,，、]", requirement)
    seen: set[str] = set()
    tasks: List[str] = []
    for part in parts:
        cleaned = part.strip()
        if len(cleaned) < 2:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        tasks.append(cleaned)
    return tasks


def _is_empty_value(value: Any) -> bool:
    if value is None:
        return True
    if value == "":
        return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def _collect_bindings(obj: Any, *, path_prefix: str = "params") -> List[Tuple[str, str]]:
    bindings: List[Tuple[str, str]] = []

    if isinstance(obj, Mapping):
        if "__from__" in obj and isinstance(obj.get("__from__"), str):
            bindings.append((path_prefix, normalize_reference_path(obj["__from__"])))
        for key, value in obj.items():
            bindings.extend(_collect_bindings(value, path_prefix=f"{path_prefix}.{key}"))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            bindings.extend(_collect_bindings(item, path_prefix=f"{path_prefix}[{idx}]"))

    return bindings


def _collect_empty_fields(obj: Any, *, path_prefix: str = "params") -> List[str]:
    empties: List[str] = []

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child_prefix = f"{path_prefix}.{key}"
            if _is_empty_value(value):
                empties.append(child_prefix)
            else:
                empties.extend(_collect_empty_fields(value, path_prefix=child_prefix))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            child_prefix = f"{path_prefix}[{idx}]"
            if _is_empty_value(item):
                empties.append(child_prefix)
            else:
                empties.extend(_collect_empty_fields(item, path_prefix=child_prefix))

    return empties


def _build_action_schema_map(action_registry: Iterable[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    schemas: Dict[str, Mapping[str, Any]] = {}
    for action in action_registry:
        action_id = action.get("action_id") if isinstance(action, Mapping) else None
        if not action_id:
            continue
        schemas[action_id] = action.get("arg_schema") if isinstance(action.get("arg_schema"), Mapping) else {}
    return schemas


def recapture_workflow_progress(
    *,
    workflow: Mapping[str, Any],
    nl_requirement: str,
    action_registry: Sequence[Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Summarize completed and pending tasks for the current workflow draft."""

    tasks = _split_requirement(nl_requirement)
    node_ids = {n.get("id") for n in iter_workflow_and_loop_body_nodes(workflow)}
    action_schemas = _build_action_schema_map(action_registry or [])

    node_progress: List[Dict[str, Any]] = []
    dangling_references: List[str] = []

    for node in iter_workflow_and_loop_body_nodes(workflow):
        node_id = node.get("id") if isinstance(node, Mapping) else None
        node_type = node.get("type") if isinstance(node, Mapping) else None
        params = node.get("params") if isinstance(node, MutableMapping) else None
        action_id = node.get("action_id") if isinstance(node, Mapping) else None

        bindings = _collect_bindings(params or {}, path_prefix="params")
        missing_sources: List[str] = []
        for path, ref in bindings:
            if isinstance(ref, str) and ref.startswith("result_of."):
                parts = ref.split(".")
                if len(parts) >= 2 and parts[1] not in node_ids:
                    missing_sources.append(f"{path} -> {ref}")
                    dangling_references.append(ref)

        arg_schema = action_schemas.get(action_id or "", {})
        required_fields = arg_schema.get("required") if isinstance(arg_schema, Mapping) else None
        missing_required: List[str] = []
        if isinstance(required_fields, list):
            for field in required_fields:
                if not isinstance(field, str):
                    continue
                if not isinstance(params, Mapping) or field not in params or _is_empty_value(params.get(field)):
                    missing_required.append(field)

        empties = _collect_empty_fields(params or {}, path_prefix="params")
        node_progress.append(
            {
                "id": node_id,
                "type": node_type,
                "action_id": action_id,
                "display_name": node.get("display_name") if isinstance(node, Mapping) else None,
                "params_filled": not missing_required and not empties,
                "missing_required_params": missing_required or None,
                "empty_fields": empties or None,
                "missing_references": missing_sources or None,
            }
        )

    coverage: List[Dict[str, Any]] = []
    for task in tasks:
        supporting_nodes: List[str] = []
        normalized_task = task.lower()
        for node in iter_workflow_and_loop_body_nodes(workflow):
            parts: List[str] = []
            if isinstance(node.get("display_name"), str):
                parts.append(node["display_name"].lower())
            if isinstance(node.get("action_id"), str):
                parts.append(str(node["action_id"]).lower())
            label = " ".join(parts)
            if normalized_task and all(token in label for token in normalized_task.split() if token):
                supporting_nodes.append(str(node.get("id")))
            elif normalized_task and normalized_task in label:
                supporting_nodes.append(str(node.get("id")))

        coverage.append({"task": task, "supporting_nodes": supporting_nodes or None})

    uncovered_tasks = [c for c in coverage if not c.get("supporting_nodes")]

    return {
        "requirement_tasks": tasks,
        "coverage": coverage,
        "uncovered_tasks": uncovered_tasks,
        "node_progress": node_progress,
        "dangling_references": sorted(set(dangling_references)),
    }


RECAPTURE_TOOL = {
    "type": "function",
    "function": {
        "name": "recapture_workflow_progress",
        "description": (
            "回顾当前 workflow 和 nl_requirement 的匹配度，输出已覆盖/缺失子任务、节点参数缺口和依赖引用状态；"
            "这是一个确定性复盘工具，仅供快速回顾，不等价于 finalize_workflow 之后的覆盖度检查。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "focus": {
                    "type": "string",
                    "description": "可选说明关注点，便于日志标记。",
                }
            },
            "additionalProperties": False,
        },
    },
}


__all__ = ["recapture_workflow_progress", "RECAPTURE_TOOL"]
