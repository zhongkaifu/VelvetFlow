"""Deterministic repair tools exposed to the LLM for tool-calling."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

from velvetflow.logging_utils import log_event, log_info, log_warn
from velvetflow.models import ValidationError, Workflow


def _workflow_fingerprint(workflow: Mapping[str, Any]) -> str:
    payload = json.dumps(workflow, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _schema_default_value(field_schema: Mapping[str, Any]) -> Any:
    if not isinstance(field_schema, Mapping):
        return None

    if "default" in field_schema:
        return field_schema.get("default")

    schema_type = field_schema.get("type")
    if isinstance(schema_type, list):
        schema_type = schema_type[0] if schema_type else None

    if schema_type == "string":
        return ""
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        return []
    if schema_type == "object":
        return {}

    enum_values = field_schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    return None


def _coerce_value_to_schema_type(value: Any, field_schema: Mapping[str, Any]) -> Optional[Any]:
    if not isinstance(field_schema, Mapping):
        return None

    schema_type = field_schema.get("type")
    if isinstance(schema_type, list):
        schema_type = schema_type[0] if schema_type else None

    if schema_type == "string" and not isinstance(value, str):
        return str(value)

    if schema_type == "integer":
        if isinstance(value, bool):
            return int(value)
        if not isinstance(value, int):
            try:
                return int(value)
            except Exception:  # noqa: BLE001
                return None

    if schema_type == "number":
        if isinstance(value, bool):
            return float(value)
        if not isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:  # noqa: BLE001
                return None

    if schema_type == "boolean" and not isinstance(value, bool):
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
        return bool(value)

    if schema_type == "array" and not isinstance(value, list):
        return [value]

    if schema_type == "object" and not isinstance(value, Mapping):
        return {"value": value}

    return None


def _index_actions_by_id(action_registry: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {a.get("action_id"): dict(a) for a in action_registry if a.get("action_id")}


def _collect_related_errors(
    validation_errors: Iterable[ValidationError], node_id: str | None
) -> List[Dict[str, Any]]:
    related: List[Dict[str, Any]] = []
    for err in validation_errors:
        if node_id and err.node_id and err.node_id != node_id:
            continue
        related.append({
            "code": err.code,
            "node_id": err.node_id,
            "field": err.field,
            "message": err.message,
        })
    return related


def _split_path(path: str) -> List[str | int]:
    parts: List[str | int] = []
    for segment in path.split("."):
        if not segment:
            continue
        while segment:
            if "[" in segment:
                before, remainder = segment.split("[", 1)
                if before:
                    parts.append(before)
                idx_str, after = remainder.split("]", 1)
                parts.append(int(idx_str))
                segment = after
                if segment.startswith("."):
                    segment = segment[1:]
            else:
                parts.append(segment)
                segment = ""
    return parts


def _set_nested_field(container: MutableMapping[str, Any], path: str, value: Any) -> None:
    segments = _split_path(path)
    current: Any = container
    for idx, seg in enumerate(segments):
        is_last = idx == len(segments) - 1
        if isinstance(seg, int):
            if not isinstance(current, list):
                raise ValueError(f"路径 {path} 需要列表，但当前为 {type(current).__name__}")
            while len(current) <= seg:
                current.append({})
            if is_last:
                current[seg] = value
            else:
                nxt = current[seg]
                if not isinstance(nxt, (dict, list)):
                    nxt = {}
                    current[seg] = nxt
                current = nxt
        else:
            if not isinstance(current, MutableMapping):
                raise ValueError(f"路径 {path} 需要对象，但当前为 {type(current).__name__}")
            if is_last:
                current[seg] = value
            else:
                nxt = current.get(seg)
                if not isinstance(nxt, (dict, list)):
                    nxt = [] if isinstance(segments[idx + 1], int) else {}
                    current[seg] = nxt
                current = nxt


def _apply_local_repairs_for_unknown_params(
    current_workflow: Workflow,
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
) -> Optional[Workflow]:
    """Remove parameters that are not defined in the action schema."""

    actions_by_id = _index_actions_by_id(action_registry)
    workflow_dict = current_workflow.model_dump(by_alias=True)
    nodes: List[Dict[str, Any]] = [
        n for n in workflow_dict.get("nodes", []) if isinstance(n, Mapping)
    ]
    nodes_by_id: Dict[str, Dict[str, Any]] = {n.get("id"): n for n in nodes}

    changed = False

    for err in validation_errors:
        if err.code != "UNKNOWN_PARAM" or not err.node_id or not err.field:
            continue

        node = nodes_by_id.get(err.node_id)
        if not node or node.get("type") != "action":
            continue

        action_id = node.get("action_id")
        action_def = actions_by_id.get(action_id)
        if not action_def:
            continue

        arg_schema = action_def.get("arg_schema") or {}
        additional_allowed = bool(arg_schema.get("additionalProperties")) if isinstance(arg_schema, Mapping) else False
        if additional_allowed:
            continue

        params = node.get("params")
        if not isinstance(params, dict):
            continue

        if err.field in params:
            params.pop(err.field, None)
            changed = True

    if not changed:
        return None

    try:
        return Workflow.model_validate(workflow_dict)
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] 移除未定义参数时验证失败：{exc}")
        return None


def repair_loop_body_references(
    workflow: Mapping[str, Any], node_id: str, *, prefer_first_node: bool = True
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    patched = copy.deepcopy(workflow)
    target = next(
        (n for n in patched.get("nodes", []) if isinstance(n, Mapping) and n.get("id") == node_id),
        None,
    )
    if not isinstance(target, Mapping):
        return patched, {"applied": False, "reason": f"节点 {node_id} 未找到"}

    params = target.get("params")
    if not isinstance(params, MutableMapping):
        return patched, {"applied": False, "reason": f"节点 {node_id} 缺少 params"}

    body = params.get("body_subgraph")
    if not isinstance(body, MutableMapping):
        return patched, {"applied": False, "reason": "body_subgraph 不是对象"}

    nodes: List[Mapping[str, Any]] = [
        bn for bn in body.get("nodes", []) if isinstance(bn, Mapping) and bn.get("id")
    ]
    node_ids = [bn.get("id") for bn in nodes]
    node_id_set = set(node_ids)
    changed = False

    if node_ids:
        if body.get("entry") not in node_id_set and prefer_first_node:
            body["entry"] = node_ids[0]
            changed = True
        if body.get("exit") not in node_id_set and prefer_first_node:
            body["exit"] = node_ids[-1]
            changed = True
    for key in ["entry", "exit"]:
        if body.get(key) not in node_id_set and key in body:
            body.pop(key)
            changed = True

    cleaned_edges = []
    for edge in body.get("edges", []) or []:
        if not isinstance(edge, Mapping):
            continue
        if edge.get("from") not in node_id_set or edge.get("to") not in node_id_set:
            changed = True
            continue
        cleaned_edges.append(edge)
    if cleaned_edges or "edges" in body:
        body["edges"] = cleaned_edges

    return patched, {"applied": changed, "reason": None if changed else "无可修复项"}


def fill_action_required_params(
    workflow: Mapping[str, Any], *, node_id: str, action_registry: Iterable[Mapping[str, Any]]
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    patched = copy.deepcopy(workflow)
    nodes = patched.get("nodes", [])
    target = next((n for n in nodes if isinstance(n, Mapping) and n.get("id") == node_id), None)
    if not isinstance(target, MutableMapping):
        return patched, {"applied": False, "reason": f"节点 {node_id} 未找到"}

    if target.get("type") != "action":
        return patched, {"applied": False, "reason": "仅支持 action 节点"}

    action_id = target.get("action_id")
    actions_by_id = _index_actions_by_id(action_registry)
    action_def = actions_by_id.get(action_id)
    if not action_def:
        return patched, {"applied": False, "reason": "action_id 未注册"}

    arg_schema = action_def.get("arg_schema") or {}
    required_fields = arg_schema.get("required") if isinstance(arg_schema, Mapping) else []
    properties = arg_schema.get("properties") if isinstance(arg_schema, Mapping) else {}
    if not isinstance(properties, Mapping):
        return patched, {"applied": False, "reason": "缺少 arg_schema.properties"}

    params = target.get("params") if isinstance(target.get("params"), MutableMapping) else {}
    if "params" not in target:
        target["params"] = params

    changed = False
    coerced_fields: List[str] = []
    filled_fields: List[str] = []

    for field in required_fields or []:
        schema = properties.get(field)
        if field not in params:
            params[field] = _schema_default_value(schema or {})
            changed = True
            filled_fields.append(field)
        else:
            coerced = _coerce_value_to_schema_type(params[field], schema or {})
            if coerced is not None and coerced != params[field]:
                params[field] = coerced
                changed = True
                coerced_fields.append(field)

    summary = {
        "applied": changed,
        "filled_fields": filled_fields,
        "coerced_fields": coerced_fields,
        "reason": None if changed else "未发现缺失字段",
    }
    return patched, summary


def update_node_field(
    workflow: Mapping[str, Any], *, node_id: str, field_path: str, value: Any
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    patched = copy.deepcopy(workflow)
    node = next(
        (n for n in patched.get("nodes", []) if isinstance(n, MutableMapping) and n.get("id") == node_id),
        None,
    )
    if not isinstance(node, MutableMapping):
        return patched, {"applied": False, "reason": f"节点 {node_id} 未找到"}

    try:
        _set_nested_field(node, field_path, value)
        return patched, {"applied": True, "reason": None}
    except Exception as exc:  # noqa: BLE001
        return patched, {"applied": False, "reason": str(exc)}


def apply_repair_tool(
    *,
    tool_name: str,
    args: Mapping[str, Any],
    workflow: Mapping[str, Any],
    validation_errors: Iterable[ValidationError],
    action_registry: Iterable[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    before_hash = _workflow_fingerprint(workflow)
    node_id = args.get("node_id") if isinstance(args, Mapping) else None

    if tool_name == "fix_loop_body_references":
        patched, summary = repair_loop_body_references(
            workflow, node_id=node_id, prefer_first_node=bool(args.get("prefer_first_node", True))
        )
    elif tool_name == "fill_action_required_params":
        patched, summary = fill_action_required_params(
            workflow, node_id=node_id or "", action_registry=action_registry
        )
    elif tool_name == "update_node_field":
        patched, summary = update_node_field(
            workflow,
            node_id=node_id or "",
            field_path=str(args.get("field_path", "")),
            value=args.get("value"),
        )
    else:
        patched, summary = workflow, {"applied": False, "reason": f"未知工具 {tool_name}"}

    after_hash = _workflow_fingerprint(patched)
    related_errors = _collect_related_errors(validation_errors, node_id)

    log_event(
        "repair_tool_call",
        {
            "tool": tool_name,
            "args": args,
            "node_id": node_id,
            "workflow_before": before_hash,
            "workflow_after": after_hash,
            "related_errors": related_errors,
            "summary": summary,
        },
    )

    if summary.get("applied"):
        log_info(
            f"[RepairTool] {tool_name} 已应用到节点 {node_id or '<unknown>'}，变更摘要: {summary}"
        )
    else:
        log_warn(f"[RepairTool] {tool_name} 未修改 workflow：{summary.get('reason')}")

    return patched, summary


REPAIR_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "fix_loop_body_references",
            "description": "清理 loop.body_subgraph 中指向缺失节点的 entry/exit/edges，可选用首尾节点兜底。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "loop 节点 ID"},
                    "prefer_first_node": {
                        "type": "boolean",
                        "description": "若 entry/exit 缺失，是否使用 body 节点的首/尾节点兜底。",
                        "default": True,
                    },
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fill_action_required_params",
            "description": "根据动作 arg_schema 自动填充必填字段的占位值，并尝试做基本类型矫正。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "action 节点 ID"},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_node_field",
            "description": "以可重复方式写入节点的字段值（支持 params.foo 或 params.items[0].field 形式的路径）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string", "description": "要修改的节点 ID"},
                    "field_path": {
                        "type": "string",
                        "description": "点/下标路径，例如 params.source.__from__ 或 params.items[0].field",
                    },
                    "value": {
                        "description": "要写入的 JSON 值，需与目标 schema 类型兼容。",
                    },
                },
                "required": ["node_id", "field_path", "value"],
            },
        },
    },
]


__all__ = [
    "apply_repair_tool",
    "fill_action_required_params",
    "_apply_local_repairs_for_unknown_params",
    "fix_loop_body_references",
    "update_node_field",
    "REPAIR_TOOLS",
]
