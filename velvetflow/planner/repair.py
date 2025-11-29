"""LLM-driven repair helpers used when validation fails."""

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence


from velvetflow.logging_utils import (
    log_error,
    log_warn,
)
from velvetflow.models import Node, PydanticValidationError, ValidationError, Workflow
from velvetflow.planner.action_guard import ensure_registered_actions


def _convert_pydantic_errors(
    workflow_raw: Any, error: PydanticValidationError
) -> List[ValidationError]:
    """Map Pydantic validation errors to generic ValidationError objects."""

    nodes = []
    if isinstance(workflow_raw, dict):
        nodes = workflow_raw.get("nodes") or []

    def _node_id_from_index(index: int) -> Optional[str]:
        if 0 <= index < len(nodes):
            node = nodes[index]
            if isinstance(node, dict):
                return node.get("id")
            if isinstance(node, Node):
                return node.id
        return None

    validation_errors: List[ValidationError] = []
    for err in error.errors():
        loc = err.get("loc", ()) or ()
        msg = err.get("msg", "")

        node_id: Optional[str] = None
        field: Optional[str] = None

        if loc:
            if loc[0] == "nodes" and len(loc) >= 2 and isinstance(loc[1], int):
                node_id = _node_id_from_index(loc[1])
                if len(loc) >= 3:
                    field = str(loc[2])
            elif loc[0] == "edges" and len(loc) >= 2 and isinstance(loc[1], int):
                if len(loc) >= 3 and isinstance(loc[-1], str):
                    field = str(loc[-1])
                else:
                    field = "edges"
            else:
                field = ".".join(str(part) for part in loc)

        validation_errors.append(
            ValidationError(
                code="INVALID_SCHEMA",
                node_id=node_id,
                field=field,
                message=msg,
            )
        )

    return validation_errors


def _make_failure_validation_error(message: str) -> ValidationError:
    return ValidationError(
        code="INVALID_SCHEMA", node_id=None, field=None, message=message
    )


def _summarize_validation_errors_for_llm(
    errors: Sequence[ValidationError],
    *,
    workflow: Mapping[str, Any] | None = None,
    action_registry: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Convert validation errors to an LLM-friendly, human-readable summary."""

    if not errors:
        return "未提供可用的错误信息。"

    lines: List[str] = []
    loop_hints: List[str] = []
    schema_hints: List[str] = []

    action_map: Dict[str, Mapping[str, Any]] = {}
    node_to_action: Dict[str, str] = {}
    if action_registry:
        action_map = {
            str(a.get("action_id")): a
            for a in action_registry
            if isinstance(a, Mapping) and a.get("action_id")
        }

    if workflow and isinstance(workflow, Mapping):
        for n in workflow.get("nodes", []) or []:
            if not isinstance(n, Mapping):
                continue
            node_id = n.get("id")
            action_id = n.get("action_id") if n.get("type") == "action" else None
            if isinstance(node_id, str) and isinstance(action_id, str):
                node_to_action[node_id] = action_id
    for idx, err in enumerate(errors, start=1):
        locations = []
        if err.node_id:
            locations.append(f"节点 {err.node_id}")
        if err.field:
            locations.append(f"字段 {err.field}")

        location = f"（{', '.join(locations)}）" if locations else ""
        lines.append(f"{idx}. [{err.code}]{location}：{err.message}")

        # 提供针对 loop 节点的额外修复提示，帮助模型快速定位核心问题
        if err.field and (
            "loop_kind" in err.field
            or err.field.startswith("exports.items")
            or err.field in {"source", "item_alias"}
        ):
            loop_hints.append(
                "- loop 节点需同时包含 params.loop_kind（for_each/while）、params.source、"
                "params.item_alias；exports.items.fields 只能包含 body_subgraph 或 source 元素 schema 中存在的字段"
            )

        if err.node_id and err.field:
            action_id = node_to_action.get(err.node_id)
            action_def = action_map.get(action_id) if action_id else None
            properties = (
                action_def.get("arg_schema", {}).get("properties")
                if isinstance(action_def, Mapping)
                else None
            )
            if isinstance(properties, Mapping) and err.field in properties:
                field_schema = properties.get(err.field)
                if isinstance(field_schema, Mapping):
                    expected_type = field_schema.get("type") or field_schema.get("anyOf")
                    schema_hints.append(
                        f"- 动作 {action_id} 的字段 {err.field} 期望类型/结构：{expected_type}，请按 schema 修正。"
                    )

    if schema_hints:
        lines.append("")
        lines.append("Schema 提示：")
        lines.extend(sorted(set(schema_hints)))

    if loop_hints:
        lines.append("")
        lines.append("Loop 修复提示：")
        lines.extend(sorted(set(loop_hints)))

    return "\n".join(lines)


def _safe_repair_invalid_loop_body(workflow_raw: Mapping[str, Any]) -> Workflow:
    """Best-effort patch for loop body graphs with missing nodes.

    当 loop.body_subgraph 的 edges/entry/exit 引用缺失节点时，校验会抛出
    异常并导致 AutoRepair 的回退逻辑失败。这里先补齐缺失节点（默认使用
    ``end`` 类型）后再做校验，确保至少可以返回结构化的 fallback workflow。
    """

    workflow_dict: Dict[str, Any] = dict(workflow_raw)
    nodes = workflow_dict.get("nodes") if isinstance(workflow_dict.get("nodes"), list) else []

    for node in nodes:
        if not isinstance(node, Mapping) or node.get("type") != "loop":
            continue

        params = node.get("params") if isinstance(node.get("params"), dict) else {}
        body = params.get("body_subgraph") if isinstance(params, Mapping) else None
        if not isinstance(body, dict):
            continue

        body_nodes = body.get("nodes") if isinstance(body.get("nodes"), list) else []
        body_ids = {bn.get("id") for bn in body_nodes if isinstance(bn, Mapping)}

        def _append_missing(node_id: str):
            if node_id in body_ids:
                return
            placeholder = {"id": node_id, "type": "end"}
            body_nodes.append(placeholder)
            body_ids.add(node_id)

        entry = body.get("entry")
        if isinstance(entry, str):
            _append_missing(entry)

        exit_node = body.get("exit")
        if isinstance(exit_node, str):
            _append_missing(exit_node)

        for edge in body.get("edges") or []:
            if not isinstance(edge, Mapping):
                continue
            frm = edge.get("from")
            to = edge.get("to")
            if isinstance(frm, str):
                _append_missing(frm)
            if isinstance(to, str):
                _append_missing(to)

        body["nodes"] = body_nodes
        params["body_subgraph"] = body
        node["params"] = params

    workflow_dict["nodes"] = nodes
    return Workflow.model_validate(workflow_dict)


def _repair_with_llm_and_fallback(
    *,
    broken_workflow: Dict[str, Any],
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
    search_service,
    reason: str,
) -> Workflow:
    log_warn(
        f"[AutoRepair] {reason}，repair_workflow_with_llm 已被移除，返回包含基本修复的回退版本。"
    )

    try:
        fallback = _safe_repair_invalid_loop_body(broken_workflow)
        fallback = ensure_registered_actions(
            fallback,
            action_registry=action_registry,
            search_service=search_service,
        )
        if not isinstance(fallback, Workflow):
            fallback = Workflow.model_validate(fallback)
        log_warn("[AutoRepair] 已跳过 LLM 修复，直接使用回退 workflow。")
        return fallback
    except Exception as err:  # noqa: BLE001
        log_error(
            f"[AutoRepair] 回退到基础 workflow 失败，将返回空的 fallback workflow：{err}"
        )
        return Workflow(workflow_name="fallback_workflow", nodes=[], declared_edges=[])


__all__ = [
    "_convert_pydantic_errors",
    "_make_failure_validation_error",
    "_repair_with_llm_and_fallback",
]
