# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Workflow-level validation entrypoints."""
from collections import deque
from typing import Any, Dict, List, Mapping, Optional

from velvetflow.models import ValidationError, Workflow
from velvetflow.loop_dsl import index_loop_body_nodes

from .binding_checks import (
    _check_output_path_against_schema,
    _collect_param_bindings,
    _index_actions_by_id,
    validate_param_binding,
)
from .error_handling import _RepairingErrorList
from .node_rules import (
    _filter_params_by_supported_fields,
    _strip_illegal_exports,
    _validate_nodes_recursive,
)


def precheck_loop_body_graphs(workflow_raw: Mapping[str, Any] | Any) -> List[ValidationError]:
    """Detect loop body graphs that refer to nonexistent nodes."""

    errors: List[ValidationError] = []

    if not isinstance(workflow_raw, Mapping):
        return errors

    nodes = workflow_raw.get("nodes")
    if not isinstance(nodes, list):
        return errors

    for node in nodes:
        if not isinstance(node, Mapping) or node.get("type") != "loop":
            continue

        loop_id = node.get("id")
        params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
        body = params.get("body_subgraph") if isinstance(params, Mapping) else None
        if not isinstance(body, Mapping):
            continue

        body_nodes = [bn for bn in body.get("nodes", []) or [] if isinstance(bn, Mapping)]
        if not body_nodes:
            errors.append(
                ValidationError(
                    code="INVALID_LOOP_BODY",
                    node_id=loop_id,
                    field="body_subgraph.nodes",
                    message=f"loop 节点 '{loop_id}' 的 body_subgraph.nodes 不能为空。",
                )
            )

    return errors


def _index_nodes_by_id(workflow: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in workflow.get("nodes", [])}


def run_lightweight_static_rules(
    workflow: Dict[str, Any], action_registry: List[Dict[str, Any]]
) -> List[ValidationError]:
    """Run compact static checks and compress issues into a single summary error."""

    messages: List[str] = []

    nodes_by_id = _index_nodes_by_id(workflow)
    node_ids = set(nodes_by_id.keys())
    actions_by_id = _index_actions_by_id(action_registry)
    loop_body_parents = index_loop_body_nodes(workflow)

    invalid_edges: List[str] = []
    for e in workflow.get("edges", []) or []:
        frm = e.get("from")
        to = e.get("to")
        if frm not in node_ids or to not in node_ids:
            invalid_edges.append(f"{frm}->{to}")
    if invalid_edges:
        messages.append(
            "无效边引用: " + ", ".join(sorted(set(invalid_edges)))
        )

    binding_issues: List[str] = []
    for node in workflow.get("nodes", []) or []:
        nid = node.get("id")
        params = node.get("params")
        bindings = _collect_param_bindings(params) if isinstance(params, Mapping) else []
        for binding in bindings:
            err = _check_output_path_against_schema(
                binding.get("source"), nodes_by_id, actions_by_id, loop_body_parents
            )
            if err:
                binding_issues.append(f"{nid}:{binding.get('path')} -> {err}")
    if binding_issues:
        messages.append("参数绑定引用无效: " + "; ".join(binding_issues[:10]))

    if not messages:
        return []

    summary = "；".join(messages)
    return [
        ValidationError(
            code="STATIC_RULES_SUMMARY",
            node_id=None,
            field=None,
            message=summary,
        )
    ]


def validate_completed_workflow(
    workflow: Dict[str, Any],
    action_registry: List[Dict[str, Any]],
) -> List[ValidationError]:
    errors: List[ValidationError] = _RepairingErrorList(workflow)

    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])

    nodes_by_id = _index_nodes_by_id(workflow)
    loop_body_parents = index_loop_body_nodes(workflow)
    node_ids = set(nodes_by_id.keys())
    actions_by_id = _index_actions_by_id(action_registry)

    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        errors.set_context({"node": node})
        if _strip_illegal_exports(node):
            errors.append(
                ValidationError(
                    code="INVALID_SCHEMA",
                    node_id=node.get("id"),
                    field="exports",
                    message=(
                        "检测到非 loop 节点携带 exports，已删除该字段，请使用工具在合法位置重新生成 exports。"
                    ),
                )
            )
        removed_fields = _filter_params_by_supported_fields(
            node=node, actions_by_id=actions_by_id
        )
        for field in removed_fields:
            errors.append(
                ValidationError(
                    code="UNKNOWN_PARAM",
                    node_id=node.get("id"),
                    field=field,
                    message="节点 params 包含不支持的字段，已在校验阶段移除。",
                )
            )
        errors.set_context(None)

    # ---------- edges 校验 ----------
    for e in edges:
        frm = e.get("from")
        to = e.get("to")
        errors.set_context({"edge": e})
        if frm not in node_ids:
            errors.append(
                ValidationError(
                    code="INVALID_EDGE",
                    node_id=frm,
                    field="from",
                    message=f"Edge from '{frm}' -> '{to}' 中，from 节点不存在。",
                )
            )
        if to not in node_ids:
            errors.append(
                ValidationError(
                    code="INVALID_EDGE",
                    node_id=to,
                    field="to",
                    message=f"Edge from '{frm}' -> '{to}' 中，to 节点不存在。",
                )
            )
        errors.set_context(None)

    # ---------- 图连通性校验 ----------
    start_nodes = [n["id"] for n in nodes if n.get("type") == "start"]
    reachable: set = set()
    if nodes:
        adj: Dict[str, List[str]] = {}
        to_ids: set = set()
        for e in edges:
            frm = e.get("from")
            to = e.get("to")
            if frm in node_ids and to in node_ids:
                adj.setdefault(frm, []).append(to)
                to_ids.add(to)

        # Treat nodes without inbound references as additional roots so workflows
        # that rely solely on parameter bindings remain connected.
        inbound_free = [nid for nid in node_ids if nid not in to_ids]
        candidate_roots = list(dict.fromkeys((start_nodes or []) + inbound_free))

        dq = deque(candidate_roots)
        while dq:
            nid = dq.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for nxt in adj.get(nid, []):
                if nxt not in reachable:
                    dq.append(nxt)

        for nid in node_ids - reachable:
            errors.set_context({"node": nodes_by_id.get(nid, {})})
            errors.append(
                ValidationError(
                    code="DISCONNECTED_GRAPH",
                    node_id=nid,
                    field=None,
                    message=f"节点 '{nid}' 无法从 start 节点到达。",
                )
            )
            errors.set_context(None)

    # ---------- 节点校验（含 loop body） ----------
    _validate_nodes_recursive(nodes, nodes_by_id, actions_by_id, loop_body_parents, errors)

    return errors


def validate_param_binding_and_schema(
    binding: Mapping[str, Any], workflow: Dict[str, Any], action_registry: List[Dict[str, Any]]
):
    """Validate parameter bindings in the context of workflow/action schemas."""

    nodes_by_id = _index_nodes_by_id(workflow)
    loop_body_parents = index_loop_body_nodes(workflow)
    actions_by_id = _index_actions_by_id(action_registry)
    binding_err = validate_param_binding(binding)
    if binding_err:
        return binding_err

    src = binding.get("__from__")
    if isinstance(src, str):
        return _check_output_path_against_schema(src, nodes_by_id, actions_by_id, loop_body_parents)
    if isinstance(src, list):
        for item in src:
            if not isinstance(item, str):
                return "__from__ 数组元素必须是字符串"
            err = _check_output_path_against_schema(
                item, nodes_by_id, actions_by_id, loop_body_parents
            )
            if err:
                return err
    return None


__all__ = [
    "precheck_loop_body_graphs",
    "run_lightweight_static_rules",
    "validate_completed_workflow",
    "validate_param_binding_and_schema",
]
