# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Workflow-level validation entrypoints."""
from collections import deque
from typing import Any, Dict, List, Mapping, Optional

from velvetflow.models import (
    ValidationError,
    Workflow,
    infer_depends_on_from_edges,
    infer_edges_from_bindings,
)
from velvetflow.loop_dsl import index_loop_body_nodes, loop_body_has_action

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
            errors.append(
                ValidationError(
                    code="INVALID_LOOP_BODY",
                    node_id=loop_id,
                    field="body_subgraph",
                    message=f"loop 节点 '{loop_id}' 必须提供 body_subgraph。",
                )
            )
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
            continue

        seen_ids: set[str] = set()
        for nested in body_nodes:
            nid = nested.get("id") if isinstance(nested.get("id"), str) else None
            if not nid:
                continue
            if nid in seen_ids:
                errors.append(
                    ValidationError(
                        code="INVALID_LOOP_BODY",
                        node_id=loop_id,
                        field="body_subgraph.nodes",
                        message=(
                            f"loop 节点 '{loop_id}' 的 body_subgraph.nodes 中存在重复的节点 id: {nid}"
                        ),
                    )
                )
                break
            seen_ids.add(nid)
        if not loop_body_has_action(body):
            errors.append(
                ValidationError(
                    code="INVALID_LOOP_BODY",
                    node_id=loop_id,
                    field="body_subgraph.nodes",
                    message=(
                        "loop 节点的 body_subgraph 至少需要一个 action 节点，"
                        "请使用规划/修复工具为循环体补充可执行步骤。"
                    ),
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

    inferred_edges = infer_edges_from_bindings(workflow.get("nodes") or [])
    binding_issues: List[str] = []
    for node in workflow.get("nodes", []) or []:
        nid = node.get("id")
        params = node.get("params")
        bindings = _collect_param_bindings(params) if isinstance(params, Mapping) else []
        for binding in bindings:
            err = _check_output_path_against_schema(
                binding.get("source"),
                nodes_by_id,
                actions_by_id,
                loop_body_parents,
                context_node_id=nid,
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
    # 拓扑与连通性基于最新的绑定推导，避免依赖可能过期的显式快照。
    inferred_edges = infer_edges_from_bindings(nodes)
    depends_map = infer_depends_on_from_edges(nodes, inferred_edges)

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

        depends_on_val = node.get("depends_on")
        if depends_on_val is not None and not isinstance(depends_on_val, list):
            errors.append(
                ValidationError(
                    code="INVALID_SCHEMA",
                    node_id=node.get("id"),
                    field="depends_on",
                    message="depends_on 必须是节点 id 的字符串数组。",
                )
            )
        elif isinstance(depends_on_val, list):
            for dep in depends_on_val:
                if not isinstance(dep, str) or dep not in node_ids:
                    errors.append(
                        ValidationError(
                            code="INVALID_SCHEMA",
                            node_id=node.get("id"),
                            field="depends_on",
                            message="depends_on 中的节点引用无效，请确保依赖指向已存在的节点。",
                        )
                    )
        errors.set_context(None)

    # ---------- 图连通性校验 ----------
    start_nodes = [n["id"] for n in nodes if n.get("type") == "start"]
    reachable: set = set()
    if nodes:
        adj: Dict[str, List[str]] = {}
        indegree: Dict[str, int] = {nid: 0 for nid in node_ids}

        used_dep_edges = False
        for nid, deps in depends_map.items():
            for dep in deps:
                if dep in node_ids and nid in node_ids:
                    adj.setdefault(dep, []).append(nid)
                    indegree[nid] += 1
                    used_dep_edges = True

        if not used_dep_edges:
            for e in inferred_edges:
                frm = e.get("from")
                to = e.get("to")
                if frm in node_ids and to in node_ids:
                    adj.setdefault(frm, []).append(to)
                    indegree[to] += 1

        zero_indegree = [nid for nid, deg in indegree.items() if deg == 0]
        if nodes and not zero_indegree:
            errors.append(
                ValidationError(
                    code="DISCONNECTED_GRAPH",
                    node_id=None,
                    field=None,
                    message="未找到入度为 0 的节点，无法确定拓扑起点。",
                )
            )

        # Treat nodes without inbound references as additional roots so workflows
        # that rely solely on parameter bindings remain connected.
        inbound_free = zero_indegree
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

        # 检测环：拓扑排序未遍历完所有节点。
        topo_queue = deque(zero_indegree)
        visited = 0
        while topo_queue:
            nid = topo_queue.popleft()
            visited += 1
            for nxt in adj.get(nid, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    topo_queue.append(nxt)
        if nodes and visited != len(node_ids):
            errors.append(
                ValidationError(
                    code="DISCONNECTED_GRAPH",
                    node_id=None,
                    field=None,
                    message="检测到工作流包含环或互相引用，无法完成拓扑排序。",
                )
            )

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
