# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Workflow-level validation entrypoints."""
from collections import deque
import copy
import json
from typing import Any, Dict, List, Mapping, Optional

from velvetflow.models import (
    ValidationError,
    Workflow,
    infer_depends_on_from_edges,
    infer_edges_from_bindings,
)
from velvetflow.loop_dsl import (
    index_loop_body_nodes,
    iter_workflow_and_loop_body_nodes,
    loop_body_has_action,
)

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
                    message=f"Loop node '{loop_id}' must provide body_subgraph.",
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
                    message=f"Loop node '{loop_id}' must have non-empty body_subgraph.nodes.",
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
                            f"Loop node '{loop_id}' body_subgraph.nodes contains duplicate node id: {nid}"
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
                        "A loop's body_subgraph requires at least one action node;"
                        " please use planning/repair tools to add executable steps to the loop body."
                    ),
                )
            )

    return errors


def _parse_expected_format_schema(expected_format: Any) -> Optional[Dict[str, Any]]:
    """Parse expected_format into a JSON schema mapping if possible."""

    if isinstance(expected_format, Mapping):
        return dict(expected_format)

    if isinstance(expected_format, str):
        stripped = expected_format.strip()

        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = stripped.strip("`")
            if "\n" in stripped:
                stripped = stripped.split("\n", 1)[1]

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None

        if isinstance(parsed, Mapping):
            return dict(parsed)

    return None


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
        messages.append("Invalid parameter binding references: " + "; ".join(binding_issues[:10]))

    if not messages:
        return []

    summary = "; ".join(messages)
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

    actions_by_id = _index_actions_by_id(action_registry)
    if actions_by_id:
        for node in iter_workflow_and_loop_body_nodes(workflow):
            if not isinstance(node, Mapping) or node.get("type") != "action":
                continue

            action_def = actions_by_id.get(node.get("action_id"))
            schema = action_def.get("output_schema") if isinstance(action_def, Mapping) else None
            params = node.get("params") if isinstance(node.get("params"), Mapping) else {}

            schema_to_set = schema

            if isinstance(schema_to_set, Mapping) and node.get("out_params_schema") != schema_to_set:
                node["out_params_schema"] = schema_to_set

    nodes = workflow.get("nodes", [])
    # Topology and connectivity rely on up-to-date inferred bindings to avoid stale snapshots.
    inferred_edges = infer_edges_from_bindings(nodes)
    depends_map = infer_depends_on_from_edges(nodes, inferred_edges)

    nodes_by_id = _index_nodes_by_id(workflow)
    loop_body_parents = index_loop_body_nodes(workflow)
    node_ids = set(nodes_by_id.keys())

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
                        "Detected exports on a non-loop node; removed the field. Please use tools to regenerate exports in a valid location."
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
                    message="Node params contain unsupported fields and were removed during validation.",
                )
            )

        depends_on_val = node.get("depends_on")
        if depends_on_val is not None and not isinstance(depends_on_val, list):
            errors.append(
                ValidationError(
                    code="INVALID_SCHEMA",
                    node_id=node.get("id"),
                    field="depends_on",
                    message="depends_on must be a list of node id strings.",
                )
            )

        ntype = node.get("type")
        if ntype == "condition":
            for branch_field in ("true_to_node", "false_to_node"):
                target = node.get(branch_field)
                if isinstance(target, str) and target not in node_ids:
                    errors.append(
                        ValidationError(
                            code="UNDEFINED_REFERENCE",
                            node_id=node.get("id"),
                            field=branch_field,
                            message=(
                                f"condition branch {branch_field} points to nonexistent node '{target}'."
                                " Confirm the node exists or use the repair tools to clean invalid branch references."
                            ),
                        )
                    )
        if ntype == "switch":
            cases = node.get("cases") if isinstance(node.get("cases"), list) else []
            for idx, case in enumerate(cases):
                if not isinstance(case, Mapping):
                    continue
                target = case.get("to_node")
                if isinstance(target, str) and target not in node_ids:
                    errors.append(
                        ValidationError(
                            code="UNDEFINED_REFERENCE",
                            node_id=node.get("id"),
                            field=f"cases[{idx}].to_node",
                            message=(
                                f"switch branch cases[{idx}].to_node points to nonexistent node '{target}'."
                                " Confirm the node exists or use the repair tools to clean invalid branch references."
                            ),
                        )
                    )
            if "default_to_node" in node:
                default_target = node.get("default_to_node")
                if isinstance(default_target, str) and default_target not in node_ids:
                    errors.append(
                        ValidationError(
                            code="UNDEFINED_REFERENCE",
                            node_id=node.get("id"),
                            field="default_to_node",
                            message=(
                                f"switch default branch default_to_node points to nonexistent node '{default_target}'."
                                " Confirm the node exists or use the repair tools to clean invalid branch references."
                            ),
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
                            message="Invalid node reference in depends_on; ensure dependencies point to existing nodes.",
                        )
                    )
        errors.set_context(None)

    # ---------- Graph connectivity validation ----------
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
                    message="No node with indegree 0 found; unable to determine entry point.",
                )
            )

        # Treat nodes without inbound references as additional roots so workflows
        # that rely solely on parameter bindings remain connected.
        inbound_free = zero_indegree
        candidate_roots = list(dict.fromkeys(inbound_free))

        dq = deque(candidate_roots)
        while dq:
            nid = dq.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for nxt in adj.get(nid, []):
                if nxt not in reachable:
                    dq.append(nxt)

        # Detect cycles: topological sort did not traverse all nodes.
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
                    message="Detected cycles or mutual references in the workflow; topological sort failed.",
                )
            )

        for nid in node_ids - reachable:
            errors.set_context({"node": nodes_by_id.get(nid, {})})
            errors.append(
                ValidationError(
                    code="DISCONNECTED_GRAPH",
                    node_id=nid,
                    field=None,
                    message=f"Node '{nid}' is unreachable from entry nodes.",
                )
            )
            errors.set_context(None)

    # ---------- Node validation (including loop body) ----------
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
                return "__from__ array elements must be strings"
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
