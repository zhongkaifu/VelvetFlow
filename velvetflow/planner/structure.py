# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Structure planning logic for the planner."""

import asyncio
import copy
import json
import os
import re
from collections import deque
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from jinja2 import Environment, TemplateSyntaxError
from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_debug, log_event, log_info, log_section, log_warn
from velvetflow.planner.agent_runtime import Agent, Runner, function_tool
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.workflow_builder import (
    WorkflowBuilder,
    attach_condition_branches,
)
from velvetflow.loop_dsl import iter_workflow_and_loop_body_nodes
from velvetflow.planner.requirement_analysis import _normalize_requirements_payload
from velvetflow.search import HybridActionSearchService
from velvetflow.models import (
    ALLOWED_PARAM_AGGREGATORS,
    Node,
    Workflow,
    infer_depends_on_from_edges,
    infer_edges_from_bindings,
)
from velvetflow.planner.relations import get_referenced_nodes
from velvetflow.reference_utils import normalize_reference_path, parse_field_path
from velvetflow.verification.validation import (
    _check_array_item_field,
    _check_output_path_against_schema,
)
from velvetflow.verification.binding_checks import _iter_template_references


CONDITION_PARAM_FIELDS = {"expression"}

SWITCH_PARAM_FIELDS = {
    "source",
    "field",
}

LOOP_PARAM_FIELDS = {
    "loop_kind",
    "source",
    "condition",
    "item_alias",
    "body_subgraph",
    "exports",
}

ACTION_NODE_FIELDS = {
    "id",
    "type",
    "action_id",
    "display_name",
    "params",
    "out_params_schema",
    "parent_node_id",
    "depends_on",
}

CONDITION_NODE_FIELDS = {
    "id",
    "type",
    "display_name",
    "params",
    "true_to_node",
    "false_to_node",
    "parent_node_id",
    "depends_on",
}

_JINJA_ENV = Environment()


def _is_valid_jinja(template: str) -> bool:
    try:
        _JINJA_ENV.parse(template)
        return True
    except TemplateSyntaxError:
        return False


def _balance_quotes(text: str) -> str:
    trimmed = text.strip()
    single_count = trimmed.count("'")
    double_count = trimmed.count('"')

    if single_count % 2 == 1:
        if not trimmed.startswith("'"):
            trimmed = "'" + trimmed
        if trimmed.count("'") % 2 == 1:
            trimmed = trimmed + "'"

    if double_count % 2 == 1:
        if not trimmed.startswith('"'):
            trimmed = '"' + trimmed
        if trimmed.count('"') % 2 == 1:
            trimmed = trimmed + '"'

    return trimmed


def _normalize_ternary_literals(expr: str) -> str:
    ternary = re.match(r"^(?P<true>.+?)\s+if\s+(?P<cond>.+?)\s+else\s+(?P<false>.+)$", expr)
    if not ternary:
        return expr

    true_part = _balance_quotes(ternary.group("true"))
    false_part = _balance_quotes(ternary.group("false"))
    return f"{true_part} if {ternary.group('cond')} else {false_part}"


def _fix_common_template_errors(expr: str) -> str:
    raw = expr.strip()
    has_wrapped = raw.startswith("{{") and raw.endswith("}}")
    body = raw[2:-2].strip() if has_wrapped else raw

    body = _normalize_ternary_literals(body)

    if re.search(r"\|\s*length\s*>", body):
        body = re.sub(r"(.*?\|\s*length)\s*>(.+)", r"(\1) >\2", body)

    return f"{{{{ {body} }}}}" if has_wrapped else body


def _normalize_template_string(value: str) -> tuple[str, bool]:
    fixed = value

    looks_like_ref = any(token in value for token in ["result_of.", "loop.", "exports."])
    has_braces = "{{" in value and "}}" in value

    if not has_braces and looks_like_ref:
        fixed = f"{{{{ {value.strip()} }}}}"

    if not _is_valid_jinja(fixed):
        candidate = _fix_common_template_errors(fixed)
        if _is_valid_jinja(candidate):
            return candidate, candidate != value
    return fixed, fixed != value


def _normalize_params_templates(obj: Any) -> tuple[Any, bool]:
    changed = False

    def _walk(val: Any) -> Any:
        nonlocal changed
        if isinstance(val, Mapping):
            normalized: Dict[str, Any] = {}
            for k, v in val.items():
                normalized[k] = _walk(v)
            return normalized
        if isinstance(val, list):
            return [_walk(item) for item in val]
        if isinstance(val, str):
            new_val, is_changed = _normalize_template_string(val)
            changed = changed or is_changed
            return new_val
        return val

    return _walk(obj), changed

SWITCH_NODE_FIELDS = {
    "id",
    "type",
    "display_name",
    "params",
    "cases",
    "default_to_node",
    "parent_node_id",
    "depends_on",
}

LOOP_NODE_FIELDS = {
    "id",
    "type",
    "display_name",
    "params",
    "parent_node_id",
    "depends_on",
}

def _attach_inferred_edges(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild derived edges so LLMs can see the implicit wiring."""

    copied = copy.deepcopy(workflow)
    nodes = list(iter_workflow_and_loop_body_nodes(copied))
    copied["edges"] = infer_edges_from_bindings(nodes)
    return attach_condition_branches(copied)


def _ensure_start_node(builder: WorkflowBuilder) -> str:
    for node in builder.nodes.values():
        if isinstance(node, Mapping) and node.get("type") == "start":
            node_id = node.get("id")
            return node_id if isinstance(node_id, str) else "start"

    base_id = "start"
    node_id = base_id
    counter = 1
    while node_id in builder.nodes:
        node_id = f"{base_id}_{counter}"
        counter += 1

    builder.add_node(node_id=node_id, node_type="start", display_name="Start", params={})
    return node_id


def _hydrate_builder_from_workflow(
    *, builder: WorkflowBuilder, workflow: Mapping[str, Any]
) -> None:
    """Load an existing workflow DAG into a mutable ``WorkflowBuilder``.

    This allows the planner to continue expanding an already constructed
    workflow instead of always starting from scratch.
    """

    if not isinstance(workflow, Mapping):
        return

    hydrated_workflow = attach_condition_branches(copy.deepcopy(workflow))

    name = hydrated_workflow.get("workflow_name")
    description = hydrated_workflow.get("description")
    if isinstance(name, str):
        builder.workflow_name = name
    if isinstance(description, str):
        builder.description = description

    visited: set[str] = set()

    def _normalize_loop_params(params: Mapping[str, Any] | None) -> tuple[dict, list]:
        if not isinstance(params, Mapping):
            return {}, []

        params_copy = copy.deepcopy(params)
        body = params_copy.get("body_subgraph")

        if isinstance(body, Mapping):
            body_nodes = body.get("nodes") if isinstance(body.get("nodes"), list) else []
            params_copy["body_subgraph"] = {k: v for k, v in body.items() if k != "nodes"}
            params_copy["body_subgraph"].setdefault("nodes", [])
        elif isinstance(body, list):
            body_nodes = body
            params_copy["body_subgraph"] = {"nodes": []}
        else:
            body_nodes = []
            if "body_subgraph" in params_copy:
                params_copy["body_subgraph"] = {"nodes": []}

        return params_copy, list(body_nodes or [])

    def _normalize_parallel_params(params: Mapping[str, Any] | None) -> tuple[dict, list[tuple[str, list[Any]]]]:
        if not isinstance(params, Mapping):
            return {}, []

        params_copy = copy.deepcopy(params)
        branches = params_copy.get("branches") if isinstance(params_copy.get("branches"), list) else []
        branch_nodes: list[tuple[str, list[Any]]] = []
        normalized_branches: list[dict] = []

        for branch in branches or []:
            if not isinstance(branch, Mapping):
                continue

            branch_copy = copy.deepcopy(branch)
            sub_nodes = branch_copy.get("sub_graph_nodes") if isinstance(branch_copy.get("sub_graph_nodes"), list) else []
            branch_copy["sub_graph_nodes"] = []

            branch_id = branch_copy.get("id") if isinstance(branch_copy.get("id"), str) else None
            if branch_id:
                branch_nodes.append((branch_id, list(sub_nodes)))

            normalized_branches.append(branch_copy)

        params_copy["branches"] = normalized_branches
        return params_copy, branch_nodes

    edges = hydrated_workflow.get("edges") if isinstance(hydrated_workflow.get("edges"), list) else []
    depends_on_map = infer_depends_on_from_edges(iter_workflow_and_loop_body_nodes(hydrated_workflow), edges)

    def _walk(nodes: Sequence[Any] | None, parent_id: str | None = None) -> None:
        for node in nodes or []:
            if not isinstance(node, Mapping):
                continue

            node_id = node.get("id")
            node_type = node.get("type")
            if not isinstance(node_id, str) or not isinstance(node_type, str):
                continue
            if node_id in visited:
                continue

            visited.add(node_id)
            params = node.get("params") if isinstance(node.get("params"), Mapping) else node.get("params")
            depends_on = (
                node.get("depends_on")
                if isinstance(node.get("depends_on"), list)
                else depends_on_map.get(node_id)
                if depends_on_map
                else None
            )
            inferred_parent = parent_id or (node.get("parent_node_id") if isinstance(node.get("parent_node_id"), str) else None)

            body_nodes: list[Any] = []
            branch_nodes: list[tuple[str, list[Any]]] = []
            params_to_use: Mapping[str, Any] | None = None

            if node_type == "loop":
                params_to_use, body_nodes = _normalize_loop_params(params if isinstance(params, Mapping) else {})
            elif node_type == "parallel":
                params_to_use, branch_nodes = _normalize_parallel_params(params if isinstance(params, Mapping) else {})

            builder.add_node(
                node_id=node_id,
                node_type=node_type,
                action_id=node.get("action_id") if isinstance(node.get("action_id"), str) else None,
                display_name=node.get("display_name") if isinstance(node.get("display_name"), str) else node.get("display_name"),
                params=params_to_use if node_type == "loop" else (copy.deepcopy(params) if isinstance(params, Mapping) else params),
                out_params_schema=node.get("out_params_schema") if isinstance(node.get("out_params_schema"), Mapping) else None,
                true_to_node=node.get("true_to_node") if isinstance(node.get("true_to_node"), str) else node.get("true_to_node"),
                false_to_node=node.get("false_to_node") if isinstance(node.get("false_to_node"), str) else node.get("false_to_node"),
                cases=node.get("cases") if isinstance(node.get("cases"), list) else None,
                default_to_node=node.get("default_to_node") if isinstance(node.get("default_to_node"), str) else node.get("default_to_node"),
                parent_node_id=inferred_parent,
                depends_on=depends_on,
            )

            if node_type == "loop" and body_nodes:
                _walk(body_nodes, parent_id=node_id)
            elif node_type == "parallel" and branch_nodes:
                for branch_id, nodes_in_branch in branch_nodes:
                    _walk(nodes_in_branch, parent_id=f"{node_id}:{branch_id}")

    _walk(hydrated_workflow.get("nodes"))


def _normalize_sub_graph_nodes(
    raw: Any, *, builder: WorkflowBuilder
) -> tuple[List[str], Optional[Dict[str, Any]]]:
    """Validate and normalize a list of node ids to attach to a loop body."""

    if raw is None:
        return [], None

    if not isinstance(raw, list):
        return [], {"message": "sub_graph_nodes 需要是节点 id 的数组。"}

    non_str_indices = [idx for idx, value in enumerate(raw) if not isinstance(value, str)]
    normalized = [value for value in raw if isinstance(value, str)]
    missing_nodes = [nid for nid in normalized if nid not in builder.nodes]

    if non_str_indices or missing_nodes:
        return [], {
            "message": "sub_graph_nodes 应为已创建节点的 id 字符串列表。",
            "invalid_indices": non_str_indices,
            "missing_nodes": missing_nodes,
        }

    return normalized, None


def _attach_sub_graph_nodes(builder: WorkflowBuilder, loop_id: str, node_ids: List[str]):
    """Mark the given nodes as belonging to the loop's body_subgraph."""

    for nid in node_ids:
        node = builder.nodes.get(nid)
        if isinstance(node, dict):
            node["parent_node_id"] = loop_id


def _filter_supported_params(
    *,
    node_type: str,
    params: Any,
    action_schemas: Mapping[str, Mapping[str, Any]],
    action_id: Optional[str] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """Keep only supported param fields for the given node type.

    Returns the sanitized params dict and a list of removed field names.
    """

    if not isinstance(params, Mapping):
        return {}, []

    allowed_fields: Optional[set[str]] = None
    if node_type == "condition":
        allowed_fields = set(CONDITION_PARAM_FIELDS)
    elif node_type == "switch":
        allowed_fields = set(SWITCH_PARAM_FIELDS)
    elif node_type == "loop":
        allowed_fields = set(LOOP_PARAM_FIELDS)
    elif node_type == "action" and action_id:
        schema = action_schemas.get(action_id, {}) if isinstance(action_id, str) else {}
        properties = schema.get("arg_schema", {}).get("properties") if isinstance(schema.get("arg_schema"), Mapping) else None
        if isinstance(properties, Mapping):
            allowed_fields = set(properties.keys())

    if not allowed_fields:
        return dict(params), []

    cleaned: Dict[str, Any] = {k: v for k, v in params.items() if k in allowed_fields}
    removed = [k for k in params if k not in allowed_fields]

    return cleaned, removed


def _sanitize_builder_node_params(
    builder: WorkflowBuilder, node_id: str, action_schemas: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    node = builder.nodes.get(node_id)
    if not isinstance(node, Mapping):
        return []

    params = node.get("params") or {}
    cleaned, removed = _filter_supported_params(
        node_type=str(node.get("type")),
        params=params,
        action_schemas=action_schemas,
        action_id=node.get("action_id") if isinstance(node.get("action_id"), str) else None,
    )

    if removed:
        node["params"] = cleaned

    return removed


def _sanitize_builder_node_fields(builder: WorkflowBuilder, node_id: str) -> List[str]:
    node = builder.nodes.get(node_id)
    if not isinstance(node, Mapping):
        return []

    node_type = node.get("type")
    allowed_fields: Optional[set[str]] = None
    if node_type == "action":
        allowed_fields = set(ACTION_NODE_FIELDS)
    elif node_type == "condition":
        allowed_fields = set(CONDITION_NODE_FIELDS)
    elif node_type == "switch":
        allowed_fields = set(SWITCH_NODE_FIELDS)
    elif node_type == "loop":
        allowed_fields = set(LOOP_NODE_FIELDS)

    if not allowed_fields:
        return []

    removed_keys = [key for key in list(node.keys()) if key not in allowed_fields]
    for key in removed_keys:
        node.pop(key, None)

    return removed_keys


def _build_action_schema_map(action_registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    action_schemas: Dict[str, Dict[str, Any]] = {}
    for action in action_registry:
        aid = action.get("action_id")
        if not aid:
            continue
        action_schemas[aid] = {
            "name": action.get("name", ""),
            "description": action.get("description", ""),
            "domain": action.get("domain", ""),
            "arg_schema": action.get("arg_schema"),
            "output_schema": action.get("output_schema"),
        }
    return action_schemas


def _extract_loop_body_context(
    loop_node: Mapping[str, Any], action_schemas: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    params = loop_node.get("params") if isinstance(loop_node, Mapping) else None
    body = params.get("body_subgraph") if isinstance(params, Mapping) else None
    if not isinstance(body, Mapping):
        return {"nodes": []}

    context_nodes = []
    for child in body.get("nodes", []) or []:
        if not isinstance(child, Mapping):
            continue
        action_id = child.get("action_id")
        schema = action_schemas.get(action_id, {}) if isinstance(action_id, str) else {}
        context_nodes.append(
            {
                "id": child.get("id"),
                "type": child.get("type"),
                "action_id": action_id,
                "display_name": child.get("display_name"),
                "output_schema": schema.get("output_schema"),
            }
        )

    return {"nodes": context_nodes}


def _validate_loop_exports(
    *, loop_node: Mapping[str, Any], exports: Mapping[str, Any]
) -> List[str]:
    params = loop_node.get("params") if isinstance(loop_node.get("params"), Mapping) else {}
    body = params.get("body_subgraph") if isinstance(params, Mapping) else None
    if not isinstance(body, Mapping):
        body = {}

    body_nodes = [bn for bn in body.get("nodes", []) or [] if isinstance(bn, Mapping)]
    body_ids = {bn.get("id") for bn in body_nodes if isinstance(bn.get("id"), str)}

    errors: List[str] = []

    if not isinstance(exports, Mapping):
        return ["exports 必须是对象"]

    for key, value in exports.items():
        if not isinstance(key, str):
            errors.append("exports 的 key 必须是字符串")
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append(f"exports.{key} 必须是非空 Jinja 表达式字符串")
            continue

        refs = list(_iter_template_references(value))
        if not refs:
            errors.append(f"exports.{key} 必须引用 loop body 节点的输出字段")
            continue

        has_body_ref = False
        for ref in refs:
            try:
                tokens = parse_field_path(ref)
            except Exception:
                continue
            if len(tokens) < 3 or tokens[0] != "result_of":
                continue
            ref_node = tokens[1]
            if isinstance(ref_node, str) and ref_node in body_ids:
                has_body_ref = True
            else:
                errors.append(f"exports.{key} 只能引用 body_subgraph 内的节点输出")
                break
        if not has_body_ref and not errors:
            errors.append(f"exports.{key} 必须引用 loop body 节点的输出字段")

    return errors


def _fallback_loop_exports(
    loop_node: Mapping[str, Any], action_schemas: Mapping[str, Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    params = loop_node.get("params") if isinstance(loop_node, Mapping) else None
    if not isinstance(params, Mapping):
        return None
    body = params.get("body_subgraph")
    if not isinstance(body, Mapping):
        return None

    body_nodes = [bn for bn in body.get("nodes", []) or [] if isinstance(bn, Mapping)]
    body_ids = [bn.get("id") for bn in body_nodes if isinstance(bn.get("id"), str)]
    exit_node = body.get("exit") if isinstance(body.get("exit"), str) else None
    from_node = exit_node if exit_node in body_ids else (body_ids[0] if body_ids else None)
    if not from_node:
        return None

    target_node = next((bn for bn in body_nodes if bn.get("id") == from_node), None)
    field_name = "status"
    if isinstance(target_node, Mapping):
        action_id = target_node.get("action_id")
        schema = action_schemas.get(action_id, {}) if isinstance(action_id, str) else {}
        props = (
            schema.get("output_schema", {}).get("properties")
            if isinstance(schema.get("output_schema"), Mapping)
            else None
        )
        if isinstance(props, Mapping):
            field_name = next((k for k in props.keys() if isinstance(k, str)), field_name)

    return {
        "items": f"{{{{ result_of.{from_node}.{field_name} }}}}",
    }


def _ensure_loop_items_fields(
    *,
    exports: Mapping[str, Any],
    loop_node: Mapping[str, Any],
    action_schemas: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Placeholder pass-through for loop exports."""

    return dict(exports)


def _prepare_skeleton_for_next_stage(
    *,
    builder: WorkflowBuilder,
    action_registry: List[Dict[str, Any]],
    search_service: HybridActionSearchService,
) -> Dict[str, Any]:
    skeleton = _attach_inferred_edges(builder.to_workflow())
    skeleton = ensure_registered_actions(
        skeleton, action_registry=action_registry, search_service=search_service
    )
    return _attach_inferred_edges(skeleton)



def _build_combined_prompt() -> str:
    structure_prompt = (
        "You are a general-purpose business workflow orchestration assistant.\n"
        "The user's natural language requirement has been decomposed in previous steps into a structured checklist, including description, intent, inputs, constraints, status, and mapped_node (mapping to workflow nodes).\n"
        "If you need to supplement or modify the decomposition and status, update the structured checklist during the requirement analysis phase and then rerun planning.\n"
        "[Workflow DSL syntax and semantics (must follow)]\n"
        "- workflow = {workflow_name, description, nodes: []}; only return valid JSON (edges will be automatically inferred by the system based on node bindings, no need to generate them).\n"
        "- Node basic structure: {id, type, display_name, params, depends_on, action_id?, out_params_schema?, loop/subgraph/branches?}.\n"
        "  type allows start/action/condition/loop/parallel; include a start node to mark the workflow entry.\n"
        "  Action nodes must specify action_id (from the action library) and params; only action nodes allow out_params_schema.\n"
        "  Condition node params can only include expression (a single Jinja expression returning a boolean); true_to_node/false_to_node must be top-level fields (string or null), not inside params.\n"
        "  Loop nodes contain loop_kind/iter/source/body_subgraph/exports. Values in exports must reference fields of nodes inside body_subgraph (e.g., {{ result_of.node.field }}); outside the loop you may only reference exports.<key>. body_subgraph only needs the nodes array—no entry/exit/edges.\n"
        "  Branches of a parallel node are a non-empty array, each element containing id/entry_node/sub_graph_nodes.\n"
        "- Within params you must directly use Jinja expressions to reference upstream results (e.g., {{ result_of.<node_id>.<field_path> }} or {{ loop.item.xxx }}); object-style bindings are forbidden."
        "  Every params value you output during planning (including condition/switch/loop nodes) will be parsed by the Jinja engine. Any reference not using Jinja2 syntax will be treated as an error and trigger auto-fix; strictly follow Jinja template syntax."
        "  The <node_id> must exist and its fields must align with the upstream output_schema or loop.exports.\n"
        "There is an Action Registry in the system with many business actions; you may only query it via search_business_actions.\n"
        "Construction steps:\n"
        "1) Use set_workflow_meta to set the workflow name and description.\n"
        "2) When business actions are needed, you must first call search_business_actions to query candidates; add_action_node.action_id must come from the most recent candidates.id.\n"
        "3) When condition/switch/loop nodes are needed, you must first create them with add_condition_node/add_switch_node/add_loop_node; expressions and references in params must strictly follow Jinja template syntax.\n"
        "4) Call update_node_params to complete and validate params for created nodes.\n"
        "6) If an existing node needs to be modified (adding display_name/params/branch targets/parent nodes, etc.), call update_action_node or update_condition_node with the fields to overwrite; after calling, be sure to check whether related upstream/downstream nodes also need updates to stay consistent.\n"
        "7) Condition nodes must explicitly provide true_to_node and false_to_node. Values can be a node id (continue execution) or null (indicating that branch ends); express dependencies through input/output references in node params—no need to draw edges explicitly.\n"
        "8) Maintain depends_on (array of strings) for each node, listing its direct upstream dependencies; when a node is targeted by condition.true_to_node/false_to_node, you must add that condition node to the target node's depends_on.\n"
        "9) After the structure is complete, continue to fill params for all nodes and ensure update_node_params validation passes.\n\n"
        "Important note: Only action nodes need out_params_schema; condition nodes do not have this property. The format of out_params_schema should be {\"param_name\": \"type\"}; list only the names and types of output parameters of the business action, without extra descriptions or examples.\n\n"
        "[Very important principles]\n"
        "1. You must design the workflow strictly around the natural language requirement in the current conversation:\n"
        "   - Trigger method (scheduled / event / manual)\n"
        "   - Data query/read\n"
        "   - Filtering conditions\n"
        "   - Aggregation/statistics/summary\n"
        "   - Notification / write / persistence / calling downstream systems\n"
        "2. Data inside a loop node can only be exposed externally through loop.exports. Downstream references to loop results must use result_of.<loop_id>.exports.<key>; directly referencing nodes in the loop body is forbidden.\n"
        "3. loop.exports should be defined under params.exports; do not place them inside body_subgraph. body_subgraph only contains the nodes array—no entry/exit/edges.\n"
        "4. Nested loops are allowed, but child loops must be explicitly included in the parent loop's body_subgraph via parent_node_id or sub_graph_nodes;"
        "   external nodes must still access loop data via loop.exports instead of directly pointing to nodes in the subgraph.\n\n"
        "You must ensure the workflow structure covers every subtask in the user's natural language requirement, not just the first half:\n"
        "For example, if the requirement includes: trigger + query + filter + summarize + notify, you cannot stop at trigger + query;\n"
        "you must explicitly include nodes and data flows for filter, summarize, notify, etc. in the structure.\n"
        "When the workflow structure is complete, ensure the overall structure covers all subtasks."
    )
    param_prompt = (
        "【Parameter completion rules】\n"
        "You are a workflow parameter completion assistant. Handle one node at a time and fill the current node's params according to the given arg_schema.\n"
        "You must submit and validate via tools: call update_node_params to submit the completion result and validate; if a validation error is received, re-analyze and submit again.\n"
        "When a field needs to reference another node's output, you must write it directly as a Jinja expression or template string, such as {{ result_of.<node_id>.<field> }};"
        "node_id can only come from allowed_node_ids, field must exist in that node's output_schema, and it must be written as a Jinja template.\n"
        "All of your params values will be parsed directly by the Jinja2 engine; any non-Jinja syntax (including leftover binding objects or pseudo-code) will be treated as an error and trigger auto-fix—always output strings or literals that Jinja can render.\n"
        "In loop scenarios you may use loop.item/loop.index and fields exposed by loop.exports.*, still referenced via Jinja expressions; directly referencing loop body nodes is forbidden.\n"
        "When referencing the loop.exports structure you must point to a specific field or sub-structure inside exports; you cannot write only result_of.<loop_id>.exports.\n"
        "Each key in loop.exports collects per-iteration results into a list; each exports value must reference a body_subgraph node field, such as {{ result_of.<body_node>.<field> }}.\n"
        "Params for all nodes must not be empty; if there is no obvious default value, analyze upstream/downstream semantics and then use the tool to fill them.\n"
        "Complete aggregation/filtering/formatting with Jinja filters or expressions (for example, {{ result_of.a.items | selectattr('score', '>', 80) | list | length }} or {{ result_of.a.items | map(attribute='name') | join(', ') }}); do not output object-style bindings."
    )
    return f"{structure_prompt}\n\n{param_prompt}"


def _find_start_nodes_for_params(workflow: Workflow) -> List[str]:
    starts = [n.id for n in workflow.nodes if n.type == "start"]
    if starts:
        return starts

    to_ids = {e.to_node for e in workflow.edges}
    candidates = [n.id for n in workflow.nodes if n.id not in to_ids]
    if candidates:
        return candidates

    return [workflow.nodes[0].id] if workflow.nodes else []


def _traverse_order(workflow: Workflow) -> List[str]:
    """按 start -> downstream 的顺序遍历，保证上游节点先被处理。"""

    adj: Dict[str, List[str]] = {}
    for e in workflow.edges:
        adj.setdefault(e.from_node, []).append(e.to_node)

    visited: set[str] = set()
    order: List[str] = []
    dq: deque[str] = deque(_find_start_nodes_for_params(workflow))

    while dq:
        nid = dq.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        order.append(nid)
        for nxt in adj.get(nid, []):
            if nxt not in visited:
                dq.append(nxt)

    for node in workflow.nodes:
        if node.id not in visited:
            order.append(node.id)

    return order


def _looks_like_entity_key(field_name: str) -> bool:
    lower = field_name.lower()
    return any(token in lower for token in ["id", "code", "key", "uuid"])


def _extract_binding_sources(
    obj: Any, *, path_prefix: str = "params"
) -> List[Tuple[str, str]]:
    """Return all bindings within a params object as (path, __from__) pairs."""

    bindings: List[Tuple[str, str]] = []

    if isinstance(obj, dict):
        if isinstance(obj.get("__from__"), str):
            bindings.append(
                (path_prefix, normalize_reference_path(obj["__from__"]))
            )
        for k, v in obj.items():
            child_prefix = f"{path_prefix}.{k}"
            bindings.extend(_extract_binding_sources(v, path_prefix=child_prefix))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            bindings.extend(
                _extract_binding_sources(item, path_prefix=f"{path_prefix}[{idx}]")
            )

    return bindings


def _build_binding_memory(
    filled_params: Mapping[str, Mapping[str, Any]], processed_node_ids: Sequence[str]
) -> Dict[str, str]:
    """Track key entity bindings from already补全的节点，供跨节点一致性检查使用。"""

    memory: Dict[str, str] = {}

    for nid in processed_node_ids:
        params = filled_params.get(nid)
        if not isinstance(params, Mapping):
            continue
        for path, source in _extract_binding_sources(params):
            field = path.split(".")[-1]
            if _looks_like_entity_key(field) and field not in memory:
                memory[field] = source

    return memory


def _summarize_output_fields_from_schema(
    schema: Mapping[str, Any] | None,
) -> List[str]:
    if not isinstance(schema, Mapping):
        return []
    props = schema.get("properties")
    if isinstance(props, Mapping):
        return [k for k in props.keys() if isinstance(k, str)]
    return []


def _summarize_node_outputs(
    node: Node, action_schemas: Mapping[str, Dict[str, Any]]
) -> List[str]:
    if node.type == "loop":
        exports = node.params.get("exports") if isinstance(node.params, Mapping) else {}
        fields: List[str] = []
        if isinstance(exports, Mapping):
            fields.extend([k for k in exports.keys() if isinstance(k, str)])
        return fields

    if not node.action_id:
        return []
    schema = getattr(node, "out_params_schema", None) or action_schemas.get(
        node.action_id, {}
    ).get("output_schema")
    return _summarize_output_fields_from_schema(schema)


def _build_global_context(
    *,
    workflow: Workflow,
    action_schemas: Mapping[str, Dict[str, Any]],
    filled_params: Mapping[str, Mapping[str, Any]],
    processed_node_ids: Sequence[str],
    binding_memory: Mapping[str, str],
) -> Dict[str, Any]:
    upstream_map: Dict[str, List[str]] = {}
    for e in workflow.edges:
        upstream_map.setdefault(e.to_node, []).append(e.from_node)

    node_summaries: List[Dict[str, Any]] = []
    for n in workflow.nodes:
        schema = action_schemas.get(n.action_id, {}) if n.action_id else {}
        node_summaries.append(
            {
                "id": n.id,
                "type": n.type,
                "action_id": n.action_id,
                "display_name": n.display_name,
                "domain": schema.get("domain"),
                "out_params_schema": getattr(n, "out_params_schema", None)
                or schema.get("output_schema"),
                "output_fields": _summarize_node_outputs(n, action_schemas),
                "arg_required_fields": (
                    schema.get("arg_schema", {}).get("required")
                    if isinstance(schema.get("arg_schema"), Mapping)
                    else None
                ),
                "upstream": upstream_map.get(n.id, []),
                "params_snapshot": filled_params.get(n.id)
                if n.id in processed_node_ids
                else None,
            }
        )

    return {
        "workflow": {
            "name": workflow.workflow_name,
            "description": workflow.description,
        },
        "node_summaries": node_summaries,
        "entity_binding_hints": [
            {"field": field, "source": src} for field, src in binding_memory.items()
        ],
    }


def _collect_binding_issues(
    params: Dict[str, Any],
    upstream_nodes: List[Node],
    action_schemas: Dict[str, Dict[str, Any]],
    binding_memory: Mapping[str, str] | None = None,
) -> List[str]:
    """Validate __from__ / field references against upstream schemas.

    The function only checks bindings that point to ``result_of.<node_id>``
    sources. When a binding or its nested pipeline/count_if field refers to a
    missing node or field, the issue is returned as a human-readable string so
    that the caller can surface it early and trigger automated repair tools.
    """

    nodes_by_id: Dict[str, Dict[str, Any]] = {
        n.id: n.model_dump(by_alias=True) for n in upstream_nodes
    }
    actions_by_id = {aid: schema for aid, schema in action_schemas.items()}

    issues: List[str] = []

    def _walk(obj: Any, path_prefix: str = "params") -> None:
        if isinstance(obj, dict):
            if "__from__" in obj:
                src = normalize_reference_path(obj.get("__from__"))
                schema_err = _check_output_path_against_schema(
                    src, nodes_by_id, actions_by_id
                )
                if schema_err:
                    issues.append(f"{path_prefix}: {schema_err}")

                field_name = path_prefix.split(".")[-1]
                remembered = (binding_memory or {}).get(field_name)
                if remembered and normalize_reference_path(remembered) != src:
                    issues.append(
                        f"{path_prefix}: 字段 {field_name} 需要与之前的绑定保持一致（之前来源 {remembered}，当前 {src}）。"
                    )

                agg = obj.get("__agg__")
                if agg is not None and agg not in ALLOWED_PARAM_AGGREGATORS:
                    issues.append(
                        f"{path_prefix}: __agg__ 取值非法（{agg}），可选值：{', '.join(ALLOWED_PARAM_AGGREGATORS)}。"
                    )

                field_checks: List[tuple[str, str]] = []
                if agg == "count_if":
                    fld = obj.get("field")
                    if isinstance(fld, str):
                        field_checks.append(("count_if.field", fld))
                if agg == "pipeline":
                    steps = obj.get("steps")
                    if isinstance(steps, list):
                        for idx, step in enumerate(steps):
                            if not isinstance(step, dict):
                                continue
                            if step.get("op") == "filter" and isinstance(
                                step.get("field"), str
                            ):
                                field_checks.append(
                                    (
                                        f"pipeline.steps[{idx}].field",
                                        step["field"],
                                    )
                                )

                for field_label, fld in field_checks:
                    item_err = _check_array_item_field(
                        src, fld, nodes_by_id, actions_by_id
                    )
                    if item_err:
                        issues.append(f"{path_prefix}.{field_label}: {item_err}")

            for k, v in obj.items():
                new_prefix = f"{path_prefix}.{k}" if path_prefix else k
                _walk(v, new_prefix)

        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                _walk(v, f"{path_prefix}[{idx}]")

    _walk(params)

    return issues


def _validate_node_params(
    *,
    node: Node,
    params: Dict[str, Any],
    upstream_nodes: List[Node],
    action_schemas: Dict[str, Dict[str, Any]],
    binding_memory: Mapping[str, str] | None = None,
) -> List[str]:
    """Validate a single node's params after every tool submission.

    The checks cover:
    - required/unknown fields against the node's arg_schema when available
    - binding legality for result_of references via ``_collect_binding_issues``
    """

    errors: List[str] = []

    if not isinstance(params, dict):
        return ["params 必须是对象"]

    schema = action_schemas.get(node.action_id or "", {}).get("arg_schema", {})
    required_fields = []
    properties = None
    allow_additional = True

    if isinstance(schema, dict):
        required_fields = schema.get("required") or []
        properties = (
            schema.get("properties") if isinstance(schema.get("properties"), dict) else None
        )
        allow_additional = bool(schema.get("additionalProperties", True))

    for field in required_fields:
        if field not in params:
            errors.append(f"缺少必填字段 {field}")

    if isinstance(properties, dict) and not allow_additional:
        for field in params:
            if field not in properties:
                errors.append(f"参数 {field} 未在 arg_schema 中定义")

    errors.extend(
        _collect_binding_issues(
            params,
            upstream_nodes=upstream_nodes,
            action_schemas=action_schemas,
            binding_memory=binding_memory,
        )
    )

    return errors



def plan_workflow_structure_with_llm(
    parsed_requirement: Mapping[str, Any],
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    max_rounds: int = 100,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    existing_workflow: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("请先设置 OPENAI_API_KEY 环境变量再进行结构规划。")

    parsed_requirement = _normalize_requirements_payload(parsed_requirement)
    parsed_requirement = copy.deepcopy(parsed_requirement)

    builder = WorkflowBuilder()
    if existing_workflow:
        _hydrate_builder_from_workflow(builder=builder, workflow=existing_workflow)
    _ensure_start_node(builder)
    action_schemas = _build_action_schema_map(action_registry)
    last_action_candidates: List[str] = []
    all_action_candidates: List[str] = []
    all_action_candidates_info: List[Dict[str, Any]] = []
    latest_skeleton: Dict[str, Any] = {}

    def _emit_progress(label: str, workflow_obj: Mapping[str, Any]) -> None:
        if not progress_callback:
            return
        try:
            progress_callback(label, workflow_obj)
        except Exception:
            log_debug(f"[StructurePlanner] progress_callback {label} 调用失败，已忽略。")

    def _log_tool_call(tool_name: str, payload: Mapping[str, Any] | None = None) -> None:
        if payload:
            log_info(
                f"[StructurePlanner] tool={tool_name}",
                json.dumps(payload, ensure_ascii=False),
            )
        else:
            log_info(f"[StructurePlanner] tool={tool_name}")

    def _log_tool_result(tool_name: str, result: Mapping[str, Any]) -> None:
        log_info(
            f"[StructurePlanner] tool_result={tool_name}",
            json.dumps(result, ensure_ascii=False),
        )

    def _log_tool_status(tool_name: str, result: Mapping[str, Any]) -> None:
        status = result.get("status") if isinstance(result, Mapping) else None
        success = status not in {"error", "failed", "failure"}
        log_info(
            f"[StructurePlanner] tool_status={tool_name}",
            json.dumps(
                {"status": status or "unknown", "success": success},
                ensure_ascii=False,
            ),
        )

    def _return_tool_result(tool_name: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
        _log_tool_result(tool_name, result)
        _log_tool_status(tool_name, result)
        return result

    def _emit_canvas_update(label: str, workflow_obj: Mapping[str, Any] | None = None) -> None:
        if not progress_callback:
            return
        try:
            snapshot = workflow_obj or _attach_inferred_edges(builder.to_workflow())
            progress_callback(label, snapshot)
        except Exception:
            log_debug(f"[StructurePlanner] canvas callback {label} 调用失败，已忽略。")

    def _snapshot(label: str) -> Dict[str, Any]:
        nonlocal latest_skeleton
        workflow_snapshot = _attach_inferred_edges(builder.to_workflow())
        latest_skeleton = workflow_snapshot
        _emit_progress(label, workflow_snapshot)
        return workflow_snapshot
    system_prompt = _build_combined_prompt()

    def _build_validation_error(message: str, **extra: Any) -> Dict[str, Any]:
        payload = {"status": "error", "message": message}
        payload.update(extra)
        return payload

    @function_tool(strict_mode=False)
    def search_business_actions(query: str, top_k: int = 5) -> Mapping[str, Any]:
        """Search business action candidates to choose a valid ``action_id`` during planning.

        Use case: Query the available action list before creating or updating an action node.

        Args:
            query: Search keywords that describe the business action capability or name.
            top_k: Maximum number of candidates to return.

        Returns:
            A result dictionary containing the search status, raw action list, and simplified
            candidate information.
        """
        _log_tool_call("search_business_actions", {"query": query, "top_k": top_k})
        actions_raw = search_service.search(query=query, top_k=int(top_k))
        candidates = [
            {
                "id": a.get("action_id"),
                "name": a.get("name", ""),
                "description": a.get("description", ""),
                "category": a.get("domain") or "general",
            }
            for a in actions_raw
            if a.get("action_id")
        ]
        candidate_ids = [c["id"] for c in candidates]
        last_action_candidates[:] = candidate_ids
        for action_id in candidate_ids:
            if action_id not in all_action_candidates:
                all_action_candidates.append(action_id)
        for candidate in candidates:
            if candidate["id"] and all(
                existing["id"] != candidate["id"] for existing in all_action_candidates_info
            ):
                all_action_candidates_info.append(candidate)
        _emit_canvas_update("search_business_actions")
        result = {
            "status": "ok",
            "query": query,
            "actions": actions_raw,
            "candidates": candidates,
        }
        return _return_tool_result("search_business_actions", result)

    @function_tool(strict_mode=False)
    def list_retrieved_business_action() -> Mapping[str, Any]:
        """List the business action candidates that have been retrieved so far."""
        _log_tool_call("list_retrieved_business_action")
        result = {
            "status": "ok",
            "actions": list(all_action_candidates_info),
        }
        return _return_tool_result("list_retrieved_business_action", result)

    @function_tool(strict_mode=False)
    def set_workflow_meta(workflow_name: str, description: Optional[str] = None) -> Mapping[str, Any]:
        """Set the workflow name and description.

        Use case: Initialize metadata before planning the workflow structure to keep later
        stages readable.

        Args:
            workflow_name: Workflow name.
            description: Optional workflow description.

        Returns:
            A result dictionary containing the status and operation type.
        """
        _log_tool_call(
            "set_workflow_meta",
            {"workflow_name": workflow_name, "description": description},
        )
        builder.set_meta(workflow_name, description)
        _snapshot("meta_updated")
        result = {"status": "ok", "type": "meta_set"}
        return _return_tool_result("set_workflow_meta", result)

    @function_tool(strict_mode=False)
    def add_action_node(
        id: str,
        action_id: str,
        display_name: Optional[str] = None,
        out_params_schema: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        parent_node_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Add an action node that binds to a retrieved business action.

        Use case: Add concrete business capabilities and connect upstream dependencies during
        structure planning.

        Args:
            id: Unique node identifier.
            action_id: Business action ID that must come from the most recent search
                candidates.
            display_name: Optional node display name.
            out_params_schema: Optional mapping of the action output parameter schema.
            params: Optional dictionary of action input parameters.
            depends_on: Optional list of upstream node IDs.
            parent_node_id: Optional parent node ID for subgraphs or loop scenarios.

        Returns:
            A result dictionary containing the status, type, and node ID; returns an error
            message on failure.
        """
        _log_tool_call(
            "add_action_node",
            {
                "id": id,
                "action_id": action_id,
                "display_name": display_name,
                "parent_node_id": parent_node_id,
            },
        )
        if not all_action_candidates:
            result = _build_validation_error("action 节点必须在调用 search_business_actions 之后创建。")
            return _return_tool_result("add_action_node", result)
        if action_id not in all_action_candidates:
            result = _build_validation_error(
                "action_id 必须是 search_business_actions 返回过的 candidates.id 之一。",
                allowed_action_ids=all_action_candidates,
            )
            return _return_tool_result("add_action_node", result)
        if parent_node_id is not None and not isinstance(parent_node_id, str):
            result = _build_validation_error("parent_node_id 需要是字符串或 null。")
            return _return_tool_result("add_action_node", result)

        cleaned_params, removed_fields = _filter_supported_params(
            node_type="action",
            params=params or {},
            action_schemas=action_schemas,
            action_id=action_id,
        )
        builder.add_node(
            node_id=id,
            node_type="action",
            action_id=action_id,
            display_name=display_name,
            out_params_schema=out_params_schema,
            params=cleaned_params,
            parent_node_id=parent_node_id if isinstance(parent_node_id, str) else None,
            depends_on=depends_on or [],
        )
        removed_node_fields = _sanitize_builder_node_fields(builder, id)
        _snapshot(f"add_action_{id}")
        if removed_fields or removed_node_fields:
            result = _build_validation_error(
                "action 节点仅支持 id/type/action_id/display_name/params/out_params_schema 字段，params 仅支持 arg_schema 字段。",
                removed_param_fields=removed_fields,
                removed_node_fields=removed_node_fields,
                node_id=id,
            )
            return _return_tool_result("add_action_node", result)
        result = {"status": "ok", "type": "node_added", "node_id": id}
        return _return_tool_result("add_action_node", result)

    @function_tool(strict_mode=False)
    def add_loop_node(
        id: str,
        loop_kind: str,
        source: Any,
        item_alias: str,
        display_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        sub_graph_nodes: Optional[List[str]] = None,
        depends_on: Optional[List[str]] = None,
        parent_node_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Add a loop node and bind its loop subgraph.

        Use case: Create a loop structure when iterating over a collection or running a
        ``while`` loop.

        Args:
            id: Unique node identifier.
            loop_kind: Loop type (``for_each``/``while``).
            source: Loop data source (Jinja expression or mapping object).
            item_alias: Alias for each loop item.
            display_name: Optional node display name.
            params: Optional loop parameter dictionary.
            sub_graph_nodes: Optional list of node IDs inside the loop subgraph.
            depends_on: Optional list of upstream node IDs.
            parent_node_id: Optional parent node ID for nested loops.

        Returns:
            A result dictionary containing the status, type, and node ID; returns an error
            message on failure.
        """
        _log_tool_call(
            "add_loop_node",
            {
                "id": id,
                "loop_kind": loop_kind,
                "item_alias": item_alias,
                "parent_node_id": parent_node_id,
            },
        )
        if parent_node_id is not None and not isinstance(parent_node_id, str):
            result = _build_validation_error("parent_node_id 需要是字符串或 null。")
            return _return_tool_result("add_loop_node", result)
        invalid_fields: List[str] = []
        if loop_kind not in {"for_each", "while"}:
            invalid_fields.append("loop_kind")
        if not isinstance(source, (str, Mapping)):
            invalid_fields.append("source")
        if not isinstance(item_alias, str):
            invalid_fields.append("item_alias")
        if invalid_fields:
            result = _build_validation_error(
                "loop 节点需要合法的 loop_kind/source/item_alias 参数。",
                invalid_fields=invalid_fields,
            )
            return _return_tool_result("add_loop_node", result)

        normalized_nodes, sub_graph_error = _normalize_sub_graph_nodes(sub_graph_nodes, builder=builder)
        if sub_graph_error:
            result = {"status": "error", **sub_graph_error}
            return _return_tool_result("add_loop_node", result)

        merged_params = dict(params or {})
        merged_params.update({"loop_kind": loop_kind, "source": source, "item_alias": item_alias})
        cleaned_params, removed_fields = _filter_supported_params(
            node_type="loop", params=merged_params, action_schemas=action_schemas
        )
        builder.add_node(
            node_id=id,
            node_type="loop",
            action_id=None,
            display_name=display_name,
            out_params_schema=None,
            params=cleaned_params,
            parent_node_id=parent_node_id if isinstance(parent_node_id, str) else None,
            depends_on=depends_on or [],
        )
        _attach_sub_graph_nodes(builder, id, normalized_nodes)
        removed_node_fields = _sanitize_builder_node_fields(builder, id)
        _snapshot(f"add_loop_{id}")
        if removed_fields or removed_node_fields:
            result = _build_validation_error(
                "loop 节点的 params 仅支持 loop_kind/source/condition/item_alias/body_subgraph/exports，exports 应使用 {key: Jinja表达式} 结构。",
                removed_fields=removed_fields,
                removed_node_fields=removed_node_fields,
                node_id=id,
            )
            return _return_tool_result("add_loop_node", result)
        result = {"status": "ok", "type": "node_added", "node_id": id}
        return _return_tool_result("add_loop_node", result)

    @function_tool(strict_mode=False)
    def add_condition_node(
        id: str,
        true_to_node: Optional[str],
        false_to_node: Optional[str],
        display_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        depends_on: Optional[List[str]] = None,
        parent_node_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Add a condition node and declare branch targets.

        Use case: Create a condition node when routing to different branches based on an
        expression.

        Args:
            id: Unique node identifier.
            true_to_node: Node ID to jump to when the condition is true, or ``null``.
            false_to_node: Node ID to jump to when the condition is false, or ``null``.
            display_name: Optional node display name.
            params: Condition parameter dictionary that must include ``expression``.
            depends_on: Optional list of upstream node IDs.
            parent_node_id: Optional parent node ID for subgraph or loop scenarios.

        Returns:
            A result dictionary containing the status, type, and node ID; returns an error
            message on failure.
        """
        _log_tool_call(
            "add_condition_node",
            {
                "id": id,
                "true_to_node": true_to_node,
                "false_to_node": false_to_node,
                "parent_node_id": parent_node_id,
            },
        )
        if parent_node_id is not None and not isinstance(parent_node_id, str):
            result = _build_validation_error("parent_node_id 需要是字符串或 null。")
            return _return_tool_result("add_condition_node", result)
        if true_to_node is not None and not isinstance(true_to_node, str):
            result = _build_validation_error("true_to_node 只能是节点 id 或 null。", invalid_fields=["true_to_node"])
            return _return_tool_result("add_condition_node", result)
        if false_to_node is not None and not isinstance(false_to_node, str):
            result = _build_validation_error("false_to_node 只能是节点 id 或 null。", invalid_fields=["false_to_node"])
            return _return_tool_result("add_condition_node", result)
        normalized_params: Dict[str, Any] = dict(params or {})
        expr_val = normalized_params.get("expression")
        if isinstance(expr_val, str):
            fixed_expr, _ = _normalize_template_string(expr_val)
            normalized_params["expression"] = fixed_expr
        if not isinstance(expr_val, str) or not expr_val.strip():
            result = _build_validation_error(
                "condition.params.expression 必须是返回布尔值的 Jinja 表达式。",
                invalid_fields=["expression"],
            )
            return _return_tool_result("add_condition_node", result)

        cleaned_params, removed_fields = _filter_supported_params(
            node_type="condition", params=normalized_params, action_schemas=action_schemas
        )
        builder.add_node(
            node_id=id,
            node_type="condition",
            display_name=display_name,
            params=cleaned_params,
            true_to_node=true_to_node if isinstance(true_to_node, str) else None,
            false_to_node=false_to_node if isinstance(false_to_node, str) else None,
            parent_node_id=parent_node_id if isinstance(parent_node_id, str) else None,
            depends_on=depends_on or [],
        )
        removed_node_fields = _sanitize_builder_node_fields(builder, id)
        _snapshot(f"add_condition_{id}")
        if removed_fields or removed_node_fields:
            result = _build_validation_error(
                "condition 节点的 params 仅支持 expression。",
                removed_fields=removed_fields,
                removed_node_fields=removed_node_fields,
                node_id=id,
            )
            return _return_tool_result("add_condition_node", result)
        result = {"status": "ok", "type": "node_added", "node_id": id}
        return _return_tool_result("add_condition_node", result)

    @function_tool(strict_mode=False)
    def add_switch_node(
        id: str,
        cases: List[Dict[str, Any]],
        display_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        default_to_node: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        parent_node_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Add a switch node that routes according to multi-branch cases.

        Use case: Create a switch node when multiple conditional branches are required.

        Args:
            id: Unique node identifier.
            cases: Branch list where each item contains matching information and ``to_node``.
            display_name: Optional node display name.
            params: Optional switch parameter dictionary (``source``/``field``).
            default_to_node: Default branch node ID or ``null``.
            depends_on: Optional list of upstream node IDs.
            parent_node_id: Optional parent node ID for subgraphs or loop scenarios.

        Returns:
            A result dictionary containing the status, type, and node ID; returns an error
            message on failure.
        """
        _log_tool_call(
            "add_switch_node",
            {
                "id": id,
                "case_count": len(cases) if isinstance(cases, list) else None,
                "default_to_node": default_to_node,
                "parent_node_id": parent_node_id,
            },
        )
        if parent_node_id is not None and not isinstance(parent_node_id, str):
            result = _build_validation_error("parent_node_id 需要是字符串或 null。")
            return _return_tool_result("add_switch_node", result)
        if not isinstance(cases, list):
            result = _build_validation_error("switch 节点需要提供 cases 数组。")
            return _return_tool_result("add_switch_node", result)
        normalized_cases: list[Dict[str, Any]] = []
        invalid_case_indices: list[int] = []
        for idx, case in enumerate(cases):
            if not isinstance(case, Mapping):
                invalid_case_indices.append(idx)
                continue
            to_node = case.get("to_node") if "to_node" in case else None
            if to_node is not None and not isinstance(to_node, str):
                invalid_case_indices.append(idx)
                continue
            normalized_cases.append(dict(case))
        if invalid_case_indices:
            result = _build_validation_error(
                "cases 中的 to_node 需要是字符串或 null。",
                invalid_case_indices=invalid_case_indices,
            )
            return _return_tool_result("add_switch_node", result)
        if default_to_node is not None and not isinstance(default_to_node, str):
            result = _build_validation_error(
                "default_to_node 需要是字符串或 null。",
                invalid_fields=["default_to_node"],
            )
            return _return_tool_result("add_switch_node", result)
        if params is not None and not isinstance(params, Mapping):
            result = _build_validation_error("switch 节点的 params 需要是对象。")
            return _return_tool_result("add_switch_node", result)

        cleaned_params, removed_fields = _filter_supported_params(
            node_type="switch", params=params or {}, action_schemas=action_schemas
        )
        builder.add_node(
            node_id=id,
            node_type="switch",
            display_name=display_name,
            params=cleaned_params,
            cases=normalized_cases,
            default_to_node=default_to_node if isinstance(default_to_node, str) else None,
            parent_node_id=parent_node_id if isinstance(parent_node_id, str) else None,
            depends_on=depends_on or [],
        )
        removed_node_fields = _sanitize_builder_node_fields(builder, id)
        _snapshot(f"add_switch_{id}")
        if removed_fields or removed_node_fields:
            result = _build_validation_error(
                "switch 节点的 params 仅支持 source/field 字段。",
                removed_fields=removed_fields,
                removed_node_fields=removed_node_fields,
                node_id=id,
            )
            return _return_tool_result("add_switch_node", result)
        result = {"status": "ok", "type": "node_added", "node_id": id}
        return _return_tool_result("add_switch_node", result)

    def _update_node_common(node_id: str, expected_type: str) -> Optional[Dict[str, Any]]:
        if not isinstance(node_id, str):
            return _build_validation_error(f"更新 {expected_type} 需要提供字符串类型的 id。")
        if node_id not in builder.nodes:
            return _build_validation_error(f"节点 {node_id} 尚未创建，无法更新。")
        if builder.nodes.get(node_id, {}).get("type") != expected_type:
            return _build_validation_error(f"节点 {node_id} 类型不是 {expected_type}。")
        return None

    @function_tool(strict_mode=False)
    def update_action_node(
        id: str,
        display_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        out_params_schema: Optional[Dict[str, Any]] = None,
        action_id: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        parent_node_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Update the properties or parameters of an existing action node.

        Use case: Add display names, parameters, or reselect an action for an existing action
        node.

        Args:
            id: Unique node identifier.
            display_name: Optional node display name.
            params: Optional action parameter dictionary.
            out_params_schema: Optional output parameter schema.
            action_id: Optional business action ID (must come from the most recent search
                candidates).
            depends_on: Optional list of upstream node IDs.
            parent_node_id: Optional parent node ID.

        Returns:
            A result dictionary containing the status, type, and node ID; returns an error
            message on failure.
        """
        _log_tool_call(
            "update_action_node",
            {"id": id, "action_id": action_id, "parent_node_id": parent_node_id},
        )
        precheck = _update_node_common(id, "action")
        if precheck:
            return _return_tool_result("update_action_node", precheck)
        if parent_node_id is not None and not isinstance(parent_node_id, str):
            result = _build_validation_error("parent_node_id 需要是字符串或 null。")
            return _return_tool_result("update_action_node", result)
        if action_id is not None:
            if not all_action_candidates:
                result = _build_validation_error("更新 action_id 前请先调用 search_business_actions。")
                return _return_tool_result("update_action_node", result)
            if action_id not in all_action_candidates:
                result = _build_validation_error(
                    "action_id 必须是 search_business_actions 返回过的 candidates.id 之一。",
                    allowed_action_ids=all_action_candidates,
                )
                return _return_tool_result("update_action_node", result)
        if params is not None and not isinstance(params, Mapping):
            result = _build_validation_error("action 节点的 params 需要是对象。")
            return _return_tool_result("update_action_node", result)

        updates: Dict[str, Any] = {}
        if display_name is not None:
            updates["display_name"] = display_name
        if out_params_schema is not None:
            updates["out_params_schema"] = out_params_schema
        if action_id is not None:
            updates["action_id"] = action_id
        if parent_node_id is not None:
            updates["parent_node_id"] = parent_node_id
        if params is not None:
            cleaned_params, removed_param_fields = _filter_supported_params(
                node_type="action",
                params=params or {},
                action_schemas=action_schemas,
                action_id=action_id if isinstance(action_id, str) else builder.nodes.get(id, {}).get("action_id"),
            )
            updates["params"] = cleaned_params
        else:
            removed_param_fields = []

        if depends_on is not None:
            updates["depends_on"] = depends_on

        builder.update_node(id, **updates)
        removed_param_fields.extend(_sanitize_builder_node_params(builder, id, action_schemas))
        removed_node_fields = _sanitize_builder_node_fields(builder, id)
        _snapshot(f"update_action_{id}")
        if removed_param_fields or removed_node_fields:
            result = _build_validation_error(
                "action 节点仅支持 id/type/action_id/display_name/params/out_params_schema 字段，params 仅支持 arg_schema 字段。",
                removed_param_fields=removed_param_fields,
                removed_node_fields=removed_node_fields,
                node_id=id,
            )
            return _return_tool_result("update_action_node", result)
        result = {"status": "ok", "type": "node_updated", "node_id": id}
        return _return_tool_result("update_action_node", result)

    @function_tool(strict_mode=False)
    def update_condition_node(
        id: str,
        display_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        true_to_node: Optional[str] = None,
        false_to_node: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        parent_node_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Update the expression or branch targets of an existing condition node.

        Use case: Fix the condition expression or adjust the true/false jump targets.

        Args:
            id: Unique node identifier.
            display_name: Optional node display name.
            params: Optional condition parameter dictionary (``expression``).
            true_to_node: Node ID to jump to when the condition is true, or ``null``.
            false_to_node: Node ID to jump to when the condition is false, or ``null``.
            depends_on: Optional list of upstream node IDs.
            parent_node_id: Optional parent node ID.

        Returns:
            A result dictionary containing the status, type, and node ID; returns an error
            message on failure.
        """
        _log_tool_call(
            "update_condition_node",
            {
                "id": id,
                "true_to_node": true_to_node,
                "false_to_node": false_to_node,
                "parent_node_id": parent_node_id,
            },
        )
        precheck = _update_node_common(id, "condition")
        if precheck:
            return _return_tool_result("update_condition_node", precheck)
        if parent_node_id is not None and not isinstance(parent_node_id, str):
            result = _build_validation_error("parent_node_id 需要是字符串或 null。")
            return _return_tool_result("update_condition_node", result)
        if true_to_node is not None and not isinstance(true_to_node, str):
            result = _build_validation_error("true_to_node 只能是节点 id 或 null。", invalid_fields=["true_to_node"])
            return _return_tool_result("update_condition_node", result)
        if false_to_node is not None and not isinstance(false_to_node, str):
            result = _build_validation_error("false_to_node 只能是节点 id 或 null。", invalid_fields=["false_to_node"])
            return _return_tool_result("update_condition_node", result)
        if params is not None and not isinstance(params, Mapping):
            result = _build_validation_error("condition 节点的 params 需要是对象。")
            return _return_tool_result("update_condition_node", result)

        updates: Dict[str, Any] = {}
        if display_name is not None:
            updates["display_name"] = display_name
        if true_to_node is not None:
            updates["true_to_node"] = true_to_node
        if false_to_node is not None:
            updates["false_to_node"] = false_to_node
        if parent_node_id is not None:
            updates["parent_node_id"] = parent_node_id
        if params is not None:
            normalized_params = dict(params or {})
            expr_val = normalized_params.get("expression")
            if not isinstance(expr_val, str) or not expr_val.strip():
                result = _build_validation_error(
                    "condition 节点的 params.expression 必须是返回布尔值的 Jinja 表达式。",
                    invalid_fields=["expression"],
                )
                return _return_tool_result("update_condition_node", result)
            cleaned_params, removed_param_fields = _filter_supported_params(
                node_type="condition", params=normalized_params, action_schemas=action_schemas
            )
            updates["params"] = cleaned_params
        else:
            removed_param_fields = []
        if depends_on is not None:
            updates["depends_on"] = depends_on

        builder.update_node(id, **updates)
        removed_param_fields.extend(_sanitize_builder_node_params(builder, id, action_schemas))
        removed_node_fields = _sanitize_builder_node_fields(builder, id)
        _snapshot(f"update_condition_{id}")
        if removed_param_fields or removed_node_fields:
            result = _build_validation_error(
                "condition 节点仅支持 id/type/display_name/params/true_to_node/false_to_node 字段，params 仅支持 expression。",
                removed_fields=removed_param_fields,
                removed_node_fields=removed_node_fields,
                node_id=id,
            )
            return _return_tool_result("update_condition_node", result)
        result = {"status": "ok", "type": "node_updated", "node_id": id}
        return _return_tool_result("update_condition_node", result)

    @function_tool(strict_mode=False)
    def update_switch_node(
        id: str,
        display_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        cases: Optional[List[Dict[str, Any]]] = None,
        default_to_node: Optional[str] = None,
        depends_on: Optional[List[str]] = None,
        parent_node_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Update the cases or default branch of an existing switch node.

        Use case: Add or correct multi-branch routing rules or the default jump target.

        Args:
            id: Unique node identifier.
            display_name: Optional node display name.
            params: Optional switch parameter dictionary (``source``/``field``).
            cases: Optional list of cases.
            default_to_node: Default branch node ID or ``null``.
            depends_on: Optional list of upstream node IDs.
            parent_node_id: Optional parent node ID.

        Returns:
            A result dictionary containing the status, type, and node ID; returns an error
            message on failure.
        """
        _log_tool_call(
            "update_switch_node",
            {
                "id": id,
                "case_count": len(cases) if isinstance(cases, list) else None,
                "default_to_node": default_to_node,
                "parent_node_id": parent_node_id,
            },
        )
        precheck = _update_node_common(id, "switch")
        if precheck:
            return _return_tool_result("update_switch_node", precheck)
        if parent_node_id is not None and not isinstance(parent_node_id, str):
            result = _build_validation_error("parent_node_id 需要是字符串或 null。")
            return _return_tool_result("update_switch_node", result)
        if default_to_node is not None and not isinstance(default_to_node, str):
            result = _build_validation_error(
                "default_to_node 只能是节点 id 或 null。",
                invalid_fields=["default_to_node"],
            )
            return _return_tool_result("update_switch_node", result)
        if params is not None and not isinstance(params, Mapping):
            result = _build_validation_error("switch 节点的 params 需要是对象。")
            return _return_tool_result("update_switch_node", result)

        normalized_cases: list[Dict[str, Any]] = []
        invalid_case_indices: list[int] = []
        if cases is not None:
            if not isinstance(cases, list):
                result = _build_validation_error("switch 的 cases 需要是数组。")
                return _return_tool_result("update_switch_node", result)
            for idx, case in enumerate(cases):
                if not isinstance(case, Mapping):
                    invalid_case_indices.append(idx)
                    continue
                to_node = case.get("to_node") if "to_node" in case else None
                if to_node is not None and not isinstance(to_node, str):
                    invalid_case_indices.append(idx)
                    continue
                normalized_cases.append(dict(case))
        if invalid_case_indices:
            result = _build_validation_error(
                "cases 中的 to_node 需要是字符串或 null。",
                invalid_case_indices=invalid_case_indices,
            )
            return _return_tool_result("update_switch_node", result)

        updates: Dict[str, Any] = {}
        if display_name is not None:
            updates["display_name"] = display_name
        if parent_node_id is not None:
            updates["parent_node_id"] = parent_node_id
        if default_to_node is not None:
            updates["default_to_node"] = default_to_node
        if cases is not None:
            updates["cases"] = normalized_cases
        if params is not None:
            cleaned_params, removed_param_fields = _filter_supported_params(
                node_type="switch", params=params or {}, action_schemas=action_schemas
            )
            updates["params"] = cleaned_params
        else:
            removed_param_fields = []
        if depends_on is not None:
            updates["depends_on"] = depends_on

        builder.update_node(id, **updates)
        removed_param_fields.extend(_sanitize_builder_node_params(builder, id, action_schemas))
        removed_node_fields = _sanitize_builder_node_fields(builder, id)
        _snapshot(f"update_switch_{id}")
        if removed_param_fields or removed_node_fields:
            result = _build_validation_error(
                "switch 节点仅支持 id/type/display_name/params/cases/default_to_node 字段，params 仅支持 source/field。",
                removed_fields=removed_param_fields,
                removed_node_fields=removed_node_fields,
                node_id=id,
            )
            return _return_tool_result("update_switch_node", result)
        result = {"status": "ok", "type": "node_updated", "node_id": id}
        return _return_tool_result("update_switch_node", result)

    @function_tool(strict_mode=False)
    def update_loop_node(
        id: str,
        display_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        sub_graph_nodes: Optional[List[str]] = None,
        depends_on: Optional[List[str]] = None,
        parent_node_id: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Update the parameters or subgraph of an existing loop node.

        Use case: Correct loop parameters, the set of subgraph nodes, or dependencies.

        Args:
            id: Unique node identifier.
            display_name: Optional node display name.
            params: Optional loop parameter dictionary.
            sub_graph_nodes: Optional list of node IDs inside the loop subgraph.
            depends_on: Optional list of upstream node IDs.
            parent_node_id: Optional parent node ID.

        Returns:
            A result dictionary containing the status, type, and node ID; returns an error
            message on failure.
        """
        _log_tool_call(
            "update_loop_node",
            {"id": id, "parent_node_id": parent_node_id},
        )
        precheck = _update_node_common(id, "loop")
        if precheck:
            return _return_tool_result("update_loop_node", precheck)
        if parent_node_id is not None and not isinstance(parent_node_id, str):
            result = _build_validation_error("parent_node_id 需要是字符串或 null。")
            return _return_tool_result("update_loop_node", result)
        if params is not None and not isinstance(params, Mapping):
            result = _build_validation_error("loop 节点的 params 需要是对象。")
            return _return_tool_result("update_loop_node", result)

        normalized_nodes, sub_graph_error = _normalize_sub_graph_nodes(sub_graph_nodes, builder=builder)
        if sub_graph_error:
            result = {"status": "error", **sub_graph_error}
            return _return_tool_result("update_loop_node", result)

        updates: Dict[str, Any] = {}
        if display_name is not None:
            updates["display_name"] = display_name
        if parent_node_id is not None:
            updates["parent_node_id"] = parent_node_id
        if params is not None:
            cleaned_params, removed_param_fields = _filter_supported_params(
                node_type="loop", params=params or {}, action_schemas=action_schemas
            )
            updates["params"] = cleaned_params
        else:
            removed_param_fields = []
        if depends_on is not None:
            updates["depends_on"] = depends_on

        builder.update_node(id, **updates)
        _attach_sub_graph_nodes(builder, id, normalized_nodes)
        removed_param_fields.extend(_sanitize_builder_node_params(builder, id, action_schemas))
        removed_node_fields = _sanitize_builder_node_fields(builder, id)
        _snapshot(f"update_loop_{id}")
        if removed_param_fields or removed_node_fields:
            result = _build_validation_error(
                "loop 节点仅支持 id/type/display_name/params 字段，params 仅支持 loop_kind/source/condition/item_alias/body_subgraph/exports，exports 应使用 {key: Jinja表达式} 结构。",
                removed_fields=removed_param_fields,
                removed_node_fields=removed_node_fields,
                node_id=id,
            )
            return _return_tool_result("update_loop_node", result)
        result = {"status": "ok", "type": "node_updated", "node_id": id}
        return _return_tool_result("update_loop_node", result)

    @function_tool(strict_mode=False)
    def dump_model() -> Mapping[str, Any]:
        """Export the current workflow snapshot for debugging or display.

        Use case: Invoke this when you need to inspect the current builder state.

        Returns:
            A result dictionary containing node/edge statistics and the workflow snapshot.
        """
        _log_tool_call("dump_model")
        snapshot = _snapshot("dump_model")
        result = {
            "status": "ok",
            "type": "dump_model",
            "summary": {
                "node_count": len(snapshot.get("nodes") or []),
                "edge_count": len(snapshot.get("edges") or []),
            },
            "workflow": snapshot,
        }
        return _return_tool_result("dump_model", result)

    def _collect_template_node_refs(obj: Any) -> set[str]:
        refs: set[str] = set()

        def _walk(val: Any) -> None:
            if isinstance(val, Mapping):
                for item in val.values():
                    _walk(item)
            elif isinstance(val, list):
                for item in val:
                    _walk(item)
            elif isinstance(val, str):
                for expr in _iter_template_references(val):
                    for match in re.findall(r"result_of\.([A-Za-z0-9_-]+)", expr):
                        refs.add(match)

        _walk(obj)
        return refs

    @function_tool(strict_mode=False)
    def check_workflow() -> Mapping[str, Any]:
        """Check whether action nodes reference upstream nodes via Jinja templates."""
        _log_tool_call("check_workflow")
        snapshot = _attach_inferred_edges(builder.to_workflow())
        nodes = list(iter_workflow_and_loop_body_nodes(snapshot))
        edges = infer_edges_from_bindings(nodes)
        node_ids = {
            node.get("id")
            for node in nodes
            if isinstance(node, Mapping) and isinstance(node.get("id"), str)
        }
        incoming_counts: Dict[str, int] = {nid: 0 for nid in node_ids if isinstance(nid, str)}
        for edge in edges:
            if not isinstance(edge, Mapping):
                continue
            to_node = edge.get("to") if "to" in edge else edge.get("to_node")
            if isinstance(to_node, str) and to_node in incoming_counts:
                incoming_counts[to_node] += 1

        nodes_without_refs: List[str] = []
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            if node.get("type") != "action":
                continue
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            has_incoming = incoming_counts.get(node_id, 0) > 0
            param_refs = _collect_template_node_refs(node.get("params", {}))
            has_param_refs = any(
                ref in node_ids and ref != node_id for ref in param_refs if isinstance(ref, str)
            )
            if not has_incoming and not has_param_refs:
                nodes_without_refs.append(node_id)

        has_issues = bool(nodes_without_refs)
        status = "needs_more_work" if has_issues else "ok"
        if has_issues:
            feedback = (
                "Some action nodes have no upstream references. "
                f"Update params to reference upstream outputs for nodes: {', '.join(nodes_without_refs)}."
            )
        else:
            feedback = "All action nodes reference upstream outputs."

        result = {
            "status": status,
            "type": "check_workflow",
            "nodes_without_references": nodes_without_refs,
            "has_issues": has_issues,
            "feedback": feedback,
            "workflow": snapshot,
        }
        return _return_tool_result("check_workflow", result)

    filled_params: Dict[str, Dict[str, Any]] = {
        nid: copy.deepcopy(node.get("params", {}))
        for nid, node in builder.nodes.items()
        if isinstance(node, Mapping)
    }
    validated_node_ids: List[str] = []

    def _build_workflow_for_params() -> Workflow:
        workflow_dict = _attach_inferred_edges(builder.to_workflow())
        return Workflow.model_validate(workflow_dict)

    def _build_param_context(node_id: str) -> Dict[str, Any]:
        workflow = _build_workflow_for_params()
        nodes_by_id = {n.id: n for n in workflow.nodes}
        node = nodes_by_id.get(node_id)
        if not node:
            return {}

        upstream_nodes = get_referenced_nodes(workflow, node_id)
        allowed_node_ids = [n.id for n in upstream_nodes]
        binding_memory = _build_binding_memory(filled_params, validated_node_ids)
        global_context = _build_global_context(
            workflow=workflow,
            action_schemas=action_schemas,
            filled_params=filled_params,
            processed_node_ids=validated_node_ids,
            binding_memory=binding_memory,
        )
        upstream_context = []
        for n in upstream_nodes:
            action_schema = action_schemas.get(n.action_id, {}) if n.action_id else {}
            upstream_context.append(
                {
                    "id": n.id,
                    "type": n.type,
                    "action_id": n.action_id,
                    "output_schema": action_schema.get("output_schema"),
                    "params": filled_params.get(n.id, n.params),
                }
            )
        target_action_schema = action_schemas.get(node.action_id, {}) if node.action_id else {}
        return {
            "target_node": {
                "id": node.id,
                "type": node.type,
                "action_id": node.action_id,
                "display_name": node.display_name,
                "existing_params": filled_params.get(node.id, node.params),
            },
            "arg_schema": target_action_schema.get("arg_schema"),
            "allowed_node_ids": allowed_node_ids,
            "allowed_upstream_nodes": upstream_context,
            "global_context": global_context,
        }

    @function_tool(strict_mode=False)
    def get_param_context(id: str) -> Mapping[str, Any]:
        """Get the parameter-completion context for a specific node."""
        _log_tool_call("get_param_context", {"id": id})
        if not isinstance(id, str) or id not in builder.nodes:
            result = {"status": "error", "message": "节点不存在，无法获取参数上下文。"}
            return _return_tool_result("get_param_context", result)
        payload = _build_param_context(id)
        if not payload:
            result = {"status": "error", "message": "无法生成参数上下文。"}
            return _return_tool_result("get_param_context", result)
        result = {"status": "ok", "payload": payload}
        return _return_tool_result("get_param_context", result)

    @function_tool(strict_mode=False)
    def update_node_params(id: str, params: Dict[str, Any]) -> Mapping[str, Any]:
        """Update and validate parameters for the specified node."""
        _log_tool_call("update_node_params", {"id": id})
        if not isinstance(id, str) or id not in builder.nodes:
            result = {"status": "error", "message": "节点不存在，无法更新参数。"}
            return _return_tool_result("update_node_params", result)
        if not isinstance(params, Mapping):
            result = {"status": "error", "message": "params 需要是对象。"}
            return _return_tool_result("update_node_params", result)

        normalized_params, _ = _normalize_params_templates(dict(params))

        workflow = _build_workflow_for_params()
        nodes_by_id = {n.id: n for n in workflow.nodes}
        node = nodes_by_id.get(id)
        if not node:
            result = {"status": "error", "errors": ["节点不存在于当前 workflow。"]}
            return _return_tool_result("update_node_params", result)

        upstream_nodes = get_referenced_nodes(workflow, id)
        binding_memory = _build_binding_memory(filled_params, validated_node_ids)
        errors = _validate_node_params(
            node=node,
            params=dict(normalized_params),
            upstream_nodes=upstream_nodes,
            action_schemas=action_schemas,
            binding_memory=binding_memory,
        )

        if errors:
            result = {
                "status": "error",
                "errors": errors,
                "hint": "参数未通过校验，未更新 workflow，请根据错误修正后再提交。",
            }
            return _return_tool_result("update_node_params", result)

        filled_params[id] = dict(normalized_params)
        if id not in validated_node_ids:
            validated_node_ids.append(id)
        builder.update_node(id, params=dict(normalized_params))
        _snapshot(f"validate_params_{id}")
        result = {"status": "ok", "params": normalized_params}
        return _return_tool_result("update_node_params", result)
    
    agent = Agent(
        name="WorkflowStructurePlanner",
        instructions=system_prompt,
        tools=[
            search_business_actions,
            list_retrieved_business_action,
            set_workflow_meta,
            add_action_node,
            add_loop_node,
            add_condition_node,
            add_switch_node,
            update_action_node,
            update_condition_node,
            update_switch_node,
            update_loop_node,
            get_param_context,
            update_node_params,
            check_workflow,
            dump_model,
        ],
        model=OPENAI_MODEL
    )

    log_section("结构规划 - Agent SDK")

    def _format_prompt_item(item: Any) -> Any:
        """Ensure the initial Agent SDK prompt is role-tagged for chat models."""

        if isinstance(item, Mapping) and "role" not in item:
            return {
                "role": "user",
                "content": json.dumps(item, ensure_ascii=False),
            }
        return item

    def _run_agent(prompt: Any) -> None:
        base_prompt = prompt if isinstance(prompt, list) else [prompt]
        initial_prompt = [_format_prompt_item(item) for item in base_prompt]
        try:
            Runner.run_sync(agent, initial_prompt, max_turns=max_rounds)  # type: ignore[arg-type]
        except TypeError:
            coro = Runner.run(agent, initial_prompt)  # type: ignore[call-arg]
            if asyncio.iscoroutine(coro):
                asyncio.run(coro)

    _run_agent(
        {"user_requirements": parsed_requirement, "existing_workflow": existing_workflow or {}}
    )
    if not latest_skeleton:
        latest_skeleton = _prepare_skeleton_for_next_stage(
            builder=builder, action_registry=action_registry, search_service=search_service
        )

    return latest_skeleton


__all__ = ["plan_workflow_structure_with_llm"]
