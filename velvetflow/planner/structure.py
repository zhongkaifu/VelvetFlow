# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Structure planning logic for the planner."""

import asyncio
import copy
import json
import os
from collections import deque
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_debug, log_event, log_info, log_section, log_warn
from velvetflow.planner.agent_runtime import Agent, Runner, function_tool
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.workflow_builder import (
    WorkflowBuilder,
    attach_condition_branches,
)
from velvetflow.loop_dsl import iter_workflow_and_loop_body_nodes
from velvetflow.search import HybridActionSearchService
from velvetflow.models import ALLOWED_PARAM_AGGREGATORS, Node, Workflow, infer_edges_from_bindings
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
    """Placeholder for legacy API; returns exports unchanged."""

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
    structure_prompt = """
You are a general business workflow orchestration assistant.
First break down the user's natural-language requirement into a structured checklist and call plan_user_requirement to save it.
The breakdown must follow:
{
  "requirements": [
    {
      "description": "Task description",
      "intent": "The intention of this task",
      "inputs": ["A list of input of this task"],
      "constraints": ["the list of constraints that this task must obey"]
    }
  ],
  "assumptions": ["The assumptions of user's requirement"]
}
Call get_user_requirement whenever you need to review the breakdown.
[Workflow DSL syntax and semantics (must be followed)]
- workflow = {workflow_name, description, nodes: []}. Return valid JSON only (edges are auto-inferred from bindings; do not generate them).
- Node structure: {id, type, display_name, params, depends_on, action_id?, out_params_schema?, loop/subgraph/branches?}.
  Allowed types: action/condition/loop/parallel. No start/end/exit nodes are needed.
  Action nodes must include action_id (from the registry) and params; only action nodes may carry out_params_schema.
  Condition node params may only contain expression (a single Jinja expression returning a boolean); true_to_node/false_to_node must be top-level fields (string or null), not inside params.
  Loop nodes include loop_kind/iter/source/body_subgraph/exports. Each exports value must reference a body_subgraph node field (e.g. {{ result_of.node.field }}). Outside the loop, only exports.<key> can be referenced. body_subgraph only needs a nodes array—no entry/exit/edges.
  Parallel nodes require branches as a non-empty array; each item includes id/entry_node/sub_graph_nodes.
- Inside params, directly use Jinja expressions to reference upstream results (e.g. {{ result_of.<node_id>.<field_path> }} or {{ loop.item.xxx }}). The legacy __from__/__agg__ DSL is no longer allowed.
  Every params value you output during planning (including condition/switch/loop nodes) will be parsed by the Jinja engine. Any non-Jinja syntax will be treated as an error and trigger auto-repair, so strictly follow Jinja templates.
  Each <node_id> must exist and fields must align with the upstream output_schema or loop.exports.
The system has an Action Registry with many business actions. You may only query it via search_business_actions.
Build order:
1) First call plan_user_requirement to save the requirement breakdown.
2) Use set_workflow_meta to set the workflow name and description.
2) When business actions are needed, call search_business_actions first; add_action_node.action_id must come from the latest candidates.id.
3) For condition/switch/loop nodes, call add_condition_node/add_switch_node/add_loop_node first. Expressions and references in params must strictly follow Jinja templating.
4) Call update_node_params to fill and validate params for created nodes.
6) To modify existing nodes (display_name/params/branches/parent_node, etc.), call update_action_node or update_condition_node with the fields to overwrite; then check whether related upstream/downstream nodes also need updates to stay consistent.
7) Condition nodes must explicitly set true_to_node and false_to_node. Values can be node ids (continue execution) or null (branch ends). Express dependencies via params bindings; explicit edges are unnecessary.
8) Maintain depends_on (string array) for each node to list direct upstream dependencies. When a node is referenced by condition.true_to_node/false_to_node, include that condition node in the target's depends_on.
9) After the structure is complete, continue filling params for every node and ensure update_node_params validation passes.

Important note: Only action nodes need out_params_schema; condition nodes do not. out_params_schema should be {"field_name": "type"}, listing only output field names and types without extra descriptions or examples.

[Critical principles]
1. Design the workflow strictly around the current natural-language requirement, covering:
   - Triggering (schedule / event / manual)
   - Data query/read
   - Filter/selection conditions
   - Aggregation/statistics/summarization
   - Notification / write / persistence / downstream calls
2. Loop node data can only be exposed via loop.exports. Downstream references must use result_of.<loop_id>.exports.<key>; do not directly reference body subgraph nodes.
3. Define loop.exports under params.exports, not inside body_subgraph. body_subgraph only contains the nodes array—no entry/exit/edges.
4. Nested loops are allowed, but use parent_node_id or sub_graph_nodes to include child loops within the parent's body_subgraph; downstream nodes must still read loop data via loop.exports instead of pointing directly to subgraph nodes.

Ensure the workflow structure covers every sub-task in the user's requirement—not just the first half.
For example, if the requirement includes trigger + query + filter + summarize + notify, you must include nodes and data flows for filtering, summarizing, and notifying.
After the structure is complete, confirm that the entire workflow covers all sub-tasks.
"""
    param_prompt = """
[Parameter completion rules]
You are a workflow parameter completion assistant. Handle one node at a time and fill the current node's params according to the provided arg_schema.
Submit and validate via tools: call update_node_params to submit completions and validate. If validation fails, analyze and resubmit.
When a field needs another node's output, write it directly as a Jinja expression or template string such as {{ result_of.<node_id>.<field> }}.
node_id must come from allowed_node_ids and the field must exist in that node's output_schema. Do not use the legacy __from__/__agg__ binding DSL.
All params values will be parsed by the Jinja2 engine. Any non-Jinja syntax (including leftover binding objects or pseudo-code) will be treated as an error and trigger auto-repair, so output only strings or literals that Jinja can render.
In loops, you may use loop.item/loop.index and fields exposed by loop.exports.*, still via Jinja expressions; do not reference loop body nodes directly.
When referencing loop.exports, point to a specific field or sub-structure inside exports—do not stop at result_of.<loop_id>.exports.
Each key in loop.exports collects results from every iteration into a list. Each exports value must reference a body_subgraph node field, e.g., {{ result_of.<body_node>.<field> }}.
Params for all nodes must not be empty. If no obvious defaults exist, analyze upstream/downstream context and use the tools to fill them.
Use Jinja filters or expressions for aggregation/filtering/formatting (e.g., {{ result_of.a.items | selectattr('score', '>', 80) | list | length }} or {{ result_of.a.items | map(attribute='name') | join(', ') }}); do not output objects containing __agg__.
"""
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
    nl_requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    max_rounds: int = 100,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
    return_requirement: bool = False,
) -> Dict[str, Any] | tuple[Dict[str, Any], Dict[str, Any] | None]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("请先设置 OPENAI_API_KEY 环境变量再进行结构规划。")

    builder = WorkflowBuilder()
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

    parsed_requirement: Dict[str, Any] | None = None

    @function_tool(strict_mode=False)
    def plan_user_requirement(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        """保存解析后的用户需求拆解，供后续规划与校验。

        payload 格式必须为：
        {
          "requirements": [
            {
              "description": "...",
              "intent": "...",
              "inputs": ["..."],
              "constraints": ["..."]
            }
          ],
          "assumptions": ["..."]
        }
        """
        nonlocal parsed_requirement
        _log_tool_call("plan_user_requirement", payload)
        requirements = payload.get("requirements")
        assumptions = payload.get("assumptions")
        if not isinstance(requirements, list):
            result = {
                "status": "error",
                "message": "requirements 必须是列表。",
            }
            return _return_tool_result("plan_user_requirement", result)
        if assumptions is not None and not isinstance(assumptions, list):
            result = {
                "status": "error",
                "message": "assumptions 必须是字符串列表。",
            }
            return _return_tool_result("plan_user_requirement", result)
        for idx, item in enumerate(requirements):
            if not isinstance(item, Mapping):
                result = {
                    "status": "error",
                    "message": f"requirements[{idx}] 必须是对象。",
                }
                return _return_tool_result("plan_user_requirement", result)
            for key in ("description", "intent", "inputs", "constraints"):
                if key not in item:
                    result = {
                        "status": "error",
                        "message": f"requirements[{idx}] 缺少字段 {key}。",
                    }
                    return _return_tool_result("plan_user_requirement", result)
        normalized_payload = {
            "requirements": list(requirements),
            "assumptions": list(assumptions or []),
        }
        parsed_requirement = normalized_payload
        result = {"status": "ok", "requirement": normalized_payload}
        return _return_tool_result("plan_user_requirement", result)

    @function_tool(strict_mode=False)
    def get_user_requirement() -> Mapping[str, Any]:
        """获取已保存的用户需求拆解，便于回顾与校验."""
        _log_tool_call("get_user_requirement")
        if not parsed_requirement:
            result = {
                "status": "error",
                "message": "尚未保存需求拆解，请先调用 plan_user_requirement。",
            }
            return _return_tool_result("get_user_requirement", result)
        result = {"status": "ok", "requirement": parsed_requirement}
        return _return_tool_result("get_user_requirement", result)

    @function_tool(strict_mode=False)
    def search_business_actions(query: str, top_k: int = 5) -> Mapping[str, Any]:
        """检索业务动作候选，用于规划时选择合法的 action_id。

        适用场景：在创建/更新 action 节点之前，先查询可用动作列表。

        Args:
            query: 检索关键词，描述业务动作能力或名称。
            top_k: 返回候选数量上限。

        Returns:
            包含检索状态、原始动作列表与精简候选信息的结果字典。
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
        """列出当前已获取的业务动作候选信息。"""
        _log_tool_call("list_retrieved_business_action")
        result = {
            "status": "ok",
            "actions": list(all_action_candidates_info),
        }
        return _return_tool_result("list_retrieved_business_action", result)

    @function_tool(strict_mode=False)
    def set_workflow_meta(workflow_name: str, description: Optional[str] = None) -> Mapping[str, Any]:
        """设置工作流的名称与描述。

        适用场景：在规划工作流结构前初始化元信息，便于后续阶段阅读。

        Args:
            workflow_name: 工作流名称。
            description: 工作流描述，可选。

        Returns:
            包含状态与操作类型的结果字典。
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
        """添加 action 节点，绑定已检索到的业务动作。

        适用场景：在结构规划阶段加入具体业务能力，并挂接上游依赖。

        Args:
            id: 节点唯一标识。
            action_id: 业务动作 ID，必须来自最近一次检索候选。
            display_name: 节点展示名称，可选。
            out_params_schema: action 输出参数 schema 映射，可选。
            params: action 输入参数字典，可选。
            depends_on: 直接依赖的上游节点 ID 列表，可选。
            parent_node_id: 父节点 ID（用于子图/循环场景），可选。

        Returns:
            包含状态、类型与节点 ID 的结果字典；失败时返回错误信息。
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
        """添加 loop 节点并绑定循环子图。

        适用场景：需要对集合进行遍历或 while 循环时，创建 loop 结构。

        Args:
            id: 节点唯一标识。
            loop_kind: 循环类型（for_each/while）。
            source: 循环数据源（Jinja 表达式或映射对象）。
            item_alias: 循环项别名。
            display_name: 节点展示名称，可选。
            params: loop 参数字典，可选。
            sub_graph_nodes: 循环子图节点 ID 列表，可选。
            depends_on: 直接依赖的上游节点 ID 列表，可选。
            parent_node_id: 父节点 ID（用于嵌套循环），可选。

        Returns:
            包含状态、类型与节点 ID 的结果字典；失败时返回错误信息。
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
        """添加 condition 节点并声明分支走向。

        适用场景：需要根据表达式判断走向不同分支时创建条件节点。

        Args:
            id: 节点唯一标识。
            true_to_node: 条件为真时跳转的节点 ID 或 null。
            false_to_node: 条件为假时跳转的节点 ID 或 null。
            display_name: 节点展示名称，可选。
            params: 条件参数字典，需包含 expression。
            depends_on: 直接依赖的上游节点 ID 列表，可选。
            parent_node_id: 父节点 ID（用于子图/循环场景），可选。

        Returns:
            包含状态、类型与节点 ID 的结果字典；失败时返回错误信息。
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
        """添加 switch 节点，按多分支 case 路由。

        适用场景：需要多条件分支跳转时创建 switch 节点。

        Args:
            id: 节点唯一标识。
            cases: 分支列表，每个元素包含匹配信息与 to_node。
            display_name: 节点展示名称，可选。
            params: switch 参数字典（source/field），可选。
            default_to_node: 默认分支节点 ID 或 null。
            depends_on: 直接依赖的上游节点 ID 列表，可选。
            parent_node_id: 父节点 ID（用于子图/循环场景），可选。

        Returns:
            包含状态、类型与节点 ID 的结果字典；失败时返回错误信息。
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
        """更新已有 action 节点的属性或参数。

        适用场景：对已创建的 action 节点补充显示名、参数或重选动作。

        Args:
            id: 节点唯一标识。
            display_name: 节点展示名称，可选。
            params: action 参数字典，可选。
            out_params_schema: 输出参数 schema，可选。
            action_id: 业务动作 ID（需来自最近一次检索候选），可选。
            depends_on: 直接依赖的上游节点 ID 列表，可选。
            parent_node_id: 父节点 ID，可选。

        Returns:
            包含状态、类型与节点 ID 的结果字典；失败时返回错误信息。
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
        """更新已有 condition 节点的表达式或分支指向。

        适用场景：修正条件表达式或调整 true/false 跳转目标。

        Args:
            id: 节点唯一标识。
            display_name: 节点展示名称，可选。
            params: 条件参数字典（expression），可选。
            true_to_node: 条件为真时跳转的节点 ID 或 null，可选。
            false_to_node: 条件为假时跳转的节点 ID 或 null，可选。
            depends_on: 直接依赖的上游节点 ID 列表，可选。
            parent_node_id: 父节点 ID，可选。

        Returns:
            包含状态、类型与节点 ID 的结果字典；失败时返回错误信息。
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
        """更新已有 switch 节点的 case 或默认分支。

        适用场景：补充/修正多分支路由规则或默认跳转。

        Args:
            id: 节点唯一标识。
            display_name: 节点展示名称，可选。
            params: switch 参数字典（source/field），可选。
            cases: case 列表，可选。
            default_to_node: 默认分支节点 ID 或 null，可选。
            depends_on: 直接依赖的上游节点 ID 列表，可选。
            parent_node_id: 父节点 ID，可选。

        Returns:
            包含状态、类型与节点 ID 的结果字典；失败时返回错误信息。
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
        """更新已有 loop 节点的参数或子图。

        适用场景：修正循环参数、子图节点集合或依赖关系。

        Args:
            id: 节点唯一标识。
            display_name: 节点展示名称，可选。
            params: loop 参数字典，可选。
            sub_graph_nodes: 循环子图节点 ID 列表，可选。
            depends_on: 直接依赖的上游节点 ID 列表，可选。
            parent_node_id: 父节点 ID，可选。

        Returns:
            包含状态、类型与节点 ID 的结果字典；失败时返回错误信息。
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
        """导出当前 workflow 快照，便于调试或回显。

        适用场景：需要查看当前构建器状态时调用。

        Returns:
            包含节点/边统计与 workflow 快照的结果字典。
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
        """获取指定节点的参数补全上下文。"""
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
        """更新并校验指定节点参数。"""
        _log_tool_call("update_node_params", {"id": id})
        if not isinstance(id, str) or id not in builder.nodes:
            result = {"status": "error", "message": "节点不存在，无法更新参数。"}
            return _return_tool_result("update_node_params", result)
        if not isinstance(params, Mapping):
            result = {"status": "error", "message": "params 需要是对象。"}
            return _return_tool_result("update_node_params", result)

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
            params=dict(params),
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

        filled_params[id] = dict(params)
        if id not in validated_node_ids:
            validated_node_ids.append(id)
        builder.update_node(id, params=dict(params))
        _snapshot(f"validate_params_{id}")
        result = {"status": "ok", "params": params}
        return _return_tool_result("update_node_params", result)
    
    agent = Agent(
        name="WorkflowStructurePlanner",
        instructions=system_prompt,
        tools=[
            plan_user_requirement,
            get_user_requirement,
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
            dump_model,
        ],
        model=OPENAI_MODEL
    )

    log_section("结构规划 - Agent SDK")

    def _run_agent(prompt: Any) -> None:
        try:
            Runner.run_sync(agent, prompt, max_turns=max_rounds)  # type: ignore[arg-type]
        except TypeError:
            coro = Runner.run(agent, prompt)  # type: ignore[call-arg]
            if asyncio.iscoroutine(coro):
                asyncio.run(coro)

    _run_agent(nl_requirement)
    if not latest_skeleton:
        latest_skeleton = _prepare_skeleton_for_next_stage(
            builder=builder, action_registry=action_registry, search_service=search_service
        )

    if return_requirement:
        return latest_skeleton, parsed_requirement
    return latest_skeleton


__all__ = ["plan_workflow_structure_with_llm"]
