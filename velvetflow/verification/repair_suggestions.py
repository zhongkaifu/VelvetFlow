"""Auto-repair and suggestion engine for workflow validation.

This module layers deterministic AST repair templates, lightweight constraint
solving, and heuristic probability-driven fills on top of the existing
validation pipeline. It is intentionally conservative: suggestions are
returned as structured patches rather than mutating inputs in-place so callers
can decide which hints to apply.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from velvetflow.models import RepairSuggestion, ValidationError
from velvetflow.workflow_parser import parse_workflow_source
from velvetflow.verification.binding_checks import (
    _get_field_schema,
    _get_node_output_schema,
    _index_actions_by_id,
)


def _clone_workflow(workflow_raw: Any) -> Dict[str, Any]:
    parse_result = parse_workflow_source(workflow_raw)
    ast = parse_result.ast if parse_result.ast is not None else workflow_raw
    return copy.deepcopy(ast if isinstance(ast, Mapping) else {})


def _schema_types(schema: Mapping[str, Any]) -> set[str]:
    typ = schema.get("type")
    if isinstance(typ, list):
        return {str(t) for t in typ}
    if isinstance(typ, str):
        return {typ}
    return set()


def _schemas_compatible(expected: Optional[Mapping[str, Any]], actual: Optional[Mapping[str, Any]]) -> bool:
    if not expected or not actual:
        return True
    etypes = _schema_types(expected)
    atypes = _schema_types(actual)
    if not etypes or not atypes:
        return True
    return bool(etypes & atypes)


def _index_nodes_by_id(workflow: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    nodes = workflow.get("nodes") if isinstance(workflow, Mapping) else None
    if not isinstance(nodes, list):
        return {}
    return {n.get("id"): n for n in nodes if isinstance(n, Mapping) and n.get("id")}


def _infer_candidate_sources(
    expected_schema: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    actions_by_id: Mapping[str, Mapping[str, Any]],
) -> List[Tuple[str, Mapping[str, Any]]]:
    candidates: List[Tuple[str, Mapping[str, Any]]] = []
    for node_id, node in nodes_by_id.items():
        if node.get("type") != "action":
            continue
        output_schema = _get_node_output_schema(node, actions_by_id)
        if not isinstance(output_schema, Mapping):
            continue

        props = output_schema.get("properties") if isinstance(output_schema.get("properties"), Mapping) else {}
        for field_name, field_schema in props.items():
            if not isinstance(field_schema, Mapping):
                continue
            if _schemas_compatible(expected_schema, field_schema):
                path = f"result_of.{node_id}.{field_name}" if field_name else f"result_of.{node_id}"
                candidates.append((path, field_schema))
    return candidates


def _resolve_missing_params(
    workflow: MutableMapping[str, Any],
    action_registry: Sequence[Mapping[str, Any]],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    actions_by_id: Mapping[str, Mapping[str, Any]],
    errors: Sequence[ValidationError],
    suggestions: List[RepairSuggestion],
) -> None:
    missing_fields: List[Tuple[str, str, Mapping[str, Any]]] = []

    # Prefer explicit error hints to drive template matching.
    for err in errors:
        if err.code == "MISSING_REQUIRED_PARAM" and err.node_id and err.field:
            node = nodes_by_id.get(err.node_id)
            action_def = actions_by_id.get(node.get("action_id")) if node else None
            field_schema = (
                _get_field_schema(action_def.get("arg_schema") or {}, err.field)
                if isinstance(action_def, Mapping)
                else None
            )
            if isinstance(field_schema, Mapping):
                missing_fields.append((err.node_id, err.field, field_schema))

    # Also scan required schema fields defensively.
    for node_id, node in nodes_by_id.items():
        if node.get("type") != "action":
            continue
        action_def = actions_by_id.get(node.get("action_id")) if isinstance(node.get("action_id"), str) else None
        arg_schema = action_def.get("arg_schema") if isinstance(action_def, Mapping) else None
        if not isinstance(arg_schema, Mapping):
            continue
        required = arg_schema.get("required") if isinstance(arg_schema.get("required"), list) else []
        params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
        for field in required:
            if not isinstance(field, str):
                continue
            if field in params and params.get(field):
                continue
            field_schema = _get_field_schema(arg_schema, field)
            if isinstance(field_schema, Mapping):
                missing_fields.append((node_id, field, field_schema))

    for node_id, field, expected_schema in missing_fields:
        candidates = _infer_candidate_sources(expected_schema, nodes_by_id, actions_by_id)
        if not candidates:
            continue
        source_path, chosen_schema = candidates[0]
        path = f"nodes.{node_id}.params.{field}" if isinstance(node_id, str) else f"nodes.params.{field}"
        rationale = f"匹配输出 schema {chosen_schema.get('type')} 填充必填参数"
        suggestion = RepairSuggestion(
            strategy="ast_template",
            description=f"为 {node_id}.{field} 注入与 schema 兼容的绑定 {source_path}",
            path=path,
            patch={"__from__": source_path},
            confidence=0.82,
            rationale=rationale,
        )
        suggestions.append(suggestion)
        node = nodes_by_id.get(node_id)
        if isinstance(node, MutableMapping):
            params = node.setdefault("params", {})
            if isinstance(params, MutableMapping):
                params[field] = suggestion.patch


def _solve_parameter_bindings(
    workflow: MutableMapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    actions_by_id: Mapping[str, Mapping[str, Any]],
    suggestions: List[RepairSuggestion],
) -> None:
    # Collect empty bindings with candidate providers and solve with backtracking.
    tasks: List[Tuple[str, str, Mapping[str, Any]]] = []
    for node_id, node in nodes_by_id.items():
        if node.get("type") != "action":
            continue
        action_def = actions_by_id.get(node.get("action_id")) if isinstance(node.get("action_id"), str) else None
        arg_schema = action_def.get("arg_schema") if isinstance(action_def, Mapping) else None
        if not isinstance(arg_schema, Mapping):
            continue
        properties = arg_schema.get("properties") if isinstance(arg_schema.get("properties"), Mapping) else {}
        params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
        for field, schema in properties.items():
            if not isinstance(field, str) or not isinstance(schema, Mapping):
                continue
            if field in params and params.get(field) not in (None, ""):
                continue
            tasks.append((node_id, field, schema))

    assignments: Dict[Tuple[str, str], str] = {}

    def backtrack(idx: int) -> bool:
        if idx >= len(tasks):
            return True
        node_id, field, schema = tasks[idx]
        candidates = _infer_candidate_sources(schema, nodes_by_id, actions_by_id)
        for source_path, candidate_schema in candidates:
            if source_path in assignments.values():
                continue
            assignments[(node_id, field)] = source_path
            if backtrack(idx + 1):
                return True
            assignments.pop((node_id, field), None)
        return False

    solved = backtrack(0)
    if not solved:
        return

    for (node_id, field), source_path in assignments.items():
        path = f"nodes.{node_id}.params.{field}" if isinstance(node_id, str) else f"nodes.params.{field}"
        suggestion = RepairSuggestion(
            strategy="constraint_solver",
            description=f"约束求解为 {node_id}.{field} 选择来源 {source_path}",
            path=path,
            patch={"__from__": source_path},
            confidence=0.77,
            rationale="满足所有必填参数可满足性约束",
        )
        suggestions.append(suggestion)
        node = nodes_by_id.get(node_id)
        if isinstance(node, MutableMapping):
            params = node.setdefault("params", {})
            if isinstance(params, MutableMapping):
                params[field] = suggestion.patch


def _statistical_fill(
    workflow: MutableMapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    actions_by_id: Mapping[str, Mapping[str, Any]],
    suggestions: List[RepairSuggestion],
) -> None:
    # Estimate most likely default from schema defaults/examples/enum counts.
    param_frequency: Dict[str, Dict[Any, int]] = {}
    for node in nodes_by_id.values():
        if node.get("type") != "action":
            continue
        params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
        for key, val in params.items():
            if val is None or val == "":
                continue
            param_frequency.setdefault(key, {})[str(val)] = param_frequency.setdefault(key, {}).get(str(val), 0) + 1

    for node_id, node in nodes_by_id.items():
        if node.get("type") != "action":
            continue
        action_def = actions_by_id.get(node.get("action_id")) if isinstance(node.get("action_id"), str) else None
        arg_schema = action_def.get("arg_schema") if isinstance(action_def, Mapping) else None
        if not isinstance(arg_schema, Mapping):
            continue
        properties = arg_schema.get("properties") if isinstance(arg_schema.get("properties"), Mapping) else {}
        params = node.setdefault("params", {}) if isinstance(node, MutableMapping) else {}
        for field, schema in properties.items():
            if not isinstance(schema, Mapping) or params.get(field) not in (None, ""):
                continue
            default = schema.get("default")
            enum_vals = schema.get("enum") if isinstance(schema.get("enum"), list) else []
            examples = schema.get("examples") if isinstance(schema.get("examples"), list) else []
            ranked: List[Any] = []
            if default is not None:
                ranked.append(default)
            ranked.extend(enum_vals)
            ranked.extend(examples)
            freq_map = param_frequency.get(field, {})
            if freq_map:
                deduped: List[Any] = []
                seen: set[str] = set()
                for candidate in ranked or list(freq_map.keys()):
                    key = str(candidate)
                    if key in seen:
                        continue
                    seen.add(key)
                    deduped.append(candidate)
                ranked = sorted(deduped, key=lambda v: freq_map.get(str(v), 0), reverse=True)
            if not ranked:
                continue
            choice = ranked[0]
            suggestion = RepairSuggestion(
                strategy="statistical_fill",
                description=f"为 {node_id}.{field} 选择概率最高的默认值 {choice}",
                path=f"nodes.{node_id}.params.{field}",
                patch=choice,
                confidence=0.65,
                rationale="结合 schema 默认值/枚举与历史频次打分",
            )
            suggestions.append(suggestion)
            params[field] = choice


def _connect_disconnected_nodes(
    workflow: MutableMapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    suggestions: List[RepairSuggestion],
) -> None:
    edges = workflow.setdefault("edges", []) if isinstance(workflow, MutableMapping) else []
    if not isinstance(edges, list):
        return
    connected = {e.get("to") for e in edges if isinstance(e, Mapping)}
    all_nodes = list(nodes_by_id.keys())
    candidates = [n for n in all_nodes if n not in connected]
    if len(all_nodes) < 2:
        return
    if not candidates:
        return
    hub = all_nodes[0]
    for node_id in candidates:
        if node_id == hub:
            continue
        patch = {"from": hub, "to": node_id, "condition": None}
        edges.append(patch)
        suggestions.append(
            RepairSuggestion(
                strategy="ast_template",
                description=f"自动连接未入度节点 {node_id} -> 由 {hub} 驱动", 
                path=f"edges[{len(edges)-1}]",
                patch=patch,
                confidence=0.55,
                rationale="消除未连接节点，保证可达性",
            )
        )


def generate_repair_suggestions(
    workflow_raw: Any,
    action_registry: Sequence[Mapping[str, Any]],
    *,
    errors: Sequence[ValidationError] | None = None,
) -> tuple[Dict[str, Any], List[RepairSuggestion]]:
    """Generate repair patches based on the workflow AST and validation issues."""

    workflow = _clone_workflow(workflow_raw)
    nodes_by_id = _index_nodes_by_id(workflow)
    actions_by_id = _index_actions_by_id(list(action_registry))

    suggestions: List[RepairSuggestion] = []

    _resolve_missing_params(workflow, action_registry, nodes_by_id, actions_by_id, errors or [], suggestions)
    _solve_parameter_bindings(workflow, nodes_by_id, actions_by_id, suggestions)
    _statistical_fill(workflow, nodes_by_id, actions_by_id, suggestions)
    _connect_disconnected_nodes(workflow, nodes_by_id, suggestions)

    return workflow, suggestions


__all__ = ["generate_repair_suggestions"]
