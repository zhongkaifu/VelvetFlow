"""Semantic analysis utilities for workflow validation.

This module builds symbol tables for nodes/aliases and performs lightweight
type/contract checks for parameter bindings. It complements the grammar parser
by catching undefined references, duplicate names, and schema incompatibility
before the planner attempts repairs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional

from velvetflow.loop_dsl import index_loop_body_nodes
from velvetflow.models import ValidationError
from velvetflow.reference_utils import normalize_reference_path

from .binding_checks import (
    _check_output_path_against_schema,
    _collect_param_bindings,
    _get_field_schema,
    _get_output_schema_at_path,
    _index_actions_by_id,
)


def _declare_node_symbol(
    nodes_by_id: MutableMapping[str, Mapping[str, Any]],
    node: Mapping[str, Any],
    errors: List[ValidationError],
) -> None:
    node_id = node.get("id")
    if not isinstance(node_id, str):
        errors.append(
            ValidationError(
                code="INVALID_SYMBOL",
                node_id=None,
                field="nodes",
                message="节点缺少 id 或 id 类型不是字符串。",
            )
        )
        return

    if node_id in nodes_by_id:
        errors.append(
            ValidationError(
                code="DUPLICATE_SYMBOL",
                node_id=node_id,
                field="nodes",
                message=f"检测到重复的节点 id '{node_id}'，请更换唯一名称。",
            )
        )
    else:
        nodes_by_id[node_id] = node


def _gather_loop_body_symbols(
    loop_node: Mapping[str, Any],
    nodes_by_id: MutableMapping[str, Mapping[str, Any]],
    errors: List[ValidationError],
) -> None:
    params = loop_node.get("params") if isinstance(loop_node, Mapping) else {}
    body = params.get("body_subgraph") if isinstance(params, Mapping) else None
    if not isinstance(body, Mapping):
        return

    alias = params.get("item_alias")
    if isinstance(alias, str) and alias in nodes_by_id:
        errors.append(
            ValidationError(
                code="NAME_CONFLICT",
                node_id=str(loop_node.get("id")),
                field="params.item_alias",
                message=(
                    f"循环别名 '{alias}' 与已有节点/符号冲突，建议更换不重复的名称。"
                ),
            )
        )

    body_nodes = body.get("nodes") if isinstance(body.get("nodes"), list) else []
    local_ids: set[str] = set()
    for nested in body_nodes:
        if not isinstance(nested, Mapping):
            continue
        nid = nested.get("id")
        if isinstance(nid, str):
            if nid in local_ids:
                errors.append(
                    ValidationError(
                        code="DUPLICATE_SYMBOL",
                        node_id=nid,
                        field="params.body_subgraph.nodes",
                        message=f"loop 子图内的节点 id '{nid}' 重复。",
                    )
                )
            local_ids.add(nid)
        _declare_node_symbol(nodes_by_id, nested, errors)

    entry = body.get("entry") if isinstance(body.get("entry"), str) else None
    exit_node = body.get("exit") if isinstance(body.get("exit"), str) else None
    for ref_field, ref_value in ("entry", entry), ("exit", exit_node):
        if ref_value and ref_value not in local_ids:
            errors.append(
                ValidationError(
                    code="UNDEFINED_REFERENCE",
                    node_id=str(loop_node.get("id")),
                    field=f"params.body_subgraph.{ref_field}",
                    message=f"loop 子图 {ref_field}='{ref_value}' 未在 body_subgraph.nodes 中定义。",
                )
            )


def _build_symbol_table(workflow: Mapping[str, Any], errors: List[ValidationError]) -> Dict[str, Mapping[str, Any]]:
    nodes_by_id: Dict[str, Mapping[str, Any]] = {}
    nodes = workflow.get("nodes")
    if isinstance(nodes, list):
        for node in nodes:
            if not isinstance(node, Mapping):
                continue
            _declare_node_symbol(nodes_by_id, node, errors)
            if node.get("type") == "loop":
                _gather_loop_body_symbols(node, nodes_by_id, errors)
    return nodes_by_id


def _check_edges_and_branches(
    workflow: Mapping[str, Any], nodes_by_id: Mapping[str, Mapping[str, Any]], errors: List[ValidationError]
) -> None:
    edges = workflow.get("edges") if isinstance(workflow.get("edges"), list) else []
    for idx, edge in enumerate(edges):
        if not isinstance(edge, Mapping):
            continue
        frm = edge.get("from")
        to = edge.get("to")
        for field_name, target in ("from", frm), ("to", to):
            if isinstance(target, str) and target not in nodes_by_id:
                errors.append(
                    ValidationError(
                        code="UNDEFINED_REFERENCE",
                        node_id=None,
                        field=f"edges[{idx}].{field_name}",
                        message=f"边指向了未定义的节点 '{target}'。",
                    )
                )

    for node_id, node in nodes_by_id.items():
        if node.get("type") != "condition":
            continue
        for branch_field in ("true_to_node", "false_to_node"):
            target = node.get(branch_field)
            if isinstance(target, str) and target not in nodes_by_id:
                errors.append(
                    ValidationError(
                        code="UNDEFINED_REFERENCE",
                        node_id=node_id,
                        field=branch_field,
                        message=f"condition 节点引用了不存在的分支节点 '{target}'。",
                    )
                )


def _schema_types(schema: Mapping[str, Any]) -> set[str]:
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return {str(t) for t in schema_type}
    if isinstance(schema_type, str):
        return {schema_type}
    return set()


def _schemas_compatible(expected: Optional[Mapping[str, Any]], actual: Optional[Mapping[str, Any]]) -> bool:
    if not expected or not actual:
        return True
    expected_types = _schema_types(expected)
    actual_types = _schema_types(actual)
    if not expected_types or not actual_types:
        return True
    return bool(expected_types & actual_types)


def _apply_binding_aggregator(
    actual_schema: Optional[Mapping[str, Any]], binding: Mapping[str, Any]
) -> Optional[Mapping[str, Any]]:
    if not actual_schema:
        return actual_schema

    agg = binding.get("agg")
    if not agg:
        binding_obj = binding.get("binding") if isinstance(binding.get("binding"), Mapping) else {}
        agg = binding_obj.get("__agg__") if isinstance(binding_obj, Mapping) else None

    if not isinstance(agg, str):
        return actual_schema

    agg = agg or "identity"
    if agg in {"count", "count_if"}:
        return {"type": "integer"}
    if agg in {"join", "format_join", "filter_map"}:
        return {"type": "string"}
    if agg == "pipeline":
        binding_obj = binding.get("binding") if isinstance(binding.get("binding"), Mapping) else {}
        steps = binding_obj.get("steps") if isinstance(binding_obj, Mapping) else None
        if isinstance(steps, list) and any(
            isinstance(step, Mapping) and step.get("op") == "format_join" for step in steps
        ):
            return {"type": "string"}
    return actual_schema


def _check_binding_contracts(
    workflow: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    actions_by_id: Mapping[str, Mapping[str, Any]],
    errors: List[ValidationError],
) -> None:
    loop_body_parents = index_loop_body_nodes(workflow)

    for node_id, node in nodes_by_id.items():
        if node.get("type") != "action":
            continue

        action_def = actions_by_id.get(node.get("action_id")) if isinstance(node.get("action_id"), str) else None
        arg_schema = action_def.get("arg_schema") if isinstance(action_def, Mapping) else None
        params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
        if not isinstance(arg_schema, Mapping) or not params:
            continue

        bindings = _collect_param_bindings(params)
        for binding in bindings:
            field_path = binding.get("path") or "params"
            if field_path.startswith("params."):
                normalized_field = field_path[len("params.") :]
            elif field_path == "params":
                normalized_field = ""
            else:
                normalized_field = field_path
            expected_schema = _get_field_schema(arg_schema, normalized_field)

            source = binding.get("source")
            sources: List[str] = []
            if isinstance(source, str):
                sources = [source]
            elif isinstance(source, list):
                sources = [s for s in source if isinstance(s, str)]

            for src in sources:
                normalized_src = normalize_reference_path(src)
                actual_schema = _get_output_schema_at_path(
                    normalized_src, nodes_by_id, actions_by_id, loop_body_parents
                )
                effective_schema = _apply_binding_aggregator(actual_schema, binding)

                if not _schemas_compatible(expected_schema, effective_schema):
                    errors.append(
                        ValidationError(
                            code="CONTRACT_VIOLATION",
                            node_id=node_id,
                            field=field_path,
                            message=(
                                f"参数绑定期望类型 {expected_schema.get('type') if expected_schema else '未知'}"
                                f"，但来源 {normalized_src} 的类型为 {effective_schema.get('type') if effective_schema else '未知'}。"
                                "请在下游 params 中添加格式或类型转换（例如设置合适的 __agg__ 或 pipeline 步骤）以适配输入需求。"
                            ),
                        )
                    )
                else:
                    schema_err = _check_output_path_against_schema(
                        normalized_src, nodes_by_id, actions_by_id, loop_body_parents
                    )
                    if schema_err:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=node_id,
                                field=field_path,
                                message=f"参数绑定来源 {normalized_src} 无效：{schema_err}",
                            )
                        )


def analyze_workflow_semantics(
    workflow: Mapping[str, Any], action_registry: List[Mapping[str, Any]]
) -> List[ValidationError]:
    """Perform semantic validation (symbol table + contract checks)."""

    errors: List[ValidationError] = []
    nodes_by_id = _build_symbol_table(workflow, errors)
    actions_by_id = _index_actions_by_id(list(action_registry))

    _check_edges_and_branches(workflow, nodes_by_id, errors)
    _check_binding_contracts(workflow, nodes_by_id, actions_by_id, errors)

    return errors


__all__ = ["analyze_workflow_semantics"]
