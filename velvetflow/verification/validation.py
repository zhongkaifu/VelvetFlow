"""Static validation helpers for VelvetFlow workflows."""

import re
from collections import deque
from typing import Any, Dict, List, Mapping, Optional

from velvetflow.loop_dsl import build_loop_output_schema, index_loop_body_nodes
from velvetflow.models import ValidationError, Workflow


def _index_actions_by_id(action_registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Lightweight action registry indexer without importing planner package.

    The planner package re-exports verification helpers, so importing from
    ``velvetflow.planner`` inside this module would trigger a circular import.
    Keeping this helper local avoids that dependency while preserving the
    expected behavior.
    """

    return {a.get("action_id"): a for a in action_registry or [] if isinstance(a, dict)}


def precheck_loop_body_graphs(workflow_raw: Mapping[str, Any] | Any) -> List[ValidationError]:
    """Detect loop body graphs that refer to nonexistent nodes.

    Pydantic 会在校验时将 loop.body_subgraph 当作独立 workflow 再做一次
    校验，如果 body_subgraph.edges/entry/exit 指向不存在的节点，会直接
    抛出 ValueError。这里提前做轻量校验，用更易读的错误提示阻断后续
    的深度校验，避免出现“边引用不存在节点”的模糊错误信息。
    """

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
        body_ids = {bn.get("id") for bn in body_nodes if isinstance(bn.get("id"), str)}

        allowed_body_types = {"action", "condition", "loop", "parallel", "start", "end"}
        for idx, body_node in enumerate(body_nodes):
            ntype = body_node.get("type")
            if ntype not in allowed_body_types:
                errors.append(
                    ValidationError(
                        code="INVALID_LOOP_BODY",
                        node_id=loop_id,
                        field=f"body_subgraph.nodes[{idx}].type",
                        message=(
                            "loop 节点 '{loop_id}' 的 body_subgraph 包含非法节点类型 "
                            f"'{ntype}'，允许类型: {sorted(allowed_body_types)}"
                        ),
                    )
                )

        entry = body.get("entry")
        if isinstance(entry, str) and entry not in body_ids:
            errors.append(
                ValidationError(
                    code="INVALID_LOOP_BODY",
                    node_id=loop_id,
                    field="body_subgraph.entry",
                    message=(
                        f"loop 节点 '{loop_id}' 的 body_subgraph.entry 指向不存在的节点 '{entry}'"
                    ),
                )
            )

        exit_node = body.get("exit")
        if isinstance(exit_node, str) and exit_node not in body_ids:
            errors.append(
                ValidationError(
                    code="INVALID_LOOP_BODY",
                    node_id=loop_id,
                    field="body_subgraph.exit",
                    message=(
                        f"loop 节点 '{loop_id}' 的 body_subgraph.exit 指向不存在的节点 '{exit_node}'"
                    ),
                )
            )

        for idx, edge in enumerate(body.get("edges") or []):
            if not isinstance(edge, Mapping):
                continue
            frm = edge.get("from")
            to = edge.get("to")
            if isinstance(frm, str) and frm not in body_ids:
                errors.append(
                    ValidationError(
                        code="INVALID_LOOP_BODY",
                        node_id=loop_id,
                        field=f"body_subgraph.edges[{idx}].from",
                        message=(
                            f"loop 节点 '{loop_id}' 的 body_subgraph.edges[{idx}].from 指向不存在的节点 '{frm}'"
                        ),
                    )
                )
            if isinstance(to, str) and to not in body_ids:
                errors.append(
                    ValidationError(
                        code="INVALID_LOOP_BODY",
                        node_id=loop_id,
                        field=f"body_subgraph.edges[{idx}].to",
                        message=(
                            f"loop 节点 '{loop_id}' 的 body_subgraph.edges[{idx}].to 指向不存在的节点 '{to}'"
                        ),
                    )
                )

    return errors


TEMPLATE_PATTERN = re.compile(r"\{\{\s*([^{}]+?)\s*\}\}")


def _index_nodes_by_id(workflow: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in workflow.get("nodes", [])}


def _validate_nodes_recursive(
    nodes: List[Mapping[str, Any]],
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Mapping[str, str],
    errors: List[ValidationError],
):
    """Validate nodes (and nested loop body nodes) against planner rules."""

    for n in nodes or []:
        if not isinstance(n, Mapping):
            continue

        nid = n.get("id")
        ntype = n.get("type")
        action_id = n.get("action_id")
        params = n.get("params", {})

        template_refs = _collect_template_refs(params) if params is not None else []
        for ref in template_refs:
            errors.append(
                ValidationError(
                    code="INVALID_TEMPLATE_BINDING",
                    node_id=nid,
                    field=ref.get("path"),
                    message=(
                        f"节点 '{nid}' 的参数 {ref.get('path')} 使用了模板 {ref.get('template')}，"
                        "workflow DSL 不支持 Jinja/ Mustache 模板，请改用 __from__ 参数绑定"
                        "（例如 {\"__from__\": \"result_of.<node>.<field>\", \"__agg__\": \"identity\"}）。"
                    ),
                )
            )

        # 1) action 节点
        if ntype == "action" and action_id:
            action_def = actions_by_id.get(action_id)
            if not action_def:
                errors.append(
                    ValidationError(
                        code="UNKNOWN_ACTION_ID",
                        node_id=nid,
                        field="action_id",
                        message=f"节点 '{nid}' 的 action_id '{action_id}' 不在 Action Registry 中。",
                    )
                )
            else:
                schema = action_def.get("arg_schema") or {}
                required_fields = (schema.get("required") or []) if isinstance(schema, dict) else []

                if not isinstance(params, dict) or len(params) == 0:
                    if required_fields:
                        for field in required_fields:
                            errors.append(
                                ValidationError(
                                    code="MISSING_REQUIRED_PARAM",
                                    node_id=nid,
                                    field=field,
                                    message=(
                                        f"action 节点 '{nid}' 的 params 为空，但 action '{action_id}' 有必填字段 '{field}'。"
                                    ),
                                )
                            )
                else:
                    for field in required_fields:
                        if field not in params:
                            errors.append(
                                ValidationError(
                                    code="MISSING_REQUIRED_PARAM",
                                    node_id=nid,
                                    field=field,
                                    message=(
                                        f"action 节点 '{nid}' 的 params 缺少必填字段 '{field}' (action_id='{action_id}')"
                                    ),
                                )
                            )

            # 绑定 DSL 静态校验
            def _walk_params_for_from(obj: Any, path_prefix: str = ""):
                if isinstance(obj, dict):
                    if "__from__" in obj:
                        schema_err = validate_param_binding(obj)
                        if schema_err:
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field=path_prefix or "params",
                                    message=f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）无效：{schema_err}",
                                )
                            )

                        source = obj.get("__from__")
                        if isinstance(source, str):
                            schema_err = _check_output_path_against_schema(
                                source,
                                nodes_by_id,
                                actions_by_id,
                                loop_body_parents,
                                current_node_id=nid,
                            )
                            if schema_err:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=path_prefix or "params",
                                        message=(
                                            f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）引用无效：{schema_err}"
                                        ),
                                    )
                                )

                        if obj.get("__agg__") == "pipeline":
                            steps = obj.get("steps", [])
                            if isinstance(steps, list):
                                for idx, step in enumerate(steps):
                                    if not isinstance(step, Mapping):
                                        continue
                                    if step.get("op") == "filter":
                                        fld = step.get("field")
                                        item_err = _check_array_item_field(
                                            source,
                                            fld,
                                            nodes_by_id,
                                            actions_by_id,
                                            loop_body_parents,
                                        )
                                        if item_err:
                                            errors.append(
                                                ValidationError(
                                                    code="SCHEMA_MISMATCH",
                                                    node_id=nid,
                                                    field=f"{path_prefix or 'params'}.pipeline.steps[{idx}].field",
                                                    message=(
                                                        f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）中 pipeline.steps[{idx}].field='{fld}' 无效：{item_err}"
                                                    ),
                                                )
                                            )

                    for k, v in obj.items():
                        new_prefix = f"{path_prefix}.{k}" if path_prefix else k
                        _walk_params_for_from(v, new_prefix)
                elif isinstance(obj, list):
                    for idx, v in enumerate(obj):
                        new_prefix = f"{path_prefix}[{idx}]"
                        _walk_params_for_from(v, new_prefix)

            _walk_params_for_from(params)

        # 2) condition 节点
        if ntype == "condition":
            if not isinstance(params, dict) or len(params) == 0:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="params",
                        message=f"condition 节点 '{nid}' 的 params 为空，至少需要 kind/source 等字段。",
                    )
                )
            else:
                kind = params.get("kind")
                if not kind:
                    errors.append(
                        ValidationError(
                            code="MISSING_REQUIRED_PARAM",
                            node_id=nid,
                            field="kind",
                            message=f"condition 节点 '{nid}' 缺少 kind 字段。",
                        )
                    )
                else:
                    allowed_kinds = {
                        "list_not_empty",
                        "any_greater_than",
                        "equals",
                        "contains",
                        "not_equals",
                        "greater_than",
                        "less_than",
                        "between",
                        "all_less_than",
                        "is_empty",
                        "not_empty",
                        "is_not_empty",
                        "multi_band",
                    }
                    if kind not in allowed_kinds:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="kind",
                                message=f"condition 节点 '{nid}' 的 kind='{kind}' 未被支持。",
                            )
                        )
                    required_fields_map = {
                        "list_not_empty": ["source"],
                        "any_greater_than": ["source", "field", "threshold"],
                        "all_less_than": ["source", "field", "threshold"],
                        "equals": ["source", "value"],
                        "not_equals": ["source", "value"],
                        "greater_than": ["source", "threshold"],
                        "less_than": ["source", "threshold"],
                        "between": ["source", "min", "max"],
                        "contains": ["source", "field", "value"],
                        "multi_band": ["source", "bands"],
                        "is_empty": ["source"],
                        "not_empty": ["source"],
                        "is_not_empty": ["source"],
                    }
                    for field in required_fields_map.get(kind, []):
                        if field not in params:
                            errors.append(
                                ValidationError(
                                    code="MISSING_REQUIRED_PARAM",
                                    node_id=nid,
                                    field=field,
                                    message=(
                                        f"condition 节点 '{nid}' (kind={kind}) 缺少字段 '{field}'。"
                                    ),
                                )
                            )

                    if kind in {"any_greater_than", "all_less_than", "contains"}:
                        src = params.get("source")
                        fld = params.get("field")
                        if isinstance(src, str) and isinstance(fld, str):
                            item_err = _check_array_item_field(src, fld, nodes_by_id, actions_by_id)
                            if item_err:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field="field",
                                        message=(
                                            f"condition 节点 '{nid}' 的 field='{fld}' 无效：{item_err}"
                                        ),
                                    )
                                )

        # 3) loop 节点
        if ntype == "loop":
            loop_kind = (params or {}).get("loop_kind")
            if not loop_kind:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="loop_kind",
                        message=f"loop 节点 '{nid}' 缺少 loop_kind 参数。",
                    )
                )
            elif loop_kind not in {"for_each", "while"}:
                errors.append(
                    ValidationError(
                        code="SCHEMA_MISMATCH",
                        node_id=nid,
                        field="loop_kind",
                        message=f"loop 节点 '{nid}' 的 loop_kind '{loop_kind}' 无效（只支持 for_each / while）。",
                    )
                )

            source = params.get("source") if isinstance(params, dict) else None
            if not source:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="source",
                        message=f"loop 节点 '{nid}' 缺少 source。",
                    )
                )
            elif isinstance(source, Mapping):
                err = validate_param_binding_and_schema(source, {"nodes": list(nodes_by_id.values())}, list(actions_by_id.values()))
                if err:
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=nid,
                            field="source",
                            message=f"loop 节点 '{nid}' 的 source 无效：{err}",
                        )
                    )
            elif isinstance(source, str):
                schema_err = _check_output_path_against_schema(
                    source,
                    nodes_by_id,
                    actions_by_id,
                    loop_body_parents,
                    current_node_id=nid,
                )
                if schema_err:
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=nid,
                            field="source",
                            message=f"loop 节点 '{nid}' 的 source 引用无效：{schema_err}",
                        )
                    )

            # exports 静态校验
            exports = params.get("exports") if isinstance(params, dict) else None
            if not isinstance(exports, Mapping):
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="exports",
                        message=f"loop 节点 '{nid}' 需要定义 exports 才能向外暴露结果。",
                    )
                )
            else:
                body_nodes: List[str] = []
                body_node_map: Dict[str, Mapping[str, Any]] = {}
                body_graph = params.get("body_subgraph") or {}
                if not isinstance(body_graph, Mapping):
                    errors.append(
                        ValidationError(
                            code="INVALID_LOOP_BODY",
                            node_id=nid,
                            field="body_subgraph",
                            message=f"loop 节点 '{nid}' 定义了 exports，但缺少 body_subgraph（需包含 nodes/edges）。",
                        )
                    )
                else:
                    raw_body_nodes = [bn for bn in body_graph.get("nodes", []) or [] if isinstance(bn, Mapping)]
                    raw_body_edges = [e for e in body_graph.get("edges", []) or [] if isinstance(e, Mapping)]
                    body_node_map = {
                        bn.get("id")
                        : bn
                        for bn in raw_body_nodes
                        if isinstance(bn.get("id"), str)
                    }
                    body_nodes = list(body_node_map.keys())

                    if len(raw_body_nodes) == 0:
                        errors.append(
                            ValidationError(
                                code="INVALID_LOOP_BODY",
                                node_id=nid,
                                field="body_subgraph.nodes",
                                message=f"loop 节点 '{nid}' 的 body_subgraph.nodes 为空，无法执行循环。",
                            )
                        )
                    if len(raw_body_edges) == 0:
                        errors.append(
                            ValidationError(
                                code="INVALID_LOOP_BODY",
                                node_id=nid,
                                field="body_subgraph.edges",
                                message=f"loop 节点 '{nid}' 的 body_subgraph.edges 为空，无法连接循环内部节点。",
                            )
                        )
                items_spec = exports.get("items") if isinstance(exports, Mapping) else None
                if items_spec is not None:
                    if not isinstance(items_spec, Mapping):
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="exports.items",
                                message=f"loop 节点 '{nid}' 的 exports.items 必须是对象。",
                            )
                        )
                    else:
                        from_node = items_spec.get("from_node")
                        fields = items_spec.get("fields")
                        if not isinstance(from_node, str) or from_node not in body_nodes:
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field="exports.items.from_node",
                                    message=f"loop 节点 '{nid}' 的 exports.items.from_node 必须引用 body_subgraph 中的节点。",
                                )
                            )
                        if not isinstance(fields, list) or not fields or not all(
                            isinstance(f, str) for f in fields
                        ):
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field="exports.items.fields",
                                    message=f"loop 节点 '{nid}' 的 exports.items.fields 需要字符串数组。",
                                )
                            )
                        elif isinstance(from_node, str):
                            target_node = body_node_map.get(from_node)
                            action_id = (
                                target_node.get("action_id")
                                if isinstance(target_node, Mapping)
                                else None
                            )
                            action_def = (
                                actions_by_id.get(action_id)
                                if isinstance(action_id, str)
                                else None
                            )
                            output_schema = (action_def or {}).get("output_schema")

                            if not isinstance(output_schema, Mapping):
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field="exports.items.fields",
                                        message=(
                                            f"loop 节点 '{nid}' 的 exports.items.from_node='{from_node}'"
                                            " 缺少可用的 output_schema，无法暴露字段。"
                                        ),
                                    )
                                )
                            else:
                                for fld in fields:
                                    if _schema_path_error(output_schema, [fld]):
                                        errors.append(
                                            ValidationError(
                                                code="SCHEMA_MISMATCH",
                                                node_id=nid,
                                                field="exports.items.fields",
                                                message=(
                                                    f"loop 节点 '{nid}' 的 exports.items.fields 包含未知字段 '{fld}'"
                                                ),
                                            )
                                        )

                aggregates = exports.get("aggregates") if isinstance(exports, Mapping) else None
                if aggregates is not None:
                    if not isinstance(aggregates, list):
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="exports.aggregates",
                                message=f"loop 节点 '{nid}' 的 exports.aggregates 需要数组。",
                            )
                        )
                    else:
                        for idx, agg in enumerate(aggregates):
                            if not isinstance(agg, Mapping):
                                continue
                            expr = agg.get("expr", {})
                            kind = (agg.get("kind") or "").lower()
                            source = agg.get("source") or ""
                            field = expr.get("field") if isinstance(expr, Mapping) else None
                            from_node = agg.get("from_node") if isinstance(agg, Mapping) else None

                            if not isinstance(kind, str) or kind not in {
                                "count",
                                "count_if",
                                "max",
                                "min",
                                "sum",
                                "avg",
                                "collect",
                            }:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=f"exports.aggregates[{idx}].kind",
                                        message=f"loop 节点 '{nid}' 的聚合 kind 仅支持 collect/count/count_if/max/min/sum/avg。",
                                    )
                                )
                                continue

                            if not isinstance(expr, Mapping):
                                errors.append(
                                    ValidationError(
                                        code="MISSING_REQUIRED_PARAM",
                                        node_id=nid,
                                        field=f"exports.aggregates[{idx}].expr",
                                        message=f"loop 节点 '{nid}' 的聚合缺少 expr。",
                                    )
                                )
                                continue

                            if not source:
                                errors.append(
                                    ValidationError(
                                        code="MISSING_REQUIRED_PARAM",
                                        node_id=nid,
                                        field=f"exports.aggregates[{idx}].source",
                                        message=f"loop 节点 '{nid}' 的聚合缺少 source。",
                                    )
                                )
                            elif isinstance(source, str):
                                schema_err = _check_output_path_against_schema(
                                    source,
                                    nodes_by_id,
                                    actions_by_id,
                                    loop_body_parents,
                                    current_node_id=nid,
                                )
                                if schema_err:
                                    errors.append(
                                        ValidationError(
                                            code="SCHEMA_MISMATCH",
                                            node_id=nid,
                                            field=f"exports.aggregates[{idx}].source",
                                            message=f"loop 节点 '{nid}' 的聚合 source 无效：{schema_err}",
                                        )
                                    )

                            allowed_from_nodes = set(body_nodes)
                            # 允许使用 loop 自身的 id 作为 from_node，以便直接基于 exports.items 进行聚合
                            # （避免在 exports 中继续引用具体的子图节点）。
                            allowed_from_nodes.add(nid)

                            if not isinstance(from_node, str) or from_node not in allowed_from_nodes:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=f"exports.aggregates[{idx}].from_node",
                                        message=(
                                            f"loop 节点 '{nid}' 的聚合 from_node 必须指向 body_subgraph 节点或 loop 节点自身。"
                                        ),
                                    )
                                )

                            if kind == "count_if":
                                if not all(
                                    isinstance(expr.get(k), (int, float, str, bool)) for k in ["value"]
                                ):
                                    errors.append(
                                        ValidationError(
                                            code="MISSING_REQUIRED_PARAM",
                                            node_id=nid,
                                            field=f"exports.aggregates[{idx}].expr.value",
                                            message=f"loop 节点 '{nid}' 的 count_if 缺少比较值。",
                                        )
                                    )
                                if not isinstance(expr.get("op"), str):
                                    errors.append(
                                        ValidationError(
                                            code="MISSING_REQUIRED_PARAM",
                                            node_id=nid,
                                            field=f"exports.aggregates[{idx}].expr.op",
                                            message=f"loop 节点 '{nid}' 的 count_if 需要 op。",
                                        )
                                    )
                                if not isinstance(field, str):
                                    errors.append(
                                        ValidationError(
                                            code="MISSING_REQUIRED_PARAM",
                                            node_id=nid,
                                            field=f"exports.aggregates[{idx}].expr.field",
                                            message=f"loop 节点 '{nid}' 的 count_if 需要 field。",
                                        )
                                    )
                            elif kind in {"max", "min", "sum", "avg", "collect"}:
                                if not isinstance(field, str):
                                    errors.append(
                                        ValidationError(
                                            code="MISSING_REQUIRED_PARAM",
                                            node_id=nid,
                                            field=f"exports.aggregates[{idx}].expr.field",
                                            message=f"loop 节点 '{nid}' 的 {kind} 需要 field。",
                                        )
                                    )

            # 递归校验 body_subgraph
            body_graph = params.get("body_subgraph") if isinstance(params, Mapping) else None
            body_nodes = body_graph.get("nodes") if isinstance(body_graph, Mapping) else None
            if isinstance(body_nodes, list) and body_nodes:
                extended_nodes_by_id = dict(nodes_by_id)
                for body_node in body_nodes:
                    if isinstance(body_node, Mapping) and isinstance(body_node.get("id"), str):
                        extended_nodes_by_id[body_node["id"]] = body_node
                _validate_nodes_recursive(
                    body_nodes,
                    extended_nodes_by_id,
                    actions_by_id,
                    loop_body_parents,
                    errors,
                )

        # 4) parallel 节点
        if ntype == "parallel":
            branches = params.get("branches") if isinstance(params, dict) else None
            if not isinstance(branches, list) or not branches:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="branches",
                        message=f"parallel 节点 '{nid}' 需要非空 branches 列表。",
                    )
                )
def _collect_param_bindings(obj: Any, prefix: str = "") -> List[Dict[str, str]]:
    """Collect bindings that carry a __from__ reference for lightweight checks."""

    bindings: List[Dict[str, str]] = []

    if isinstance(obj, Mapping):
        if "__from__" in obj:
            bindings.append({"path": prefix or "params", "source": obj.get("__from__")})
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            bindings.extend(_collect_param_bindings(value, new_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            bindings.extend(_collect_param_bindings(value, new_prefix))

    return bindings


def _collect_template_refs(obj: Any, prefix: str = "") -> List[Dict[str, str]]:
    """Collect Jinja-style template references like ``{{foo.bar}}`` for linting."""

    refs: List[Dict[str, str]] = []

    if isinstance(obj, str):
        for match in TEMPLATE_PATTERN.finditer(obj):
            refs.append(
                {
                    "path": prefix or "params",
                    "template": match.group(0),
                    "expr": match.group(1).strip(),
                }
            )
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            refs.extend(_collect_template_refs(value, new_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            refs.extend(_collect_template_refs(value, new_prefix))

    return refs


def _check_output_path_against_schema(
    source_path: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Optional[Mapping[str, str]] = None,
    *,
    current_node_id: Optional[str] = None,
) -> Optional[str]:
    """
    对诸如 "result_of.fetch_temperatures.data" 或 "result_of.node_id.foo.bar" 做静态校验：
    - result_of.<node_id> 必须存在
    - 该 node 必须有 action_id
    - 对应 action 的 output_schema 必须包含第一层字段（data / foo）

    返回:
      - None: 校验通过
      - str: 具体错误信息
    """
    if not isinstance(source_path, str):
        return f"source/__from__ 应该是字符串，但收到类型: {type(source_path)}"

    parts = source_path.split(".")
    if len(parts) < 2 or parts[0] != "result_of":
        return None

    node_id = parts[1]
    rest_path = parts[2:]

    loop_body_parents = loop_body_parents or {}

    if node_id not in nodes_by_id:
        if node_id in loop_body_parents:
            parent_loop = loop_body_parents.get(node_id)
            current_parent = loop_body_parents.get(current_node_id)
            # 允许 loop 自己或其 body 内的节点引用同一个 body 节点；其余情况视为越界引用。
            if parent_loop not in {current_node_id, current_parent}:
                return (
                    f"循环外不允许直接引用子图节点 '{node_id}'，请通过 loop 节点 '{parent_loop}' 的 exports 暴露输出。"
                )
            # body_subgraph 内部或 loop 自身可引用 body 节点
            pass
        else:
            return f"路径 '{source_path}' 引用的节点 '{node_id}' 不存在。"

    node = nodes_by_id[node_id]
    action_id = node.get("action_id")
    node_type = node.get("type")

    # loop 节点使用虚拟 output_schema 校验 exports。
    if node_type == "loop":
        loop_schema = build_loop_output_schema(node.get("params") or {})
        if not loop_schema:
            return f"loop 节点 '{node_id}' 未定义 exports，无法暴露给外部引用。"
        err = _schema_path_error(loop_schema, rest_path)
        if err:
            return f"路径 '{source_path}' 无效：{err}"
        return None

    # 控制节点（如 condition / start / end）没有 action_id，也没有可用的 output_schema。
    # 在这种情况下跳过 schema 校验，允许下游继续引用其动态结果。
    if not action_id and node_type in {"condition", "start", "end"}:
        return None

    if not action_id:
        return f"路径 '{source_path}' 引用的节点 '{node_id}' 没有 action_id，无法从 output_schema 校验。"

    action_def = actions_by_id.get(action_id)
    if not action_def:
        return f"路径 '{source_path}' 引用的节点 '{node_id}' 的 action_id='{action_id}' 不在 Action Registry 中。"

    output_schema = action_def.get("output_schema")
    arg_schema = action_def.get("arg_schema")

    if not rest_path:
        return None

    if rest_path[0] == "params":
        arg_fields = rest_path[1:]
        if not arg_fields:
            return None
        if not isinstance(arg_schema, Mapping):
            return f"路径 '{source_path}' 无效：action_id='{action_id}' 缺少 arg_schema，无法校验 params 字段。"

        err = _schema_path_error(arg_schema, arg_fields)
        if err:
            return f"路径 '{source_path}' 无效：{err}"
        return None

    if not isinstance(output_schema, dict):
        return f"action_id='{action_id}' 没有定义 output_schema，无法校验路径 '{source_path}'。"

    err = _schema_path_error(output_schema, rest_path)
    if err:
        return f"路径 '{source_path}' 无效：{err}"

    return None


def _schema_path_error(schema: Mapping[str, Any], fields: List[str]) -> Optional[str]:
    """Check whether a dotted field path exists in a JSON schema."""

    if not isinstance(schema, Mapping):
        return "output_schema 不是对象，无法校验字段路径。"

    # 字段列表可能已经按点拆分，也可能仍然包含带点的路径（例如来自 params.field）。
    normalized_fields: List[str] = []
    for f in fields:
        if isinstance(f, str):
            normalized_fields.extend(part for part in f.split(".") if part)
        else:
            normalized_fields.append(f)

    current: Mapping[str, Any] = schema
    idx = 0
    while idx < len(normalized_fields):
        name = normalized_fields[idx]
        typ = current.get("type")

        if typ == "array":
            current = current.get("items") or {}
            continue

        if typ == "object" or typ is None:
            props = current.get("properties") or {}
            if name not in props:
                return f"字段 '{name}' 不存在，已知字段有: {list(props.keys())}"
            current = props[name]
            idx += 1
            continue

        if idx == len(normalized_fields) - 1:
            return None

        return f"字段路径 '{'.'.join(normalized_fields)}' 与 schema 类型 '{typ}' 不匹配（期望 object/array）。"

    return None


def validate_param_binding(binding: Any) -> Optional[str]:
    """Validate the shape of a single parameter binding."""

    if not isinstance(binding, Mapping):
        return "参数绑定必须是对象。"

    if "__from__" not in binding:
        return "缺少 __from__ 字段"

    source_path = binding["__from__"]
    if not isinstance(source_path, str):
        return "__from__ 必须是字符串"

    allowed_aggs = {
        "identity",
        "count",
        "sum",
        "avg",
        "max",
        "min",
        "format_join",
        "count_if",
        "pipeline",
    }
    agg = binding.get("__agg__", "identity")
    if agg not in allowed_aggs:
        return f"__agg__ 不支持值 {agg}"

    if agg == "count_if":
        if "field" not in binding or "op" not in binding or "value" not in binding:
            return "count_if 需要提供 field/op/value"

    if agg == "pipeline":
        steps = binding.get("steps")
        if not isinstance(steps, list) or not steps:
            return "pipeline 需要非空 steps 数组"
        for idx, step in enumerate(steps):
            if not isinstance(step, Mapping):
                return f"pipeline.steps[{idx}] 必须是对象"
            op = step.get("op")
            if op not in {"filter", "map", "format_join", "sort", "limit"}:
                return f"pipeline.steps[{idx}].op 不支持值 {op}"

    return None


def _get_array_item_schema_from_output(
    source: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Optional[Mapping[str, str]] = None,
) -> Optional[Mapping[str, Any]]:
    err = _check_output_path_against_schema(source, nodes_by_id, actions_by_id, loop_body_parents)
    if err:
        return None

    parts = source.split(".")
    node_id = parts[1]
    first_field = parts[2] if len(parts) >= 3 else None
    node = nodes_by_id.get(node_id)
    node_type = node.get("type") if node else None
    if node_type == "loop":
        loop_params = node.get("params") or {}
        if first_field == "items":
            items_spec = (loop_params.get("exports") or {}).get("items")
            items_schema = _get_loop_items_schema_from_exports(
                items_spec, node, nodes_by_id, actions_by_id
            )
            if items_schema:
                return items_schema

        schema = build_loop_output_schema(loop_params) or {}
        props = schema.get("properties") or {}
        return (props.get(first_field) or {}).get("items") if first_field in props else None

    action_def = actions_by_id.get(node.get("action_id")) if node else None
    if not action_def:
        return None

    schema = (action_def.get("output_schema") or {}).get("properties", {})
    if first_field not in schema:
        return None

    return (schema.get(first_field) or {}).get("items")


def _get_field_schema(schema: Mapping[str, Any], field: str) -> Optional[Mapping[str, Any]]:
    """Return the schema of a dotted field path inside an object/array schema."""

    if not isinstance(schema, Mapping) or not isinstance(field, str):
        return None

    current: Mapping[str, Any] = schema
    parts = field.split(".")

    for part in parts:
        if not isinstance(current, Mapping):
            return None

        current_type = current.get("type")
        if current_type == "array":
            current = current.get("items") or {}
            current_type = current.get("type") if isinstance(current, Mapping) else None

        if current_type in {"object", None}:
            props = current.get("properties") or {}
            current = props.get(part)
            if current is None:
                return None
            continue

        if part == parts[-1]:
            return current

        return None

    return current


def _suggest_numeric_subfield(schema: Mapping[str, Any], prefix: str = "") -> Optional[str]:
    """Find a numeric leaf field path within a schema to help auto-repair."""

    if not isinstance(schema, Mapping):
        return None

    typ = schema.get("type")
    if typ in {"number", "integer"}:
        return prefix or ""

    if typ == "array":
        return _suggest_numeric_subfield(schema.get("items") or {}, prefix)

    if typ == "object" or typ is None:
        props = schema.get("properties") or {}
        for name, subschema in props.items():
            candidate = _suggest_numeric_subfield(
                subschema,
                f"{prefix}.{name}" if prefix else name,
            )
            if candidate:
                return candidate

    return None


def _get_loop_items_schema_from_exports(
    items_spec: Any,
    loop_node: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    actions_by_id: Mapping[str, Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Derive the loop items array schema using the referenced action's output schema."""

    if not isinstance(items_spec, Mapping):
        return None

    from_node_id = items_spec.get("from_node")
    fields = items_spec.get("fields")
    if not isinstance(from_node_id, str) or not isinstance(fields, list):
        return None

    body_nodes = (
        (loop_node.get("params") or {}).get("body_subgraph") or {}
    ).get("nodes", [])
    src_node = next(
        (n for n in body_nodes if isinstance(n, Mapping) and n.get("id") == from_node_id),
        None,
    )

    action_def = None
    if isinstance(src_node, Mapping):
        action_id = src_node.get("action_id")
        if isinstance(action_id, str):
            action_def = actions_by_id.get(action_id)

    output_schema = (action_def or {}).get("output_schema") if action_def else None

    item_props: Dict[str, Any] = {}
    for fld in fields:
        if not isinstance(fld, str):
            continue
        field_schema = _get_field_schema(output_schema, fld) if output_schema else None
        item_props[fld] = field_schema or {}

    return {"type": "object", "properties": item_props} if item_props else None


def _check_array_item_field(
    source: str,
    field: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Optional[Mapping[str, str]] = None,
) -> Optional[str]:
    item_schema = _get_array_item_schema_from_output(
        source, nodes_by_id, actions_by_id, loop_body_parents
    )
    if not item_schema:
        return None

    return _schema_path_error(item_schema, [field])


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
                binding.get("source"),
                nodes_by_id,
                actions_by_id,
                loop_body_parents,
                current_node_id=nid,
            )
            if err:
                binding_issues.append(f"{nid}:{binding.get('path')} -> {err}")
    if binding_issues:
        messages.append("参数绑定引用无效: " + "; ".join(binding_issues[:10]))

    template_issues: List[str] = []
    for node in workflow.get("nodes", []) or []:
        nid = node.get("id")
        params = node.get("params")
        refs = _collect_template_refs(params) if params is not None else []
        for ref in refs:
            template_issues.append(
                f"{nid}:{ref.get('path')} -> {ref.get('template')} (expr={ref.get('expr')})"
            )
    if template_issues:
        messages.append(
            "发现模板占位符（{{}}）: "
            + "; ".join(template_issues[:10])
            + "。workflow DSL 不支持模板，请改用 __from__ 绑定。"
        )

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
    errors: List[ValidationError] = []

    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])

    nodes_by_id = _index_nodes_by_id(workflow)
    loop_body_parents = index_loop_body_nodes(workflow)
    node_ids = set(nodes_by_id.keys())
    actions_by_id = _index_actions_by_id(action_registry)

    # ---------- edges 校验 ----------
    for e in edges:
        frm = e.get("from")
        to = e.get("to")
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

    # ---------- 图连通性校验 ----------
    start_nodes = [n["id"] for n in nodes if n.get("type") == "start"]
    reachable: set = set()
    if nodes and start_nodes:
        adj: Dict[str, List[str]] = {}
        for e in edges:
            frm = e.get("from")
            to = e.get("to")
            if frm in node_ids and to in node_ids:
                adj.setdefault(frm, []).append(to)

        dq = deque(start_nodes)
        while dq:
            nid = dq.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for nxt in adj.get(nid, []):
                if nxt not in reachable:
                    dq.append(nxt)

        for nid in node_ids - reachable:
            errors.append(
                ValidationError(
                    code="DISCONNECTED_GRAPH",
                    node_id=nid,
                    field=None,
                    message=f"节点 '{nid}' 无法从 start 节点到达。",
                )
            )

    # ---------- 节点校验（含 loop body） ----------
    _validate_nodes_recursive(nodes, nodes_by_id, actions_by_id, loop_body_parents, errors)

    return errors


def validate_param_binding_and_schema(binding: Mapping[str, Any], workflow: Dict[str, Any], action_registry: List[Dict[str, Any]]):
    """Validate parameter bindings in the context of workflow/action schemas."""

    nodes_by_id = _index_nodes_by_id(workflow)
    loop_body_parents = index_loop_body_nodes(workflow)
    actions_by_id = _index_actions_by_id(action_registry)
    binding_err = validate_param_binding(binding)
    if binding_err:
        return binding_err

    src = binding.get("__from__")
    return (
        _check_output_path_against_schema(src, nodes_by_id, actions_by_id, loop_body_parents)
        if isinstance(src, str)
        else None
    )


__all__ = [
    "precheck_loop_body_graphs",
    "validate_param_binding",
    "validate_completed_workflow",
    "validate_param_binding_and_schema",
    "run_lightweight_static_rules",
]
