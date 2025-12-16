# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Node-level validation logic for workflows."""
from typing import Any, Dict, List, Mapping, Optional

from velvetflow.models import ValidationError
from velvetflow.reference_utils import normalize_reference_path, parse_field_path
from velvetflow.loop_dsl import loop_body_has_action

from .binding_checks import (
    _check_array_item_field,
    _check_output_path_against_schema,
    _get_array_item_schema_from_output,
    _get_field_schema,
    _get_field_schema_from_item,
    _get_loop_items_schema_from_exports,
    _get_node_output_schema,
    _get_output_schema_at_path,
    _iter_empty_param_fields,
    _iter_template_references,
    _maybe_decode_binding_string,
    _project_schema_through_agg,
    _schema_path_error,
    _suggest_numeric_subfield,
    _walk_schema_with_tokens,
    validate_param_binding,
)

CONDITION_PARAM_FIELDS = {"expression"}

LOOP_PARAM_FIELDS = {
    "loop_kind",
    "source",
    "condition",
    "item_alias",
    "body_subgraph",
    "exports",
}


def _filter_params_by_supported_fields(
    *,
    node: Mapping[str, Any],
    actions_by_id: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    params = node.get("params")
    if not isinstance(params, Mapping):
        return []

    node_type = node.get("type")
    allowed_fields: set[str] | None = None

    if node_type == "condition":
        allowed_fields = set(CONDITION_PARAM_FIELDS)
    elif node_type == "loop":
        allowed_fields = set(LOOP_PARAM_FIELDS)
    elif node_type == "action":
        action_id = node.get("action_id")
        action_def = actions_by_id.get(action_id) if isinstance(action_id, str) else None
        properties = (action_def or {}).get("arg_schema", {}).get("properties")
        if isinstance(properties, Mapping):
            allowed_fields = set(properties.keys())

    if not allowed_fields:
        return []

    removed = [key for key in params if key not in allowed_fields]
    if removed:
        node_params = {k: v for k, v in params.items() if k in allowed_fields}
        node["params"] = node_params

    return removed


def _is_self_reference_path(path: Any, node_id: str | None) -> bool:
    """Return True if ``path`` points to ``result_of.<node_id>`` (self-cycle)."""

    if not isinstance(path, str) or not node_id:
        return False

    normalized = normalize_reference_path(path)
    if not isinstance(normalized, str):
        return False

    try:
        parts = parse_field_path(normalized)
    except Exception:
        return False

    return len(parts) >= 2 and parts[0] == "result_of" and parts[1] == node_id


def _resolve_condition_schema(
    source_path: str,
    field: str | None,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    actions_by_id: Mapping[str, Mapping[str, Any]],
    loop_body_parents: Mapping[str, str],
    alias_schemas: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    """Return the schema of the condition target (source + optional field)."""

    normalized_source = normalize_reference_path(source_path)
    field_path = field if isinstance(field, str) and field else None

    alias_schema = (alias_schemas or {}).get(normalized_source)
    if isinstance(alias_schema, Mapping):
        if field_path:
            try:
                tokens = parse_field_path(field_path)
            except Exception:
                return None
            return _walk_schema_with_tokens(alias_schema, tokens)
        return alias_schema

    if not isinstance(normalized_source, str):
        return None

    combined_path = (
        f"{normalized_source}.{field_path}" if field_path else normalized_source
    )
    return _get_output_schema_at_path(
        combined_path, nodes_by_id, actions_by_id, loop_body_parents
    )


def _strip_illegal_exports(node: Mapping[str, Any]) -> bool:
    """Remove ``params.exports`` from non-loop nodes to avoid repeated errors."""

    if node.get("type") == "loop":
        return False

    params = node.get("params")
    if not isinstance(params, Mapping) or "exports" not in params:
        return False

    new_params = {k: v for k, v in params.items() if k != "exports"}
    node["params"] = new_params
    return True


def _validate_nodes_recursive(
    nodes: List[Mapping[str, Any]],
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Mapping[str, str],
    errors: List[ValidationError],
    alias_schemas: Optional[Dict[str, Mapping[str, Any]]] = None,
):
    """Validate nodes (and nested loop body nodes) against planner rules."""

    node_ids = set(nodes_by_id.keys())

    for n in nodes or []:
        if not isinstance(n, Mapping):
            continue

        if hasattr(errors, "set_context"):
            errors.set_context({"node": n})

        nid = n.get("id")
        ntype = n.get("type")
        action_id = n.get("action_id")
        params = n.get("params", {})

        def _flag_self_reference(field_path: str, ref: str) -> None:
            errors.append(
                ValidationError(
                    code="SELF_REFERENCE",
                    node_id=nid,
                    field=field_path,
                    message=(
                        f"节点 '{nid}' 的字段 '{field_path}' 自引用 {ref}"
                        "，会造成循环依赖。请将此错误上下文提交给 LLM 分析原因，"
                        "并使用可用工具改为引用上游节点的输出或拆分节点。"
                    ),
                )
            )

        if ntype == "action" and (not isinstance(params, Mapping) or len(params) == 0):
            errors.append(
                ValidationError(
                    code="EMPTY_PARAMS",
                    node_id=nid,
                    field="params",
                    message=(
                        "action 节点的 params 为空，需要通过 LLM 分析原因并调用工具补全必需字段或绑定。"
                    ),
                )
            )

        empty_fields = list(_iter_empty_param_fields(params))
        for path in empty_fields:
            context_parts = []
            if ntype:
                context_parts.append(f"type={ntype}")
            if action_id:
                context_parts.append(f"action_id={action_id}")
            context_hint = f"（{', '.join(context_parts)}）" if context_parts else ""

            errors.append(
                ValidationError(
                    code="EMPTY_PARAM_VALUE",
                    node_id=nid,
                    field=path,
                    message=(
                        f"节点 '{nid}' 的参数 '{path}' 值为空，请结合上下文补全真实值或绑定来源{context_hint}。"
                        "可参考输入/输出 schema 使用工具进行修复。"
                    ),
                )
            )

        # exports 只能用于 loop 节点
        if "exports" in params and ntype != "loop":
            errors.append(
                ValidationError(
                    code="INVALID_SCHEMA",
                    node_id=nid,
                    field="exports",
                    message="exports 仅允许出现在 loop 节点上，用于暴露子图结果。",
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
                properties = schema.get("properties") if isinstance(schema, Mapping) else None
                allow_additional = bool(schema.get("additionalProperties")) if isinstance(schema, Mapping) else False

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

                    if isinstance(properties, Mapping) and not allow_additional:
                        unknown_fields = [k for k in params if k not in properties]
                        for field in unknown_fields:
                            errors.append(
                                ValidationError(
                                    code="UNKNOWN_PARAM",
                                    node_id=nid,
                                    field=field,
                                    message=(
                                        f"action 节点 '{nid}' 的参数 '{field}' 未在 action '{action_id}' 的 arg_schema 中定义。"
                                    ),
                                )
                            )

            # 绑定 DSL 静态校验
            def _walk_params_for_from(obj: Any, path_prefix: str = ""):
                if isinstance(obj, dict):
                    if "__from__" in obj:
                        schema_err = validate_param_binding(obj)
                        if schema_err:
                            field_label = path_prefix or "params"
                            if path_prefix and not path_prefix.startswith("params"):
                                field_label = f"params.{path_prefix}"
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field=field_label,
                                    message=f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）无效：{schema_err}",
                                )
                            )

                        source = obj.get("__from__")
                        sources: List[str] = []

                        if isinstance(source, str):
                            sources = [source]
                        elif isinstance(source, list):
                            for idx, item in enumerate(source):
                                if isinstance(item, str):
                                    sources.append(item)
                                else:
                                    errors.append(
                                        ValidationError(
                                            code="INVALID_SCHEMA",
                                            node_id=nid,
                                            field=path_prefix or "params",
                                            message=(
                                                f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）"
                                                f"的 __from__[{idx}] 类型无效，期望字符串。"
                                            ),
                                        )
                                    )

                        for src_idx, src in enumerate(sources):
                            if isinstance(src, str) and src != normalize_reference_path(src):
                                # Templated references are resolved at runtime; skip strict checks.
                                continue
                            if _is_self_reference_path(src, nid):
                                field_label = path_prefix or "params"
                                if path_prefix and not path_prefix.startswith("params"):
                                    field_label = f"params.{path_prefix}"
                                _flag_self_reference(field_label, src)
                            schema_err = _check_output_path_against_schema(
                                src,
                                nodes_by_id,
                                actions_by_id,
                                loop_body_parents,
                                context_node_id=nid,
                            )
                            if schema_err:
                                suffix = f"[{src_idx}]" if len(sources) > 1 else ""
                                field_label = path_prefix or "params"
                                if path_prefix and not path_prefix.startswith("params"):
                                    field_label = f"params.{path_prefix}"
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=field_label,
                                        message=(
                                            f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）引用无效"
                                            f"{suffix}：{schema_err}"
                                        ),
                                    )
                                )

                        agg_spec = obj.get("__agg__")
                        agg_op = agg_spec.get("op") if isinstance(agg_spec, Mapping) else agg_spec
                        if agg_op == "pipeline":
                            steps = obj.get("steps", [])
                            if isinstance(steps, list):
                                for idx, step in enumerate(steps):
                                    if not isinstance(step, Mapping):
                                        continue
                                    if step.get("op") == "filter":
                                        fld = step.get("field")
                                        for src_idx, src in enumerate(sources or [source]):
                                            if not isinstance(src, str):
                                                continue
                                            item_err = _check_array_item_field(
                                                src,
                                                fld,
                                                nodes_by_id,
                                                actions_by_id,
                                                loop_body_parents,
                                                alias_schemas,
                                            )
                                            if item_err:
                                                suffix = f"[{src_idx}]" if len(sources) > 1 else ""
                                                errors.append(
                                                    ValidationError(
                                                        code="SCHEMA_MISMATCH",
                                                        node_id=nid,
                                                        field=f"{path_prefix or 'params'}.pipeline.steps[{idx}].field",
                                                        message=(
                                                            f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）中 pipeline.steps[{idx}].field='{fld}'"
                                                            f" 无效{suffix}：{item_err}"
                                                        ),
                                                    )
                                                )

                    for k, v in list(obj.items()):
                        new_prefix = f"{path_prefix}.{k}" if path_prefix else k
                        _walk_params_for_from(v, new_prefix)
                elif isinstance(obj, list):
                    for idx, v in enumerate(obj):
                        new_prefix = f"{path_prefix}[{idx}]"
                        _walk_params_for_from(v, new_prefix)
                elif isinstance(obj, str):
                    decoded = _maybe_decode_binding_string(obj)
                    if decoded:
                        _walk_params_for_from(decoded, path_prefix)

            _walk_params_for_from(params)

            def _walk_params_for_templates(obj: Any, path_prefix: str = "") -> None:
                if isinstance(obj, str):
                    for ref in _iter_template_references(obj):
                        ref_path = normalize_reference_path(ref)
                        try:
                            ref_parts = parse_field_path(ref_path)
                        except Exception:
                            continue

                        missing_target_node = False
                        if ref_path.startswith("result_of."):
                            target_node = None
                            if ref_parts and len(ref_parts) >= 2 and isinstance(ref_parts[1], str):
                                target_node = ref_parts[1]

                            if target_node and target_node not in nodes_by_id:
                                missing_target_node = True
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=path_prefix,
                                        message=(
                                            f"action 节点 '{nid}' 的模板引用 '{ref}' 无效："
                                            f"引用了不存在的节点 '{target_node}'。"
                                        ),
                                    )
                                )

                            if missing_target_node:
                                continue

                        if _is_self_reference_path(ref_path, nid):
                            field_label = path_prefix or "params"
                            if path_prefix and not path_prefix.startswith("params"):
                                field_label = f"params.{path_prefix}"
                            _flag_self_reference(field_label, ref_path)

                        schema_err = None
                        if ref_parts:
                            alias = ref_parts[0]
                            if alias_schemas and alias in alias_schemas:
                                schema_err = _schema_path_error(alias_schemas[alias], ref_parts[1:])
                            else:
                                schema_err = _check_output_path_against_schema(
                                    ref_path,
                                    nodes_by_id,
                                    actions_by_id,
                                    loop_body_parents,
                                    context_node_id=nid,
                                )

                        if schema_err:
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field=path_prefix,
                                    message=(
                                        f"action 节点 '{nid}' 的模板引用 '{ref}' 无效：{schema_err}"
                                    ),
                                )
                            )
                elif isinstance(obj, Mapping):
                    for key, value in list(obj.items()):
                        new_prefix = f"{path_prefix}.{key}" if path_prefix else str(key)
                        _walk_params_for_templates(value, new_prefix)
                elif isinstance(obj, list):
                    for idx, value in enumerate(obj):
                        new_prefix = f"{path_prefix}[{idx}]"
                        _walk_params_for_templates(value, new_prefix)

            _walk_params_for_templates(params)

        # 2) condition 节点
        if ntype == "condition":
            true_to_node = n.get("true_to_node")
            false_to_node = n.get("false_to_node")
            true_branch_empty = true_to_node is None or (
                isinstance(true_to_node, str) and not true_to_node.strip()
            )
            false_branch_empty = false_to_node is None or (
                isinstance(false_to_node, str) and not false_to_node.strip()
            )
            if true_branch_empty and false_branch_empty:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="true_to_node/false_to_node",
                        message=(
                            "condition 节点的 true_to_node 和 false_to_node 不能同时为空，"
                            "请让 LLM 分析并使用工具修复该错误。"
                        ),
                    )
                )


            expr_val = params.get("expression") if isinstance(params, Mapping) else None
            if not isinstance(expr_val, str) or not expr_val.strip():
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="expression",
                        message=(
                            "condition 节点需要提供返回布尔值的 Jinja 表达式，"
                            "请使用 params.expression 指定逻辑，其他字段（kind/source/field/threshold 等）已弃用。"
                        ),
                    )
                )
        # 3) loop 节点
        if ntype == "loop":
            body_graph = params.get("body_subgraph") if isinstance(params, Mapping) else None
            body_exports = body_graph.get("exports") if isinstance(body_graph, Mapping) else None
            body_nodes = body_graph.get("nodes") if isinstance(body_graph, Mapping) else None
            if "exports" in params and not isinstance(body_graph, Mapping):
                errors.append(
                    ValidationError(
                        code="INVALID_SCHEMA",
                        node_id=nid,
                        field="exports",
                        message="loop 节点定义 exports 时必须提供 body_subgraph。",
                    )
                )
            if isinstance(body_exports, Mapping):
                errors.append(
                    ValidationError(
                        code="INVALID_SCHEMA",
                        node_id=nid,
                        field="body_subgraph.exports",
                        message="loop.exports 应定义在 params.exports，请从 body_subgraph 中移除。",
                    )
                )

            if isinstance(body_nodes, list) and body_nodes and not loop_body_has_action(body_graph):
                errors.append(
                    ValidationError(
                        code="INVALID_LOOP_BODY",
                        node_id=nid,
                        field="body_subgraph.nodes",
                        message=(
                            "loop 节点的 body_subgraph 至少需要一个 action 节点，"
                            "请使用规划/修复工具补充可执行步骤。"
                        ),
                    )
                )

            loop_kind = (params or {}).get("loop_kind")
            allowed_loop_kinds = {"foreach", "for_each", "while"}
            if not loop_kind:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="loop_kind",
                        message=f"loop 节点 '{nid}' 的 params.loop_kind 不能为空。",
                    )
                )
            elif loop_kind not in allowed_loop_kinds:
                errors.append(
                    ValidationError(
                        code="SCHEMA_MISMATCH",
                        node_id=nid,
                        field="loop_kind",
                        message=f"loop 节点 '{nid}' 的 loop_kind='{loop_kind}' 未被支持。",
                    )
                )

            source = params.get("source")
            item_alias = params.get("item_alias")
            if not source:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="source",
                        message=f"loop 节点 '{nid}' 需要 source 字段指明循环输入。",
                    )
                )
            elif isinstance(source, str):
                if _is_self_reference_path(source, nid):
                    _flag_self_reference("source", source)
                src_err = _check_output_path_against_schema(
                    source,
                    nodes_by_id,
                    actions_by_id,
                    loop_body_parents,
                    context_node_id=nid,
                )
                if src_err:
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=nid,
                            field="source",
                            message=f"loop 节点 '{nid}' 的 source 无效：{src_err}",
                        )
                    )
                else:
                    normalized_source = normalize_reference_path(source)
                    container_schema = _get_output_schema_at_path(
                        normalized_source,
                        nodes_by_id,
                        actions_by_id,
                        loop_body_parents,
                    )
                    actual_type = (
                        container_schema.get("type")
                        if isinstance(container_schema, Mapping)
                        else None
                    )
                    if actual_type not in {"array"}:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="source",
                                message=(
                                    f"loop 节点 '{nid}' 的 source 应该引用数组/序列"
                                    f"，但解析到的类型为 {actual_type or '未知'}，路径: {normalized_source}"
                                ),
                            )
                        )

                    if isinstance(item_alias, str):
                        nested_schema = _get_array_item_schema_from_output(
                            normalized_source,
                            nodes_by_id,
                            actions_by_id,
                            loop_body_parents,
                            context_node_id=nid,
                        )

                        if nested_schema:
                            alias_schemas = dict(alias_schemas or {})
                            alias_schemas[normalize_reference_path(item_alias)] = (
                                nested_schema
                            )
            elif isinstance(source, Mapping):
                src_err = validate_param_binding(source)
                if src_err:
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=nid,
                            field="source",
                            message=f"loop 节点 '{nid}' 的 source 无效：{src_err}",
                        )
                    )
                elif isinstance(source.get("__from__"), str):
                    normalized_source = normalize_reference_path(source["__from__"])
                    if _is_self_reference_path(normalized_source, nid):
                        _flag_self_reference("source", normalized_source)
                    schema_err = _check_output_path_against_schema(
                        normalized_source,
                        nodes_by_id,
                        actions_by_id,
                        loop_body_parents,
                        context_node_id=nid,
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
                    else:
                        container_schema = _get_output_schema_at_path(
                            normalized_source,
                            nodes_by_id,
                            actions_by_id,
                            loop_body_parents,
                        )
                        actual_type = (
                            container_schema.get("type")
                            if isinstance(container_schema, Mapping)
                            else None
                        )
                        if actual_type not in {"array"}:
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field="source",
                                    message=(
                                        f"loop 节点 '{nid}' 的 source 应该引用数组/序列"
                                        f"，但解析到的类型为 {actual_type or '未知'}，路径: {normalized_source}"
                                    ),
                                )
                            )

                        nested_schema = _get_array_item_schema_from_output(
                            normalized_source,
                            nodes_by_id,
                            actions_by_id,
                            loop_body_parents,
                            context_node_id=nid,
                        )
                        if nested_schema:
                            alias_schemas = dict(alias_schemas or {})
                            alias_schemas[normalize_reference_path(item_alias or "")] = nested_schema
            else:
                errors.append(
                    ValidationError(
                        code="INVALID_SCHEMA",
                        node_id=nid,
                        field="source",
                        message="loop 节点的 source 必须是字符串或绑定对象。",
                    )
                )

            if not isinstance(item_alias, str):
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="item_alias",
                        message=f"loop 节点 '{nid}' 需要 item_alias 用于循环体中引用当前元素。",
                    )
                )

            if loop_kind == "while" and not params.get("condition"):
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="condition",
                        message=f"loop 节点 '{nid}' 的 while 循环需要 condition。",
                    )
                )

            exports = params.get("exports")
            if exports is not None and not isinstance(exports, Mapping):
                errors.append(
                    ValidationError(
                        code="SCHEMA_MISMATCH",
                        node_id=nid,
                        field="exports",
                        message=f"loop 节点 '{nid}' 的 exports 必须是对象。",
                    )
                )
            else:
                items = exports.get("items") if isinstance(exports, Mapping) else None
                if items is not None:
                    if not isinstance(items, Mapping):
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="exports.items",
                                message=f"loop 节点 '{nid}' 的 exports.items 必须是对象。",
                            )
                        )
                    else:
                        from_node = items.get("from_node")
                        fields = items.get("fields")
                        if not isinstance(from_node, str):
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field="exports.items.from_node",
                                    message=f"loop 节点 '{nid}' 的 exports.items.from_node 需要字符串。",
                                )
                            )
                        elif not isinstance(fields, list) or not all(
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
                        elif isinstance(from_node, str) and exports:
                            body_nodes = (
                                params.get("body_subgraph") or {}
                            ).get("nodes", [])
                            body_node_map = {
                                bn.get("id"): bn for bn in body_nodes if isinstance(bn, Mapping)
                            }
                            target_node = body_node_map.get(from_node)
                            if not target_node:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field="exports.items.from_node",
                                        message=f"loop 节点 '{nid}' 的 exports.items.from_node 必须引用 body_subgraph 中的节点。",
                                    )
                                )
                            else:
                                output_schema = _get_node_output_schema(
                                    target_node, actions_by_id
                                )

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
                        body_nodes = (params.get("body_subgraph") or {}).get(
                            "nodes", []
                        )
                        extended_nodes_by_id = dict(nodes_by_id)
                        body_node_ids = set()
                        for bn in body_nodes:
                            if (
                                isinstance(bn, Mapping)
                                and isinstance(bn.get("id"), str)
                            ):
                                body_node_ids.add(bn.get("id"))
                                extended_nodes_by_id[bn.get("id")] = bn
                        for idx, agg in enumerate(aggregates):
                            if not isinstance(agg, Mapping):
                                continue
                            expr = agg.get("expr", {})
                            kind = (agg.get("kind") or "").lower()
                            source = agg.get("source") or ""
                            field = expr.get("field") if isinstance(expr, Mapping) else None
                            from_node = agg.get("from_node") if isinstance(agg, Mapping) else None

                            if not isinstance(kind, str) or kind not in {"count", "count_if", "max", "min", "sum", "avg"}:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=f"exports.aggregates[{idx}].kind",
                                        message=f"loop 节点 '{nid}' 的聚合 kind 仅支持 count/count_if/max/min/sum/avg。",
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
                                    extended_nodes_by_id,
                                    actions_by_id,
                                    loop_body_parents,
                                    context_node_id=nid,
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

                            if not isinstance(from_node, str) or from_node not in body_node_ids:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=f"exports.aggregates[{idx}].from_node",
                                        message=f"loop 节点 '{nid}' 的聚合 from_node 必须指向 body_subgraph 节点。",
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
                                if "op" not in expr:
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
                            elif kind in {"max", "min", "sum", "avg"}:
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
                loop_alias_schemas = dict(alias_schemas or {})
                if isinstance(source, str) and isinstance(item_alias, str):
                    item_schema = _get_array_item_schema_from_output(
                        source, nodes_by_id, actions_by_id, loop_body_parents
                    )
                    if item_schema:
                        loop_alias_schemas[item_alias] = item_schema

                for body_node in body_nodes:
                    if isinstance(body_node, Mapping) and isinstance(body_node.get("id"), str):
                        extended_nodes_by_id[body_node["id"]] = body_node
                _validate_nodes_recursive(
                    body_nodes,
                    extended_nodes_by_id,
                    actions_by_id,
                    loop_body_parents,
                    errors,
                    loop_alias_schemas,
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

        if hasattr(errors, "set_context"):
            errors.set_context(None)


def _is_numeric_schema_type(schema_type: Any) -> bool:
    """Return True if the schema type represents a numeric value."""

    if isinstance(schema_type, list):
        return any(t in {"number", "integer"} for t in schema_type)

    return schema_type in {"number", "integer"}


def _is_array_schema_type(schema_type: Any) -> bool:
    """Return True if the schema type represents an array."""

    if isinstance(schema_type, list):
        return "array" in schema_type

    return schema_type == "array"


def _is_array_or_string_schema_type(schema_type: Any) -> bool:
    """Return True if the schema type represents an array or string."""

    if isinstance(schema_type, list):
        return any(t in {"array", "string"} for t in schema_type)

    return schema_type in {"array", "string"}


__all__ = [
    "CONDITION_PARAM_FIELDS",
    "LOOP_PARAM_FIELDS",
    "_filter_params_by_supported_fields",
    "_is_numeric_schema_type",
    "_is_array_schema_type",
    "_is_array_or_string_schema_type",
    "_strip_illegal_exports",
    "_validate_nodes_recursive",
]
