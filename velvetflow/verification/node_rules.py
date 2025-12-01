"""Node-level validation logic for workflows."""
from typing import Any, Dict, List, Mapping, Optional

from velvetflow.models import ValidationError
from velvetflow.reference_utils import normalize_reference_path, parse_field_path

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
    _schema_path_error,
    _suggest_numeric_subfield,
    _walk_schema_with_tokens,
    validate_param_binding,
)

CONDITION_PARAM_FIELDS = {
    "kind",
    "source",
    "field",
    "value",
    "threshold",
    "min",
    "max",
    "bands",
}

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
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field=path_prefix or "params",
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
                            schema_err = _check_output_path_against_schema(
                                src, nodes_by_id, actions_by_id, loop_body_parents
                            )
                            if schema_err:
                                suffix = f"[{src_idx}]" if len(sources) > 1 else ""
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=path_prefix or "params",
                                        message=(
                                            f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）引用无效"
                                            f"{suffix}：{schema_err}"
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

                        schema_err = None
                        if ref_parts:
                            alias = ref_parts[0]
                            if alias_schemas and alias in alias_schemas:
                                schema_err = _schema_path_error(alias_schemas[alias], ref_parts[1:])
                            else:
                                schema_err = _check_output_path_against_schema(
                                    ref_path, nodes_by_id, actions_by_id, loop_body_parents
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
            condition = params if isinstance(params, dict) else {}
            kind = condition.get("kind")

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
                    "compare",
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
                    "compare": ["source", "value"],
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

                source_ref = params.get("source")
                source_paths: List[str] = []
                if isinstance(source_ref, Mapping):
                    source_err = validate_param_binding(source_ref)
                    if source_err:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="source",
                                message=(
                                    f"condition 节点 '{nid}' 的 source 绑定无效：{source_err}"
                                ),
                            )
                        )
                    elif isinstance(source_ref.get("__from__"), str):
                        source_paths.append(source_ref["__from__"])
                elif isinstance(source_ref, str):
                    source_paths.append(source_ref)
                elif isinstance(source_ref, list):
                    for idx, item in enumerate(source_ref):
                        if isinstance(item, Mapping):
                            item_err = validate_param_binding(item)
                            if item_err:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field="source",
                                        message=(
                                            f"condition 节点 '{nid}' 的 source[{idx}] 绑定无效：{item_err}"
                                        ),
                                    )
                                )
                                continue
                            if isinstance(item.get("__from__"), str):
                                source_paths.append(item["__from__"])
                        elif isinstance(item, str):
                            source_paths.append(item)
                        elif isinstance(item, (int, float, bool)):
                            continue
                        elif item is not None:
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field="source",
                                    message=(
                                        "condition 节点 '{nid}' 的 source[{idx}] 类型无效，期望字符串或包含 __from__ 的对象，收到 {typ}"
                                        .format(nid=nid, idx=idx, typ=type(item))
                                    ),
                                )
                            )
                elif isinstance(source_ref, (int, float, bool)):
                    pass
                elif source_ref is not None:
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=nid,
                            field="source",
                            message=(
                                f"condition 节点 '{nid}' 的 source 类型无效，期望字符串或包含 __from__ 的对象，收到 {type(source_ref)}"
                            ),
                        )
                    )

                field_path = params.get("field") if isinstance(params.get("field"), str) else None
                target_schemas: List[tuple[str, Mapping[str, Any] | None]] = []
                for source_path in source_paths:
                    normalized_source = normalize_reference_path(source_path)
                    target_schema = _resolve_condition_schema(
                        normalized_source,
                        field_path,
                        nodes_by_id,
                        actions_by_id,
                        loop_body_parents,
                        alias_schemas,
                    )
                    target_schemas.append((normalized_source, target_schema))
                    if field_path and target_schema is None:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="field",
                                message=(
                                    f"condition 节点 '{nid}' 的引用 '{normalized_source}.{field_path}' 无法在 schema 中找到或缺少类型信息。"
                                ),
                            )
                        )

                if kind == "list_not_empty":
                    for normalized_source, target_schema in target_schemas:
                        if target_schema:
                            target_type = target_schema.get("type")
                            if not _is_array_schema_type(target_type):
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field="field" if field_path else "source",
                                        message=(
                                            f"condition 节点 '{nid}' 的引用 '{normalized_source if not field_path else f'{normalized_source}.{field_path}'}' 类型为 {target_type}，"
                                            "期望数组用于 list_not_empty。"
                                        ),
                                    )
                                )

                if kind in {"any_greater_than", "all_less_than", "greater_than", "less_than", "between"}:
                    threshold = params.get("threshold")
                    if kind == "between":
                        min_val = params.get("min")
                        max_val = params.get("max")
                        for field_name, val in ("min", min_val), ("max", max_val):
                            if not isinstance(val, (int, float)):
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=field_name,
                                        message=(
                                            f"condition 节点 '{nid}' 的 {field_name} 必须是数字，用于范围比较。"
                                        ),
                                    )
                                )
                    else:
                        if not isinstance(threshold, (int, float)):
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field="threshold",
                                    message=(
                                        f"condition 节点 '{nid}' 的 threshold 必须是数字，用于比较。"
                                    ),
                                )
                            )

                    if field_path and kind in {
                        "any_greater_than",
                        "all_less_than",
                        "greater_than",
                        "less_than",
                        "between",
                    }:
                        for normalized_source, target_schema in target_schemas:
                            if target_schema:
                                field_type = target_schema.get("type")
                                is_numeric_type = _is_numeric_schema_type(field_type)
                                if not is_numeric_type:
                                    errors.append(
                                        ValidationError(
                                            code="SCHEMA_MISMATCH",
                                            node_id=nid,
                                            field="field",
                                            message=(
                                                f"condition 节点 '{nid}' 的字段 '{field_path}' 类型为 {field_type}，"
                                                "无法与数值阈值比较。"
                                            ),
                                        )
                                    )

                if kind == "contains" and field_path:
                    for normalized_source, target_schema in target_schemas:
                        if not target_schema:
                            continue

                        target_type = target_schema.get("type")
                        if target_type and not _is_array_or_string_schema_type(target_type):
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field="field",
                                    message=(
                                        f"condition 节点 '{nid}' 的引用 '{normalized_source}.{field_path}' 类型为 {target_type}，"
                                        "需要数组或字符串以支持 contains。"
                                    ),
                                )
                            )

        # 3) loop 节点
        if ntype == "loop":
            if nid in loop_body_parents:
                parent_loop = loop_body_parents.get(nid)
                errors.append(
                    ValidationError(
                        code="INVALID_SCHEMA",
                        node_id=nid,
                        field="parent_node_id",
                        message=(
                            f"不允许嵌套循环：loop 节点 '{nid}' 位于父循环 '{parent_loop}' 的 body_subgraph 中。"
                        ),
                    )
                )
            body_graph = params.get("body_subgraph") if isinstance(params, Mapping) else None
            body_exports = body_graph.get("exports") if isinstance(body_graph, Mapping) else None
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
                src_err = _check_output_path_against_schema(
                    source, nodes_by_id, actions_by_id, loop_body_parents
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
                    schema_err = _check_output_path_against_schema(
                        normalized_source, nodes_by_id, actions_by_id, loop_body_parents
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
                        nested_schema = _get_array_item_schema_from_output(
                            normalized_source, nodes_by_id, actions_by_id, loop_body_parents
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
                                    source, nodes_by_id, actions_by_id, loop_body_parents
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

                            if not isinstance(from_node, str) or (params and from_node not in (params.get("body_subgraph", {}).get("nodes", []) or [])):
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
