"""Static validation helpers for VelvetFlow workflows."""

import json
from collections import deque
from contextlib import contextmanager
from typing import Any, Dict, List, Mapping, Optional, Set

from velvetflow.loop_dsl import build_loop_output_schema, index_loop_body_nodes
from velvetflow.models import ValidationError, Workflow
from velvetflow.reference_utils import normalize_reference_path, parse_field_path

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


class _RepairingErrorList(list):
    """Record validation errors while stripping offending fields in-place.

    校验阶段如果发现字段错误，会直接删除触发错误的字段，再返回错误信息给 LLM，方便
    通过工具重新生成该字段。通过 ``contextualize`` 设置上下文时，会尝试根据上下文
    信息（当前节点或边）删除对应字段。
    """

    def __init__(self, workflow: Mapping[str, Any]):
        super().__init__()
        self._context: Optional[Mapping[str, Any]] = None

    @contextmanager
    def contextualize(self, ctx: Mapping[str, Any]):
        prev = self._context
        self._context = ctx
        try:
            yield
        finally:
            self._context = prev

    def set_context(self, ctx: Optional[Mapping[str, Any]]):
        self._context = ctx

    def append(self, error: ValidationError):  # type: ignore[override]
        self._repair(error)
        return super().append(error)

    def extend(self, values):  # type: ignore[override]
        for val in values:
            self.append(val)

    def _repair(self, error: ValidationError) -> None:
        ctx = self._context or {}
        if not isinstance(ctx, Mapping):
            return

        field = getattr(error, "field", None)
        if not field:
            return

        target = ctx.get("node") or ctx.get("edge")
        if isinstance(target, Mapping):
            self._remove_field(target, field)

    def _remove_field(self, container: Mapping[str, Any], field: str) -> None:
        try:
            parts = parse_field_path(field)
        except Exception:
            return

        if not parts:
            return

        current: Any = container
        for part in parts[:-1]:
            if isinstance(part, int):
                if isinstance(current, list) and 0 <= part < len(current):
                    current = current[part]
                else:
                    return
            else:
                if isinstance(current, Mapping):
                    current = current.get(part)
                else:
                    return

        last = parts[-1]
        if isinstance(last, int):
            if isinstance(current, list) and 0 <= last < len(current):
                current.pop(last)
        elif isinstance(current, dict):
            current.pop(last, None)


def _index_actions_by_id(action_registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {a["action_id"]: a for a in action_registry}


def _get_node_output_schema(
    node: Mapping[str, Any] | None, actions_by_id: Mapping[str, Mapping[str, Any]]
) -> Optional[Mapping[str, Any]]:
    if isinstance(node, Mapping):
        node_schema = node.get("out_params_schema")
        if isinstance(node_schema, Mapping):
            return node_schema

        action_id = node.get("action_id")
        if isinstance(action_id, str):
            action_def = actions_by_id.get(action_id)
            if isinstance(action_def, Mapping):
                output_schema = action_def.get("output_schema")
                if isinstance(output_schema, Mapping):
                    return output_schema

    return None


def _maybe_decode_binding_string(raw: str) -> Optional[Any]:
    """Attempt to parse a JSON-like binding stored as string.

    Planner 生成的 workflow 有时会携带字符串化的参数绑定（例如 "{\"__from__\": ...}"），
    否则这些绑定只能在执行时才暴露错误。通过在验证阶段尝试解析，可提早暴露
    引用错误并让 LLM 通过工具修复。
    """

    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text.startswith("{"):
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, Mapping) and "__from__" in parsed:
        return parsed

    return None


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


def _strip_illegal_exports(node: Mapping[str, Any]) -> bool:
    """Remove ``params.exports`` from non-loop nodes to avoid repeated errors.

    校验阶段如果发现非 loop 节点携带 ``exports`` 字段，会直接删除该字段并返回
    ``ValidationError``，提示 LLM 使用合适的工具重新生成合法的 exports。这样可以
    避免错误字段阻塞后续修复流程。返回值表示是否执行了删除动作。
    """

    if node.get("type") == "loop":
        return False

    params = node.get("params")
    if not isinstance(params, Mapping) or "exports" not in params:
        return False

    new_params = {k: v for k, v in params.items() if k != "exports"}
    node["params"] = new_params
    return True


def precheck_loop_body_graphs(workflow_raw: Mapping[str, Any] | Any) -> List[ValidationError]:
    """Detect loop body graphs that refer to nonexistent nodes.

    Loop 子图无需再声明 entry/exit 或 edges，执行流程会根据参数绑定推断。
    这里仅保留对节点基本结构的检查，避免因缺少 body_subgraph.nodes 而导致
    难以理解的错误信息。
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
                        if isinstance(source, str):
                            schema_err = _check_output_path_against_schema(
                                source, nodes_by_id, actions_by_id, loop_body_parents
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
                                            alias_schemas,
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
                elif isinstance(obj, str):
                    decoded = _maybe_decode_binding_string(obj)
                    if decoded:
                        _walk_params_for_from(decoded, path_prefix)

            _walk_params_for_from(params)

            # 模板引用静态校验（例如 "{{news_item.title}}"）
            def _walk_params_for_template_refs(obj: Any, path_prefix: str = ""):
                if isinstance(obj, str):
                    normalized = normalize_reference_path(obj)
                    if isinstance(normalized, str):
                        parts = normalized.split(".")
                        alias = parts[0] if parts else None

                        if alias_schemas and alias in alias_schemas:
                            alias_schema = alias_schemas.get(alias)
                            err = _schema_path_error(alias_schema or {}, parts[1:])
                            if err:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=path_prefix or "params",
                                        message=(
                                            f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）"
                                            f"引用无效：别名 '{alias}' {err}"
                                        ),
                                    )
                                )
                        if normalized.startswith("result_of."):
                            schema_err = _check_output_path_against_schema(
                                normalized, nodes_by_id, actions_by_id, loop_body_parents
                            )
                            if schema_err:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=path_prefix or "params",
                                        message=(
                                            f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）"
                                            f"引用无效：{schema_err}"
                                        ),
                                    )
                                )
                elif isinstance(obj, Mapping):
                    for key, value in obj.items():
                        new_prefix = f"{path_prefix}.{key}" if path_prefix else str(key)
                        _walk_params_for_template_refs(value, new_prefix)
                elif isinstance(obj, list):
                    for idx, value in enumerate(obj):
                        new_prefix = f"{path_prefix}[{idx}]" if path_prefix else f"[{idx}]"
                        _walk_params_for_template_refs(value, new_prefix)

            _walk_params_for_template_refs(params)

        # 2) condition 节点
        if ntype == "condition":
            for field_name in ("true_to_node", "false_to_node"):
                field_missing = field_name not in n
                target = n.get(field_name)

                if field_missing:
                    errors.append(
                        ValidationError(
                            code="MISSING_REQUIRED_PARAM",
                            node_id=nid,
                            field=field_name,
                            message=f"condition 节点 '{nid}' 缺少 {field_name} 字段（可填 null 表示分支结束）。",
                        )
                    )
                    continue

                if target is None or target == "null":
                    continue
                if not isinstance(target, str):
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=nid,
                            field=field_name,
                            message=f"condition 节点 '{nid}' 的 {field_name} 需要是字符串或 null。",
                        )
                    )
                elif target not in node_ids:
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=nid,
                            field=field_name,
                            message=f"condition 节点 '{nid}' 的 {field_name} 指向未知节点 '{target}'。",
                        )
                    )

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
                    source_path: Optional[str] = None
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
                            source_path = source_ref["__from__"]
                    elif isinstance(source_ref, str):
                        source_path = source_ref
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

                    if kind in {"any_greater_than", "all_less_than", "contains"}:
                        fld = params.get("field")
                        if isinstance(source_path, str) and isinstance(fld, str):
                            item_err = _check_array_item_field(
                                source_path,
                                fld,
                                nodes_by_id,
                                actions_by_id,
                                loop_body_parents,
                                alias_schemas,
                            )
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

                        if kind in {"any_greater_than", "all_less_than"}:
                            fld = params.get("field")
                            if isinstance(source_path, str) and isinstance(fld, str):
                                item_schema = _get_array_item_schema_from_output(
                                    source_path, nodes_by_id, actions_by_id
                                )
                                field_schema = _get_field_schema_from_item(item_schema, fld)
                                if field_schema:
                                    field_type = field_schema.get("type")
                                    is_numeric_type = _is_numeric_schema_type(field_type)
                                    if not is_numeric_type:
                                        errors.append(
                                            ValidationError(
                                                code="SCHEMA_MISMATCH",
                                                node_id=nid,
                                                field="field",
                                                message=(
                                                    f"condition 节点 '{nid}' 的字段 '{fld}' 类型为 {field_type}，"
                                                    "无法与数值阈值比较。"
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
                    source, nodes_by_id, actions_by_id, loop_body_parents
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

            item_alias = params.get("item_alias") if isinstance(params, Mapping) else None
            if not isinstance(item_alias, str) or not item_alias.strip():
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="item_alias",
                        message=f"loop 节点 '{nid}' 缺少 item_alias 或其值为空字符串。",
                    )
                )

            if loop_kind == "while":
                condition = params.get("condition") if isinstance(params, Mapping) else None
                if not condition:
                    errors.append(
                        ValidationError(
                            code="MISSING_REQUIRED_PARAM",
                            node_id=nid,
                            field="condition",
                            message=f"loop 节点 '{nid}' (loop_kind=while) 需要 condition。",
                        )
                    )
                elif isinstance(condition, Mapping):
                    err = validate_param_binding_and_schema(
                        condition, {"nodes": list(nodes_by_id.values())}, list(actions_by_id.values())
                    )
                    if err:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="condition",
                                message=f"loop 节点 '{nid}' 的 condition 无效：{err}",
                            )
                        )
                elif isinstance(condition, str):
                    schema_err = _check_output_path_against_schema(
                        condition, nodes_by_id, actions_by_id, loop_body_parents
                    )
                    if schema_err:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="condition",
                                message=f"loop 节点 '{nid}' 的 condition 引用无效：{schema_err}",
                            )
                        )

            # exports 静态校验
            body_graph = params.get("body_subgraph") if isinstance(params, Mapping) else None
            exports = params.get("exports") if isinstance(params, dict) else None
            if not isinstance(exports, Mapping) and isinstance(body_graph, Mapping):
                body_exports = body_graph.get("exports")
                if isinstance(body_exports, Mapping):
                    exports = body_exports

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
                if isinstance(body_graph, Mapping):
                    body_node_map = {
                        bn.get("id"):
                        bn
                        for bn in body_graph.get("nodes", [])
                        if isinstance(bn, Mapping) and isinstance(bn.get("id"), str)
                    }
                    body_nodes = list(body_node_map.keys())
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
                        if not isinstance(from_node, str) or (body_nodes and from_node not in body_nodes):
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
                        elif isinstance(from_node, str) and body_nodes:
                            target_node = body_node_map.get(from_node)
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

                            if not isinstance(from_node, str) or (body_nodes and from_node not in body_nodes):
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
    elif isinstance(obj, str):
        decoded = _maybe_decode_binding_string(obj)
        if decoded:
            bindings.extend(_collect_param_bindings(decoded, prefix))

    return bindings


def _check_output_path_against_schema(
    source_path: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Optional[Mapping[str, str]] = None,
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
    source_path = normalize_reference_path(source_path)
    if not isinstance(source_path, str):
        return f"source/__from__ 应该是字符串，但收到类型: {type(source_path)}"

    try:
        parts = parse_field_path(source_path)
    except Exception:
        return f"source/__from__ 不是合法的路径字符串: {source_path}"

    if len(parts) < 2 or parts[0] != "result_of":
        return None

    node_id = parts[1]
    rest_path = parts[2:]

    loop_body_parents = loop_body_parents or {}

    if node_id not in nodes_by_id:
        if node_id in loop_body_parents:
            parent_loop = loop_body_parents.get(node_id)
            return (
                f"循环外不允许直接引用子图节点 '{node_id}'，请通过 loop 节点 '{parent_loop}' 的 exports 暴露输出。"
            )
        return f"路径 '{source_path}' 引用的节点 '{node_id}' 不存在。"

    node = nodes_by_id[node_id]
    action_id = node.get("action_id")
    node_type = node.get("type")

    # loop 节点使用虚拟 output_schema 校验 exports。
    if node_type == "loop":
        loop_schema = build_loop_output_schema(node.get("params") or {})
        if not loop_schema:
            return f"loop 节点 '{node_id}' 未定义 exports，无法暴露给外部引用。"

        effective_path = rest_path[1:] if rest_path and rest_path[0] == "exports" else rest_path
        err = _schema_path_error(loop_schema, effective_path)
        if err:
            return f"路径 '{source_path}' 无效：{err}"
        return None
    # 控制节点（如 condition / start / end）没有 action_id，也没有可用的 output_schema。
    # 若下游尝试访问子字段（例如误写了 exports/items），应当直接报错，避免静默跳过校验。
    if not action_id and node_type in {"condition", "start", "end"}:
        if node_type == "start":
            return None
        if rest_path:
            path_str = ".".join(rest_path)
            return (
                f"路径 '{source_path}' 引用的 {node_type} 节点没有 output_schema，",
                f"无法访问子字段 '{path_str}'。",
            )
        return None

    if not action_id:
        return f"路径 '{source_path}' 引用的节点 '{node_id}' 没有 action_id，无法从 output_schema 校验。"

    action_def = actions_by_id.get(action_id)
    if not action_def:
        return f"路径 '{source_path}' 引用的节点 '{node_id}' 的 action_id='{action_id}' 不在 Action Registry 中。"

    output_schema = _get_node_output_schema(node, actions_by_id)
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


def _normalize_field_tokens(fields: List[Any]) -> List[Any]:
    normalized_fields: List[Any] = []
    for f in fields:
        if isinstance(f, str):
            try:
                normalized_fields.extend(parse_field_path(f))
            except Exception:
                normalized_fields.append(f)
        else:
            normalized_fields.append(f)

    return normalized_fields


def _get_builtin_field_schema(schema: Mapping[str, Any], field: str) -> Optional[Mapping[str, Any]]:
    if not isinstance(schema, Mapping):
        return None

    if field != "length":
        return None

    typ = schema.get("type")
    types: Set[Any] = set()
    if isinstance(typ, list):
        types.update(typ)
    elif typ is not None:
        types.add(typ)

    supports_length = not types or any(t in {"array", "object", "string"} for t in types)
    if supports_length:
        return {"type": "integer"}

    return None


def _walk_schema_with_tokens(
    schema: Mapping[str, Any], normalized_fields: List[Any]
) -> Optional[Mapping[str, Any]]:
    if not isinstance(schema, Mapping):
        return None

    current: Mapping[str, Any] = schema
    idx = 0
    while idx < len(normalized_fields):
        name = normalized_fields[idx]
        typ = current.get("type")

        if isinstance(name, int):
            if typ != "array":
                return None
            current = current.get("items") or {}
            idx += 1
            continue

        builtin_schema = _get_builtin_field_schema(current, name)
        if builtin_schema is not None:
            current = builtin_schema
            idx += 1
            continue

        if typ == "array":
            current = current.get("items") or {}
            continue

        if typ == "object" or typ is None:
            props = current.get("properties") or {}
            current = props.get(name)
            if current is None:
                return None
            idx += 1
            continue

        if idx == len(normalized_fields) - 1:
            return current

        return None

    return current


def _schema_path_error(schema: Mapping[str, Any], fields: List[Any]) -> Optional[str]:
    """Check whether a dotted/Indexed field path exists in a JSON schema."""

    if not isinstance(schema, Mapping):
        return "output_schema 不是对象，无法校验字段路径。"

    normalized_fields = _normalize_field_tokens(fields)

    current: Mapping[str, Any] = schema
    idx = 0
    while idx < len(normalized_fields):
        name = normalized_fields[idx]
        typ = current.get("type")

        if isinstance(name, int):
            if typ != "array":
                return f"字段路径 '{'.'.join(map(str, normalized_fields))}' 与 schema 类型 '{typ}' 不匹配（期望 array）。"
            current = current.get("items") or {}
            idx += 1
            continue

        builtin_schema = _get_builtin_field_schema(current, name)
        if builtin_schema is not None:
            current = builtin_schema
            idx += 1
            continue

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

        return f"字段路径 '{'.'.join(map(str, normalized_fields))}' 与 schema 类型 '{typ}' 不匹配（期望 object/array）。"

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
        "join",
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
    *,
    skip_path_check: bool = False,
) -> Optional[Mapping[str, Any]]:
    normalized_source = normalize_reference_path(source)
    if not skip_path_check:
        err = _check_output_path_against_schema(
            normalized_source, nodes_by_id, actions_by_id, loop_body_parents
        )
        if err:
            return None

    try:
        parts = parse_field_path(normalized_source)
    except Exception:
        return None

    if len(parts) < 2 or parts[0] != "result_of":
        return None

    node_id = parts[1]
    first_field = None
    for token in parts[2:]:
        if isinstance(token, str):
            first_field = token
            break
    node = nodes_by_id.get(node_id)
    node_type = node.get("type") if node else None
    if node_type == "loop":
        loop_params = node.get("params") or {}
        if first_field == "items":
            body_exports = (
                (loop_params.get("body_subgraph") or {}).get("exports")
                if isinstance(loop_params, Mapping)
                else None
            )
            items_spec = (body_exports or loop_params.get("exports") or {}).get(
                "items"
            )
            items_schema = _get_loop_items_schema_from_exports(
                items_spec, node, nodes_by_id, actions_by_id
            )
            if items_schema:
                return items_schema

        schema = build_loop_output_schema(loop_params) or {}
        props = schema.get("properties") or {}
        return (props.get(first_field) or {}).get("items") if first_field in props else None

    schema_obj = _get_node_output_schema(node, actions_by_id) or {}
    schema = schema_obj.get("properties", {}) if isinstance(schema_obj, Mapping) else {}
    if first_field not in schema:
        return None

    return (schema.get(first_field) or {}).get("items")


def _get_output_schema_at_path(
    source: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Optional[Mapping[str, str]] = None,
) -> Optional[Mapping[str, Any]]:
    normalized_source = normalize_reference_path(source)
    try:
        parts = parse_field_path(normalized_source)
    except Exception:
        return None

    if len(parts) < 2 or parts[0] != "result_of":
        return None

    node_id = parts[1]
    rest_path = parts[2:]
    node = nodes_by_id.get(node_id)
    node_type = node.get("type") if node else None

    if node_type == "loop":
        params = node.get("params") if isinstance(node, Mapping) else {}
        loop_schema = build_loop_output_schema(params or {})
        if not loop_schema:
            return None

        effective_path = rest_path[1:] if rest_path and rest_path[0] == "exports" else rest_path
        return _walk_schema_with_tokens(
            loop_schema, _normalize_field_tokens(list(effective_path))
        )

    schema_obj = _get_node_output_schema(node, actions_by_id) or {}
    return _walk_schema_with_tokens(schema_obj, _normalize_field_tokens(list(rest_path)))


def _get_field_schema_from_item(
    item_schema: Optional[Mapping[str, Any]], field: str
) -> Optional[Mapping[str, Any]]:
    """Extract the schema definition for a field inside an array item schema."""

    if not item_schema:
        return None

    try:
        parts = parse_field_path(field)
    except Exception:
        return None

    current: Mapping[str, Any] = item_schema
    for token in parts:
        typ = current.get("type")

        if isinstance(token, int):
            if typ != "array":
                return None
            current = current.get("items") or {}
            continue

        if typ == "array":
            current = current.get("items") or {}

        if typ in {None, "object"}:
            props = current.get("properties") or {}
            if token not in props:
                return None
            current = props[token]
            continue

        return None

    return current


def _is_numeric_schema_type(schema_type: Any) -> bool:
    """Return True if the schema type represents a numeric value."""

    if isinstance(schema_type, list):
        return any(t in {"number", "integer"} for t in schema_type)

    return schema_type in {"number", "integer"}


def _get_field_schema(schema: Mapping[str, Any], field: str) -> Optional[Mapping[str, Any]]:
    """Return the schema of a dotted field path inside an object/array schema."""

    if not isinstance(schema, Mapping) or not isinstance(field, str):
        return None

    current: Mapping[str, Any] = schema
    try:
        parts = parse_field_path(field)
    except Exception:
        return None

    for part in parts:
        if not isinstance(current, Mapping):
            return None

        current_type = current.get("type")
        if isinstance(part, int):
            if current_type != "array":
                return None
            current = current.get("items") or {}
            continue

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

    output_schema = _get_node_output_schema(src_node, actions_by_id)

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
    alias_schemas: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Optional[str]:
    normalized_source = normalize_reference_path(source)

    alias_schema = (alias_schemas or {}).get(normalized_source)
    if alias_schema:
        return _schema_path_error(alias_schema, [field])

    path_error = _check_output_path_against_schema(
        normalized_source, nodes_by_id, actions_by_id, loop_body_parents
    )
    if path_error:
        return path_error

    if field == "length":
        container_schema = _get_output_schema_at_path(
            normalized_source, nodes_by_id, actions_by_id, loop_body_parents
        )
        if container_schema is None:
            item_schema = _get_array_item_schema_from_output(
                normalized_source,
                nodes_by_id,
                actions_by_id,
                loop_body_parents,
                skip_path_check=True,
            )
            if item_schema is not None:
                container_schema = {"type": "array", "items": item_schema}

        if _get_builtin_field_schema(container_schema or {}, field) is not None:
            return None

    item_schema = _get_array_item_schema_from_output(
        normalized_source,
        nodes_by_id,
        actions_by_id,
        loop_body_parents,
        skip_path_check=True,
    )
    if not item_schema:
        return f"路径 '{normalized_source}' 不是数组输出，无法访问字段 '{field}'。"

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
                        "检测到非 loop 节点携带 exports，已删除该字段，请使用工具在合法位置重"
                        "新生成 exports。"
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
