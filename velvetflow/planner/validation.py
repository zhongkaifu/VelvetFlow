"""Static validation helpers for planner workflows."""

from collections import deque
from typing import Any, Dict, List, Mapping, Optional

from velvetflow.models import ValidationError, Workflow
from velvetflow.planner.action_guard import _index_actions_by_id


def _index_nodes_by_id(workflow: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in workflow.get("nodes", [])}


def _check_output_path_against_schema(
    source_path: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
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

    if node_id not in nodes_by_id:
        return f"路径 '{source_path}' 引用的节点 '{node_id}' 不存在。"

    node = nodes_by_id[node_id]
    action_id = node.get("action_id")
    node_type = node.get("type")

    # 控制节点（如 condition / loop）没有 action_id，也没有可用的 output_schema。
    # 在这种情况下跳过 schema 校验，允许下游继续引用其动态结果。
    if not action_id and node_type in {"condition", "loop", "start", "end"}:
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

    current: Mapping[str, Any] = schema
    idx = 0
    while idx < len(fields):
        name = fields[idx]
        typ = current.get("type")

        if typ == "array":
            current = current.get("items") or {}
            continue

        if typ == "object":
            props = current.get("properties") or {}
            if name not in props:
                return f"字段 '{name}' 不存在，已知字段有: {list(props.keys())}"
            current = props[name]
            idx += 1
            continue

        return f"字段路径 '{'.'.join(fields)}' 与 schema 类型 '{typ}' 不匹配（期望 object/array）。"

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
    source: str, nodes_by_id: Dict[str, Dict[str, Any]], actions_by_id: Dict[str, Dict[str, Any]]
) -> Optional[Mapping[str, Any]]:
    err = _check_output_path_against_schema(source, nodes_by_id, actions_by_id)
    if err:
        return None

    parts = source.split(".")
    node_id = parts[1]
    first_field = parts[2] if len(parts) >= 3 else None
    node = nodes_by_id.get(node_id)
    action_def = actions_by_id.get(node.get("action_id")) if node else None
    if not action_def:
        return None

    schema = (action_def.get("output_schema") or {}).get("properties", {})
    if first_field not in schema:
        return None

    return (schema.get(first_field) or {}).get("items")


def _check_array_item_field(
    source: str,
    field: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    item_schema = _get_array_item_schema_from_output(source, nodes_by_id, actions_by_id)
    if not item_schema:
        return None

    return _schema_path_error(item_schema, [field])


def validate_completed_workflow(
    workflow: Dict[str, Any],
    action_registry: List[Dict[str, Any]],
) -> List[ValidationError]:
    errors: List[ValidationError] = []

    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])

    nodes_by_id = _index_nodes_by_id(workflow)
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

    # ---------- 节点校验 ----------
    for n in nodes:
        nid = n["id"]
        ntype = n.get("type")
        action_id = n.get("action_id")
        params = n.get("params", {})

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
                                source, nodes_by_id, actions_by_id
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
                                            source, fld, nodes_by_id, actions_by_id
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
                                        message=f"condition 节点 '{nid}' 的 field='{fld}' 无效：{item_err}",
                                    )
                                )

                source = params.get("source")
                source_path: Optional[str] = None
                if isinstance(source, dict) and "__from__" in source:
                    binding_err = validate_param_binding(source)
                    if binding_err:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="source",
                                message=(
                                    f"condition 节点 '{nid}' 的 source 绑定无效：{binding_err}"
                                ),
                            )
                        )
                    source_path = source.get("__from__") if isinstance(source.get("__from__"), str) else None
                elif isinstance(source, str):
                    source_path = source

                if source_path:
                    schema_err = _check_output_path_against_schema(source_path, nodes_by_id, actions_by_id)
                    if schema_err:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="source",
                                message=f"condition 节点 '{nid}' 的 source 引用无效：{schema_err}",
                            )
                        )

                    if kind in {"any_greater_than", "all_less_than", "contains"}:
                        fld = params.get("field")
                        if isinstance(fld, str):
                            item_err = _check_array_item_field(source_path, fld, nodes_by_id, actions_by_id)
                            if item_err:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field="field",
                                        message=f"condition 节点 '{nid}' 的 field='{fld}' 无效：{item_err}",
                                    )
                                )

        # 3) loop 节点
        if ntype == "loop":
            loop_kind = params.get("loop_kind") if isinstance(params, dict) else None
            if loop_kind not in {"for_each", "while"}:
                errors.append(
                    ValidationError(
                        code="SCHEMA_MISMATCH",
                        node_id=nid,
                        field="loop_kind",
                        message=f"loop 节点 '{nid}' 的 loop_kind 必须是 for_each 或 while。",
                    )
                )
            source = params.get("source") if isinstance(params, dict) else None
            if not source:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="source",
                        message=f"loop 节点 '{nid}' 缺少 source 字段。",
                    )
                )
            elif isinstance(source, str):
                schema_err = _check_output_path_against_schema(source, nodes_by_id, actions_by_id)
                if schema_err:
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=nid,
                            field="source",
                            message=f"loop 节点 '{nid}' 的 source 引用无效：{schema_err}",
                        )
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

    return errors


def validate_param_binding_and_schema(binding: Mapping[str, Any], workflow: Dict[str, Any], action_registry: List[Dict[str, Any]]):
    """Validate parameter bindings in the context of workflow/action schemas."""

    nodes_by_id = _index_nodes_by_id(workflow)
    actions_by_id = _index_actions_by_id(action_registry)
    binding_err = validate_param_binding(binding)
    if binding_err:
        return binding_err

    src = binding.get("__from__")
    return _check_output_path_against_schema(src, nodes_by_id, actions_by_id) if isinstance(src, str) else None


__all__ = [
    "validate_param_binding",
    "validate_completed_workflow",
    "validate_param_binding_and_schema",
]
