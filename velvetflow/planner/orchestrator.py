# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Top-level orchestrator for the two-pass planner.

本模块聚合了结构规划、参数补全与错误修复三个阶段，主要通过 LLM 进行
workflow 结构和参数的两阶段推理，然后在必要时向 LLM 提供错误上下文进行
自修复。为了降低调用方的认知负担，核心入口 `plan_workflow_with_two_pass`
会处理：

* 骨架阶段：根据自然语言需求触发 LLM 生成节点的初稿，并保证 Action
  Registry 覆盖。
* 参数阶段：在约定的 prompt 语境下补全每个 action 的参数，并校验返回
  结构符合 `Workflow` Pydantic 模型。
* 修复阶段：当校验失败时，将 ValidationError 列表和上一次合法结构回传给
  LLM，请求其输出完全序列化的 workflow 字典；若多轮修复仍失败则返回最后
  一次通过校验的版本。

返回值统一为 `Workflow` 对象，调用方无需关心 LLM 返回的原始 JSON 格式。
"""

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    TraceContext,
    child_span,
    log_error,
    log_event,
    log_info,
    log_json,
    log_llm_usage,
    log_section,
    log_success,
    log_warn,
    use_trace_context,
)
from velvetflow.models import PydanticValidationError, ValidationError, Workflow
from velvetflow.loop_dsl import index_loop_body_nodes, iter_workflow_and_loop_body_nodes
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.params import fill_params_with_llm
from velvetflow.planner.repair import (
    _convert_pydantic_errors,
    _make_failure_validation_error,
    _repair_with_llm_and_fallback,
)
from velvetflow.planner.repair_tools import (
    align_loop_body_alias_references,
    fill_loop_exports_defaults,
    fix_missing_loop_exports_items,
    normalize_binding_paths,
    _apply_local_repairs_for_unknown_params,
)
from velvetflow.planner.structure import plan_workflow_structure_with_llm
from velvetflow.planner.update import update_workflow_with_llm
from velvetflow.reference_utils import parse_field_path
from velvetflow.verification import (
    precheck_loop_body_graphs,
    run_lightweight_static_rules,
    validate_completed_workflow,
)
from velvetflow.verification.validation import (
    _get_array_item_schema_from_output,
    _get_field_schema_from_item,
    _get_output_schema_at_path,
    _is_numeric_schema_type,
)
from velvetflow.search import HybridActionSearchService


def _index_actions_by_id(
    action_registry: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Build a quick lookup table for action metadata keyed by `action_id`."""

    return {a["action_id"]: a for a in action_registry}


def _schema_primary_type(schema: Optional[Mapping[str, Any]]) -> Any:
    if not isinstance(schema, Mapping):
        return None

    typ = schema.get("type")
    if isinstance(typ, list):
        return typ[0] if typ else None
    return typ


def _schema_types(schema: Optional[Mapping[str, Any]]) -> set[str]:
    if not isinstance(schema, Mapping):
        return set()

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return {str(t) for t in schema_type}
    if isinstance(schema_type, str):
        return {schema_type}
    return set()


def _schemas_compatible(expected: Optional[Mapping[str, Any]], actual: Optional[Mapping[str, Any]]) -> bool:
    """Lightweight JSON Schema compatibility check for planner-time validation."""

    if not expected:
        return True
    if not actual:
        return False

    expected_types = _schema_types(expected)
    actual_types = _schema_types(actual)
    if not expected_types or not actual_types:
        return True

    overlapping = expected_types & actual_types
    if not overlapping:
        return False

    if "array" in overlapping:
        expected_items = expected.get("items") if isinstance(expected.get("items"), Mapping) else None
        actual_items = actual.get("items") if isinstance(actual.get("items"), Mapping) else None
        if expected_items or actual_items:
            return _schemas_compatible(expected_items, actual_items)

    return True


def _iter_bindings_with_schema(
    value: Any, schema: Optional[Mapping[str, Any]], path_prefix: str
) -> Iterable[tuple[str, Mapping[str, Any], Optional[Mapping[str, Any]]]]:
    """Yield all bindings together with their expected schema context."""

    if isinstance(value, Mapping):
        if "__from__" in value:
            yield path_prefix, value, schema
        for key, val in value.items():
            next_prefix = f"{path_prefix}.{key}" if path_prefix else str(key)
            child_schema: Optional[Mapping[str, Any]] = None
            if isinstance(schema, Mapping):
                schema_type = schema.get("type")
                if schema_type == "array":
                    child_schema = schema.get("items")
                elif schema_type in {None, "object"}:
                    props = schema.get("properties") or {}
                    child_schema = props.get(key) if isinstance(props, Mapping) else None
            yield from _iter_bindings_with_schema(val, child_schema, next_prefix)
        return

    if isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
        item_schema = schema.get("items") if isinstance(schema, Mapping) else None
        for idx, item in enumerate(value):
            yield from _iter_bindings_with_schema(item, item_schema, f"{path_prefix}[{idx}]")


def _binding_io_schemas(
    source_schema: Optional[Mapping[str, Any]], binding: Mapping[str, Any]
) -> tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]]]:
    """Infer aggregator input/output schemas for a binding."""

    if not isinstance(binding, Mapping):
        return source_schema, None

    agg = binding.get("__agg__")
    agg_input: Optional[Mapping[str, Any]] = None
    agg_output: Optional[Mapping[str, Any]] = None

    if isinstance(agg, Mapping):
        if isinstance(agg.get("input_type"), str):
            agg_input = {"type": agg["input_type"]}
        if isinstance(agg.get("output_type"), str):
            agg_output = {"type": agg["output_type"]}
        agg = agg.get("op")

    if agg in {"count", "count_if"}:
        agg_input = agg_input or {"type": "array"}
        agg_output = agg_output or {"type": "integer"}
    if agg in {"join", "format_join", "filter_map"}:
        agg_input = agg_input or {"type": "array"}
        agg_output = agg_output or {"type": "string"}
    if agg == "pipeline":
        agg_input = agg_input or {"type": "array"}
        steps = binding.get("steps")
        if isinstance(steps, list) and any(
            isinstance(step, Mapping) and step.get("op") == "format_join"
            for step in steps
        ):
            agg_output = agg_output or {"type": "string"}

    return agg_output or source_schema, agg_input


def _find_unregistered_action_nodes(
    workflow_dict: Dict[str, Any],
    actions_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Find workflow nodes whose `action_id` is missing or not registered."""

    invalid: List[Dict[str, str]] = []
    for node in iter_workflow_and_loop_body_nodes(workflow_dict):
        if node.get("type") != "action":
            continue
        action_id = node.get("action_id")
        if not action_id or action_id not in actions_by_id:
            invalid.append({
                "id": node.get("id", "<unknown>"),
                "action_id": action_id or "",
            })
    return invalid


def _attach_out_params_schema(
    workflow: Workflow, actions_by_id: Mapping[str, Mapping[str, Any]]
) -> Workflow:
    wf_dict = workflow.model_dump(by_alias=True)
    changed = False

    for node in iter_workflow_and_loop_body_nodes(wf_dict):
        if not isinstance(node, Mapping) or node.get("type") != "action":
            continue

        action_id = node.get("action_id")
        schema = actions_by_id.get(action_id, {}).get("output_schema") if action_id else None
        if not isinstance(schema, Mapping):
            continue

        if node.get("out_params_schema") != schema:
            node["out_params_schema"] = schema
            changed = True

    return workflow if not changed else Workflow.model_validate(wf_dict)


def _evaluate_schema_alignment(
    node_params: Mapping[str, Any], candidate_schema: Mapping[str, Any]
) -> tuple[float, List[str], List[str], List[str]]:
    """Score how well the node params align with the candidate arg_schema."""

    if not isinstance(node_params, Mapping):
        return 0.0, [], [], ["节点 params 不是映射，无法比较"]

    if not isinstance(candidate_schema, Mapping):
        return 0.0, [], sorted(node_params.keys()), ["候选 action 缺少 arg_schema，跳过对齐评分"]

    properties = candidate_schema.get("properties")
    if not isinstance(properties, Mapping):
        return 0.0, [], sorted(node_params.keys()), ["arg_schema.properties 缺失，跳过对齐评分"]

    provided_keys = set(node_params.keys())
    schema_keys = set(properties.keys())
    matched = sorted(provided_keys & schema_keys)
    unmatched = sorted(provided_keys - schema_keys)

    if not provided_keys:
        return 0.6, matched, unmatched, ["节点未提供参数键名，给予中等默认置信度"]

    if not schema_keys:
        return 0.0, matched, unmatched, ["候选 arg_schema 无 properties，无法比对参数键名"]

    coverage = len(matched) / max(len(provided_keys), len(schema_keys))
    notes: List[str] = []
    if unmatched:
        notes.append("节点参数键名在候选 arg_schema 中不存在：" + ",".join(unmatched))

    return coverage, matched, unmatched, notes


def _auto_replace_unregistered_actions(
    workflow_dict: Dict[str, Any],
    invalid_nodes: List[Dict[str, str]],
    search_service: HybridActionSearchService,
    actions_by_id: Mapping[str, Dict[str, Any]],
) -> Optional[Workflow]:
    """Attempt a one-shot auto replacement for unknown action_id to avoid LLM hops."""

    changed = False
    MIN_SEARCH_CONFIDENCE = 0.6
    MIN_SCHEMA_SCORE = 0.4

    for item in invalid_nodes:
        node_id = item.get("id")
        if not node_id:
            continue

        node = next(
            (n for n in iter_workflow_and_loop_body_nodes(workflow_dict) if n.get("id") == node_id),
            None,
        )
        if not isinstance(node, Mapping):
            continue

        display_name = node.get("display_name") or ""
        original_action_id = node.get("action_id") or ""

        query = display_name or original_action_id
        if not query:
            continue

        candidates = search_service.search(query=query, top_k=1) if search_service else []
        candidate = candidates[0] if candidates else None
        candidate_id = candidate.get("action_id") if candidate else None
        search_confidence = candidate.get("score") if candidate else None
        if not candidate_id or candidate_id not in actions_by_id:
            continue

        if search_confidence is None or search_confidence < MIN_SEARCH_CONFIDENCE:
            log_warn(
                f"[ActionGuard] 节点 '{node_id}' 的候选动作 '{candidate_id}' 置信度 {search_confidence}"
                f" 低于阈值 {MIN_SEARCH_CONFIDENCE}，跳过自动替换。"
            )
            continue

        schema_score, matched_params, unmatched_params, schema_notes = _evaluate_schema_alignment(
            node.get("params", {}), candidate.get("arg_schema", {})
        )
        if schema_score < MIN_SCHEMA_SCORE:
            log_warn(
                f"[ActionGuard] 节点 '{node_id}' 的候选动作 '{candidate_id}' 参数匹配度 {schema_score:.2f}"
                f" 低于阈值 {MIN_SCHEMA_SCORE}，跳过自动替换。"
            )
            continue

        node["action_id"] = candidate_id
        changed = True
        log_info(
            f"[ActionGuard] 节点 '{node_id}' 的 action_id='{original_action_id}' 未注册，",
            f"已自动替换为 '{candidate_id}'（search={search_confidence:.2f}, schema={schema_score:.2f}）。"
        )

    if not changed:
        return None

    try:
        return Workflow.model_validate(workflow_dict)
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[ActionGuard] 自动替换 action_id 失败：{exc}")
        return None


def _schema_default_value(field_schema: Mapping[str, Any]) -> Any:
    """Return a predictable placeholder value for a JSON schema field."""

    if not isinstance(field_schema, Mapping):
        return None

    if "default" in field_schema:
        return field_schema.get("default")

    schema_type = field_schema.get("type")
    if isinstance(schema_type, list):
        schema_type = schema_type[0] if schema_type else None

    if schema_type == "string":
        return ""
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        return []
    if schema_type == "object":
        return {}

    enum_values = field_schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    return None


def _coerce_value_to_schema_type(value: Any, field_schema: Mapping[str, Any]) -> Optional[Any]:
    """Coerce a value into the basic JSON schema type when possible."""

    if not isinstance(field_schema, Mapping):
        return None

    schema_type = field_schema.get("type")
    if isinstance(schema_type, list):
        schema_type = schema_type[0] if schema_type else None

    if schema_type == "string" and not isinstance(value, str):
        return str(value)

    if schema_type == "integer":
        if isinstance(value, bool):
            return int(value)
        if not isinstance(value, int):
            try:
                return int(value)
            except Exception:  # noqa: BLE001
                return None

    if schema_type == "number":
        if isinstance(value, bool):
            return float(value)
        if not isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:  # noqa: BLE001
                return None

    if schema_type == "boolean" and not isinstance(value, bool):
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
        return bool(value)

    if schema_type == "array" and not isinstance(value, list):
        return [value]

    if schema_type == "object" and not isinstance(value, Mapping):
        return {"value": value}

    return None


def _align_reference_field_types(
    workflow: Workflow, *, action_registry: List[Dict[str, Any]]
) -> tuple[Workflow, List[ValidationError]]:
    """Ensure param binding types match the referenced output schema.

    当某个参数通过 ``__from__`` 引用上游节点输出时，检查该字段期望的
    JSON Schema 类型是否与引用的输出类型一致：

    * 若检测到类型不一致且可预测的转换规则（如数组包装、字符串化）可用，
      则直接在本地修改 workflow，避免进入 LLM 修复流程。
    * 无法自动转换时，返回带有上下文信息的 ``ValidationError``，以便后续
      交给 LLM 处理。
    """

    def _coerce_binding(binding: Mapping[str, Any], field_schema: Mapping[str, Any]) -> Optional[Any]:
        schema_type = _schema_primary_type(field_schema)
        if schema_type == "array" and not isinstance(binding, list):
            return [binding]

        if schema_type == "string":
            agg = binding.get("__agg__", "identity")
            if agg in {None, "", "identity"}:
                coerced = dict(binding)
                coerced["__agg__"] = "pipeline"
                steps = coerced.get("steps") or []
                coerced["steps"] = steps + [{"op": "format_join"}]
                return coerced

        return None

    wf_dict = workflow.model_dump(by_alias=True)
    actions_by_id = _index_actions_by_id(action_registry)
    nodes_by_id = {n.get("id"): n for n in iter_workflow_and_loop_body_nodes(wf_dict)}

    loop_body_parents: Dict[str, str] = {}
    for node in wf_dict.get("nodes", []) or []:
        if not isinstance(node, Mapping) or node.get("type") != "loop":
            continue
        params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
        body_nodes = (
            params.get("body_subgraph", {}).get("nodes") if isinstance(params, Mapping) else []
        )
        for bn in body_nodes or []:
            if isinstance(bn, Mapping) and bn.get("id"):
                loop_body_parents[bn["id"]] = node.get("id", "")

    errors: List[ValidationError] = []
    changed = False

    for node in wf_dict.get("nodes", []) or []:
        if not isinstance(node, Mapping) or node.get("type") != "action":
            continue

        node_id = node.get("id")
        params = node.get("params") if isinstance(node.get("params"), Mapping) else None
        if not params:
            continue

        arg_schema = actions_by_id.get(node.get("action_id"), {}).get("arg_schema")
        for path, binding, binding_schema in _iter_bindings_with_schema(
            params, arg_schema, "params"
        ):
            from_path = binding.get("__from__") if isinstance(binding, Mapping) else None
            if not isinstance(from_path, str):
                continue

            source_schema = _get_output_schema_at_path(
                from_path,
                nodes_by_id,
                actions_by_id,
                loop_body_parents=loop_body_parents,
            )
            effective_schema, agg_input_schema = _binding_io_schemas(
                source_schema, binding
            )
            if agg_input_schema and not _schemas_compatible(
                agg_input_schema, source_schema
            ):
                errors.append(
                    ValidationError(
                        code="SCHEMA_MISMATCH",
                        node_id=node_id,
                        field=path.replace("params.", "", 1),
                        message=(
                            f"参数 {path} 使用 __agg__ 需要类型为 {agg_input_schema.get('type')} 的输入，"
                            f"但来源 {from_path} 的类型为 {', '.join(sorted(_schema_types(source_schema))) or '未知'}"
                        ),
                    )
                )
                continue

            target_type = _schema_primary_type(binding_schema)
            source_type = _schema_primary_type(effective_schema)

            if target_type is None or source_type is None or target_type == source_type:
                continue

            coerced = _coerce_binding(binding, binding_schema or {})
            if coerced is not None:
                parent = params
                try:
                    tokens = parse_field_path(path.replace("params.", "", 1))
                except Exception:
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=node_id,
                            field=path.replace("params.", "", 1),
                            message=(
                                f"参数 {path} 的路径无法解析，跳过自动类型转换。"
                            ),
                        )
                    )
                    continue
                cursor: Any = parent
                for token in tokens[:-1]:
                    if isinstance(token, int):
                        cursor = cursor[token]
                    else:
                        cursor = cursor.get(token)
                last = tokens[-1]
                if isinstance(cursor, list) and isinstance(last, int):
                    cursor[last] = coerced
                elif isinstance(cursor, Mapping):
                    cursor[last] = coerced
                changed = True
                continue

            errors.append(
                ValidationError(
                    code="SCHEMA_MISMATCH",
                    node_id=node_id,
                    field=path.replace("params.", "", 1),
                    message=(
                        "参数 {field} 引用 {source} 的输出类型为 {source_type}，"
                        "但期望 {target_type}。无法自动转换，请使用工具调整或添加转换逻辑。"
                    ).format(
                        field=path,
                        source=from_path,
                        source_type=source_type,
                        target_type=target_type,
                    ),
                )
            )

    if not changed:
        return workflow, errors

    try:
        coerced_workflow = Workflow.model_validate(wf_dict)
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] 引用类型矫正后验证失败：{exc}")
        return workflow, errors

    return coerced_workflow, errors


def _align_binding_and_param_types(
    workflow: Workflow, *, action_registry: List[Dict[str, Any]]
) -> tuple[Workflow, List[ValidationError]]:
    """Match binding source types with target param schemas early in planning.

    This step builds a lightweight type environment from upstream node outputs,
    infers the effective type of each ``__from__`` binding (including
    ``__agg__`` transformations), and checks it against the expected type in
    the action's ``arg_schema``. Mismatches are surfaced as validation errors,
    with an early auto-repair rule (R1): when the source is an object and the
    target expects a primitive (string/number/integer/boolean), automatically
    pick the only compatible field if unique; otherwise, emit a repair
    suggestion listing candidates.
    """

    wf_dict = workflow.model_dump(by_alias=True)
    actions_by_id = _index_actions_by_id(action_registry)
    nodes_by_id = {n.get("id"): n for n in iter_workflow_and_loop_body_nodes(wf_dict)}
    loop_body_parents = index_loop_body_nodes(wf_dict)

    errors: List[ValidationError] = []
    changed = False

    for node in wf_dict.get("nodes", []) or []:
        if not isinstance(node, Mapping) or node.get("type") != "action":
            continue

        node_id = node.get("id")
        params = node.get("params") if isinstance(node.get("params"), Mapping) else None
        if not params:
            continue

        arg_schema = actions_by_id.get(node.get("action_id"), {}).get("arg_schema")
        if not isinstance(arg_schema, Mapping):
            continue

        for path, binding, binding_schema in _iter_bindings_with_schema(
            params, arg_schema, "params"
        ):
            if not isinstance(binding_schema, Mapping):
                continue

            from_path = binding.get("__from__") if isinstance(binding, Mapping) else None
            if not isinstance(from_path, str):
                continue

            source_schema = _get_output_schema_at_path(
                from_path,
                nodes_by_id,
                actions_by_id,
                loop_body_parents=loop_body_parents,
            )
            expected_path = path.replace("params.", "", 1)
            effective_schema, agg_input_schema = _binding_io_schemas(source_schema, binding)

            if agg_input_schema and not _schemas_compatible(agg_input_schema, source_schema):
                errors.append(
                    ValidationError(
                        code="SCHEMA_MISMATCH",
                        node_id=node_id,
                        field=expected_path,
                        message=(
                            f"参数 {path} 使用 __agg__ 需要类型为 {agg_input_schema.get('type')} 的输入，"
                            f"但来源 {from_path} 的类型为 {', '.join(sorted(_schema_types(source_schema))) or '未知'}"
                        ),
                    )
                )
                continue

            if _schemas_compatible(binding_schema, effective_schema):
                continue

            target_types = _schema_types(binding_schema)
            source_types = _schema_types(source_schema)
            expected_path = path.replace("params.", "", 1)

            primitive_targets = {"string", "number", "integer", "boolean"}
            if "object" in source_types and target_types & primitive_targets:
                props = (
                    source_schema.get("properties")
                    if isinstance(source_schema, Mapping)
                    and isinstance(source_schema.get("properties"), Mapping)
                    else {}
                )
                compatible_fields = [
                    name
                    for name, prop_schema in props.items()
                    if isinstance(prop_schema, Mapping)
                    and _schemas_compatible(binding_schema, prop_schema)
                ]

                if len(compatible_fields) == 1:
                    binding["__from__"] = f"{from_path}.{compatible_fields[0]}"
                    changed = True
                    continue

                if len(compatible_fields) > 1:
                    candidates = ", ".join(
                        f"{from_path}.{field}" for field in compatible_fields
                    )
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=node_id,
                            field=expected_path,
                            message=(
                                f"参数 {path} 期望 {', '.join(sorted(target_types))}，"
                                f"但来源 {from_path} 是 object，存在多种兼容字段: {candidates}"
                            ),
                        )
                    )
                    continue

            errors.append(
                ValidationError(
                    code="SCHEMA_MISMATCH",
                    node_id=node_id,
                    field=expected_path,
                    message=(
                        f"参数 {path} 期望 {', '.join(sorted(target_types)) or '未知'}，"
                        f"但来源 {from_path} 的类型为 {', '.join(sorted(source_types)) or '未知'}"
                    ),
                )
            )

    if not changed:
        return workflow, errors

    try:
        coerced_workflow = Workflow.model_validate(wf_dict)
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] 引用字段类型对齐失败：{exc}")
        return workflow, errors

    return coerced_workflow, errors


def _coerce_condition_param_types(
    workflow: Workflow,
    *,
    action_registry: Optional[List[Dict[str, Any]]] = None,
) -> tuple[Workflow, List[ValidationError]]:
    """Validate condition params against kind-specific type expectations.

    在进入常规校验/修复流程前，先根据 condition 节点 params.kind 的语义
    做一次类型匹配检查：

    * 若发现数值比较类的字段（threshold/min/max/bands[*].min/bands[*].max）
      与期望的 number 不匹配，尝试转换为 float/int。
    * regex_match 的 pattern 会被强制转换为字符串。
    * 无法转换的字段会生成 ValidationError，引导后续 LLM 做结构化修复。
    """

    def _to_number(value: Any) -> Optional[float]:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
        try:
            return float(value)
        except Exception:  # noqa: BLE001
            return None

    kind_numeric_fields = {
        "any_greater_than": ["threshold"],
        "all_less_than": ["threshold"],
        "greater_than": ["threshold", "value"],
        "less_than": ["threshold"],
        "between": ["min", "max"],
        "max_in_range": ["min", "max"],
    }

    errors: List[ValidationError] = []
    wf_dict = workflow.model_dump(by_alias=True)
    changed = False

    actions_by_id = {
        a.get("action_id"): a for a in (action_registry or []) if a.get("action_id")
    }
    nodes_by_id = {
        n.get("id"): n for n in iter_workflow_and_loop_body_nodes(wf_dict)
    }

    def _resolve_field_schema(source: Any, field: Any) -> Optional[Mapping[str, Any]]:
        if not isinstance(source, (Mapping, str)) or not isinstance(field, str):
            return None

        source_path = None
        if isinstance(source, Mapping):
            from_path = source.get("__from__")
            if isinstance(from_path, str):
                source_path = from_path
        elif isinstance(source, str):
            source_path = source

        if not source_path:
            return None

        item_schema = _get_array_item_schema_from_output(
            source_path, nodes_by_id, actions_by_id, skip_path_check=True
        )
        return _get_field_schema_from_item(item_schema, field)

    def _schema_type_label(schema: Any) -> str:
        if isinstance(schema, list):
            return "/".join(sorted(str(s) for s in schema))
        return str(schema)

    for node in iter_workflow_and_loop_body_nodes(wf_dict):
        if not isinstance(node, Mapping) or node.get("type") != "condition":
            continue

        nid = node.get("id")
        params = node.get("params") if isinstance(node.get("params"), Mapping) else None
        kind = params.get("kind") if isinstance(params, Mapping) else None
        if not params or not kind:
            continue

        field_schema = _resolve_field_schema(params.get("source"), params.get("field"))
        field_schema_type = field_schema.get("type") if isinstance(field_schema, Mapping) else None

        for field in kind_numeric_fields.get(kind, []):
            if field not in params:
                continue
            coerced = _to_number(params[field])
            if coerced is None:
                errors.append(
                    ValidationError(
                        code="SCHEMA_MISMATCH",
                        node_id=nid,
                        field=field,
                        message=(
                            f"condition 节点 '{nid}' (kind={kind}) 的字段 '{field}' 需要数值类型，无法自动转换当前值"
                        ),
                    )
                )
            elif coerced != params[field]:
                params[field] = coerced
                changed = True

        if kind == "max_in_range" and isinstance(params.get("bands"), list):
            for idx, band in enumerate(params.get("bands") or []):
                if not isinstance(band, Mapping):
                    continue
                for numeric_key in ("min", "max"):
                    if numeric_key not in band:
                        continue
                    coerced = _to_number(band[numeric_key])
                    if coerced is None:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field=f"bands[{idx}].{numeric_key}",
                                message=(
                                    f"condition 节点 '{nid}' (kind=max_in_range) 的 bands[{idx}].{numeric_key} 需要数值类型，无法自动转换"
                                ),
                            )
                        )
                    elif coerced != band[numeric_key]:
                        band[numeric_key] = coerced
                        changed = True

        if kind == "regex_match" and "pattern" in params:
            pattern = params.get("pattern")
            if pattern is not None and not isinstance(pattern, str):
                params["pattern"] = str(pattern)
                changed = True

        if field_schema_type is not None:
            numeric_kinds = {
                "any_greater_than",
                "all_less_than",
                "greater_than",
                "less_than",
                "between",
                "max_in_range",
            }
            text_kinds = {"contains", "regex_match"}

            if kind in numeric_kinds and not _is_numeric_schema_type(field_schema_type):
                errors.append(
                    ValidationError(
                        code="SCHEMA_MISMATCH",
                        node_id=nid,
                        field="kind",
                        message=(
                            "condition 节点 '{nid}' 的 field 类型为 {field_type}，应选择字符串/布尔比较类"
                            " kind（如 equals/not_equals/contains/regex_match），而不是数值比较类 {kind}。"
                        ).format(
                            nid=nid,
                            field_type=_schema_type_label(field_schema_type),
                            kind=kind,
                        ),
                    )
                )

            if kind in text_kinds and _is_numeric_schema_type(field_schema_type):
                errors.append(
                    ValidationError(
                        code="SCHEMA_MISMATCH",
                        node_id=nid,
                        field="kind",
                        message=(
                            "condition 节点 '{nid}' 的 field 类型为 {field_type}，应选择数值比较类 kind"
                            "（如 any_greater_than/all_less_than/greater_than/less_than/between/max_in_range）"
                            "，而不是字符串匹配类 {kind}。"
                        ).format(
                            nid=nid,
                            field_type=_schema_type_label(field_schema_type),
                            kind=kind,
                        ),
                    )
                )

    if not changed:
        return workflow, errors

    try:
        coerced_workflow = Workflow.model_validate(wf_dict)
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] 条件节点类型矫正后验证失败：{exc}")
        return workflow, errors

    return coerced_workflow, errors


def _apply_local_repairs_for_missing_params(
    current_workflow: Workflow,
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
) -> Optional[Workflow]:
    """Handle predictable missing/typing issues before invoking the LLM.

    当补参结果存在缺失的必填字段或简单的类型问题时，尝试根据 arg_schema
    填充占位符或类型矫正，以减少进入 LLM 自修复的次数。
    """

    actions_by_id = _index_actions_by_id(action_registry)
    workflow_dict = current_workflow.model_dump(by_alias=True)
    nodes: List[Dict[str, Any]] = [
        n for n in workflow_dict.get("nodes", []) if isinstance(n, Mapping)
    ]
    nodes_by_id: Dict[str, Dict[str, Any]] = {n.get("id"): n for n in nodes}

    changed = False

    for err in validation_errors:
        if err.code != "MISSING_REQUIRED_PARAM" or not err.node_id:
            continue

        node = nodes_by_id.get(err.node_id)
        if not node or node.get("type") != "action":
            continue

        action_id = node.get("action_id")
        action_def = actions_by_id.get(action_id)
        if not action_def:
            continue

        arg_schema = action_def.get("arg_schema") or {}
        properties = arg_schema.get("properties") if isinstance(arg_schema, Mapping) else None
        if not isinstance(properties, Mapping):
            continue

        field_schema = properties.get(err.field) if err.field else None
        if not field_schema:
            continue

        params = node.get("params")
        if not isinstance(params, dict):
            params = {}
            node["params"] = params
            changed = True

        if err.field and err.field not in params:
            params[err.field] = _schema_default_value(field_schema)
            changed = True
            continue

        if err.field and err.field in params:
            coerced = _coerce_value_to_schema_type(params[err.field], field_schema)
            if coerced is not None and coerced != params[err.field]:
                params[err.field] = coerced
                changed = True

    if not changed:
        return None

    try:
        return Workflow.model_validate(workflow_dict)
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] 本地修正失败：{exc}")
        return None


def _apply_local_repairs_for_schema_mismatch(
    current_workflow: Workflow,
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
) -> Optional[Workflow]:
    """Coerce obvious类型不匹配的 action 参数，减少进入 LLM 的次数。"""

    actions_by_id = _index_actions_by_id(action_registry)
    workflow_dict = current_workflow.model_dump(by_alias=True)
    nodes: List[Dict[str, Any]] = [
        n for n in workflow_dict.get("nodes", []) if isinstance(n, Mapping)
    ]
    nodes_by_id: Dict[str, Dict[str, Any]] = {n.get("id"): n for n in nodes}

    changed = False

    for err in validation_errors:
        if err.code != "SCHEMA_MISMATCH" or not err.node_id or not err.field:
            continue

        node = nodes_by_id.get(err.node_id)
        if not node or node.get("type") != "action":
            continue

        action_id = node.get("action_id")
        action_def = actions_by_id.get(action_id)
        if not action_def:
            continue

        properties = (
            action_def.get("arg_schema", {}).get("properties")
            if isinstance(action_def, Mapping)
            else None
        )
        if not isinstance(properties, Mapping):
            continue

        field_schema = properties.get(err.field)
        if not isinstance(field_schema, Mapping):
            continue

        params = node.get("params")
        if not isinstance(params, dict) or err.field not in params:
            continue

        coerced = _coerce_value_to_schema_type(params[err.field], field_schema)
        if coerced is not None and coerced != params[err.field]:
            params[err.field] = coerced
            changed = True

    if not changed:
        return None

    try:
        return Workflow.model_validate(workflow_dict)
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] schema 类型矫正失败：{exc}")
        return None


def _ensure_actions_registered_or_repair(
    workflow: Workflow,
    action_registry: List[Dict[str, Any]],
    search_service: HybridActionSearchService,
    reason: str,
) -> Workflow:
    """Ensure all action nodes reference a registered action, otherwise trigger repair.

    如果 Workflow 中的 action 节点缺失或引用未注册的 `action_id`，该函数会
    生成对应的 `ValidationError`，并将错误上下文、Action Registry 以及调用
    原因传递给 LLM 进行自动修复。LLM 需要返回一个完整的 workflow JSON，
    随后再由 Pydantic 转为 `Workflow` 实例。
    """

    guarded = ensure_registered_actions(
        workflow,
        action_registry=action_registry,
        search_service=search_service,
    )
    guarded = guarded if isinstance(guarded, Workflow) else Workflow.model_validate(guarded)

    actions_by_id = _index_actions_by_id(action_registry)
    guarded = _attach_out_params_schema(guarded, actions_by_id)
    invalid_nodes = _find_unregistered_action_nodes(
        guarded.model_dump(by_alias=True),
        actions_by_id=actions_by_id,
    )
    if not invalid_nodes:
        return guarded

    auto_repaired = _auto_replace_unregistered_actions(
        guarded.model_dump(by_alias=True),
        invalid_nodes=invalid_nodes,
        search_service=search_service,
        actions_by_id=actions_by_id,
    )
    if auto_repaired is not None:
        log_info(
            "[ActionGuard] 已基于 display_name/action_id 的相似度完成自动替换，进入一次性校验。"
        )
        return _attach_out_params_schema(auto_repaired, actions_by_id)

    validation_errors = [
        ValidationError(
            code="UNKNOWN_ACTION_ID",
            node_id=item["id"],
            field="action_id",
            message=(
                f"节点 '{item['id']}' 的 action_id '{item['action_id']}' 不在 Action Registry 中，"
                "请替换为已注册的动作。"
                if item.get("action_id")
                else f"节点 '{item['id']}' 缺少 action_id，请补充已注册的动作。"
            ),
        )
        for item in invalid_nodes
    ]

    repaired = _repair_with_llm_and_fallback(
        broken_workflow=guarded.model_dump(by_alias=True),
        validation_errors=validation_errors,
        action_registry=action_registry,
        search_service=search_service,
        reason=reason,
    )
    final_workflow = repaired if isinstance(repaired, Workflow) else Workflow.model_validate(repaired)
    return _attach_out_params_schema(final_workflow, actions_by_id)


def _validate_and_repair_workflow(
    current_workflow: Workflow,
    *,
    action_registry: List[Dict[str, Any]],
    search_service: HybridActionSearchService,
    max_repair_rounds: int,
    last_good_workflow: Workflow | None = None,
    trace_event_prefix: str = "planner",
) -> Workflow:
    failed_repair_history: Dict[str, List[str]] = {}
    pending_attempts: Dict[str, Dict[str, Any]] = {}

    def _error_key(err: ValidationError) -> str:
        return f"{err.code}:{err.node_id or 'global'}:{err.field or 'global'}"

    last_good_workflow = last_good_workflow or current_workflow

    for repair_round in range(max_repair_rounds + 1):
        log_section(f"校验 + 自修复轮次 {repair_round}")
        log_json("当前 workflow", current_workflow.model_dump(by_alias=True))

        current_workflow, coercion_errors = _coerce_condition_param_types(
            current_workflow, action_registry=action_registry
        )

        current_workflow, binding_type_errors = _align_binding_and_param_types(
            current_workflow, action_registry=action_registry
        )

        current_workflow, reference_errors = _align_reference_field_types(
            current_workflow, action_registry=action_registry
        )

        loop_exports_workflow, loop_export_summary = fill_loop_exports_defaults(
            current_workflow.model_dump(by_alias=True),
            action_registry=action_registry,
        )
        if loop_export_summary.get("applied"):
            log_info(
                "[AutoRepair] loop.exports 缺失或不完整，已自动填充：",
                f"summary={loop_export_summary}",
            )
            current_workflow = Workflow.model_validate(loop_exports_workflow)
            last_good_workflow = current_workflow

        aligned_alias_workflow, alias_summary = align_loop_body_alias_references(
            current_workflow.model_dump(by_alias=True),
            action_registry=action_registry,
        )
        if alias_summary.get("applied"):
            log_info(
                "[AutoRepair] loop.body_subgraph 引用了失效的 alias，已自动对齐到当前 item_alias。",
                f"summary={alias_summary}",
            )
            current_workflow = Workflow.model_validate(aligned_alias_workflow)
            last_good_workflow = current_workflow

        normalized_workflow, normalize_summary = normalize_binding_paths(
            current_workflow.model_dump(by_alias=True)
        )
        if normalize_summary.get("applied"):
            log_info(
                "[AutoRepair] 已规范化绑定路径，减少进入 LLM 的次数。",
                f"summary={normalize_summary}",
            )
            current_workflow = Workflow.model_validate(normalized_workflow)
            last_good_workflow = current_workflow

        static_errors = run_lightweight_static_rules(
            current_workflow.model_dump(by_alias=True),
            action_registry=action_registry,
        )

        errors: List[ValidationError] = []
        errors.extend(coercion_errors)
        errors.extend(binding_type_errors)
        errors.extend(reference_errors)

        if static_errors:
            errors.extend(static_errors)
        else:
            errors.extend(
                validate_completed_workflow(
                    current_workflow.model_dump(by_alias=True),
                    action_registry=action_registry,
                )
            )

        fixed_workflow, fix_summary = fix_missing_loop_exports_items(
            current_workflow.model_dump(by_alias=True), errors
        )
        if fix_summary.get("applied"):
            log_info(
                "[AutoRepair] 检测到 loop.exports 引用缺少 items 段，已自动补全。",
                f"replacements={fix_summary.get('replacements')}",
            )
            current_workflow = Workflow.model_validate(fixed_workflow)
            last_good_workflow = current_workflow
            continue

        current_errors_by_key = {_error_key(e): e for e in errors}
        if pending_attempts:
            for key, attempt in list(pending_attempts.items()):
                if key in current_errors_by_key:
                    note = (
                        f"轮次 {attempt.get('round')} 使用 {attempt.get('method')} 未修复："
                        f"{current_errors_by_key[key].message}"
                    )
                    failed_repair_history.setdefault(key, []).append(note)
                pending_attempts.pop(key, None)

        if not errors:
            log_success("校验通过，无需进一步修复")
            last_good_workflow = current_workflow
            log_event(
                f"{trace_event_prefix}_completed",
                {
                    "status": "success",
                    "repair_rounds": repair_round,
                    "has_errors": False,
                },
            )
            return current_workflow

        code_counts: Dict[str, int] = {}
        for e in errors:
            code_counts[e.code] = code_counts.get(e.code, 0) + 1

        log_event(
            "validation_errors_distribution",
            {
                "round": repair_round,
                "counts": code_counts,
                "total": len(errors),
            },
        )
        log_warn("校验未通过，错误列表：")
        for e in errors:
            log_error(
                f"[code={e.code}] node={e.node_id} field={e.field} message={e.message}"
            )

        locally_repaired = _apply_local_repairs_for_unknown_params(
            current_workflow=current_workflow,
            validation_errors=errors,
            action_registry=action_registry,
        )
        if locally_repaired is None:
            locally_repaired = _apply_local_repairs_for_missing_params(
                current_workflow=current_workflow,
                validation_errors=errors,
                action_registry=action_registry,
            )
        if locally_repaired is None:
            locally_repaired = _apply_local_repairs_for_schema_mismatch(
                current_workflow=current_workflow,
                validation_errors=errors,
                action_registry=action_registry,
            )
        if locally_repaired is not None:
            log_info("[AutoRepair] 检测到可预测的字段缺失/多余/类型不匹配问题，已在本地修正并重新校验。")
            current_workflow = locally_repaired
            last_good_workflow = current_workflow

            errors = validate_completed_workflow(
                current_workflow.model_dump(by_alias=True),
                action_registry=action_registry,
            )

            if not errors:
                log_success("本地修正后校验通过，无需调用 LLM 继续修复。")
                log_event(
                    f"{trace_event_prefix}_completed",
                    {
                        "status": "success_after_local_repair",
                        "repair_rounds": repair_round,
                        "has_errors": False,
                    },
                )
                return current_workflow

                log_warn("本地修正后仍有错误，将继续进入 LLM 修复流程：")
                for e in errors:
                    log_error(
                        f"[code={e.code}] node={e.node_id} field={e.field} message={e.message}"
                    )

        if repair_round == max_repair_rounds:
            log_warn("已到最大修复轮次，仍有错误，返回最后一个合法结构版本")
            log_event(
                f"{trace_event_prefix}_completed",
                {
                    "status": "reached_max_rounds",
                    "repair_rounds": repair_round,
                    "last_good_returned": True,
                },
            )
            return last_good_workflow

        current_errors_by_key = {_error_key(e): e for e in errors}
        previous_attempts = {
            key: failed_repair_history.get(key, [])
            for key in current_errors_by_key
            if failed_repair_history.get(key)
        }

        log_info(f"调用 LLM 进行第 {repair_round + 1} 次修复")
        current_workflow = _repair_with_llm_and_fallback(
            broken_workflow=current_workflow.model_dump(by_alias=True),
            validation_errors=errors,
            action_registry=action_registry,
            search_service=search_service,
            reason=f"修复轮次 {repair_round + 1}",
            previous_attempts=previous_attempts,
        )
        for key in current_errors_by_key:
            pending_attempts[key] = {
                "round": repair_round + 1,
                "method": "LLM 修复",
            }
        current_workflow = _ensure_actions_registered_or_repair(
            current_workflow,
            action_registry=action_registry,
            search_service=search_service,
            reason=f"修复轮次 {repair_round + 1} 后校验 action_id",
        )
        last_good_workflow = current_workflow

    log_event(
        f"{trace_event_prefix}_completed",
        {
            "status": "exhausted_rounds",
            "repair_rounds": max_repair_rounds,
            "last_good_returned": True,
        },
    )
    return last_good_workflow


def plan_workflow_with_two_pass(
    nl_requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    max_rounds: int = 10,
    max_repair_rounds: int = 3,
    trace_context: TraceContext | None = None,
    trace_id: str | None = None,
) -> Workflow:
    """Plan a workflow in two passes with structured validation and LLM repair.

    参数
    ----
    nl_requirement:
        面向 LLM 的自然语言需求描述，必须明确业务目标、主要输入/输出。
    search_service:
        用于辅助 LLM 选择与需求匹配的业务 Action 的检索服务。
    action_registry:
        已注册的业务动作列表，传入的每个元素应包含唯一的 `action_id`。
    max_rounds:
        结构规划阶段允许的 LLM 迭代轮次，决定覆盖率/连通性纠错次数。
    max_repair_rounds:
        参数补全后进入校验 + 自修复的最大轮次，超过后返回最后通过校验的
        版本。

    返回
    ----
    Workflow
        通过 Pydantic 校验、所有 action_id 均注册的最终 Workflow 对象。

    LLM 返回格式约定
    ----------------
    * 结构规划与参数补全阶段均要求 LLM 返回可被 `Workflow` 解析的 JSON 对象，
      至少包含 `nodes` 数组（连线会通过节点绑定自动推导）。
    * 修复阶段会附带 `validation_errors`，LLM 需直接输出修正后的完整 JSON，
      不需要包含额外解释文本。

    边界条件
    --------
    * 如果结构规划或补参阶段抛出异常/校验失败，会在保留最近一次合法
      Workflow 的前提下，使用 LLM 自修复。
    * 当达到 `max_repair_rounds` 仍存在错误时，返回最近一次通过校验的版本，
      保证调用方始终获得一个合法的 `Workflow` 实例。
    """

    context = trace_context or TraceContext.create(trace_id=trace_id, span_name="orchestrator")
    with use_trace_context(context):
        log_event(
            "plan_start",
            {
                "nl_requirement": nl_requirement,
                "max_rounds": max_rounds,
                "max_repair_rounds": max_repair_rounds,
            },
            context=context,
        )

        with child_span("structure_planning"):
            skeleton_raw = plan_workflow_structure_with_llm(
                nl_requirement=nl_requirement,
                search_service=search_service,
                action_registry=action_registry,
                max_rounds=max_rounds,
                max_coverage_refine_rounds=2,
            )
        log_section("第一阶段结果：Workflow Skeleton")
        log_json("Workflow Skeleton", skeleton_raw)

        precheck_errors = precheck_loop_body_graphs(skeleton_raw)
        if precheck_errors:
            log_warn(
                "[plan_workflow_with_two_pass] 结构规划阶段检测到 loop body 引用缺失节点，交给 LLM 修复。"
            )
            skeleton = _repair_with_llm_and_fallback(
                broken_workflow=skeleton_raw if isinstance(skeleton_raw, dict) else {},
                validation_errors=precheck_errors,
                action_registry=action_registry,
                search_service=search_service,
                reason="结构规划阶段校验失败",
            )
        else:
            try:
                skeleton = Workflow.model_validate(skeleton_raw)
            except PydanticValidationError as e:
                log_warn(
                    "[plan_workflow_with_two_pass] 结构规划阶段校验失败，将错误交给 LLM 修复后继续。"
                )
                validation_errors = _convert_pydantic_errors(skeleton_raw, e)
                if not validation_errors:
                    validation_errors = [_make_failure_validation_error(str(e))]
                skeleton = _repair_with_llm_and_fallback(
                    broken_workflow=skeleton_raw if isinstance(skeleton_raw, dict) else {},
                    validation_errors=validation_errors,
                    action_registry=action_registry,
                    search_service=search_service,
                    reason="结构规划阶段校验失败",
                )
        log_event("plan_structure_done", {"workflow": skeleton.model_dump()})
        skeleton = _ensure_actions_registered_or_repair(
            skeleton,
            action_registry=action_registry,
            search_service=search_service,
            reason="结构规划后修正未注册的 action_id",
        )
        last_good_workflow: Workflow = skeleton
        try:
            with child_span("params_completion"):
                completed_workflow_raw = fill_params_with_llm(
                    workflow_skeleton=skeleton.model_dump(by_alias=True),
                    action_registry=action_registry,
                    model=OPENAI_MODEL,
                )
        except Exception as err:  # noqa: BLE001
            log_warn(
                f"[plan_workflow_with_two_pass] 参数补全阶段发生异常，将错误交给 LLM 自修复：{err}"
            )
            current_workflow = _repair_with_llm_and_fallback(
                broken_workflow=skeleton.model_dump(by_alias=True),
                validation_errors=[
                    _make_failure_validation_error(f"参数补全阶段失败：{err}")
                ],
                action_registry=action_registry,
                search_service=search_service,
                reason="参数补全异常，尝试直接基于 skeleton 修复",
            )
            current_workflow = _ensure_actions_registered_or_repair(
                current_workflow,
                action_registry=action_registry,
                search_service=search_service,
                reason="参数补全异常修复后校验 action_id",
            )
            last_good_workflow = current_workflow
            completed_workflow_raw = current_workflow.model_dump(by_alias=True)
        else:
            precheck_errors = precheck_loop_body_graphs(completed_workflow_raw)
            if precheck_errors:
                log_warn(
                    "[plan_workflow_with_two_pass] 警告：补参结果包含 loop body 节点缺失，将交给 LLM 修复。"
                )
                current_workflow = _repair_with_llm_and_fallback(
                    broken_workflow=completed_workflow_raw,
                    validation_errors=precheck_errors,
                    action_registry=action_registry,
                    search_service=search_service,
                    reason="补参结果校验失败",
                )
                current_workflow = _ensure_actions_registered_or_repair(
                    current_workflow,
                    action_registry=action_registry,
                    search_service=search_service,
                    reason="补参结果修复后校验 action_id",
                )
                last_good_workflow = current_workflow
            else:
                try:
                    completed_workflow = Workflow.model_validate(completed_workflow_raw)
                    current_workflow = _ensure_actions_registered_or_repair(
                        completed_workflow,
                        action_registry=action_registry,
                        search_service=search_service,
                        reason="参数补全后修正未注册的 action_id",
                    )
                    last_good_workflow = current_workflow
                except PydanticValidationError as e:
                    log_warn(
                        f"[plan_workflow_with_two_pass] 警告：fill_params_with_llm 返回的结构无法通过校验，{e}"
                    )
                    validation_errors = _convert_pydantic_errors(completed_workflow_raw, e)
                    if validation_errors:
                        current_workflow = _repair_with_llm_and_fallback(
                            broken_workflow=completed_workflow_raw,
                            validation_errors=validation_errors,
                            action_registry=action_registry,
                            search_service=search_service,
                            reason="补参结果校验失败",
                        )
                        current_workflow = _ensure_actions_registered_or_repair(
                            current_workflow,
                            action_registry=action_registry,
                            search_service=search_service,
                            reason="补参结果修复后校验 action_id",
                        )
                        last_good_workflow = current_workflow
                    else:
                        current_workflow = last_good_workflow

        return _validate_and_repair_workflow(
            current_workflow,
            action_registry=action_registry,
            search_service=search_service,
            max_repair_rounds=max_repair_rounds,
            last_good_workflow=last_good_workflow,
            trace_event_prefix="planner",
        )


def update_workflow_with_two_pass(
    existing_workflow: Mapping[str, Any],
    requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    *,
    max_repair_rounds: int = 3,
    model: str = OPENAI_MODEL,
    trace_context: TraceContext | None = None,
    trace_id: str | None = None,
) -> Workflow:
    """Update an existing workflow using the same validation/repair pipeline as planning.

    The ``existing_workflow`` is treated as a trusted DSL 模版 whenever possible. We
    attempt to preserve untouched nodes/edges while satisfying the new ``requirement``
    through incremental changes plus validation-driven self-repair.
    """

    context = trace_context or TraceContext.create(trace_id=trace_id, span_name="update_orchestrator")
    with use_trace_context(context):
        log_event(
            "update_start",
            {
                "max_repair_rounds": max_repair_rounds,
            },
            context=context,
        )

        try:
            base_workflow = Workflow.model_validate(existing_workflow)
        except PydanticValidationError as e:
            log_warn("[update_workflow_with_two_pass] 输入 workflow 校验失败，进入修复流程。")
            validation_errors = _convert_pydantic_errors(existing_workflow, e) or [
                _make_failure_validation_error(str(e))
            ]
            base_workflow = _repair_with_llm_and_fallback(
                broken_workflow=existing_workflow if isinstance(existing_workflow, dict) else {},
                validation_errors=validation_errors,
                action_registry=action_registry,
                search_service=search_service,
                reason="更新前的输入校验失败",
            )

        base_workflow = _ensure_actions_registered_or_repair(
            base_workflow,
            action_registry=action_registry,
            search_service=search_service,
            reason="更新前校验 action_id",
        )
        last_good_workflow = base_workflow

        with child_span("llm_update"):
            updated_raw = update_workflow_with_llm(
                workflow_raw=base_workflow.model_dump(by_alias=True),
                requirement=requirement,
                action_registry=action_registry,
                model=model,
            )

        precheck_errors = precheck_loop_body_graphs(updated_raw)
        if precheck_errors:
            log_warn("[update_workflow_with_two_pass] 更新结果包含 loop body 缺失节点，将交给 LLM 修复。")
            updated_workflow = _repair_with_llm_and_fallback(
                broken_workflow=updated_raw if isinstance(updated_raw, dict) else {},
                validation_errors=precheck_errors,
                action_registry=action_registry,
                search_service=search_service,
                reason="更新结果校验失败",
            )
        else:
            try:
                updated_workflow = Workflow.model_validate(updated_raw)
            except PydanticValidationError as e:
                log_warn(
                    "[update_workflow_with_two_pass] 更新结果无法通过校验，进入自修复。"
                )
                validation_errors = _convert_pydantic_errors(updated_raw, e) or [
                    _make_failure_validation_error(str(e))
                ]
                updated_workflow = _repair_with_llm_and_fallback(
                    broken_workflow=updated_raw if isinstance(updated_raw, dict) else {},
                    validation_errors=validation_errors,
                    action_registry=action_registry,
                    search_service=search_service,
                    reason="更新结果校验失败",
                )

        updated_workflow = _ensure_actions_registered_or_repair(
            updated_workflow,
            action_registry=action_registry,
            search_service=search_service,
            reason="更新结果校验 action_id",
        )
        last_good_workflow = updated_workflow

        return _validate_and_repair_workflow(
            updated_workflow,
            action_registry=action_registry,
            search_service=search_service,
            max_repair_rounds=max_repair_rounds,
            last_good_workflow=last_good_workflow,
            trace_event_prefix="update",
        )


__all__ = ["plan_workflow_with_two_pass", "update_workflow_with_two_pass"]
