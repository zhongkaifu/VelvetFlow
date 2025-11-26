"""Top-level orchestrator for the two-pass planner.

本模块聚合了结构规划、参数补全与错误修复三个阶段，主要通过 LLM 进行
workflow 结构和参数的两阶段推理，然后在必要时向 LLM 提供错误上下文进行
自修复。为了降低调用方的认知负担，核心入口 `plan_workflow_with_two_pass`
会处理：

* 骨架阶段：根据自然语言需求触发 LLM 生成节点/边的初稿，并保证 Action
  Registry 覆盖。
* 参数阶段：在约定的 prompt 语境下补全每个 action 的参数，并校验返回
  结构符合 `Workflow` Pydantic 模型。
* 修复阶段：当校验失败时，将 ValidationError 列表和上一次合法结构回传给
  LLM，请求其输出完全序列化的 workflow 字典；若多轮修复仍失败则返回最后
  一次通过校验的版本。

返回值统一为 `Workflow` 对象，调用方无需关心 LLM 返回的原始 JSON 格式。
"""

from typing import Any, Dict, List, Mapping, Optional

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    log_error,
    log_event,
    log_info,
    log_json,
    log_section,
    log_success,
    log_warn,
)
from velvetflow.models import PydanticValidationError, ValidationError, Workflow
from velvetflow.loop_dsl import iter_workflow_and_loop_body_nodes
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.coverage import check_requirement_coverage_with_llm
from velvetflow.planner.params import fill_params_with_llm
from velvetflow.planner.repair import (
    _convert_pydantic_errors,
    _make_failure_validation_error,
    _repair_with_llm_and_fallback,
)
from velvetflow.planner.structure import plan_workflow_structure_with_llm
from velvetflow.planner.validation import (
    run_lightweight_static_rules,
    validate_completed_workflow,
)
from velvetflow.search import HybridActionSearchService


def _index_actions_by_id(
    action_registry: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Build a quick lookup table for action metadata keyed by `action_id`."""

    return {a["action_id"]: a for a in action_registry}


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


def _auto_replace_unregistered_actions(
    workflow_dict: Dict[str, Any],
    invalid_nodes: List[Dict[str, str]],
    search_service: HybridActionSearchService,
    actions_by_id: Mapping[str, Dict[str, Any]],
) -> Optional[Workflow]:
    """Attempt a one-shot auto replacement for unknown action_id to avoid LLM hops."""

    changed = False

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
        candidate_id = candidates[0].get("action_id") if candidates else None
        if candidate_id and candidate_id in actions_by_id:
            node["action_id"] = candidate_id
            changed = True
            log_info(
                f"[ActionGuard] 节点 '{node_id}' 的 action_id='{original_action_id}' 未注册，"
                f"已自动替换为最相近的 '{candidate_id}'。"
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


def _ensure_actions_registered_or_repair(
    workflow: Workflow,
    action_registry: List[Dict[str, Any]],
    search_service: HybridActionSearchService,
    reason: str,
) -> Workflow:
    """Ensure all action nodes reference a registered action, otherwise trigger repair.

    如果 Workflow 中的 action 节点缺失或引用未注册的 `action_id`，该函数会
    生成对应的 `ValidationError`，并将错误上下文、Action Registry 以及调用
    原因传递给 LLM 进行自动修复。LLM 需要返回一个完整的 workflow JSON
    （包含 nodes/edges），随后再由 Pydantic 转为 `Workflow` 实例。
    """

    guarded = ensure_registered_actions(
        workflow,
        action_registry=action_registry,
        search_service=search_service,
    )
    guarded = guarded if isinstance(guarded, Workflow) else Workflow.model_validate(guarded)

    actions_by_id = _index_actions_by_id(action_registry)
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
        return auto_repaired

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
    return repaired if isinstance(repaired, Workflow) else Workflow.model_validate(repaired)


def plan_workflow_with_two_pass(
    nl_requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    max_rounds: int = 10,
    max_repair_rounds: int = 3,
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
      至少包含 `nodes` 与 `edges` 数组。
    * 修复阶段会附带 `validation_errors`，LLM 需直接输出修正后的完整 JSON，
      不需要包含额外解释文本。

    边界条件
    --------
    * 如果结构规划或补参阶段抛出异常/校验失败，会在保留最近一次合法
      Workflow 的前提下，使用 LLM 自修复。
    * 当达到 `max_repair_rounds` 仍存在错误时，返回最近一次通过校验的版本，
      保证调用方始终获得一个合法的 `Workflow` 实例。
    """

    skeleton_raw = plan_workflow_structure_with_llm(
        nl_requirement=nl_requirement,
        search_service=search_service,
        action_registry=action_registry,
        max_rounds=max_rounds,
        max_coverage_refine_rounds=2,
    )
    log_section("第一阶段结果：Workflow Skeleton")
    log_json("Workflow Skeleton", skeleton_raw)

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
    coverage_hint = check_requirement_coverage_with_llm(
        nl_requirement=nl_requirement,
        workflow=skeleton.model_dump(by_alias=True),
        model=OPENAI_MODEL,
    )
    log_json("覆盖度摘要（补参提示用）", coverage_hint)
    coverage_missing_points = coverage_hint.get("missing_points") or []
    coverage_analysis = coverage_hint.get("analysis") or ""
    last_good_workflow: Workflow = skeleton
    try:
        completed_workflow_raw = fill_params_with_llm(
            workflow_skeleton=skeleton.model_dump(by_alias=True),
            action_registry=action_registry,
            nl_requirement=nl_requirement,
            coverage_missing_points=coverage_missing_points,
            coverage_analysis=coverage_analysis,
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

    for repair_round in range(max_repair_rounds + 1):
        log_section(f"校验 + 自修复轮次 {repair_round}")
        log_json("当前 workflow", current_workflow.model_dump(by_alias=True))

        static_errors = run_lightweight_static_rules(
            current_workflow.model_dump(by_alias=True),
            action_registry=action_registry,
        )

        errors = static_errors or validate_completed_workflow(
            current_workflow.model_dump(by_alias=True),
            action_registry=action_registry,
        )

        if not errors:
            log_success("校验通过，无需进一步修复")
            last_good_workflow = current_workflow
            return current_workflow

        log_warn("校验未通过，错误列表：")
        for e in errors:
            log_error(
                f"[code={e.code}] node={e.node_id} field={e.field} message={e.message}"
            )

        locally_repaired = _apply_local_repairs_for_missing_params(
            current_workflow=current_workflow,
            validation_errors=errors,
            action_registry=action_registry,
        )
        if locally_repaired is not None:
            log_info(
                "[AutoRepair] 检测到可预测的缺失字段/类型问题，已在本地修正并重新校验。"
            )
            current_workflow = locally_repaired
            last_good_workflow = current_workflow

            errors = validate_completed_workflow(
                current_workflow.model_dump(by_alias=True),
                action_registry=action_registry,
            )

            if not errors:
                log_success("本地修正后校验通过，无需调用 LLM 继续修复。")
                return current_workflow

            log_warn("本地修正后仍有错误，将继续进入 LLM 修复流程：")
            for e in errors:
                log_error(
                    f"[code={e.code}] node={e.node_id} field={e.field} message={e.message}"
                )

        if repair_round == max_repair_rounds:
            log_warn("已到最大修复轮次，仍有错误，返回最后一个合法结构版本")
            return last_good_workflow

        log_info(f"调用 LLM 进行第 {repair_round + 1} 次修复")
        current_workflow = _repair_with_llm_and_fallback(
            broken_workflow=current_workflow.model_dump(by_alias=True),
            validation_errors=errors,
            action_registry=action_registry,
            search_service=search_service,
            reason=f"修复轮次 {repair_round + 1}",
        )
        current_workflow = _ensure_actions_registered_or_repair(
            current_workflow,
            action_registry=action_registry,
            search_service=search_service,
            reason=f"修复轮次 {repair_round + 1} 后校验 action_id",
        )
        last_good_workflow = current_workflow

    return last_good_workflow


__all__ = ["plan_workflow_with_two_pass"]
