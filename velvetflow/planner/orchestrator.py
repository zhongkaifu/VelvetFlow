"""Top-level orchestrator for the two-pass planner."""

from typing import Any, Dict, List

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
from velvetflow.models import PydanticValidationError, Workflow
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.params import fill_params_with_llm
from velvetflow.planner.repair import (
    _convert_pydantic_errors,
    _make_failure_validation_error,
    _repair_with_llm_and_fallback,
)
from velvetflow.planner.structure import plan_workflow_structure_with_llm
from velvetflow.planner.validation import validate_completed_workflow
from velvetflow.search import HybridActionSearchService


def plan_workflow_with_two_pass(
    nl_requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    max_rounds: int = 10,
    max_repair_rounds: int = 3,
) -> Workflow:
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
    last_good_workflow: Workflow = skeleton
    try:
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
        last_good_workflow = current_workflow
        completed_workflow_raw = current_workflow.model_dump(by_alias=True)
    else:
        try:
            completed_workflow = Workflow.model_validate(completed_workflow_raw)
            completed_workflow = ensure_registered_actions(
                completed_workflow,
                action_registry=action_registry,
                search_service=search_service,
            )
            if isinstance(completed_workflow, Workflow):
                current_workflow = completed_workflow
            else:
                current_workflow = Workflow.model_validate(completed_workflow)
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
                last_good_workflow = current_workflow
            else:
                current_workflow = last_good_workflow

    for repair_round in range(max_repair_rounds + 1):
        log_section(f"校验 + 自修复轮次 {repair_round}")
        log_json("当前 workflow", current_workflow.model_dump(by_alias=True))

        errors = validate_completed_workflow(
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
        last_good_workflow = current_workflow

    return last_good_workflow


__all__ = ["plan_workflow_with_two_pass"]
