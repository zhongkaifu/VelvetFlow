# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Single Agent entrypoint that exposes all planner tools at once.

This module wraps structure planning, parameter completion, validation,
repair and workflow updates into one Agent SDK workflow. The Agent is
executed by :class:`Runner` so the LLM can freely choose the right tool
sequence to satisfy a user's requirement and return a validated workflow
JSON.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_section, log_warn
from velvetflow.models import ValidationError, Workflow
from velvetflow.planner.agent_runtime import Agent, Runner, function_tool
from velvetflow.planner.params import fill_params_with_llm
from velvetflow.planner.repair import repair_workflow_with_llm
from velvetflow.planner.structure import plan_workflow_structure_with_llm
from velvetflow.planner.update import update_workflow_with_llm
from velvetflow.search import HybridActionSearchService, build_default_search_service
from velvetflow.verification import validate_completed_workflow


def _workflow_to_dict(workflow: Workflow | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(workflow, Workflow):
        return workflow.model_dump(by_alias=True)
    if isinstance(workflow, Mapping):
        return Workflow.model_validate(workflow).model_dump(by_alias=True)
    raise ValueError("workflow 必须是 Mapping 或 Workflow 实例。")


def _serialize_validation_errors(errors: Sequence[ValidationError]) -> List[Dict[str, Any]]:
    return [
        {
            "code": err.code,
            "node_id": err.node_id,
            "field": err.field,
            "message": err.message,
        }
        for err in errors
    ]


def run_workflow_planning_agent(
    nl_requirement: str,
    action_registry: List[Dict[str, Any]],
    search_service: Optional[HybridActionSearchService] = None,
    base_workflow: Optional[Mapping[str, Any]] = None,
    *,
    model: str = OPENAI_MODEL,
    max_turns: int = 32,
) -> Workflow:
    """Expose planner lifecycle tools through one Agent SDK runner.

    The Agent exposes dedicated tools for structure building, parameter fill,
    validation, repair, update and final submission. The LLM can choose any
    order and combination via the Agent SDK Runner.
    """

    service = search_service or build_default_search_service()
    working_workflow: Optional[Dict[str, Any]] = _workflow_to_dict(base_workflow) if base_workflow else None
    finalized_workflow: Optional[Dict[str, Any]] = None
    latest_validation_errors: List[ValidationError] = []

    system_prompt = (
        "你是一个统一的 Workflow Agent，负责使用提供的工具完成构建、补参、校验、修复与更新。\n"
        "可以按需调用 build_workflow、fill_workflow_params、validate_workflow、repair_workflow、update_workflow、submit_final_workflow 组合出满足用户需求的流程。\n"
        "所有 params 必须使用 Jinja 表达式或字面量，禁止 __from__/__agg__，引用 loop 结果时只能使用 exports.items/aggregates。"
    )

    @function_tool(strict_mode=False)
    def build_workflow(requirement: Optional[str] = None) -> Mapping[str, Any]:
        """Generate a workflow skeleton from a natural language requirement."""

        nonlocal working_workflow, latest_validation_errors
        req = requirement or nl_requirement
        skeleton = plan_workflow_structure_with_llm(
            req,
            search_service=service,
            action_registry=action_registry,
        )
        working_workflow = Workflow.model_validate(skeleton).model_dump(by_alias=True)
        latest_validation_errors = []
        return {"status": "ok", "workflow": working_workflow}

    @function_tool(strict_mode=False)
    def fill_workflow_params(workflow: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        """Complete node parameters for the current or provided workflow."""

        nonlocal working_workflow, latest_validation_errors
        target = _workflow_to_dict(workflow or working_workflow or {})
        filled = fill_params_with_llm(target, action_registry=action_registry, model=model)
        working_workflow = Workflow.model_validate(filled).model_dump(by_alias=True)
        latest_validation_errors = []
        return {"status": "ok", "workflow": working_workflow}

    @function_tool(strict_mode=False)
    def validate_workflow(workflow: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        """Run static validation and return structured errors if any."""

        nonlocal working_workflow, latest_validation_errors
        target = _workflow_to_dict(workflow or working_workflow or {})
        errors = validate_completed_workflow(target, action_registry=action_registry)
        working_workflow = target
        latest_validation_errors = errors
        return {
            "status": "ok" if not errors else "failed",
            "workflow": target,
            "errors": _serialize_validation_errors(errors),
        }

    @function_tool(strict_mode=False)
    def repair_workflow(
        workflow: Optional[Mapping[str, Any]] = None,
        error_summary: Optional[str] = None,
    ) -> Mapping[str, Any]:
        """Repair the workflow using the latest validation errors."""

        nonlocal working_workflow, latest_validation_errors
        target = _workflow_to_dict(workflow or working_workflow or {})
        fixed = repair_workflow_with_llm(
            broken_workflow=target,
            validation_errors=latest_validation_errors,
            action_registry=action_registry,
            error_summary=error_summary,
            previous_failed_attempts=None,
            model=model,
        )
        working_workflow = Workflow.model_validate(fixed).model_dump(by_alias=True)
        latest_validation_errors = []
        return {"status": "ok", "workflow": working_workflow}

    @function_tool(strict_mode=False)
    def update_workflow(
        requirement: Optional[str] = None,
        workflow: Optional[Mapping[str, Any]] = None,
        validation_errors: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> Mapping[str, Any]:
        """Update the workflow according to a requirement or validation errors."""

        nonlocal working_workflow, latest_validation_errors
        req = requirement or nl_requirement
        target = _workflow_to_dict(workflow or working_workflow or {})
        parsed_errors: Optional[Sequence[ValidationError]] = latest_validation_errors
        if validation_errors is not None:
            parsed_errors = [
                ValidationError(
                    code=e.get("code", "INVALID_SCHEMA"),
                    node_id=e.get("node_id"),
                    field=e.get("field"),
                    message=str(e.get("message", "")),
                )
                for e in validation_errors
                if isinstance(e, Mapping)
            ]
        updated = update_workflow_with_llm(
            workflow_raw=target,
            requirement=req,
            action_registry=action_registry,
            model=model,
            validation_errors=parsed_errors or None,
        )
        working_workflow = Workflow.model_validate(updated).model_dump(by_alias=True)
        latest_validation_errors = list(parsed_errors or [])
        return {"status": "ok", "workflow": working_workflow}

    @function_tool(strict_mode=False)
    def submit_final_workflow(workflow: Optional[Mapping[str, Any]] = None) -> Mapping[str, Any]:
        """Submit the final workflow once all checks pass."""

        nonlocal working_workflow, finalized_workflow
        target = _workflow_to_dict(workflow or working_workflow or {})
        working_workflow = target
        finalized_workflow = target
        return {"status": "ok", "workflow": target}

    agent = Agent(
        name="WorkflowOrchestrator",
        instructions=system_prompt,
        tools=[
            build_workflow,
            fill_workflow_params,
            validate_workflow,
            repair_workflow,
            update_workflow,
            submit_final_workflow,
        ],
        model=model,
    )

    log_section("统一 Agent 工作流规划")
    run_input: Any = json.dumps(
        {
            "requirement": nl_requirement,
            "base_workflow": working_workflow,
            "action_registry_size": len(action_registry),
        },
        ensure_ascii=False,
    )

    try:
        Runner.run_sync(agent, run_input, max_turns=max_turns)
    except TypeError:
        coro = Runner.run(agent, run_input)  # type: ignore[call-arg]
        result = coro if not asyncio.iscoroutine(coro) else asyncio.run(coro)
        _ = result

    if finalized_workflow is None:
        if working_workflow is None:
            raise RuntimeError("Agent 未提交最终 workflow，且没有可用的工作副本。")
        if latest_validation_errors:
            raise RuntimeError(
                "Agent 结束但 workflow 仍未通过校验："
                + "; ".join(err.message for err in latest_validation_errors)
            )
        log_warn("[run_workflow_planning_agent] 未收到 submit_final_workflow，返回当前工作副本。")
        finalized_workflow = working_workflow

    return Workflow.model_validate(finalized_workflow)


__all__ = ["run_workflow_planning_agent"]
