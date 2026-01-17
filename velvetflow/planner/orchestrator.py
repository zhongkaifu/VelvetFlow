# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Top-level orchestrator for the two-pass planner.

This module orchestrates structure planning and parameter completion. It runs a
two-pass reasoning flow with an LLM to draft the workflow structure and fill
parameters, then validates the result before returning.
"""

import json
from typing import Any, Callable, Dict, List, Mapping

from velvetflow.logging_utils import (
    TraceContext,
    child_span,
    log_debug,
    log_event,
    log_json,
    log_section,
    log_warn,
    use_trace_context,
)
from velvetflow.models import PydanticValidationError, ValidationError, Workflow
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.requirement_analysis import analyze_user_requirement
from velvetflow.planner.structure import plan_workflow_structure_with_llm
from velvetflow.search import HybridActionSearchService
from velvetflow.verification import precheck_loop_body_graphs, validate_completed_workflow


def _validate_workflow(
    workflow: Workflow, action_registry: List[Dict[str, Any]]
) -> List[ValidationError]:
    """Run lightweight validation and return any errors."""

    return validate_completed_workflow(
        workflow.model_dump(by_alias=True), action_registry=action_registry
    )


def plan_workflow_with_two_pass(
    nl_requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    *,
    max_rounds: int = 10,
    max_repair_rounds: int | None = None,
    trace_context: TraceContext | None = None,
    trace_id: str | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Workflow:
    """Plan a workflow in two passes and validate the result."""

    parsed_requirement = analyze_user_requirement(
        nl_requirement,
        max_rounds=max_rounds,
    )

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
                parsed_requirement=parsed_requirement,
                search_service=search_service,
                action_registry=action_registry,
                max_rounds=max_rounds,
                progress_callback=progress_callback,
            )
        log_section("Phase one result: Workflow Skeleton")
        log_json("Workflow Skeleton", skeleton_raw)
        if progress_callback:
            try:
                progress_callback("structure_completed", skeleton_raw)
            except Exception:
                log_debug(
                    "[plan_workflow_with_two_pass] progress_callback failed during structure stage and was ignored."
                )

        precheck_errors = precheck_loop_body_graphs(skeleton_raw)
        if precheck_errors:
            log_warn(
                "[plan_workflow_with_two_pass] Structure stage found loop body referencing missing nodes."
            )

        try:
            skeleton = Workflow.model_validate(skeleton_raw)
        except PydanticValidationError as exc:
            raise

        guarded = ensure_registered_actions(
            skeleton, action_registry=action_registry, search_service=search_service
        )
        guarded = guarded if isinstance(guarded, Workflow) else Workflow.model_validate(guarded)

        validation_errors = _validate_workflow(guarded, action_registry)
        if validation_errors:
            log_warn(
                "[plan_workflow_with_two_pass] Validation produced errors; returning workflow with warnings.",
                json.dumps([err.model_dump() for err in validation_errors], ensure_ascii=False),
            )

        return guarded


def update_workflow_with_two_pass(
    existing_workflow: Mapping[str, Any],
    requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    *,
    max_rounds: int = 100,
    max_repair_rounds: int | None = None,
    trace_context: TraceContext | None = None,
    trace_id: str | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Workflow:
    """Update an existing workflow using the same two-pass planning flow."""

    context = trace_context or TraceContext.create(trace_id=trace_id, span_name="update_orchestrator")
    with use_trace_context(context):
        log_event(
            "update_start",
            {
                "max_rounds": max_rounds,
            },
            context=context,
        )

        base_workflow = Workflow.model_validate(existing_workflow)

        parsed_requirement = analyze_user_requirement(
            requirement,
            existing_workflow=base_workflow.model_dump(by_alias=True),
            max_rounds=max_rounds,
        )

        with child_span("structure_planning"):
            updated_raw = plan_workflow_structure_with_llm(
                parsed_requirement=parsed_requirement,
                search_service=search_service,
                action_registry=action_registry,
                max_rounds=max_rounds,
                progress_callback=progress_callback,
                existing_workflow=base_workflow.model_dump(by_alias=True),
            )

        updated_workflow = Workflow.model_validate(updated_raw)
        updated_workflow = ensure_registered_actions(
            updated_workflow, action_registry=action_registry, search_service=search_service
        )
        updated_workflow = (
            updated_workflow
            if isinstance(updated_workflow, Workflow)
            else Workflow.model_validate(updated_workflow)
        )

        if progress_callback:
            try:
                progress_callback("update_completed", updated_workflow.model_dump(by_alias=True))
            except Exception:
                log_debug(
                    "[update_workflow_with_two_pass] progress_callback failed during update stage and was ignored."
                )

        validation_errors = _validate_workflow(updated_workflow, action_registry)
        if validation_errors:
            log_warn(
                "[update_workflow_with_two_pass] Validation produced errors; returning workflow with warnings.",
                json.dumps([err.model_dump() for err in validation_errors], ensure_ascii=False),
            )

        return updated_workflow


__all__ = ["plan_workflow_with_two_pass", "update_workflow_with_two_pass"]
