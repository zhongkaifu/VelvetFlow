"""Requirement analysis helpers for breaking down natural language requests."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Mapping, MutableMapping

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_info, log_section
from velvetflow.planner.agent_runtime import Agent, Runner, function_tool


def _normalize_requirements_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the requirement payload structure.

    The analyzer and planner share this helper to ensure every requirement item
    includes the expected fields and that status is always provided as a string
    describing progress (for example "Not started"/"In progress"/"Completed"), and
    that mapped nodes
    are normalized as string arrays.
    """

    requirements = payload.get("requirements")
    if not isinstance(requirements, list):
        raise ValueError("requirements must be a list.")
    if len(requirements) == 0:
        raise ValueError("requirements must include at least one subtask.")

    normalized_items = []

    for idx, item in enumerate(requirements):
        if not isinstance(item, MutableMapping):
            raise ValueError(f"requirements[{idx}] must be an object.")
        for key in ["description", "intent", "inputs", "constraints", "status"]:
            if key not in item:
                raise ValueError(f"requirements[{idx}] is missing field {key}.")
        if not isinstance(item.get("status"), str):
            raise ValueError(
                f"requirements[{idx}].status must be a string describing progress (e.g., Not started / In progress / Completed)."
            )

        mapped_node = item.get("mapped_node", [])
        if mapped_node is None:
            mapped_node = []
        if not isinstance(mapped_node, list) or any(
            not isinstance(node_id, str) for node_id in mapped_node
        ):
            raise ValueError(
                f"requirements[{idx}].mapped_node must be an array of strings for mapped workflow node ids."
            )

        normalized_item = dict(item)
        normalized_item["mapped_node"] = list(mapped_node)
        normalized_items.append(normalized_item)

    assumptions = payload.get("assumptions", [])
    if assumptions is None:
        assumptions = []

    return {
        "requirements": normalized_items,
        "assumptions": list(assumptions) if isinstance(assumptions, list) else [],
    }


def _build_requirement_prompt() -> str:
    return (
        "You are a requirement analysis assistant. Carefully read the user's natural language request and break it into a structured checklist for workflow construction.\n"
        "Analysis steps:\n"
        "1) Identify the main goals and subtasks, adding any necessary contextual assumptions.\n"
        "2) For each subtask, clarify the intent, required inputs, constraints, current status, and the list of mapped workflow node ids (mapped_node, which may be empty).\n"
        "3) Collect global assumptions or prerequisites into assumptions.\n"
        "You must call the tool and return the JSON structure below; keep field meanings complete.\n"
        "{\n"
        "  \"requirements\": [\n"
        "    {\n"
        "      \"description\": \"What the subtask does\",\n"
        "      \"intent\": \"Purpose or value delivered by the subtask\",\n"
        "      \"inputs\": [\"Inputs needed to complete the subtask\"],\n"
        "      \"constraints\": [\"Constraints that must be followed\"],\n"
        "      \"status\": \"Not started / In progress / Completed / other useful hints\",\n"
        "      \"mapped_node\": [\"Mapped workflow node ids, may be empty\"]\n"
        "    }\n"
        "  ],\n"
        "  \"assumptions\": [\"Assumptions or notes about the requirement\"]\n"
        "}\n"
        "If an existing workflow is provided, use its nodes to populate mapped_node and ensure status reflects current progress accurately."
    )


def analyze_user_requirement(
    nl_requirement: str,
    *,
    existing_workflow: Mapping[str, Any] | None = None,
    max_rounds: int = 10,
) -> Dict[str, Any]:
    """Break down the raw natural-language requirement into structured items."""

    log_section("Requirement analysis - Agent SDK")

    parsed_requirement: Dict[str, Any] | None = None

    def _log_tool_call(tool_name: str, payload: Mapping[str, Any] | None = None) -> None:
        if payload:
            log_info(
                f"[RequirementAnalysis] tool={tool_name}",
                json.dumps(payload, ensure_ascii=False),
            )
        else:
            log_info(f"[RequirementAnalysis] tool={tool_name}")

    def _log_tool_result(tool_name: str, result: Mapping[str, Any]) -> None:
        log_info(
            f"[RequirementAnalysis] tool_result={tool_name}",
            json.dumps(result, ensure_ascii=False),
        )

    def _return_tool_result(tool_name: str, result: Mapping[str, Any]) -> Mapping[str, Any]:
        _log_tool_result(tool_name, result)
        return result

    @function_tool(strict_mode=False)
    def plan_user_requirement(payload: Mapping[str, Any]) -> Mapping[str, Any]:
        nonlocal parsed_requirement
        _log_tool_call("plan_user_requirement", payload)
        try:
            normalized_payload = _normalize_requirements_payload(payload)
        except ValueError as exc:  # pragma: no cover - defensive
            return _return_tool_result(
                "plan_user_requirement",
                {"status": "error", "message": str(exc)},
            )

        parsed_requirement = normalized_payload
        return _return_tool_result(
            "plan_user_requirement",
            {"status": "ok", "requirement": normalized_payload},
        )

    agent = Agent(
        name="RequirementAnalyzer",
        instructions=_build_requirement_prompt(),
        tools=[plan_user_requirement],
        model=OPENAI_MODEL,
    )

    def _format_prompt_item(item: Any) -> Any:
        """Ensure each message passed to the Agent SDK has a valid role/content."""

        if isinstance(item, Mapping) and "role" not in item:
            return {
                "role": "user",
                "content": json.dumps(item, ensure_ascii=False),
            }
        return item

    def _run_agent(prompt: Any) -> None:
        # Agents SDK expects a list of role-tagged messages; stringify dict payloads
        # into a single user message to avoid empty-role errors.
        base_prompt = prompt if isinstance(prompt, list) else [prompt]
        initial_prompt = [_format_prompt_item(item) for item in base_prompt]
        try:
            Runner.run_sync(agent, initial_prompt, max_turns=max_rounds)  # type: ignore[arg-type]
        except TypeError:  # pragma: no cover - fallback for async runner
            coro = Runner.run(agent, initial_prompt)  # type: ignore[call-arg]
            if asyncio.iscoroutine(coro):
                asyncio.run(coro)

    _run_agent(
        {
            "nl_requirement": nl_requirement,
            "existing_workflow": existing_workflow or {},
        }
    )

    if not parsed_requirement:
        # Fallback: if the agent failed to call the tool, produce a minimal
        # structured requirement so the workflow planner can proceed instead
        # of surfacing an opaque parsing error to the user.
        fallback_payload = {
            "requirements": [
                {
                    "description": nl_requirement,
                    "intent": "",
                    "inputs": [],
                    "constraints": [],
                    "status": "Not started",
                    "mapped_node": [],
                }
            ],
            "assumptions": [],
        }
        parsed_requirement = _normalize_requirements_payload(fallback_payload)

    return parsed_requirement


__all__ = ["analyze_user_requirement", "_normalize_requirements_payload"]
