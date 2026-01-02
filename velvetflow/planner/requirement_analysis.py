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
    describing progress (例如 "未开始"/"进行中"/"已完成"), and that mapped nodes
    are normalized as string arrays.
    """

    requirements = payload.get("requirements")
    if not isinstance(requirements, list):
        raise ValueError("requirements 必须是列表。")

    normalized_items = []

    for idx, item in enumerate(requirements):
        if not isinstance(item, MutableMapping):
            raise ValueError(f"requirements[{idx}] 必须是对象。")
        for key in ["description", "intent", "inputs", "constraints", "status"]:
            if key not in item:
                raise ValueError(f"requirements[{idx}] 缺少字段 {key}。")
        if not isinstance(item.get("status"), str):
            raise ValueError(
                f"requirements[{idx}].status 必须是字符串，用于标记进度（如未开始/进行中/已完成）。"
            )

        mapped_node = item.get("mapped_node", [])
        if mapped_node is None:
            mapped_node = []
        if not isinstance(mapped_node, list) or any(
            not isinstance(node_id, str) for node_id in mapped_node
        ):
            raise ValueError(
                f"requirements[{idx}].mapped_node 必须是字符串数组，用于记录映射到的 workflow 节点 id。"
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
        "你是一个需求分析助手，请将用户的自然语言需求拆解为结构化清单。\n"
        "输出格式必须为：\n"
        "{\n"
        "  \"requirements\": [\n"
        "    {\n"
        "      \"description\": \"Task description\",\n"
        "      \"intent\": \"The intention of this task\",\n"
        "      \"inputs\": [\"A list of input of this task\"],\n"
        "      \"constraints\": [\"the list of constraints that this task must obey\"],\n"
        "      \"status\": \"未开始 / 进行中 / 已完成 / 其他有助提示\",\n"
        "      \"mapped_node\": [\"workflow 节点 id 列表，可为空\"]\n"
        "    }\n"
        "  ],\n"
        "  \"assumptions\": [\"The assumptions of user's requirement\"]\n"
        "}\n"
        "请根据上下文明确每个子需求的输入、约束与当前状态，便于后续构建工作流时持续复用。"
    )


def analyze_user_requirement(
    nl_requirement: str,
    *,
    existing_workflow: Mapping[str, Any] | None = None,
    max_rounds: int = 10,
) -> Dict[str, Any]:
    """Break down the raw natural-language requirement into structured items."""

    log_section("需求分析 - Agent SDK")

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

    def _run_agent(prompt: Any) -> None:
        initial_prompt = prompt if isinstance(prompt, list) else [prompt]
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
        raise ValueError("未能解析用户需求，请重试或补充更多上下文。")

    return parsed_requirement


__all__ = ["analyze_user_requirement", "_normalize_requirements_payload"]
