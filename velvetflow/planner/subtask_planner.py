# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Lightweight helper to plan remaining subtasks for workflow construction."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Iterable, List, Mapping

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import child_span, log_error, log_llm_usage


def _split_requirement_into_subtasks(requirement: str) -> List[str]:
    """Break a natural-language requirement into ordered, deduped subtasks."""

    if not requirement:
        return []

    # Split by common Chinese/English separators and sequencing cue words.
    parts = re.split(r"[，。；;,.\n]+|然后|并且|以及|同时|接着|并|随后", requirement)
    cleaned: List[str] = []
    for part in parts:
        task = part.strip()
        if not task:
            continue
        # Remove leading connectors such as “先”/“再”.
        task = re.sub(r"^(先|再|继续|需要|请)", "", task).strip()
        if task and task not in cleaned:
            cleaned.append(task)
    return cleaned


def _summarize_workflow_nodes(workflow: Mapping[str, Any]) -> str:
    """Concatenate node metadata into a lowercase summary string."""

    nodes = workflow.get("nodes") if isinstance(workflow, Mapping) else []
    if not isinstance(nodes, Iterable) or isinstance(nodes, (str, bytes)):
        return ""

    summary_parts: List[str] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        for key in ("display_name", "action_id", "type", "id"):
            value = node.get(key)
            if isinstance(value, str) and value.strip():
                summary_parts.append(value.strip())

    return " ".join(summary_parts).lower()


def _is_subtask_covered(subtask: str, workflow_summary: str) -> bool:
    """Determine whether a subtask is already represented in the workflow."""

    normalized = subtask.strip().lower()
    if not normalized or not workflow_summary:
        return False

    if normalized in workflow_summary:
        return True

    tokens = [
        token
        for token in re.split(r"[\s,;，。；、/]+", normalized)
        if len(token) >= 2
    ]
    if not tokens:
        return False

    hits = sum(1 for token in tokens if token in workflow_summary)
    coverage_ratio = hits / len(tokens)
    return coverage_ratio >= 0.6 or hits >= 2


def plan_remaining_subtasks(
    *,
    nl_requirement: str,
    workflow: Mapping[str, Any],
    use_llm: bool | None = None,
    llm_solver: Callable[[str, Mapping[str, Any], str], Mapping[str, Any]] | None = None,
    model: str = OPENAI_MODEL,
) -> Mapping[str, Any]:
    """Return remaining subtasks based on requirement and current workflow DSL.

    When ``use_llm`` is True (or left as ``None`` with an available OPENAI_API_KEY),
    the helper will ask an LLM to reason about coverage; otherwise, a
    deterministic heuristic is used.
    """

    requirement = (nl_requirement or "").strip()
    if not requirement:
        return {
            "status": "error",
            "message": "requirement is empty; provide a natural-language request first",
            "remaining_subtasks": [],
            "covered_subtasks": [],
        }

    heuristic = _plan_with_heuristics(requirement, workflow)

    should_use_llm = use_llm if use_llm is not None else bool(os.environ.get("OPENAI_API_KEY"))
    if not should_use_llm:
        return heuristic

    try:
        llm_payload = (
            llm_solver(requirement, workflow, model)
            if llm_solver
            else _plan_with_llm(requirement, workflow, model=model)
        )
        normalized_llm = _normalize_llm_result(llm_payload)
        if not normalized_llm:
            return heuristic

        subtasks = normalized_llm.get("subtasks") or heuristic.get("subtasks") or []
        covered = normalized_llm.get("covered_subtasks") or heuristic.get("covered_subtasks") or []
        remaining = normalized_llm.get("remaining_subtasks") or heuristic.get("remaining_subtasks") or []

        return {
            **heuristic,
            "method": "llm",
            "analysis": normalized_llm.get("analysis", ""),
            "total_subtasks": len(subtasks),
            "covered_subtasks": covered,
            "remaining_subtasks": remaining,
        }
    except Exception as exc:  # pragma: no cover - defensive logging
        log_error(f"[plan_remaining_subtasks] LLM 调用失败，已回退到启发式：{exc}")
        return heuristic


def _plan_with_heuristics(requirement: str, workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates = _split_requirement_into_subtasks(requirement) or [requirement]
    workflow_summary = _summarize_workflow_nodes(workflow)

    covered: List[str] = []
    remaining: List[str] = []
    for task in candidates:
        if _is_subtask_covered(task, workflow_summary):
            covered.append(task)
        else:
            remaining.append(task)

    return {
        "status": "ok",
        "requirement": requirement,
        "total_subtasks": len(candidates),
        "subtasks": candidates,
        "covered_subtasks": covered,
        "remaining_subtasks": remaining,
        "workflow_snapshot": {
            "node_count": len(workflow.get("nodes", []))
            if isinstance(workflow, Mapping)
            else 0,
            "summary": workflow_summary,
        },
    }


def _plan_with_llm(
    requirement: str,
    workflow: Mapping[str, Any],
    *,
    model: str = OPENAI_MODEL,
    client: OpenAI | None = None,
) -> Mapping[str, Any]:
    """Ask an LLM to infer covered vs. missing subtasks from the workflow."""

    llm_client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    payload = {
        "requirement": requirement,
        "workflow": _normalize_workflow_for_llm(workflow),
        "workflow_summary": _summarize_workflow_nodes(workflow),
    }

    system_prompt = (
        "你是一名严谨的工作流规划助手。\n"
        "给定用户需求 (requirement) 和当前 workflow DSL，\n"
        "请先把 requirement 拆成 3-8 个有序的子任务（subtasks，按原始顺序）。\n"
        "然后结合 workflow 的 nodes/summary 判断哪些子任务已被覆盖 (covered_subtasks)，\n"
        "哪些仍然缺失 (remaining_subtasks)。\n"
        "输出必须是 JSON 对象，形如：\n"
        "{\n"
        "  \"subtasks\": [\"...\"],\n"
        "  \"covered_subtasks\": [\"...\"],\n"
        "  \"remaining_subtasks\": [\"...\"],\n"
        "  \"analysis\": \"简短中文分析\"\n"
        "}\n"
        "不要添加额外字段，不要输出 Markdown 代码块。"
    )

    with child_span("plan_remaining_subtasks_llm"):
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
    log_llm_usage(model, getattr(resp, "usage", None), operation="subtask_planning")

    if not resp.choices:
        raise RuntimeError("plan_remaining_subtasks LLM 未返回任何结果")

    content = resp.choices[0].message.content or ""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                text = rest

    return json.loads(text)


def _normalize_llm_result(payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        return {}

    def _clean_list(values: Any) -> List[str]:
        seen = []
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            return []
        for item in values:
            if isinstance(item, str):
                normalized = item.strip()
                if normalized and normalized not in seen:
                    seen.append(normalized)
        return seen

    return {
        "subtasks": _clean_list(payload.get("subtasks")),
        "covered_subtasks": _clean_list(payload.get("covered_subtasks")),
        "remaining_subtasks": _clean_list(payload.get("remaining_subtasks")),
        "analysis": payload.get("analysis") or payload.get("reasoning") or "",
    }


def _normalize_workflow_for_llm(workflow: Mapping[str, Any]) -> Mapping[str, Any]:
    normalized = dict(workflow) if isinstance(workflow, Mapping) else {}
    normalized.setdefault("workflow_name", "unnamed_workflow")
    normalized.setdefault("description", "")
    normalized.setdefault("nodes", [])
    normalized.pop("edges", None)
    return normalized


__all__ = ["plan_remaining_subtasks"]
