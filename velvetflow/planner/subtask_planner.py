# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Lightweight helper to plan remaining subtasks for workflow construction."""

from __future__ import annotations

import re
from typing import Any, Iterable, List, Mapping


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
    *, nl_requirement: str, workflow: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Return remaining subtasks based on requirement and current workflow DSL."""

    requirement = (nl_requirement or "").strip()
    if not requirement:
        return {
            "status": "error",
            "message": "requirement is empty; provide a natural-language request first",
            "remaining_subtasks": [],
            "covered_subtasks": [],
        }

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
        "covered_subtasks": covered,
        "remaining_subtasks": remaining,
        "workflow_snapshot": {
            "node_count": len(workflow.get("nodes", []))
            if isinstance(workflow, Mapping)
            else 0,
            "summary": workflow_summary,
        },
    }


__all__ = ["plan_remaining_subtasks"]
