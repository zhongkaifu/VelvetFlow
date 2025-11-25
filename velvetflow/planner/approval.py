"""Approval related checks for workflows."""

from typing import Any, Dict, List

from velvetflow.planner.action_guard import _index_actions_by_id


def _has_approval_node(nodes: List[Dict[str, Any]]) -> bool:
    """Heuristically detect whether an approval node already exists."""

    for node in nodes:
        if node.get("type") != "action":
            continue

        display_name = (node.get("display_name") or "").lower()
        action_id = (node.get("action_id") or "").lower()
        if "审批" in (node.get("display_name") or ""):
            return True
        if "approve" in display_name or "approval" in display_name:
            return True
        if "approve" in action_id or "approval" in action_id:
            return True

    return False


def detect_missing_approval_nodes(
    workflow: Dict[str, Any], action_registry: List[Dict[str, Any]]
) -> List[str]:
    """Identify approval-requiring actions that lack an approval step in the workflow."""

    nodes = workflow.get("nodes", []) or []
    actions_by_id = _index_actions_by_id(action_registry)

    approvals_needed: List[str] = []
    for node in nodes:
        if node.get("type") != "action":
            continue

        action_def = actions_by_id.get(node.get("action_id"))
        if action_def and action_def.get("requires_approval"):
            approvals_needed.append(node.get("action_id") or node.get("display_name") or node.get("id"))

    if not approvals_needed or _has_approval_node(nodes):
        return []

    target_text = "、".join(approvals_needed)
    return [f"动作 {target_text} 需要审批，但当前流程没有任何审批节点，请补充一个“审批”节点。"]


__all__ = ["detect_missing_approval_nodes"]
