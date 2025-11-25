"""Action registry helpers and guards used by the planner."""

import copy
from typing import Any, Dict, List, Optional, Union

from velvetflow.logging_utils import log_info, log_warn
from velvetflow.models import Workflow
from velvetflow.search import HybridActionSearchService


def _index_actions_by_id(action_registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {a["action_id"]: a for a in action_registry}


def ensure_registered_actions(
    workflow: Union[Workflow, Dict[str, Any]],
    action_registry: List[Dict[str, Any]],
    search_service: Optional[HybridActionSearchService] = None,
) -> Union[Workflow, Dict[str, Any]]:
    """
    确保所有 action 节点引用的 action_id 都存在于注册表。

    - 如果 action_id 合法，保持不变；
    - 如果 action_id 缺失或未注册，尝试用节点 display_name 进行搜索替换；
    - 如果依然无法匹配，则清空 action_id，避免携带非法 ID 进入后续阶段。
    """

    actions_by_id = _index_actions_by_id(action_registry)
    original_type = Workflow if isinstance(workflow, Workflow) else dict
    workflow_dict = (
        workflow.model_dump(by_alias=True)
        if isinstance(workflow, Workflow)
        else copy.deepcopy(workflow)
    )

    nodes = workflow_dict.get("nodes", []) if isinstance(workflow_dict, dict) else []

    for node in nodes:
        ntype = node.get("type")
        if ntype != "action":
            if node.get("action_id") is not None:
                nid = node.get("id", "<unknown>")
                log_warn(
                    f"[ActionGuard] 控制节点 '{nid}' 的 action_id 将被清空（type='{ntype}' 不需要 action_id）。"
                )
                node["action_id"] = None
            continue

        aid = node.get("action_id")
        if aid and aid in actions_by_id:
            continue

        nid = node.get("id", "<unknown>")
        display_name = node.get("display_name") or ""

        replacement: Optional[str] = None
        if search_service and display_name:
            candidates = search_service.search(query=display_name, top_k=1)
            if candidates:
                replacement = candidates[0].get("action_id")

        if replacement:
            log_info(
                f"[ActionGuard] 节点 '{nid}' 的 action_id='{aid}' 未注册，"
                f"已根据 display_name='{display_name}' 替换为 '{replacement}'。"
            )
            node["action_id"] = replacement
        else:
            if aid:
                log_warn(
                    f"[ActionGuard] 节点 '{nid}' 的 action_id='{aid}' 未注册且无法自动替换，"
                    "已清空该字段以便后续流程重新补齐。"
                )
            node["action_id"] = None

    if original_type is Workflow:
        return Workflow.model_validate(workflow_dict)
    return workflow_dict


__all__ = ["ensure_registered_actions", "_index_actions_by_id"]
