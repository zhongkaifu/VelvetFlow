"""Utilities for building workflow skeletons during planning."""

from typing import Any, Dict, List, Optional

from velvetflow.logging_utils import log_warn


class WorkflowBuilder:
    """Mutable builder used by the planner to accumulate nodes and edges."""

    def __init__(self):
        self.workflow_name: str = "unnamed_workflow"
        self.description: str = ""
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []

    def set_meta(self, name: str, description: Optional[str]):
        if name:
            self.workflow_name = name
        if description:
            self.description = description or ""

    def add_node(
        self,
        node_id: str,
        node_type: str,
        action_id: Optional[str],
        display_name: Optional[str],
        params: Optional[Dict[str, Any]],
    ):
        if node_id in self.nodes:
            log_warn(f"[Builder] 节点 {node_id} 已存在，将覆盖。")
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "action_id": action_id,
            "display_name": display_name,
            "params": params or {},
        }

    def add_edge(self, from_node: str, to_node: str, condition: Optional[str]):
        self.edges.append({"from": from_node, "to": to_node, "condition": condition})

    def to_workflow(self) -> Dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "description": self.description,
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
        }


__all__ = ["WorkflowBuilder"]
