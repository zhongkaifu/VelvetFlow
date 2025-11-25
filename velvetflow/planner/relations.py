"""Helpers for traversing workflow graph relationships."""

from collections import deque
from typing import Any, Dict, List

from velvetflow.models import Node, Workflow


def get_upstream_nodes(workflow: Workflow, target_node_id: str) -> List[Node]:
    """沿着 edges 从 target_node_id 反向 DFS/BFS，收集所有上游节点。"""

    node_map = {n.id: n for n in workflow.nodes}
    if target_node_id not in node_map:
        return []

    reverse_adj: Dict[str, List[str]] = {}
    for e in workflow.edges:
        reverse_adj.setdefault(e.to_node, []).append(e.from_node)

    visited: set[str] = set()
    dq: deque[str] = deque()
    dq.append(target_node_id)

    while dq:
        nid = dq.popleft()
        for upstream_id in reverse_adj.get(nid, []):
            if upstream_id in visited:
                continue
            visited.add(upstream_id)
            dq.append(upstream_id)

    return [node_map[uid] for uid in node_map if uid in visited]


def build_node_relations(workflow_skeleton: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    nodes = workflow_skeleton.get("nodes", [])
    edges = workflow_skeleton.get("edges", [])
    node_ids = {n["id"] for n in nodes}
    relations: Dict[str, Dict[str, List[str]]] = {
        nid: {"upstream": [], "downstream": []} for nid in node_ids
    }
    for e in edges:
        frm = e.get("from")
        to = e.get("to")
        if frm in node_ids and to in node_ids:
            relations[frm]["downstream"].append(to)
            relations[to]["upstream"].append(frm)
    return relations


__all__ = ["get_upstream_nodes", "build_node_relations"]
