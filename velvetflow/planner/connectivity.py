"""Graph connectivity helpers used across planning stages."""

from collections import deque
from typing import Any, Dict, List, Optional


def ensure_edges_connectivity(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    保留已有 edge（Planner/LLM 已经设计好的结构），并保证所有节点连通。

    - 保证所有节点至少在一条 edge 中出现（除非只有一个节点）
    - 保证从某个 start 节点能到所有节点（补充必要的边）
    """

    if not nodes:
        return []

    node_ids = [n["id"] for n in nodes]
    id_set = set(node_ids)

    # 1) 过滤非法边
    cleaned_edges: List[Dict[str, Any]] = []
    for e in edges:
        frm = e.get("from")
        to = e.get("to")
        if frm in id_set and to in id_set:
            cleaned_edges.append({"from": frm, "to": to, "condition": e.get("condition")})
    edges = cleaned_edges

    # 2) 找 start 节点
    start_nodes = [n["id"] for n in nodes if n.get("type") == "start"]
    if not start_nodes:
        to_ids = {e["to"] for e in edges}
        start_nodes = [nid for nid in node_ids if nid not in to_ids]

    if not start_nodes:
        start_nodes = [node_ids[0]]

    # 3) BFS 找 reachable
    adj: Dict[str, List[str]] = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])

    reachable: set = set()
    dq = deque(start_nodes)
    while dq:
        x = dq.popleft()
        if x in reachable:
            continue
        reachable.add(x)
        for y in adj.get(x, []):
            if y not in reachable:
                dq.append(y)

    # 4) 把不可达节点挂上去
    unreachable = [nid for nid in node_ids if nid not in reachable]
    if not unreachable:
        return edges  # 已经全连通

    from_ids = {e["from"] for e in edges}
    to_ids = {e["to"] for e in edges}
    tail_candidates = [nid for nid in node_ids if nid in from_ids and nid not in to_ids]
    if tail_candidates:
        current_tail = tail_candidates[0]
    else:
        current_tail = list(reachable)[-1] if reachable else start_nodes[0]

    for u in unreachable:
        if current_tail == u:
            continue
        edges.append({"from": current_tail, "to": u, "condition": None})
        current_tail = u

    return edges


__all__ = ["ensure_edges_connectivity"]
