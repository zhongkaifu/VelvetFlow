"""Graph utilities and traversal helpers for workflow execution."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set

from velvetflow.logging_utils import log_warn
from velvetflow.models import Node, Workflow, merge_edges


class GraphTraversalMixin:
    """Provide graph traversal helpers for executors."""

    def _derive_edges(self, workflow: Workflow) -> List[Dict[str, Any]]:
        """Rebuild implicit edges from the latest node bindings."""

        # 仅根据节点定义推导连线，统一从 start 开始遍历。
        return merge_edges(workflow.nodes)

    def _find_start_nodes(
        self, nodes: Mapping[str, Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> List[str]:
        """Locate the designated start node and enforce single entry."""

        starts = [nid for nid, n in nodes.items() if n.get("type") == "start"]
        if starts:
            return [starts[0]]

        raise ValueError("工作流缺少 start 节点，无法执行。")

    def _topological_sort(self, workflow: Workflow) -> List[Node]:
        """Return execution order while validating reachability constraints."""

        nodes = workflow.nodes
        edges = self._derive_edges(workflow)

        indegree: Dict[str, int] = {n.id: 0 for n in nodes}
        adjacency: Dict[str, List[str]] = {n.id: [] for n in nodes}
        node_lookup: Dict[str, Node] = {n.id: n for n in nodes}
        for e in edges:
            if not isinstance(e, Mapping):
                continue
            frm = e.get("from")
            to = e.get("to")
            if not frm or not to or frm not in indegree or to not in indegree:
                continue
            indegree[to] += 1
            adjacency[frm].append(to)

        stack = [nid for nid, deg in indegree.items() if deg == 0]
        if not stack:
            log_warn("所有节点均有入度，回退为声明顺序并警告不可达风险")
            return nodes

        ordered: List[str] = []
        while stack:
            nid = stack.pop()
            ordered.append(nid)
            for neighbor in adjacency[nid]:
                indegree[neighbor] -= 1
                if indegree[neighbor] == 0:
                    stack.append(neighbor)

        if len(ordered) != len(nodes):
            log_warn("检测到环路或不可达节点，回退为声明顺序")
            reachable: Set[str] = set(ordered)
            unreachable = [n.id for n in nodes if n.id not in reachable]
            if unreachable:
                log_warn(f"不可达节点: {unreachable}")
            ordered = [n.id for n in nodes]
        else:
            reachable = set(ordered)

        if reachable:
            ordered = [nid for nid in ordered if nid in reachable]

        start_nodes = [n.id for n in nodes if n.type == "start"]
        end_nodes = [n.id for n in nodes if n.type == "end"]
        if start_nodes:
            primary_start = start_nodes[0]
            ordered = [primary_start] + [nid for nid in ordered if nid != primary_start]
        if end_nodes:
            tail_end = [nid for nid in ordered if nid in end_nodes]
            ordered = [nid for nid in ordered if nid not in end_nodes] + tail_end

        return [node_lookup[nid] for nid in ordered]

    def _next_nodes(
        self,
        edges: List[Dict[str, Any]],
        nid: str,
        cond_value: Any = None,
        nodes_data: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        res: List[str] = []
        cond_label: Optional[str]
        if isinstance(cond_value, bool):
            cond_label = "true" if cond_value else "false"
        elif cond_value is None:
            cond_label = None
        else:
            cond_label = str(cond_value)

        node_lookup = nodes_data if isinstance(nodes_data, Mapping) else self.nodes  # type: ignore[attr-defined]
        node_def = node_lookup.get(nid, {}) if isinstance(node_lookup.get(nid, {}), Mapping) else {}

        if node_def.get("type") == "switch":
            cases = node_def.get("cases") if isinstance(node_def.get("cases"), list) else []
            for case in cases:
                if not isinstance(case, Mapping):
                    continue
                raw_match = case.get("match") if "match" in case else case.get("value")
                match_values = raw_match if isinstance(raw_match, list) else [raw_match]
                if cond_value in match_values:
                    target = case.get("to_node")
                    if target in {None, "null"}:
                        return []
                    if isinstance(target, str):
                        return [target]
                    return []
            if "default_to_node" in node_def:
                default_target = node_def.get("default_to_node")
                if default_target in {None, "null"}:
                    return []
                if isinstance(default_target, str):
                    return [default_target]
                return []

        if node_def.get("type") == "condition" and isinstance(cond_value, bool):
            branch_key = "true_to_node" if cond_value else "false_to_node"
            if branch_key in node_def:
                branch_target = node_def.get(branch_key)
                if branch_target in {None, "null"}:
                    return []
                if isinstance(branch_target, str):
                    return [branch_target]
                return []
            return []

        for e in edges:
            if not isinstance(e, Mapping):
                continue
            frm = e.get("from")
            if frm != nid:
                continue
            cond = e.get("condition")
            if cond is None:
                res.append(e.get("to"))
            elif cond_label is not None and cond == cond_label:
                res.append(e.get("to"))
        return [r for r in res if r not in {None, "null"}]

    def _collect_downstream_nodes(
        self, edges: List[Dict[str, Any]], start: str
    ) -> Set[str]:
        """Return all nodes reachable from ``start`` following edge definitions."""

        visited: Set[str] = set()
        stack: List[str] = [start]

        while stack:
            nid = stack.pop()
            if not isinstance(nid, str) or nid in visited:
                continue
            visited.add(nid)
            for e in edges:
                if not isinstance(e, Mapping):
                    continue
                if e.get("from") != nid:
                    continue
                nxt = e.get("to")
                if isinstance(nxt, str) and nxt not in visited:
                    stack.append(nxt)

        return visited
