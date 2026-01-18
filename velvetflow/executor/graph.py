"""Graph utilities and traversal helpers for workflow execution."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set

from velvetflow.logging_utils import log_warn
from velvetflow.models import (
    Node,
    Workflow,
    infer_depends_on_from_edges,
    infer_edges_from_bindings,
)


class GraphTraversalMixin:
    """Provide graph traversal helpers for executors."""

    def _derive_edges(self, workflow: Workflow) -> List[Dict[str, Any]]:
        """Rebuild implicit edges from the latest node bindings."""

        # 所有连线信息都由绑定实时推导，不再消费/缓存声明式 edges 字段。
        return infer_edges_from_bindings(workflow.nodes)

    def _find_start_nodes(
        self, nodes: Mapping[str, Dict[str, Any]], edges: List[Dict[str, Any]]
    ) -> List[str]:
        """Locate entry nodes by combining condition starts and inbound-free nodes."""

        all_ids = set(nodes.keys())
        to_ids = set()
        from_ids = set()
        depends_map = infer_depends_on_from_edges(nodes.values(), edges)
        for e in edges:
            if not isinstance(e, Mapping):
                continue
            to_ids.add(e.get("to"))
            from_ids.add(e.get("from"))
        for nid, deps in depends_map.items():
            for dep in deps:
                to_ids.add(nid)
                from_ids.add(dep)

        inbound_free = list(all_ids - to_ids)

        condition_starts = [
            nid for nid, n in nodes.items() if n.get("type") == "condition" and nid in inbound_free
        ]

        outbound_roots = [nid for nid in inbound_free if nid in from_ids]

        merged = list(dict.fromkeys(condition_starts + outbound_roots + inbound_free))
        return merged

    def _topological_sort(self, workflow: Workflow) -> List[Node]:
        """Return execution order while validating reachability constraints."""

        nodes = workflow.nodes
        edges = self._derive_edges(workflow)
        depends_map = infer_depends_on_from_edges(workflow.model_dump(by_alias=True).get("nodes", []), edges)

        indegree: Dict[str, int] = {n.id: 0 for n in nodes}
        adjacency: Dict[str, List[str]] = {n.id: [] for n in nodes}
        node_lookup: Dict[str, Node] = {n.id: n for n in nodes}
        used_dep_edges = False
        for nid, deps in depends_map.items():
            for dep in deps:
                if dep in indegree and nid in indegree:
                    indegree[nid] += 1
                    adjacency[dep].append(nid)
                    used_dep_edges = True

        if not used_dep_edges:
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
