"""Utility helpers to apply workflow edits from tool-calls.

This module offers a minimal editing session object that mutates a workflow
dictionary in-place. It is intentionally lightweight so that multiple planner
stages (parameter completion, verification repair) can reuse the same tool
handlers without duplicating mutation logic.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Tuple

from velvetflow.loop_dsl import iter_workflow_and_loop_body_nodes


class WorkflowEditingSession:
    """Mutate a workflow dict based on tool-call instructions."""

    def __init__(self, workflow: Dict[str, Any]):
        self.workflow: Dict[str, Any] = copy.deepcopy(workflow)

    # ---- helpers ----
    def _find_node(self, node_id: str, workflow: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        data = workflow or self.workflow
        if not isinstance(data, dict):
            return None

        for node in data.get("nodes", []) or []:
            if not isinstance(node, dict):
                continue
            if node.get("id") == node_id:
                return node
            if node.get("type") == "loop":
                params = node.get("params") or {}
                body = params.get("body_subgraph") if isinstance(params, dict) else None
                if isinstance(body, dict):
                    found = self._find_node(node_id, body)
                    if found:
                        return found
        return None

    def _edge_containers(self) -> Iterable[Tuple[Optional[str], List[Dict[str, Any]]]]:
        """Yield (scope_id, edges_list) pairs for root and loop bodies."""

        root_edges = self.workflow.get("edges")
        if not isinstance(root_edges, list):
            root_edges = []
            self.workflow["edges"] = root_edges
        yield None, root_edges

        for node in iter_workflow_and_loop_body_nodes(self.workflow):
            if not isinstance(node, dict) or node.get("type") != "loop":
                continue
            params = node.get("params") if isinstance(node.get("params"), dict) else None
            body = params.get("body_subgraph") if isinstance(params, dict) else None
            if not isinstance(body, dict):
                continue
            edges = body.get("edges")
            if not isinstance(edges, list):
                edges = []
                body["edges"] = edges
            yield node.get("id"), edges

    def _resolve_edge_list(self, scope_node_id: Optional[str]) -> Optional[List[Dict[str, Any]]]:
        for scope, edges in self._edge_containers():
            if scope == scope_node_id:
                return edges
        return None

    # ---- tool handlers ----
    def update_node_params(self, node_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        node = self._find_node(node_id)
        if not node:
            return {"status": "error", "message": f"未找到节点 {node_id}"}

        node["params"] = params or {}
        return {"status": "ok", "type": "node_params_updated", "node_id": node_id, "params": node["params"]}

    def update_node(self, node_id: str, *, action_id: Optional[str] = None, display_name: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        node = self._find_node(node_id)
        if not node:
            return {"status": "error", "message": f"未找到节点 {node_id}"}

        if action_id is not None:
            node["action_id"] = action_id
        if display_name is not None:
            node["display_name"] = display_name
        if params is not None:
            node["params"] = params

        return {
            "status": "ok",
            "type": "node_updated",
            "node_id": node_id,
            "action_id": node.get("action_id"),
            "display_name": node.get("display_name"),
            "params": node.get("params"),
        }

    def add_edge(self, from_node: str, to_node: str, condition: Optional[str], scope_node_id: Optional[str]) -> Dict[str, Any]:
        edges = self._resolve_edge_list(scope_node_id)
        if edges is None:
            return {"status": "error", "message": f"未找到 scope={scope_node_id} 对应的 edges 容器"}

        edges.append({"from": from_node, "to": to_node, "condition": condition})
        return {"status": "ok", "type": "edge_added", "scope": scope_node_id}

    def update_edge(self, from_node: str, to_node: str, condition: Optional[str], scope_node_id: Optional[str]) -> Dict[str, Any]:
        edges = self._resolve_edge_list(scope_node_id)
        if edges is None:
            return {"status": "error", "message": f"未找到 scope={scope_node_id} 对应的 edges 容器"}

        for edge in edges:
            if not isinstance(edge, dict):
                continue
            if edge.get("from") == from_node and edge.get("to") == to_node:
                edge["condition"] = condition
                return {"status": "ok", "type": "edge_updated", "scope": scope_node_id}

        return {"status": "error", "message": f"未找到 from={from_node}, to={to_node} 的 edge"}

    def remove_edge(self, from_node: str, to_node: str, scope_node_id: Optional[str]) -> Dict[str, Any]:
        edges = self._resolve_edge_list(scope_node_id)
        if edges is None:
            return {"status": "error", "message": f"未找到 scope={scope_node_id} 对应的 edges 容器"}

        before = len(edges)
        edges[:] = [e for e in edges if not (isinstance(e, dict) and e.get("from") == from_node and e.get("to") == to_node)]
        if len(edges) == before:
            return {"status": "error", "message": f"未找到 from={from_node}, to={to_node} 的 edge"}

        return {"status": "ok", "type": "edge_removed", "scope": scope_node_id}

    def finalize(self) -> Dict[str, Any]:
        return {"status": "ok", "type": "final", "workflow": self.workflow}

    def handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "update_node_params":
            return self.update_node_params(args.get("node_id", ""), args.get("params") or {})

        if name == "update_node":
            return self.update_node(
                args.get("node_id", ""),
                action_id=args.get("action_id"),
                display_name=args.get("display_name"),
                params=args.get("params"),
            )

        if name == "add_edge":
            return self.add_edge(
                args.get("from_node", ""),
                args.get("to_node", ""),
                args.get("condition"),
                args.get("scope_node_id"),
            )

        if name == "update_edge":
            return self.update_edge(
                args.get("from_node", ""),
                args.get("to_node", ""),
                args.get("condition"),
                args.get("scope_node_id"),
            )

        if name == "remove_edge":
            return self.remove_edge(
                args.get("from_node", ""),
                args.get("to_node", ""),
                args.get("scope_node_id"),
            )

        if name in {"submit_workflow", "finalize_workflow"}:
            return self.finalize()

        return {"status": "error", "message": f"未知工具 {name}"}


__all__ = ["WorkflowEditingSession"]
