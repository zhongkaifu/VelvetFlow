"""Utility helpers to apply workflow edits from tool-calls.

This module offers a minimal editing session object that mutates a workflow
dictionary in-place. It is intentionally lightweight so that multiple planner
stages (parameter completion, verification repair) can reuse the same tool
handlers without duplicating mutation logic.
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from velvetflow.loop_dsl import iter_workflow_and_loop_body_nodes


class WorkflowEditingSession:
    """Mutate a workflow dict based on tool-call instructions."""

    def __init__(self, workflow: Dict[str, Any], *, validation_fn: Optional[Callable[[Dict[str, Any]], List[Any]]] = None):
        self.workflow: Dict[str, Any] = copy.deepcopy(workflow)
        self._validation_fn = validation_fn

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
    def add_node(
        self,
        node_id: str,
        node_type: str,
        *,
        action_id: Optional[str] = None,
        display_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        scope_node_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        container: Optional[Dict[str, Any]] = self.workflow
        if scope_node_id:
            scope_node = self._find_node(scope_node_id)
            if not scope_node or scope_node.get("type") != "loop":
                return {"status": "error", "message": f"未找到 loop 节点 {scope_node_id}"}

            scope_params = scope_node.setdefault("params", {}) if isinstance(scope_node.get("params"), dict) else {}
            scope_node["params"] = scope_params
            body_subgraph = scope_params.get("body_subgraph")
            if not isinstance(body_subgraph, dict):
                body_subgraph = {"nodes": [], "edges": []}
                scope_params["body_subgraph"] = body_subgraph
            container = body_subgraph

        if not isinstance(container, dict):
            return {"status": "error", "message": "workflow 数据结构无效"}

        nodes_list = container.setdefault("nodes", []) if isinstance(container.get("nodes"), list) else []
        container["nodes"] = nodes_list

        nodes_list.append(
            {
                "id": node_id,
                "type": node_type,
                "action_id": action_id,
                "display_name": display_name,
                "params": params or {},
            }
        )
        return {"status": "ok", "type": "node_added", "node_id": node_id, "scope": scope_node_id}

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

    def remove_node(self, node_id: str, scope_node_id: Optional[str]) -> Dict[str, Any]:
        container: Optional[Dict[str, Any]] = self.workflow
        if scope_node_id:
            scope_node = self._find_node(scope_node_id)
            if not scope_node or scope_node.get("type") != "loop":
                return {"status": "error", "message": f"未找到 loop 节点 {scope_node_id}"}

            params = scope_node.get("params") if isinstance(scope_node.get("params"), dict) else None
            body_subgraph = params.get("body_subgraph") if isinstance(params, dict) else None
            if not isinstance(body_subgraph, dict):
                return {"status": "error", "message": f"loop {scope_node_id} 缺少 body_subgraph"}
            container = body_subgraph

        if not isinstance(container, dict):
            return {"status": "error", "message": "workflow 数据结构无效"}

        nodes = container.get("nodes") if isinstance(container.get("nodes"), list) else []
        container["nodes"] = nodes

        before = len(nodes)
        nodes[:] = [n for n in nodes if not (isinstance(n, dict) and n.get("id") == node_id)]
        if len(nodes) == before:
            return {"status": "error", "message": f"未找到节点 {node_id}"}

        edges = container.get("edges") if isinstance(container.get("edges"), list) else []
        container["edges"] = edges
        edges[:] = [
            e
            for e in edges
            if not (
                isinstance(e, dict)
                and (e.get("from") == node_id or e.get("to") == node_id)
            )
        ]

        return {"status": "ok", "type": "node_removed", "node_id": node_id, "scope": scope_node_id}

    def finalize(self) -> Dict[str, Any]:
        return {"status": "ok", "type": "final", "workflow": self.workflow}

    def validate(self) -> Dict[str, Any]:
        if not self._validation_fn:
            return {"status": "error", "message": "validation_fn 未配置"}

        errors = self._validation_fn(self.workflow) or []
        normalized_errors: List[Dict[str, Any]] = []
        for err in errors:
            if hasattr(err, "__dict__"):
                try:
                    normalized_errors.append(asdict(err))
                    continue
                except TypeError:
                    pass
            if isinstance(err, dict):
                normalized_errors.append(err)
            else:
                normalized_errors.append({"message": str(err)})

        return {
            "status": "ok",
            "type": "validation_result",
            "errors": normalized_errors,
            "passed": len(normalized_errors) == 0,
        }

    def handle_tool_call(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if name == "add_node":
            return self.add_node(
                args.get("id", ""),
                args.get("type", ""),
                action_id=args.get("action_id"),
                display_name=args.get("display_name"),
                params=args.get("params"),
                scope_node_id=args.get("scope_node_id"),
            )

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

        if name == "remove_node":
            return self.remove_node(args.get("node_id", ""), args.get("scope_node_id"))

        if name in {"submit_workflow", "finalize_workflow"}:
            return self.finalize()

        if name == "validate_workflow":
            return self.validate()

        return {"status": "error", "message": f"未知工具 {name}"}


__all__ = ["WorkflowEditingSession"]
