# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Utilities for building workflow skeletons during planning."""

import re
import copy

from typing import Any, Dict, Mapping, Optional

from velvetflow.logging_utils import log_warn
from velvetflow.loop_dsl import iter_workflow_and_loop_body_nodes
from velvetflow.models import infer_depends_on_from_edges, infer_edges_from_bindings
from velvetflow.reference_utils import canonicalize_template_placeholders, normalize_reference_path


def _normalize_condition_label(raw: Any) -> Optional[str]:
    if isinstance(raw, bool):
        return "true" if raw else "false"

    if isinstance(raw, str):
        lowered = raw.lower()
        if lowered in {"true", "false"}:
            return lowered

    return None


def _attach_depends_on(workflow: Dict[str, Any]) -> None:
    edges = workflow.get("edges") if isinstance(workflow.get("edges"), list) else []
    nodes = list(iter_workflow_and_loop_body_nodes(workflow))
    depends_map = infer_depends_on_from_edges(nodes, edges)

    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        nid = node.get("id")
        if isinstance(nid, str):
            node["depends_on"] = depends_map.get(nid, [])


def attach_condition_branches(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Augment condition nodes with downstream branch references.

    在 planner 构建 skeleton 阶段就写入 condition 的 true/false 下游指向，
    方便后续执行器无需再依赖 edges 来决定分支流向。
    """

    copied = {**workflow}
    edges = copied.get("edges") if isinstance(copied.get("edges"), list) else []

    node_lookup: Dict[str, Mapping[str, Any]] = {}
    for node in iter_workflow_and_loop_body_nodes(copied):
        if isinstance(node, Mapping):
            node_lookup[str(node.get("id"))] = node

    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        from_node = edge.get("from") or edge.get("from_node")
        to_node = edge.get("to") or edge.get("to_node")
        cond_label = _normalize_condition_label(edge.get("condition"))

        if not cond_label or not isinstance(from_node, str) or not isinstance(to_node, str):
            continue

        source_node = node_lookup.get(from_node)
        if not source_node or source_node.get("type") != "condition":
            continue

        if cond_label == "true":
            source_node.setdefault("true_to_node", to_node)
        elif cond_label == "false":
            source_node.setdefault("false_to_node", to_node)

    # condition 节点必须包含 true_to_node/false_to_node 字段，即便没有明确的分支流向，
    # 也应以 null 表示分支结束，避免下游执行/校验阶段缺失必填字段。
    for node in node_lookup.values():
        if node.get("type") == "condition":
            node.setdefault("true_to_node", None)
            node.setdefault("false_to_node", None)

    _attach_depends_on(copied)
    return copied


_UNSET = object()
_TEMPLATE_REF_PATTERN = re.compile(
    r"\{\{\s*([^{}]+?)\s*\}\}|\$\{\{\s*([^{}]+?)\s*\}\}|\$\{\s*([^{}]+?)\s*\}"
)


def _extract_export_expr(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    if not stripped:
        return None
    normalized = normalize_reference_path(stripped)
    return normalized.strip()


def _replace_exports_in_string(
    value: str, *, export_exprs: Mapping[str, str], loop_id: str | None
) -> str:
    if not export_exprs:
        return value

    canonical = canonicalize_template_placeholders(value)
    stripped = canonical.strip()
    for key, expr in export_exprs.items():
        if stripped == f"exports.{key}":
            return f"{{{{ {expr} }}}}"
        if loop_id and stripped == f"result_of.{loop_id}.exports.{key}":
            return f"{{{{ {expr} }}}}"

    def _match_key(expr: str) -> str | None:
        expr = expr.strip()
        if expr.startswith("exports."):
            key = expr[len("exports.") :]
        elif loop_id and expr.startswith(f"result_of.{loop_id}.exports."):
            key = expr[len(f"result_of.{loop_id}.exports.") :]
        else:
            return None
        return key if key in export_exprs else None

    def _replace(match: re.Match[str]) -> str:
        expr = match.group(1) or match.group(2) or match.group(3) or ""
        key = _match_key(expr)
        if key:
            return f"{{{{ {export_exprs[key]} }}}}"
        return match.group(0)

    return _TEMPLATE_REF_PATTERN.sub(_replace, canonical)


def _replace_exports_in_params(
    value: Any, *, export_exprs: Mapping[str, str], loop_id: str | None
) -> Any:
    if isinstance(value, Mapping):
        return {
            key: _replace_exports_in_params(val, export_exprs=export_exprs, loop_id=loop_id)
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [
            _replace_exports_in_params(item, export_exprs=export_exprs, loop_id=loop_id)
            for item in value
        ]
    if isinstance(value, str):
        return _replace_exports_in_string(value, export_exprs=export_exprs, loop_id=loop_id)
    return value


def _replace_loop_body_exports(node: Mapping[str, Any]) -> None:
    params = node.get("params") if isinstance(node.get("params"), Mapping) else None
    if not isinstance(params, Mapping):
        return
    exports = params.get("exports")
    if not isinstance(exports, Mapping):
        return
    loop_id = node.get("id") if isinstance(node.get("id"), str) else None
    export_exprs = {
        key: expr
        for key, value in exports.items()
        if isinstance(key, str) and (expr := _extract_export_expr(value))
    }
    if not export_exprs:
        return

    body = params.get("body_subgraph") if isinstance(params.get("body_subgraph"), Mapping) else None
    body_nodes = body.get("nodes") if isinstance(body, Mapping) else None
    if not isinstance(body_nodes, list):
        return

    for body_node in body_nodes:
        if not isinstance(body_node, Mapping):
            continue
        body_params = body_node.get("params")
        if isinstance(body_params, Mapping):
            body_node["params"] = _replace_exports_in_params(
                body_params, export_exprs=export_exprs, loop_id=loop_id
            )
        if body_node.get("type") == "loop":
            _replace_loop_body_exports(body_node)


class WorkflowBuilder:
    """Mutable builder used by the planner to accumulate nodes."""

    def __init__(self):
        self.workflow_name: str = "unnamed_workflow"
        self.description: str = ""
        self.nodes: Dict[str, Dict[str, Any]] = {}

    def set_meta(self, name: str, description: Optional[str]):
        if name:
            self.workflow_name = name
        if description:
            self.description = description or ""

    def add_node(
        self,
        node_id: str,
        node_type: str,
        action_id: Optional[str] = None,
        display_name: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        out_params_schema: Optional[Dict[str, Any]] = None,
        true_to_node: Optional[str] = None,
        false_to_node: Optional[str] = None,
        cases: Optional[list[Dict[str, Any]]] = None,
        default_to_node: Optional[str] = None,
        parent_node_id: Optional[str] = None,
        depends_on: Optional[list[str]] = None,
    ):
        if node_id in self.nodes:
            log_warn(f"[Builder] 节点 {node_id} 已存在，将覆盖。")
        node: Dict[str, Any] = {"id": node_id, "type": node_type}
        if node_type == "action":
            node.update(
                {
                    "action_id": action_id,
                    "display_name": display_name,
                    "params": params or {},
                    "out_params_schema": out_params_schema,
                    "parent_node_id": parent_node_id,
                    "depends_on": depends_on or [],
                }
            )
        elif node_type == "condition":
            node.update(
                {
                    "display_name": display_name,
                    "params": params or {},
                    "true_to_node": true_to_node,
                    "false_to_node": false_to_node,
                    "parent_node_id": parent_node_id,
                    "depends_on": depends_on or [],
                }
            )
        elif node_type == "switch":
            node.update(
                {
                    "display_name": display_name,
                    "params": params or {},
                    "cases": cases or [],
                    "default_to_node": default_to_node,
                    "parent_node_id": parent_node_id,
                    "depends_on": depends_on or [],
                }
            )
        elif node_type == "loop":
            node.update(
                {
                    "display_name": display_name,
                    "params": params or {},
                    "parent_node_id": parent_node_id,
                    "depends_on": depends_on or [],
                }
            )
        else:
            node.update(
                {
                    "display_name": display_name,
                    "params": params or {},
                    "parent_node_id": parent_node_id,
                    "depends_on": depends_on or [],
                }
            )

        self.nodes[node_id] = node

    def update_node(self, node_id: str, **fields: Any):
        node = self.nodes.get(node_id)
        if not isinstance(node, dict):
            log_warn(f"[Builder] 节点 {node_id} 不存在，无法更新。")
            return

        for key, value in fields.items():
            if value is _UNSET:
                continue
            node[key] = value

    def to_workflow(self) -> Dict[str, Any]:
        # 深拷贝节点，避免在转换过程中污染 builder 内部状态。
        node_copies: list[Dict[str, Any]] = [copy.deepcopy(n) for n in self.nodes.values()]
        loop_lookup: Dict[str, Dict[str, Any]] = {}
        branch_lookup: Dict[str, Dict[str, Any]] = {}

        for node in node_copies:
            node_id = node.get("id")
            if not isinstance(node_id, str):
                continue
            node_type = node.get("type")

            if node_type == "loop":
                loop_lookup[node_id] = node
            if node_type == "parallel":
                params = node.get("params") if isinstance(node.get("params"), dict) else {}
                node["params"] = params

                branches = params.get("branches") if isinstance(params.get("branches"), list) else []
                normalized_branches: list[Dict[str, Any]] = []

                for branch in branches:
                    if not isinstance(branch, Mapping):
                        continue

                    branch_copy = copy.deepcopy(branch)
                    branch_id = branch_copy.get("id") if isinstance(branch_copy.get("id"), str) else None
                    branch_copy["sub_graph_nodes"] = (
                        list(branch_copy.get("sub_graph_nodes") or [])
                        if isinstance(branch_copy.get("sub_graph_nodes"), list)
                        else []
                    )

                    if branch_id:
                        branch_lookup[f"{node_id}:{branch_id}"] = branch_copy

                    normalized_branches.append(branch_copy)

                params["branches"] = normalized_branches

        root_nodes: list[Dict[str, Any]] = []

        for node in node_copies:
            parent_id = node.get("parent_node_id") if isinstance(node.get("parent_node_id"), str) else None
            branch_parent = branch_lookup.get(parent_id) if parent_id else None
            parent_loop = loop_lookup.get(parent_id) if parent_id else None

            if branch_parent is not None:
                branch_parent_nodes = branch_parent.get("sub_graph_nodes") if isinstance(branch_parent.get("sub_graph_nodes"), list) else []
                branch_parent["sub_graph_nodes"] = branch_parent_nodes
                branch_parent_nodes.append(node)
                continue

            if parent_loop:
                params = parent_loop.get("params") if isinstance(parent_loop.get("params"), dict) else {}
                parent_loop["params"] = params

                body = params.get("body_subgraph") if isinstance(params.get("body_subgraph"), dict) else {}
                params["body_subgraph"] = body

                body_nodes = body.get("nodes") if isinstance(body.get("nodes"), list) else []
                body["nodes"] = body_nodes

                body_nodes.append(node)
            else:
                if node.get("type") in {"action", "condition", "loop"}:
                    node["parent_node_id"] = None
                root_nodes.append(node)

        workflow = {
            "workflow_name": self.workflow_name,
            "description": self.description,
            "nodes": root_nodes,
        }
        for node in workflow.get("nodes", []) or []:
            if isinstance(node, Mapping) and node.get("type") == "loop":
                _replace_loop_body_exports(node)
        # Provide implicitly derived edges as read-only context for downstream
        # tools/LLM refinement while keeping the source of truth in param
        # bindings. 遍历主图与 loop 子图的所有节点，确保子图中的引用同样被纳入。 
        workflow["edges"] = infer_edges_from_bindings(iter_workflow_and_loop_body_nodes(workflow))
        return attach_condition_branches(workflow)


__all__ = ["WorkflowBuilder", "attach_condition_branches"]
