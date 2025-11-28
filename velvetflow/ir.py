"""Intermediate Representation (IR) and optimizer for workflows.

This module normalizes different frontends (JSON/YAML/DSL/GUI) into a
workflow IR, applies common compiler optimizations, and emits target
artifacts for multiple orchestrators. The goal is to reuse the same
semantic guarantees across build and execution while improving runtime
throughput.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from velvetflow.models import Workflow, infer_edges_from_bindings

try:  # Optional dependency: PyYAML is not required at runtime.
    import yaml
except Exception:  # pragma: no cover - best effort import
    yaml = None


@dataclass
class IRNode:
    """IR node with normalized dependencies and optimization annotations."""

    id: str
    kind: str
    action_id: Optional[str]
    params: Mapping[str, Any]
    inputs: Set[str] = field(default_factory=set)
    annotations: Dict[str, Any] = field(default_factory=dict)

    def is_pure(self) -> bool:
        return bool(self.annotations.get("pure"))

    def is_static(self) -> bool:
        return bool(self.annotations.get("static"))


@dataclass
class WorkflowIR:
    nodes: Dict[str, IRNode]
    edges: List[Tuple[str, str]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def prune_unused_nodes(self) -> None:
        """Remove static, side-effect-free nodes that are not consumed."""

        consumers: Dict[str, Set[str]] = {nid: set() for nid in self.nodes}
        for src, dst in self.edges:
            if src in consumers:
                consumers[src].add(dst)

        removable = [
            nid
            for nid, node in self.nodes.items()
            if node.is_pure() and node.is_static() and not consumers.get(nid)
        ]
        if not removable:
            return

        self.nodes = {nid: node for nid, node in self.nodes.items() if nid not in removable}
        self.edges = [(src, dst) for src, dst in self.edges if src not in removable and dst not in removable]
        if removable:
            self.metadata.setdefault("pruned_nodes", []).extend(removable)

    def inline_constant_bindings(self) -> None:
        """Replace bindings that target folded constants with literal values."""

        constant_values = {
            nid: node.annotations.get("folded_output")
            for nid, node in self.nodes.items()
            if node.annotations.get("folded_output") is not None
        }
        if not constant_values:
            return

        for node in self.nodes.values():
            if node.id in constant_values:
                continue
            node.annotations["inlined_params"] = _inline_bindings(node.params, constant_values)

    def eliminate_common_subexpressions(self) -> None:
        """Merge pure nodes that compute identical expressions."""

        signature_to_node: Dict[Tuple[str, Any, Tuple[str, ...]], str] = {}
        redirected: Dict[str, str] = {}

        for node in list(self.nodes.values()):
            if not node.is_pure():
                continue
            signature = (
                node.kind,
                json.dumps(node.annotations.get("inlined_params", node.params), sort_keys=True, ensure_ascii=False),
                tuple(sorted(node.inputs)),
            )
            if signature in signature_to_node:
                redirected[node.id] = signature_to_node[signature]
                continue
            signature_to_node[signature] = node.id

        if not redirected:
            return

        new_edges: List[Tuple[str, str]] = []
        for src, dst in self.edges:
            src = redirected.get(src, src)
            dst = redirected.get(dst, dst)
            if src == dst:
                continue
            new_edges.append((src, dst))
        self.edges = list(dict.fromkeys(new_edges))

        self.nodes = {nid: node for nid, node in self.nodes.items() if nid not in redirected}
        self.metadata.setdefault("cse_redirects", {}).update(redirected)

    def plan_parallel_layers(self) -> List[List[str]]:
        """Return execution layers that can run in parallel given dependencies."""

        indegree: Dict[str, int] = {nid: 0 for nid in self.nodes}
        adjacency: Dict[str, Set[str]] = {nid: set() for nid in self.nodes}
        for src, dst in self.edges:
            if src in adjacency and dst in indegree:
                adjacency[src].add(dst)
                indegree[dst] += 1

        queue = [nid for nid, deg in indegree.items() if deg == 0]
        layers: List[List[str]] = []
        while queue:
            current_layer = list(queue)
            layers.append(current_layer)
            next_queue: List[str] = []
            for nid in current_layer:
                for dst in adjacency.get(nid, ()):  # pragma: no branch - small loop
                    indegree[dst] -= 1
                    if indegree[dst] == 0:
                        next_queue.append(dst)
            queue = next_queue
        return layers


# Builder utilities ---------------------------------------------------------

def _is_static_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "__from__" in value:
            return False
        return all(_is_static_value(v) for v in value.values())
    if isinstance(value, list):
        return all(_is_static_value(v) for v in value)
    return True


def _collect_inputs(params: Any) -> Set[str]:
    inputs: Set[str] = set()
    if isinstance(params, Mapping):
        if "__from__" in params and isinstance(params.get("__from__"), str):
            inputs.add(params["__from__"])
        for v in params.values():
            inputs.update(_collect_inputs(v))
    elif isinstance(params, list):
        for item in params:
            inputs.update(_collect_inputs(item))
    return inputs


def _inline_bindings(obj: Any, const_values: Mapping[str, Any]) -> Any:
    if isinstance(obj, Mapping):
        if set(obj.keys()) == {"__from__"}:
            ref = obj.get("__from__")
            if isinstance(ref, str):
                node_id = ref.split(".")[1] if ref.startswith("result_of.") and "." in ref else None
                if node_id and node_id in const_values:
                    return const_values[node_id]
        return {k: _inline_bindings(v, const_values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_inline_bindings(v, const_values) for v in obj]
    return obj


def _decode_frontend_source(source: Any) -> Mapping[str, Any]:
    if isinstance(source, Workflow):
        return source.model_dump(by_alias=True)
    if isinstance(source, Mapping):
        return dict(source)
    if isinstance(source, str):
        text = source.strip()
        if text.startswith("{"):
            return json.loads(text)
        if yaml is not None:
            try:
                loaded = yaml.safe_load(text)
                if isinstance(loaded, Mapping):
                    return loaded
            except Exception:
                pass
    raise ValueError("无法从给定输入解析 workflow 描述。请提供 JSON/YAML/Workflow 对象。")


def build_ir_from_workflow(source: Any, action_registry: Sequence[Mapping[str, Any]] | None = None) -> WorkflowIR:
    """Normalize any supported frontend into a WorkflowIR instance."""

    workflow_dict = _decode_frontend_source(source)
    nodes_raw = workflow_dict.get("nodes") if isinstance(workflow_dict, Mapping) else None
    if not isinstance(nodes_raw, list):
        raise ValueError("workflow 缺少 nodes 列表，无法生成 IR")

    edges_raw = workflow_dict.get("edges") if isinstance(workflow_dict, Mapping) else None
    if not isinstance(edges_raw, list):
        edges_raw = infer_edges_from_bindings(nodes_raw)

    actions_by_id = {str(a.get("action_id")): dict(a) for a in action_registry or []}

    nodes: Dict[str, IRNode] = {}
    for raw in nodes_raw:
        if not isinstance(raw, Mapping):
            continue
        nid = str(raw.get("id"))
        params = raw.get("params") if isinstance(raw.get("params"), Mapping) else raw.get("params", {})
        inputs = _collect_inputs(params)
        action_id = raw.get("action_id") if isinstance(raw.get("action_id"), str) else None
        action_def = actions_by_id.get(action_id or "")
        annotations: Dict[str, Any] = {}
        annotations["static"] = _is_static_value(params)
        if action_def:
            annotations["pure"] = bool(action_def.get("pure") or action_def.get("idempotent"))
        if raw.get("type") in {"start", "end"}:
            annotations["pure"] = True
            annotations["static"] = True
        if annotations.get("static") and annotations.get("pure"):
            # Treat a single-value param as a literal we can fold.
            if isinstance(params, Mapping) and len(params) == 1:
                ((_, only_value),) = params.items()
                annotations["folded_output"] = only_value
        nodes[nid] = IRNode(
            id=nid,
            kind=str(raw.get("type")),
            action_id=action_id,
            params=params if isinstance(params, Mapping) else {},
            inputs=inputs,
            annotations=annotations,
        )

    edges: List[Tuple[str, str]] = []
    for e in edges_raw:
        if not isinstance(e, Mapping):
            continue
        frm = e.get("from") or e.get("from_node")
        to = e.get("to") or e.get("to_node")
        if isinstance(frm, str) and isinstance(to, str):
            edges.append((frm, to))

    ir = WorkflowIR(nodes=nodes, edges=edges, metadata={"source_workflow": workflow_dict})
    return ir


# Optimizer -----------------------------------------------------------------

def optimize_ir(ir: WorkflowIR) -> WorkflowIR:
    """Apply safe, rule-based IR optimizations."""

    ir.inline_constant_bindings()
    ir.eliminate_common_subexpressions()
    ir.prune_unused_nodes()

    layers = ir.plan_parallel_layers()
    ir.metadata["parallel_layers"] = layers
    return ir


# Target generation ---------------------------------------------------------

def _render_task_template(node: IRNode) -> Dict[str, Any]:
    params = node.annotations.get("inlined_params", node.params)
    return {
        "name": node.id,
        "action": node.action_id or node.kind,
        "params": params,
    }


def generate_target_specs(ir: WorkflowIR) -> Dict[str, Any]:
    """Render orchestration-specific manifests from the optimized IR."""

    tasks = {nid: _render_task_template(node) for nid, node in ir.nodes.items()}

    dag_edges = [
        {"from": src, "to": dst}
        for src, dst in ir.edges
        if src in ir.nodes and dst in ir.nodes
    ]

    argo = {
        "entrypoint": next(iter(ir.nodes), None),
        "templates": list(tasks.values()),
        "dependencies": dag_edges,
    }
    airflow = {
        "dag_id": ir.metadata.get("source_workflow", {}).get("workflow_name", "velvetflow_dag"),
        "tasks": tasks,
        "dependencies": dag_edges,
    }
    k8s = {
        "kind": "Workflow",
        "metadata": {"name": airflow["dag_id"]},
        "spec": {"tasks": tasks, "dependencies": dag_edges},
    }
    native = {
        "parallel_layers": ir.metadata.get("parallel_layers"),
        "pruned_nodes": ir.metadata.get("pruned_nodes", []),
        "cse_redirects": ir.metadata.get("cse_redirects", {}),
    }

    return {"argo": argo, "airflow": airflow, "k8s": k8s, "native": native}


# Workflow materialization --------------------------------------------------

def materialize_workflow_from_ir(ir: WorkflowIR) -> Mapping[str, Any]:
    """Produce a workflow dict that reflects IR optimizations."""

    source = ir.metadata.get("source_workflow", {}) if isinstance(ir.metadata, Mapping) else {}
    nodes_raw = source.get("nodes") if isinstance(source, Mapping) else []
    nodes_by_id: Dict[str, Mapping[str, Any]] = {n.get("id"): n for n in nodes_raw if isinstance(n, Mapping)}

    optimized_nodes: List[Mapping[str, Any]] = []
    for nid, node in ir.nodes.items():
        raw = dict(nodes_by_id.get(nid, {}))
        if node.annotations.get("inlined_params") is not None:
            raw["params"] = node.annotations["inlined_params"]
        optimized_nodes.append(raw)

    optimized_edges = [
        {"from": src, "to": dst}
        for src, dst in ir.edges
        if src in ir.nodes and dst in ir.nodes
    ]

    materialized = dict(source)
    materialized["nodes"] = optimized_nodes
    materialized["edges"] = optimized_edges
    return materialized


__all__ = [
    "IRNode",
    "WorkflowIR",
    "build_ir_from_workflow",
    "optimize_ir",
    "generate_target_specs",
    "materialize_workflow_from_ir",
]
