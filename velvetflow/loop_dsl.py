"""Shared helpers for loop DSL schema and validation."""
from typing import Any, Dict, Iterable, Mapping, Optional


def build_loop_output_schema(loop_params: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Construct a virtual output schema for a loop node based on exports.

    The schema is an "object" with optional `items` and `aggregates` properties,
    derived from the loop's exports specification.
    """

    if not isinstance(loop_params, Mapping):
        return None

    exports = loop_params.get("exports")
    if not isinstance(exports, Mapping):
        return None

    properties: Dict[str, Any] = {
        "status": {"type": "string"},
        "loop_kind": {"type": "string"},
        "iterations": {"type": "array", "items": {"type": "object"}},
        "accumulator": {"type": "object"},
    }

    items_spec = exports.get("items")
    if isinstance(items_spec, Mapping):
        fields = items_spec.get("fields") if isinstance(items_spec.get("fields"), list) else []
        item_props = {f: {} for f in fields if isinstance(f, str)}
        properties["items"] = {
            "type": "array",
            "items": {"type": "object", "properties": item_props},
        }

    aggregates_spec = exports.get("aggregates")
    if isinstance(aggregates_spec, list):
        agg_props = {}
        for agg in aggregates_spec:
            if isinstance(agg, Mapping):
                name = agg.get("name")
                if isinstance(name, str):
                    agg_props[name] = {}
        properties["aggregates"] = {"type": "object", "properties": agg_props}

    return {"type": "object", "properties": properties} if properties else None


def index_loop_body_nodes(workflow: Mapping[str, Any]) -> Dict[str, str]:
    """Build a mapping of loop body node id -> parent loop id."""

    body_to_loop: Dict[str, str] = {}
    for node in workflow.get("nodes", []):
        if not isinstance(node, Mapping):
            continue
        if node.get("type") != "loop":
            continue
        loop_id = node.get("id")
        params = node.get("params") or {}
        body = params.get("body_subgraph")
        if isinstance(body, Mapping):
            body_nodes = body.get("nodes") or []
        elif isinstance(body, list):
            body_nodes = body
        else:
            body_nodes = []

        for child in body_nodes or []:
            if isinstance(child, Mapping) and isinstance(child.get("id"), str):
                body_to_loop[child["id"]] = loop_id
    return body_to_loop


def iter_workflow_and_loop_body_nodes(workflow: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    """Yield all nodes in a workflow, including nested loop body nodes."""

    def _iter_nodes(nodes: Iterable[Any]) -> Iterable[Mapping[str, Any]]:
        for node in nodes or []:
            if not isinstance(node, Mapping):
                continue
            yield node
            if node.get("type") == "loop":
                params = node.get("params") or {}
                body = params.get("body_subgraph")
                if isinstance(body, Mapping):
                    body_nodes = body.get("nodes") or []
                elif isinstance(body, list):
                    # Some callers supply the loop body directly as a list of nodes
                    # rather than wrapping it in a mapping. Accept both forms to avoid
                    # attribute errors during traversal.
                    body_nodes = body
                else:
                    body_nodes = []

                if isinstance(body_nodes, list):
                    yield from _iter_nodes(body_nodes)

    nodes = workflow.get("nodes") if isinstance(workflow, Mapping) else []
    yield from _iter_nodes(nodes or [])


__all__ = [
    "build_loop_output_schema",
    "index_loop_body_nodes",
    "iter_workflow_and_loop_body_nodes",
]
