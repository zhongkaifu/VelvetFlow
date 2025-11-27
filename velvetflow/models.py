"""Strongly-typed workflow DSL models without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional


class PydanticValidationError(Exception):
    """Lightweight stand-in for :class:`pydantic.ValidationError`.

    Only the ``errors()`` method is implemented because the rest of the codebase
    relies on that shape for error reporting. Each error is expected to be a
    mapping containing at least ``loc`` and ``msg`` keys.
    """

    def __init__(self, errors: List[Dict[str, Any]]):
        super().__init__("validation failed")
        self._errors = errors

    def errors(self) -> List[Dict[str, Any]]:  # pragma: no cover - trivial
        return self._errors


# Backwards-compatible stubs so imports in the rest of the code keep working.
def ConfigDict(**_: Any) -> Dict[str, Any]:  # pragma: no cover - compatibility
    return {}


def Field(default: Any = None, *, alias: str | None = None, default_factory=None, **_: Any) -> Dict[str, Any]:
    return {"default": default, "alias": alias, "default_factory": default_factory}


def field_validator(*_: str, **__: Any):  # pragma: no cover - compatibility
    def decorator(func):
        return func

    return decorator


def model_validator(*_: str, **__: Any):  # pragma: no cover - compatibility
    def decorator(func):
        return func

    return decorator


@dataclass
class ValidationError:
    code: Literal[
        "MISSING_REQUIRED_PARAM",
        "UNKNOWN_ACTION_ID",
        "UNKNOWN_PARAM",
        "DISCONNECTED_GRAPH",
        "INVALID_EDGE",
        "SCHEMA_MISMATCH",
        "INVALID_SCHEMA",
        "INVALID_LOOP_BODY",
        "STATIC_RULES_SUMMARY",
    ]
    node_id: Optional[str]
    field: Optional[str]
    message: str


@dataclass
class ParamBinding:
    """Typed representation of the parameter binding DSL."""

    __from__: str | None = None
    __agg__: Literal[
        "identity",
        "count",
        "count_if",
        "filter_map",
        "format_join",
        "pipeline",
    ] | None = None
    field: str | None = None
    op: str | None = None
    value: Any = None
    filter_field: str | None = None
    filter_op: str | None = None
    filter_value: Any = None
    map_field: str | None = None
    format: str | None = None
    sep: str | None = None
    steps: List[Dict[str, Any]] | None = None


@dataclass
class Node:
    """A workflow node definition."""

    id: str
    type: str
    action_id: Optional[str] = None
    display_name: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def model_validate(cls, data: Any) -> "Node":
        if isinstance(data, cls):
            return data

        if not isinstance(data, Mapping):
            raise PydanticValidationError([
                {"loc": ("node",), "msg": "节点必须是对象"},
            ])

        errors: List[Dict[str, Any]] = []
        node_id = data.get("id")
        node_type = data.get("type")

        if not isinstance(node_id, str):
            errors.append({"loc": ("id",), "msg": "节点 id 必须是字符串"})

        if not isinstance(node_type, str):
            errors.append({"loc": ("type",), "msg": "节点类型必须是字符串"})
        else:
            allowed = {"start", "end", "action", "condition", "loop", "parallel"}
            if node_type not in allowed:
                errors.append({
                    "loc": ("type",),
                    "msg": f"节点类型必须是 {sorted(allowed)} 之一",
                })

        if errors:
            raise PydanticValidationError(errors)

        params = data.get("params") if isinstance(data.get("params"), Mapping) else {}

        return cls(
            id=node_id,
            type=node_type,
            action_id=data.get("action_id"),
            display_name=data.get("display_name"),
            params=dict(params),
        )

    def model_dump(self, *, by_alias: bool = False) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "action_id": self.action_id,
            "display_name": self.display_name,
            "params": self.params,
        }


@dataclass
class Edge:
    """Directed edge between workflow nodes."""

    from_node: str
    to_node: str
    condition: Optional[str] = None
    note: Optional[str] = None

    @classmethod
    def model_validate(cls, data: Any) -> "Edge":
        if isinstance(data, cls):
            return data

        if not isinstance(data, Mapping):
            raise PydanticValidationError([
                {"loc": ("edge",), "msg": "边必须是对象"},
            ])

        errors: List[Dict[str, Any]] = []
        from_node = data.get("from") if "from" in data else data.get("from_node")
        to_node = data.get("to") if "to" in data else data.get("to_node")

        if not isinstance(from_node, str):
            errors.append({"loc": ("from",), "msg": "边的 from 不能为空"})
        if not isinstance(to_node, str):
            errors.append({"loc": ("to",), "msg": "边的 to 不能为空"})

        if errors:
            raise PydanticValidationError(errors)

        condition = data.get("condition")
        if isinstance(condition, bool):
            condition = "true" if condition else "false"

        return cls(
            from_node=from_node,
            to_node=to_node,
            condition=condition,
            note=data.get("note"),
        )

    def model_dump(self, *, by_alias: bool = False) -> Dict[str, Any]:
        if by_alias:
            return {"from": self.from_node, "to": self.to_node, "condition": self.condition, "note": self.note}
        return {
            "from_node": self.from_node,
            "to_node": self.to_node,
            "condition": self.condition,
            "note": self.note,
        }


@dataclass
class Workflow:
    """Complete workflow definition used across planner and executor."""

    nodes: List[Node]
    edges: List[Edge]
    workflow_name: str = "unnamed_workflow"
    description: str = ""

    @classmethod
    def model_validate(cls, data: Any) -> "Workflow":
        if isinstance(data, cls):
            return data

        if not isinstance(data, Mapping):
            raise PydanticValidationError([
                {"loc": ("workflow",), "msg": "workflow 必须是对象"},
            ])

        errors: List[Dict[str, Any]] = []
        raw_nodes = data.get("nodes")
        raw_edges = data.get("edges")

        if not isinstance(raw_nodes, list):
            errors.append({"loc": ("nodes",), "msg": "nodes 必须是数组"})
        if not isinstance(raw_edges, list):
            errors.append({"loc": ("edges",), "msg": "edges 必须是数组"})

        parsed_nodes: List[Node] = []
        parsed_edges: List[Edge] = []

        if isinstance(raw_nodes, list):
            for idx, node in enumerate(raw_nodes):
                try:
                    parsed_nodes.append(Node.model_validate(node))
                except PydanticValidationError as exc:
                    for err in exc.errors():
                        loc = ("nodes", idx, *err.get("loc", ()))
                        errors.append({"loc": loc, "msg": err.get("msg")})

        if isinstance(raw_edges, list):
            for idx, edge in enumerate(raw_edges):
                try:
                    parsed_edges.append(Edge.model_validate(edge))
                except PydanticValidationError as exc:
                    for err in exc.errors():
                        loc = ("edges", idx, *err.get("loc", ()))
                        errors.append({"loc": loc, "msg": err.get("msg")})

        if errors:
            raise PydanticValidationError(errors)

        # ensure unique node ids
        seen = set()
        for n in parsed_nodes:
            if n.id in seen:
                raise ValueError(f"重复的节点 id: {n.id}")
            seen.add(n.id)

        # ensure edges refer to existing nodes
        node_ids = {n.id for n in parsed_nodes}
        for e in parsed_edges:
            if e.from_node not in node_ids or e.to_node not in node_ids:
                raise ValueError(
                    f"边 {e.model_dump(by_alias=True)} 引用了不存在的节点，已知节点: {sorted(node_ids)}"
                )
            if e.from_node == e.to_node:
                raise ValueError("边的 from/to 不能指向同一个节点")

        workflow = cls(
            workflow_name=str(data.get("workflow_name", "unnamed_workflow")),
            description=str(data.get("description", "")),
            nodes=parsed_nodes,
            edges=parsed_edges,
        )

        return workflow.validate_loop_body_subgraphs()

    def validate_loop_body_subgraphs(self) -> "Workflow":
        """Validate nested loop body subgraphs at build time."""

        for node in self.nodes:
            if node.type != "loop":
                continue

            params = node.params if isinstance(node.params, Mapping) else {}
            body_graph = params.get("body_subgraph") if isinstance(params, Mapping) else None
            if not isinstance(body_graph, Mapping):
                continue

            body_nodes = body_graph.get("nodes")
            body_edges = body_graph.get("edges")
            if not isinstance(body_nodes, list) or not isinstance(body_edges, list):
                continue

            try:
                Workflow.model_validate(
                    {
                        "workflow_name": f"{node.id}_loop_body",
                        "nodes": body_nodes,
                        "edges": body_edges,
                    }
                )
            except PydanticValidationError as exc:
                # 标注 body_subgraph 中的具体字段，方便上层修复逻辑消费
                errors: List[Dict[str, Any]] = []
                for err in exc.errors():
                    loc = ("body_subgraph", *err.get("loc", ()))
                    errors.append({"loc": loc, "msg": err.get("msg")})

                raise PydanticValidationError(errors) from exc
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"loop 节点 '{node.id}' 的 body_subgraph 校验失败: {exc}"
                ) from exc

        return self

    def model_dump(self, *, by_alias: bool = False) -> Dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "description": self.description,
            "nodes": [n.model_dump(by_alias=by_alias) for n in self.nodes],
            "edges": [e.model_dump(by_alias=by_alias) for e in self.edges],
        }


__all__ = [
    "Edge",
    "Node",
    "ParamBinding",
    "PydanticValidationError",
    "ValidationError",
    "Workflow",
]
