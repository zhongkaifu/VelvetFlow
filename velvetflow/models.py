"""Strongly-typed workflow DSL models without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Set

from velvetflow.reference_utils import normalize_reference_path


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


def infer_edges_from_bindings(nodes: Iterable[Any]) -> List[Dict[str, Any]]:
    """Infer directed edges from node parameter bindings.

    This helper is used to generate *contextual* topology information for LLMs
    and internal checks without requiring callers to maintain a redundant
    ``edges`` array. It supports both :class:`Node` instances and mapping-like
    node dictionaries.
    """

    def _extract_node_id(node: Any) -> Optional[str]:
        if isinstance(node, Node):
            return node.id
        if isinstance(node, Mapping):
            raw_id = node.get("id")
            return raw_id if isinstance(raw_id, str) else None
        return None

    def _extract_params(node: Any) -> Mapping[str, Any]:
        if isinstance(node, Node):
            return node.params
        if isinstance(node, Mapping):
            params = node.get("params")
            if isinstance(params, Mapping):
                return params
        return {}

    def _collect_refs(value: Any) -> Iterable[str]:
        if isinstance(value, Mapping):
            if "__from__" in value:
                ref = normalize_reference_path(value.get("__from__"))
                if isinstance(ref, str) and ref.startswith("result_of."):
                    parts = ref.split(".")
                    if len(parts) >= 2 and parts[1]:
                        yield parts[1]
            for v in value.values():
                yield from _collect_refs(v)
        elif isinstance(value, list):
            for item in value:
                yield from _collect_refs(item)
        elif isinstance(value, str):
            normalized = normalize_reference_path(value)
            if normalized.startswith("result_of."):
                parts = normalized.split(".")
                if len(parts) >= 2 and parts[1]:
                    yield parts[1]
            # Also detect embedded result_of references inside templated or
            # free-form strings, e.g. "Hello {{result_of.node.foo}} world".
            for match in re.findall(r"result_of\.([A-Za-z0-9_-]+)", value):
                yield match

    node_ids = {_extract_node_id(n) for n in nodes}
    node_ids.discard(None)
    seen_pairs: Set[tuple[str | None, str | None]] = set()
    edges: List[Dict[str, Any]] = []

    for node in nodes:
        nid = _extract_node_id(node)
        if not nid:
            continue

        # 参数依赖推导出的隐式连线
        deps = {
            ref
            for ref in _collect_refs(_extract_params(node))
            if ref in node_ids and ref != nid
        }
        for dep in sorted(deps):
            pair = (dep, nid)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append({"from": dep, "to": nid, "condition": None})

        # condition 节点的 true/false 分支也被视为连线，避免被误判为“无入度”起点。
        branches = []
        if isinstance(node, Node) and node.type == "condition":
            branches = [("true", node.true_to_node), ("false", node.false_to_node)]
        elif isinstance(node, Mapping) and node.get("type") == "condition":
            branches = [
                ("true", node.get("true_to_node")),
                ("false", node.get("false_to_node")),
            ]

        for cond_label, target in branches:
            if not isinstance(target, str):
                continue
            pair = (nid, target)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append({"from": nid, "to": target, "condition": cond_label})

    return edges


@dataclass
class Node:
    """Base workflow node definition.

    Concrete node types derive from this class to attach their own fields.
    """

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

        action_id = data.get("action_id")
        if action_id is not None and not isinstance(action_id, str):
            errors.append({"loc": ("action_id",), "msg": "action_id 必须是字符串"})

        if node_type == "action":
            for branch_field in ("true_to_node", "false_to_node"):
                if branch_field in data and data.get(branch_field) is not None:
                    errors.append(
                        {
                            "loc": (branch_field,),
                            "msg": "action 节点不支持 true/false 分支字段",
                        }
                    )
            if errors:
                raise PydanticValidationError(errors)

            return ActionNode(
                id=node_id,
                action_id=action_id if isinstance(action_id, str) else None,
                display_name=data.get("display_name"),
                params=dict(params),
            )

        if node_type == "condition":
            true_to_node = data.get("true_to_node")
            false_to_node = data.get("false_to_node")
            if true_to_node is not None and not isinstance(true_to_node, str):
                errors.append({"loc": ("true_to_node",), "msg": "true_to_node 必须是字符串"})
            if false_to_node is not None and not isinstance(false_to_node, str):
                errors.append({"loc": ("false_to_node",), "msg": "false_to_node 必须是字符串"})

            if errors:
                raise PydanticValidationError(errors)

            return ConditionNode(
                id=node_id,
                display_name=data.get("display_name"),
                params=dict(params),
                true_to_node=true_to_node if isinstance(true_to_node, str) else None,
                false_to_node=false_to_node if isinstance(false_to_node, str) else None,
            )

        if errors:
            raise PydanticValidationError(errors)

        return cls(
            id=node_id,
            type=node_type,
            action_id=action_id if isinstance(action_id, str) else None,
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
class ActionNode(Node):
    """Executable action node."""

    type: Literal["action"] = "action"


@dataclass
class ConditionNode(Node):
    """Branching condition node."""

    type: Literal["condition"] = "condition"
    true_to_node: Optional[str] = None
    false_to_node: Optional[str] = None

    def model_dump(self, *, by_alias: bool = False) -> Dict[str, Any]:
        data = super().model_dump(by_alias=by_alias)
        data.update({
            "true_to_node": self.true_to_node,
            "false_to_node": self.false_to_node,
        })
        return data


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
    workflow_name: str = "unnamed_workflow"
    description: str = ""
    declared_edges: List[Edge] = field(default_factory=list)

    @property
    def edges(self) -> List[Edge]:
        """Always rebuild edges from the latest node bindings.

        Implicit wiring must faithfully reflect the *current* node set. The
        derived edges are therefore regenerated on every access instead of
        cached, ensuring callers never read stale topology after node edits.
        """

        inferred = self._infer_edges()
        declared = self.declared_edges or []
        if not declared:
            return inferred

        existing = {(e.from_node, e.to_node, e.condition) for e in inferred}
        combined = list(inferred)
        for edge in declared:
            key = (edge.from_node, edge.to_node, edge.condition)
            if key not in existing:
                combined.append(edge)
        return combined

    def _infer_edges(self) -> List[Edge]:
        """Build edges from parameter bindings declared on nodes.

        The workflow DSL expresses dependencies via ``result_of.<node>`` bindings
        rather than explicit edge arrays. This helper walks each node's params to
        discover inbound dependencies and materializes them as edge dictionaries
        for internal use (topological sort, reachability checks, etc.). The
        derived edges are not part of the external contract and are regenerated
        on demand to stay in sync with the latest node bindings.
        """

        inferred = infer_edges_from_bindings(self.nodes)
        return [Edge(from_node=e["from"], to_node=e["to"], condition=e.get("condition")) for e in inferred]

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
        if raw_edges is not None and not isinstance(raw_edges, list):
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

        if parsed_edges:
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
            declared_edges=parsed_edges,
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
            if not isinstance(body_nodes, list):
                continue

            try:
                Workflow.model_validate(
                    {
                        "workflow_name": f"{node.id}_loop_body",
                        "nodes": body_nodes,
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
        }


__all__ = [
    "Node",
    "Edge",
    "ParamBinding",
    "PydanticValidationError",
    "ValidationError",
    "infer_edges_from_bindings",
    "Workflow",
]
