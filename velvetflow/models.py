# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Strongly-typed workflow DSL models without external dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Set

from velvetflow.loop_dsl import loop_body_has_action
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

    def __str__(self) -> str:  # pragma: no cover - simple formatting
        """Return a readable summary of validation issues for logging."""

        parts: List[str] = []
        for err in self._errors:
            loc = err.get("loc") or ()
            if isinstance(loc, (list, tuple)):
                location = ".".join(str(part) for part in loc)
            else:
                location = str(loc)
            message = err.get("msg") or "validation failed"
            if location:
                parts.append(f"{location}: {message}")
            else:
                parts.append(str(message))

        if parts:
            return "; ".join(parts)
        return super().__str__()

    __repr__ = __str__


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
        "EMPTY_PARAM_VALUE",
        "EMPTY_PARAMS",
        "SELF_REFERENCE",
        "SYNTAX_ERROR",
        "GRAMMAR_VIOLATION",
    ]
    node_id: Optional[str]
    field: Optional[str]
    message: str


@dataclass
class RepairSuggestion:
    """Structured auto-repair hint produced by the validator.

    ``patch`` describes the replacement value to be applied at ``path`` within
    the workflow AST. ``strategy`` captures which repair technique generated the
    hint so callers can prioritize deterministic templates over probabilistic
    fills.
    """

    strategy: Literal[
        "ast_template",
        "constraint_solver",
        "statistical_fill",
    ]
    description: str
    path: str
    patch: Any
    confidence: float = 0.5
    rationale: Optional[str] = None


# 统一的绑定聚合枚举，供校验和提示使用。
ALLOWED_PARAM_AGGREGATORS: tuple[str, ...] = (
    "identity",
    "count",
    "count_if",
    "join",
    "filter_map",
    "format_join",
    "pipeline",
)


@dataclass
class ParamBinding:
    """Typed representation of the parameter binding DSL."""

    __from__: str | None = None
    __agg__: Literal[
        "identity",
        "count",
        "count_if",
        "join",
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


def _extract_node_id(node: Any) -> Optional[str]:
    if isinstance(node, Node):
        return node.id
    if isinstance(node, Mapping):
        raw_id = node.get("id")
        return raw_id if isinstance(raw_id, str) else None
    return None


def infer_edges_from_bindings(nodes: Iterable[Any]) -> List[Dict[str, Any]]:
    """Infer directed edges from node parameter bindings.

    This helper is used to generate *contextual* topology information for LLMs
    and internal checks without requiring callers to maintain a redundant
    ``edges`` array. It supports both :class:`Node` instances and mapping-like
    node dictionaries.
    """

    node_list = list(nodes)

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

    node_ids = {_extract_node_id(n) for n in node_list}
    node_ids.discard(None)
    seen_pairs: Set[tuple[str | None, str | None]] = set()
    edges: List[Dict[str, Any]] = []

    for node in node_list:
        nid = _extract_node_id(node)
        if not nid:
            continue

        params = _extract_params(node)

        # 参数依赖推导出的隐式连线
        deps = {
            ref
            for ref in _collect_refs(params)
            if ref in node_ids and ref != nid
        }

        # condition.source 支持直接引用节点 id，需同样计入依赖以驱动上游执行。
        if isinstance(node, Node) and node.type == "condition":
            cond_source = node.params.get("source")
            if isinstance(cond_source, str) and cond_source in node_ids and cond_source != nid:
                deps.add(cond_source)
            if isinstance(cond_source, list):
                deps.update({s for s in cond_source if isinstance(s, str) and s in node_ids and s != nid})
        elif isinstance(node, Mapping) and node.get("type") == "condition":
            cond_source = params.get("source")
            if isinstance(cond_source, str) and cond_source in node_ids and cond_source != nid:
                deps.add(cond_source)
            if isinstance(cond_source, list):
                deps.update({s for s in cond_source if isinstance(s, str) and s in node_ids and s != nid})

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
        elif isinstance(node, Node) and getattr(node, "type", None) == "switch":
            branches = [
                (str(case.get("match")), case.get("to_node"))
                for case in getattr(node, "cases", [])
                if isinstance(case, Mapping)
            ]
            if getattr(node, "default_to_node", None) is not None:
                branches.append(("default", getattr(node, "default_to_node", None)))
        elif isinstance(node, Mapping) and node.get("type") == "switch":
            cases = node.get("cases") if isinstance(node.get("cases"), list) else []
            branches = [
                (str(case.get("match")), case.get("to_node"))
                for case in cases
                if isinstance(case, Mapping)
            ]
            if "default_to_node" in node:
                branches.append(("default", node.get("default_to_node")))

        for cond_label, target in branches:
            if not isinstance(target, str):
                continue
            pair = (nid, target)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            edges.append({"from": nid, "to": target, "condition": cond_label})

    return edges


def infer_depends_on_from_edges(
    nodes: Iterable[Any], edges: Iterable[Mapping[str, Any]]
) -> Dict[str, List[str]]:
    """Infer depends_on relationships from edges and existing declarations."""

    node_list = list(nodes)
    node_ids = {_extract_node_id(n) for n in node_list}
    node_ids.discard(None)

    depends_map: Dict[str, Set[str]] = {nid: set() for nid in node_ids}

    for node in node_list:
        nid = _extract_node_id(node)
        if not nid:
            continue
        existing = None
        if isinstance(node, Node):
            existing = node.depends_on
        elif isinstance(node, Mapping):
            existing = node.get("depends_on")

        if isinstance(existing, list):
            for dep in existing:
                if isinstance(dep, str) and dep in node_ids and dep != nid:
                    depends_map.setdefault(nid, set()).add(dep)

    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        frm = edge.get("from") if "from" in edge else edge.get("from_node")
        to = edge.get("to") if "to" in edge else edge.get("to_node")
        if not isinstance(frm, str) or not isinstance(to, str):
            continue
        if frm not in node_ids or to not in node_ids or frm == to:
            continue
        depends_map.setdefault(to, set()).add(frm)

    return {nid: sorted(deps) for nid, deps in depends_map.items()}


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
    depends_on: List[str] = field(default_factory=list)

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
            allowed = {"start", "end", "action", "condition", "switch", "loop", "parallel"}
            if node_type not in allowed:
                errors.append({
                    "loc": ("type",),
                    "msg": f"节点类型必须是 {sorted(allowed)} 之一",
                })

        if errors:
            raise PydanticValidationError(errors)

        params = data.get("params") if isinstance(data.get("params"), Mapping) else {}
        depends_on_raw = data.get("depends_on")
        depends_on: List[str] = []
        if depends_on_raw is None:
            depends_on = []
        elif isinstance(depends_on_raw, list):
            depends_on = [d for d in depends_on_raw if isinstance(d, str)]
        else:
            errors.append({"loc": ("depends_on",), "msg": "depends_on 必须是字符串数组"})

        action_id = data.get("action_id")
        if action_id is not None and not isinstance(action_id, str):
            errors.append({"loc": ("action_id",), "msg": "action_id 必须是字符串"})

        if node_type == "action":
            out_params_schema = data.get("out_params_schema")
            if out_params_schema is not None and not isinstance(out_params_schema, Mapping):
                errors.append(
                    {
                        "loc": ("out_params_schema",),
                        "msg": "out_params_schema 必须是对象",
                    }
                )
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
                out_params_schema=out_params_schema if isinstance(out_params_schema, Mapping) else None,
                params=dict(params),
                depends_on=depends_on,
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
                depends_on=depends_on,
            )

        if node_type == "switch":
            cases = data.get("cases") if isinstance(data.get("cases"), list) else []
            default_to_node = data.get("default_to_node")
            if default_to_node is not None and not isinstance(default_to_node, str):
                errors.append({"loc": ("default_to_node",), "msg": "default_to_node 必须是字符串"})

            validated_cases: List[Dict[str, Any]] = []
            if cases:
                for idx, case in enumerate(cases):
                    if not isinstance(case, Mapping):
                        errors.append({"loc": ("cases", idx), "msg": "case 必须是对象"})
                        continue
                    to_node = case.get("to_node")
                    if to_node is not None and not isinstance(to_node, str):
                        errors.append({"loc": ("cases", idx, "to_node"), "msg": "to_node 必须是字符串"})
                        continue
                    validated_cases.append(dict(case))

            if errors:
                raise PydanticValidationError(errors)

            return SwitchNode(
                id=node_id,
                display_name=data.get("display_name"),
                params=dict(params),
                cases=validated_cases,
                default_to_node=default_to_node if isinstance(default_to_node, str) else None,
                depends_on=depends_on,
            )

        if errors:
            raise PydanticValidationError(errors)

        return cls(
            id=node_id,
            type=node_type,
            action_id=action_id if isinstance(action_id, str) else None,
            display_name=data.get("display_name"),
            params=dict(params),
            depends_on=depends_on,
        )

    def model_dump(self, *, by_alias: bool = False) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "action_id": self.action_id,
            "display_name": self.display_name,
            "params": self.params,
            "depends_on": self.depends_on,
        }


@dataclass
class ActionNode(Node):
    """Executable action node."""

    type: Literal["action"] = "action"
    out_params_schema: Optional[Dict[str, Any]] = None

    def model_dump(self, *, by_alias: bool = False) -> Dict[str, Any]:
        data = super().model_dump(by_alias=by_alias)
        data["out_params_schema"] = self.out_params_schema
        return data


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
class SwitchNode(Node):
    """Multi-branch switch node, similar to Python's match-case."""

    type: Literal["switch"] = "switch"
    cases: List[Dict[str, Any]] = field(default_factory=list)
    default_to_node: Optional[str] = None

    def model_dump(self, *, by_alias: bool = False) -> Dict[str, Any]:
        data = super().model_dump(by_alias=by_alias)
        data.update({"cases": self.cases, "default_to_node": self.default_to_node})
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

    @property
    def edges(self) -> List[Edge]:
        """Always rebuild edges from the latest node bindings."""

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
        if not isinstance(raw_nodes, list):
            errors.append({"loc": ("nodes",), "msg": "nodes 必须是数组"})

        parsed_nodes: List[Node] = []

        if isinstance(raw_nodes, list):
            for idx, node in enumerate(raw_nodes):
                try:
                    parsed_nodes.append(Node.model_validate(node))
                except PydanticValidationError as exc:
                    for err in exc.errors():
                        loc = ("nodes", idx, *err.get("loc", ()))
                        errors.append({"loc": loc, "msg": err.get("msg")})

        if errors:
            raise PydanticValidationError(errors)

        # ensure unique node ids
        seen = set()
        for n in parsed_nodes:
            if n.id in seen:
                raise ValueError(f"重复的节点 id: {n.id}")
            seen.add(n.id)

        workflow = cls(
            workflow_name=str(data.get("workflow_name", "unnamed_workflow")),
            description=str(data.get("description", "")),
            nodes=parsed_nodes,
        )

        return workflow.validate_loop_body_subgraphs()

    def validate_loop_body_subgraphs(self) -> "Workflow":
        """Validate nested loop body subgraphs at build time."""

        errors: List[Dict[str, Any]] = []

        for idx, node in enumerate(self.nodes):
            if node.type != "loop":
                continue

            params = node.params if isinstance(node.params, Mapping) else {}
            body_graph = params.get("body_subgraph") if isinstance(params, Mapping) else None

            if body_graph is None:
                errors.append(
                    {
                        "loc": ("nodes", idx, "params", "body_subgraph"),
                        "msg": "loop 节点必须提供 body_subgraph",
                    }
                )
                continue

            if not isinstance(body_graph, Mapping):
                errors.append(
                    {
                        "loc": ("nodes", idx, "params", "body_subgraph"),
                        "msg": "body_subgraph 必须是对象",
                    }
                )
                continue

            body_nodes = body_graph.get("nodes")
            if not isinstance(body_nodes, list):
                errors.append(
                    {
                        "loc": ("nodes", idx, "params", "body_subgraph", "nodes"),
                        "msg": "body_subgraph.nodes 必须是数组",
                    }
                )
                continue

            if not body_nodes:
                errors.append(
                    {
                        "loc": ("nodes", idx, "params", "body_subgraph", "nodes"),
                        "msg": "loop 节点的 body_subgraph.nodes 不能为空",
                    }
                )
                continue

            if not loop_body_has_action(body_graph):
                errors.append(
                    {
                        "loc": ("nodes", idx, "params", "body_subgraph", "nodes"),
                        "msg": "loop 子图必须包含至少一个 action 节点以承载执行逻辑",
                    }
                )
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
                nested_errors: List[Dict[str, Any]] = []
                for err in exc.errors():
                    loc = ("body_subgraph", *err.get("loc", ()))
                    nested_errors.append({"loc": loc, "msg": err.get("msg")})

                raise PydanticValidationError(nested_errors) from exc
            except Exception as exc:  # noqa: BLE001
                raise ValueError(
                    f"loop 节点 '{node.id}' 的 body_subgraph 校验失败: {exc}"
                ) from exc

        if errors:
            raise PydanticValidationError(errors)

        return self

    def model_dump(self, *, by_alias: bool = False) -> Dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "description": self.description,
            "nodes": [n.model_dump(by_alias=by_alias) for n in self.nodes],
        }


__all__ = [
    "Node",
    "ActionNode",
    "ConditionNode",
    "SwitchNode",
    "Edge",
    "ParamBinding",
    "PydanticValidationError",
    "ValidationError",
    "infer_edges_from_bindings",
    "infer_depends_on_from_edges",
    "Workflow",
]
