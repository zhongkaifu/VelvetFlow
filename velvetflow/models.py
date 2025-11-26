"""Strongly-typed workflow DSL models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, TypedDict

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError as PydanticValidationError,
    field_validator,
    model_validator,
)


@dataclass
class ValidationError:
    code: Literal[
        "MISSING_REQUIRED_PARAM",
        "UNKNOWN_ACTION_ID",
        "DISCONNECTED_GRAPH",
        "INVALID_EDGE",
        "SCHEMA_MISMATCH",
        "INVALID_SCHEMA",
    ]
    node_id: Optional[str]
    field: Optional[str]
    message: str


class ParamBinding(TypedDict, total=False):
    """Typed representation of the parameter binding DSL."""

    __from__: str
    __agg__: Literal[
        "identity",
        "count",
        "count_if",
        "filter_map",
        "format_join",
        "pipeline",
    ]
    field: str
    op: str
    value: Any
    filter_field: str
    filter_op: str
    filter_value: Any
    map_field: str
    format: str
    sep: str
    steps: List[Dict[str, Any]]


class Node(BaseModel):
    """A workflow node definition."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    id: str
    type: str
    action_id: Optional[str] = None
    display_name: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("type")
    @classmethod
    def validate_type(cls, value: str) -> str:
        allowed = {"start", "end", "action", "condition", "loop", "parallel"}
        if value not in allowed:
            raise ValueError(f"节点类型必须是 {sorted(allowed)} 之一")
        return value


class Edge(BaseModel):
    """Directed edge between workflow nodes."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    from_node: str = Field(..., alias="from")
    to_node: str = Field(..., alias="to")
    condition: Optional[str] = Field(default=None)
    note: Optional[str] = Field(default=None, description="Optional human-readable note")

    @field_validator("condition", mode="before")
    @classmethod
    def coerce_bool_condition(cls, value: Any) -> Any:
        """Normalize boolean conditions into string form for validation.

        LLM 或手写 DSL 可能会将条件边标记为布尔值 true/false。由于 Edge
        的类型定义要求字符串/None，这里在校验前将布尔值转换为 "true"/"false"，
        以避免不必要的类型错误。
        """

        if isinstance(value, bool):
            return "true" if value else "false"
        return value


class Workflow(BaseModel):
    """Complete workflow definition used across planner and executor."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    workflow_name: str = "unnamed_workflow"
    description: str = ""
    nodes: List[Node]
    edges: List[Edge]

    @field_validator("nodes")
    @classmethod
    def ensure_unique_node_ids(cls, nodes: List[Node]) -> List[Node]:
        seen = set()
        for n in nodes:
            if n.id in seen:
                raise ValueError(f"重复的节点 id: {n.id}")
            seen.add(n.id)
        return nodes

    @model_validator(mode="after")
    def ensure_edges_refer_existing_nodes(self):
        node_ids = {n.id for n in self.nodes}
        for e in self.edges:
            if e.from_node not in node_ids or e.to_node not in node_ids:
                raise ValueError(
                    f"边 {e.model_dump(by_alias=True)} 引用了不存在的节点，已知节点: {sorted(node_ids)}"
                )
            if e.from_node == e.to_node:
                raise ValueError("边的 from/to 不能指向同一个节点")
        return self

    @model_validator(mode="after")
    def validate_loop_body_subgraphs(self):
        """Validate nested loop body subgraphs at build time.

        This catches schema issues (e.g., unexpected fields on edges/nodes) before
        runtime execution by re-validating each loop body as an independent
        workflow.
        """

        for node in self.nodes:
            if node.type != "loop":
                continue

            body_graph = node.params.get("body_subgraph") if isinstance(node.params, dict) else None
            if not isinstance(body_graph, dict):
                continue

            # Minimal shape check before attempting validation
            body_nodes = body_graph.get("nodes")
            body_edges = body_graph.get("edges")
            if not isinstance(body_nodes, list) or not isinstance(body_edges, list):
                continue

            # If the subgraph is obviously malformed (unknown node types or
            # edges/entry/exit pointing to missing nodes), skip deep validation
            # so that upstream precheck/static rules can surface actionable
            # errors to the LLM instead of aborting workflow construction.
            body_ids = {bn.get("id") for bn in body_nodes if isinstance(bn.get("id"), str)}
            allowed_body_types = {"action", "condition", "loop", "parallel", "start", "end"}

            def _has_structural_issues() -> bool:
                for bn in body_nodes:
                    ntype = bn.get("type")
                    if ntype not in allowed_body_types:
                        return True

                entry = body_graph.get("entry")
                if isinstance(entry, str) and entry not in body_ids:
                    return True

                exit_node = body_graph.get("exit")
                if isinstance(exit_node, str) and exit_node not in body_ids:
                    return True

                for edge in body_edges:
                    frm = edge.get("from") if isinstance(edge, dict) else None
                    to = edge.get("to") if isinstance(edge, dict) else None
                    if isinstance(frm, str) and frm not in body_ids:
                        return True
                    if isinstance(to, str) and to not in body_ids:
                        return True

                return False

            if _has_structural_issues():
                continue

            try:
                Workflow.model_validate(
                    {
                        "workflow_name": f"{node.id}_loop_body",
                        "nodes": body_nodes,
                        "edges": body_edges,
                    }
                )
            except Exception as exc:
                raise ValueError(
                    f"loop 节点 '{node.id}' 的 body_subgraph 校验失败: {exc}"
                ) from exc

        return self


__all__ = [
    "Edge",
    "Node",
    "ParamBinding",
    "PydanticValidationError",
    "ValidationError",
    "Workflow",
]
