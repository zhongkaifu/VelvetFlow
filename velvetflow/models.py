"""Strongly-typed workflow DSL models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

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
    ]
    node_id: Optional[str]
    field: Optional[str]
    message: str


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

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return value
        allowed = {"true", "false"}
        if value not in allowed:
            raise ValueError(f"edge.condition 只能是 {allowed} 或 None")
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


__all__ = [
    "Edge",
    "Node",
    "PydanticValidationError",
    "ValidationError",
    "Workflow",
]
