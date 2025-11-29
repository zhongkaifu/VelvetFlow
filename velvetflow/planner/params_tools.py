"""Tool definitions for parameter completion via LLM tool-calling.

This module exposes a single helper to build a per-node tool schema that
mirrors the Workflow DSL. It is used by :mod:`velvetflow.planner.params`
so that the model must emit structured tool calls instead of free-form
JSON text when filling node parameters.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping

from velvetflow.models import Node


def _binding_schema() -> Dict[str, Any]:
    """Schema for binding expressions in params (__from__/__agg__)."""

    return {
        "type": "object",
        "properties": {
            "__from__": {"type": "string"},
            "__agg__": {
                "type": "string",
                "enum": [
                    "identity",
                    "count",
                    "count_if",
                    "format_join",
                    "filter_map",
                    "pipeline",
                ],
            },
        },
        "required": ["__from__"],
        "additionalProperties": True,
    }


def _normalize_object_schema(schema: Any) -> Dict[str, Any]:
    if isinstance(schema, Mapping) and schema.get("type") == "object":
        normalized = copy.deepcopy(schema)
        normalized.setdefault("additionalProperties", True)
        return normalized  # type: ignore[return-value]

    return {"type": "object", "additionalProperties": True}


def _condition_params_schema() -> Dict[str, Any]:
    allowed_kinds = [
        "list_not_empty",
        "any_greater_than",
        "equals",
        "contains",
        "not_equals",
        "greater_than",
        "less_than",
        "between",
        "all_less_than",
        "is_empty",
        "not_empty",
        "is_not_empty",
        "multi_band",
        "compare",
    ]

    return {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": allowed_kinds},
            "source": {
                "description": "可以是 result_of 路径或绑定对象。",
                "anyOf": [
                    {"type": "string"},
                    _binding_schema(),
                ],
            },
            "field": {"type": "string"},
            "value": {},
            "threshold": {"type": "number"},
            "min": {"type": "number"},
            "max": {"type": "number"},
            "bands": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "min": {"type": "number"},
                        "max": {"type": "number"},
                        "label": {"type": "string"},
                    },
                    "required": ["min", "max"],
                },
            },
        },
        "required": ["kind"],
        "additionalProperties": True,
    }


def _loop_params_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "loop_kind": {"type": "string", "enum": ["for_each", "while"]},
            "source": {
                "description": "for_each 迭代来源，支持绑定或 result_of 路径。",
                "anyOf": [
                    {"type": "string"},
                    _binding_schema(),
                ],
            },
            "condition": {
                "description": "loop_kind=while 时的退出条件。",
                "anyOf": [
                    {"type": "string"},
                    _binding_schema(),
                ],
            },
            "item_alias": {
                "type": "string",
                "description": "循环体内访问当前元素时使用的变量名，必填且需非空。",
            },
            "body_subgraph": {
                "type": "object",
                "properties": {
                    "nodes": {"type": "array", "items": {"type": "object"}},
                    "entry": {"type": "string"},
                    "exit": {"type": "string"},
                },
                "required": ["nodes"],
                "additionalProperties": True,
            },
            "exports": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "object",
                        "properties": {
                            "from_node": {"type": "string"},
                            "fields": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                        },
                        "required": ["from_node", "fields"],
                        "additionalProperties": True,
                    },
                    "aggregates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "from_node": {"type": "string"},
                                "source": {"type": "string"},
                                "kind": {
                                    "type": "string",
                                    "enum": ["count", "count_if", "max", "min", "sum", "avg"],
                                },
                                "expr": {
                                    "type": "object",
                                    "properties": {
                                        "field": {"type": "string"},
                                        "op": {"type": "string"},
                                        "value": {},
                                    },
                                    "additionalProperties": True,
                                },
                            },
                            "required": ["kind", "source"],
                            "additionalProperties": True,
                        },
                    },
                },
                "required": ["items"],
                "additionalProperties": True,
            },
        },
        "required": ["loop_kind", "source", "item_alias", "exports"],
        "additionalProperties": True,
    }


def _params_schema_for_node(
    node: Node, action_schemas: Dict[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    if node.type == "action":
        schema = action_schemas.get(node.action_id or "", {}).get("arg_schema")
        return _normalize_object_schema(schema)

    if node.type == "condition":
        return _condition_params_schema()

    if node.type == "loop":
        return _loop_params_schema()

    return {"type": "object", "additionalProperties": True}


def build_param_completion_tool(
    node: Node,
    action_schemas: Dict[str, Mapping[str, Any]],
    allowed_node_ids: List[str],
) -> Dict[str, Any]:
    """Build a single function tool for the target node.

    The tool enforces the expected params shape for the node type and
    injects the allowed upstream node IDs into the description so that the
    model only references valid bindings.
    """

    params_schema = _params_schema_for_node(node, action_schemas)
    description_prefix = (
        f"补全节点 {node.id}({node.type}) 的 params。"
        "必须引用 allowed_node_ids 内的节点输出作为绑定来源。"
    )
    action_desc = ""
    if node.action_id:
        schema_meta = action_schemas.get(node.action_id, {})
        action_desc = (
            f" action_id={node.action_id} ({schema_meta.get('name', '')})，"
            "请遵守 arg_schema 的必填/类型要求。"
        )

    description = (
        f"{description_prefix}{action_desc}"
        f" allowed_node_ids={','.join(allowed_node_ids) or '无上游'}。"
        "填完后调用该工具提交 params。"
    )

    return {
        "type": "function",
        "function": {
            "name": "submit_node_params",
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "enum": [node.id]},
                    "params": params_schema,
                },
                "required": ["id", "params"],
            },
        },
    }


__all__ = ["build_param_completion_tool"]
