# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Tool definitions for parameter completion via LLM tool-calling.

This module exposes a single helper to build a per-node tool schema that
mirrors the Workflow DSL. It is used by :mod:`velvetflow.planner.structure`
so that the model must emit structured tool calls instead of free-form
JSON text when filling node parameters.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping

from velvetflow.models import Node

JINJA_EXPRESSION_NOTE = (
    "所有 params 都会作为 Jinja2 模板解析，"
    "必须使用 {{ ... }} 或 {% ... %} 的合法语法引用上游结果、输入或常量。"
    "仅支持模板字符串或字面量，不允许对象式绑定。"
)


def _normalize_object_schema(schema: Any) -> Dict[str, Any]:
    if isinstance(schema, Mapping) and schema.get("type") == "object":
        normalized = copy.deepcopy(schema)
        normalized.setdefault("additionalProperties", True)
        return normalized  # type: ignore[return-value]

    return {"type": "object", "additionalProperties": True}


def _condition_params_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": (
                    "必须是返回布尔值的 Jinja 表达式，"
                    "直接写出判断逻辑，例如 {{ result_of.fetch.status == 'ok' }} 或 {{ (loop.item.values | length) > 0 }}。"
                    f"{JINJA_EXPRESSION_NOTE}"
                ),
            }
        },
        "required": ["expression"],
        "additionalProperties": True,
    }


def _loop_params_schema() -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "loop_kind": {"type": "string", "enum": ["for_each", "while"]},
            "source": {
                "description": (
                    "for_each 迭代来源，必须是合法的 Jinja 模板字符串，"
                    "例如 {{ result_of.users.items }} 或 {{ input.items }}。"
                    f"{JINJA_EXPRESSION_NOTE}"
                ),
                "type": "string",
            },
            "condition": {
                "description": (
                    "loop_kind=while 时的退出条件，必须是返回布尔值的 Jinja 表达式，"
                    "例如 {{ result_of.check.flag }}。"
                    f"{JINJA_EXPRESSION_NOTE}"
                ),
                "type": "string",
            },
            "item_alias": {
                "type": "string",
                "description": "循环体内访问当前元素时使用的变量名，必填且需非空。",
            },
            "body_subgraph": {
                "type": "object",
                "properties": {
                    "nodes": {"type": "array", "items": {"type": "object"}},
                },
                "required": ["nodes"],
                "additionalProperties": True,
            },
            "exports": {
                "type": "object",
                "description": (
                    "loop.exports 使用 {key: Jinja表达式} 结构暴露循环体字段，"
                    "每个 value 必须引用 body_subgraph 节点的字段，如 {{ result_of.node.field }}，"
                    "执行时每个 key 生成一个列表，包含每轮迭代计算得到的元素。"
                    f"{JINJA_EXPRESSION_NOTE}"
                ),
                "additionalProperties": {"type": "string"},
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
        "必须使用符合 Jinja2 语法的模板填入 params，"
        "不得输出对象式绑定。"
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
