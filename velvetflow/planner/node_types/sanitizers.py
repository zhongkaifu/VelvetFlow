# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Node field/param sanitization helpers."""

from typing import Any, Dict, List, Mapping, Optional

from velvetflow.planner.workflow_builder import WorkflowBuilder
from velvetflow.planner.node_types.action import ACTION_NODE_FIELDS
from velvetflow.planner.node_types.condition import CONDITION_NODE_FIELDS, CONDITION_PARAM_FIELDS
from velvetflow.planner.node_types.data import DATA_NODE_FIELDS, DATA_PARAM_FIELDS
from velvetflow.planner.node_types.loop import LOOP_NODE_FIELDS, LOOP_PARAM_FIELDS
from velvetflow.planner.node_types.reasoning import REASONING_NODE_FIELDS, REASONING_PARAM_FIELDS
from velvetflow.planner.node_types.switch import SWITCH_NODE_FIELDS, SWITCH_PARAM_FIELDS


def filter_supported_params(
    *,
    node_type: str,
    params: Any,
    action_schemas: Mapping[str, Mapping[str, Any]],
    action_id: Optional[str] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """Keep only supported param fields for the given node type."""

    if not isinstance(params, Mapping):
        return {}, []

    allowed_fields: Optional[set[str]] = None
    if node_type == "condition":
        allowed_fields = set(CONDITION_PARAM_FIELDS)
    elif node_type == "reasoning":
        allowed_fields = set(REASONING_PARAM_FIELDS)
    elif node_type == "data":
        allowed_fields = set(DATA_PARAM_FIELDS)
    elif node_type == "switch":
        allowed_fields = set(SWITCH_PARAM_FIELDS)
    elif node_type == "loop":
        allowed_fields = set(LOOP_PARAM_FIELDS)
    elif node_type == "action" and action_id:
        schema = action_schemas.get(action_id, {}) if isinstance(action_id, str) else {}
        properties = schema.get("arg_schema", {}).get("properties") if isinstance(schema.get("arg_schema"), Mapping) else None
        if isinstance(properties, Mapping):
            allowed_fields = set(properties.keys())

    if not allowed_fields:
        return dict(params), []

    cleaned: Dict[str, Any] = {k: v for k, v in params.items() if k in allowed_fields}
    removed = [k for k in params if k not in allowed_fields]

    return cleaned, removed


def sanitize_builder_node_params(
    builder: WorkflowBuilder, node_id: str, action_schemas: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    node = builder.nodes.get(node_id)
    if not isinstance(node, Mapping):
        return []

    params = node.get("params") or {}
    cleaned, removed = filter_supported_params(
        node_type=str(node.get("type")),
        params=params,
        action_schemas=action_schemas,
        action_id=node.get("action_id") if isinstance(node.get("action_id"), str) else None,
    )

    if removed:
        node["params"] = cleaned

    return removed


def sanitize_builder_node_fields(builder: WorkflowBuilder, node_id: str) -> List[str]:
    node = builder.nodes.get(node_id)
    if not isinstance(node, Mapping):
        return []

    node_type = node.get("type")
    allowed_fields: Optional[set[str]] = None
    if node_type == "action":
        allowed_fields = set(ACTION_NODE_FIELDS)
    elif node_type == "reasoning":
        allowed_fields = set(REASONING_NODE_FIELDS)
    elif node_type == "condition":
        allowed_fields = set(CONDITION_NODE_FIELDS)
    elif node_type == "switch":
        allowed_fields = set(SWITCH_NODE_FIELDS)
    elif node_type == "loop":
        allowed_fields = set(LOOP_NODE_FIELDS)
    elif node_type == "data":
        allowed_fields = set(DATA_NODE_FIELDS)

    if not allowed_fields:
        return []

    removed_keys = [key for key in list(node.keys()) if key not in allowed_fields]
    for key in removed_keys:
        node.pop(key, None)

    return removed_keys
