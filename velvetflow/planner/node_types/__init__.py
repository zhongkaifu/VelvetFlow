# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Node-type specific definitions and helper utilities."""

from velvetflow.planner.node_types.action import ACTION_NODE_FIELDS
from velvetflow.planner.node_types.condition import CONDITION_NODE_FIELDS, CONDITION_PARAM_FIELDS
from velvetflow.planner.node_types.data import DATA_NODE_FIELDS, DATA_PARAM_FIELDS, build_data_node_output_schema
from velvetflow.planner.node_types.loop import (
    LOOP_NODE_FIELDS,
    LOOP_PARAM_FIELDS,
    ensure_loop_items_fields,
    extract_loop_body_context,
    fallback_loop_exports,
    validate_loop_exports,
)
from velvetflow.planner.node_types.reasoning import REASONING_NODE_FIELDS, REASONING_PARAM_FIELDS
from velvetflow.planner.node_types.sanitizers import (
    filter_supported_params,
    sanitize_builder_node_fields,
    sanitize_builder_node_params,
)
from velvetflow.planner.node_types.switch import SWITCH_NODE_FIELDS, SWITCH_PARAM_FIELDS

__all__ = [
    "ACTION_NODE_FIELDS",
    "CONDITION_NODE_FIELDS",
    "CONDITION_PARAM_FIELDS",
    "DATA_NODE_FIELDS",
    "DATA_PARAM_FIELDS",
    "LOOP_NODE_FIELDS",
    "LOOP_PARAM_FIELDS",
    "REASONING_NODE_FIELDS",
    "REASONING_PARAM_FIELDS",
    "SWITCH_NODE_FIELDS",
    "SWITCH_PARAM_FIELDS",
    "build_data_node_output_schema",
    "ensure_loop_items_fields",
    "extract_loop_body_context",
    "fallback_loop_exports",
    "filter_supported_params",
    "sanitize_builder_node_fields",
    "sanitize_builder_node_params",
    "validate_loop_exports",
]
