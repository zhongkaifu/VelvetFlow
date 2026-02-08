# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Action node definitions."""

from typing import Any, Dict, List

ACTION_NODE_FIELDS = {
    "id",
    "type",
    "action_id",
    "display_name",
    "params",
    "out_params_schema",
    "parent_node_id",
    "depends_on",
}


def build_action_schema_map(action_registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Build a lookup map for action schemas keyed by action_id."""

    action_schemas: Dict[str, Dict[str, Any]] = {}
    for action in action_registry:
        aid = action.get("action_id")
        if not aid:
            continue
        action_schemas[aid] = {
            "name": action.get("name", ""),
            "description": action.get("description", ""),
            "domain": action.get("domain", ""),
            "arg_schema": action.get("arg_schema"),
            "output_schema": action.get("output_schema"),
        }
    return action_schemas
