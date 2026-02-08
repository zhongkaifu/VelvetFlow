# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Data node definitions and helpers."""

from typing import Any, Dict, Mapping

DATA_PARAM_FIELDS = {
    "schema",
    "dataset",
}

DATA_NODE_FIELDS = {
    "id",
    "type",
    "display_name",
    "params",
    "out_params_schema",
    "parent_node_id",
    "depends_on",
}


def build_data_node_output_schema(schema: Any) -> Dict[str, Any]:
    properties: Dict[str, Any] = {}
    if isinstance(schema, list):
        for field in schema:
            if not isinstance(field, Mapping):
                continue
            name = field.get("name")
            if not isinstance(name, str) or not name:
                continue
            field_type = field.get("type") if isinstance(field.get("type"), str) else "string"
            description = field.get("description") if isinstance(field.get("description"), str) else ""
            properties[name] = {"type": field_type, "description": description}

    return {
        "type": "object",
        "properties": {
            "dataset": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                },
            },
            "schema": {
                "type": "array",
                "items": {"type": "object"},
            },
        },
    }
