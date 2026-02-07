# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Loop node definitions and helpers."""

from typing import Any, Dict, List, Mapping, Optional

from velvetflow.reference_utils import parse_field_path
from velvetflow.verification.binding_checks import _iter_template_references

LOOP_PARAM_FIELDS = {
    "loop_kind",
    "source",
    "condition",
    "item_alias",
    "body_subgraph",
    "exports",
}

LOOP_NODE_FIELDS = {
    "id",
    "type",
    "display_name",
    "params",
    "parent_node_id",
    "depends_on",
}


def extract_loop_body_context(
    loop_node: Mapping[str, Any], action_schemas: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    params = loop_node.get("params") if isinstance(loop_node, Mapping) else None
    body = params.get("body_subgraph") if isinstance(params, Mapping) else None
    if not isinstance(body, Mapping):
        return {"nodes": []}

    context_nodes = []
    for child in body.get("nodes", []) or []:
        if not isinstance(child, Mapping):
            continue
        action_id = child.get("action_id")
        schema = action_schemas.get(action_id, {}) if isinstance(action_id, str) else {}
        context_nodes.append(
            {
                "id": child.get("id"),
                "type": child.get("type"),
                "action_id": action_id,
                "display_name": child.get("display_name"),
                "output_schema": schema.get("output_schema"),
            }
        )

    return {"nodes": context_nodes}


def validate_loop_exports(*, loop_node: Mapping[str, Any], exports: Mapping[str, Any]) -> List[str]:
    params = loop_node.get("params") if isinstance(loop_node.get("params"), Mapping) else {}
    body = params.get("body_subgraph") if isinstance(params, Mapping) else None
    if not isinstance(body, Mapping):
        body = {}

    body_nodes = [bn for bn in body.get("nodes", []) or [] if isinstance(bn, Mapping)]
    body_ids = {bn.get("id") for bn in body_nodes if isinstance(bn.get("id"), str)}

    errors: List[str] = []

    if not isinstance(exports, Mapping):
        return ["exports 必须是对象"]

    for key, value in exports.items():
        if not isinstance(key, str):
            errors.append("exports 的 key 必须是字符串")
            continue
        if not isinstance(value, str) or not value.strip():
            errors.append(f"exports.{key} 必须是非空 Jinja 表达式字符串")
            continue

        refs = list(_iter_template_references(value))
        if not refs:
            errors.append(f"exports.{key} 必须引用 loop body 节点的输出字段")
            continue

        has_body_ref = False
        for ref in refs:
            try:
                tokens = parse_field_path(ref)
            except Exception:
                continue
            if len(tokens) < 3 or tokens[0] != "result_of":
                continue
            ref_node = tokens[1]
            if isinstance(ref_node, str) and ref_node in body_ids:
                has_body_ref = True
            else:
                errors.append(f"exports.{key} 只能引用 body_subgraph 内的节点输出")
                break
        if not has_body_ref and not errors:
            errors.append(f"exports.{key} 必须引用 loop body 节点的输出字段")

    return errors


def fallback_loop_exports(
    loop_node: Mapping[str, Any], action_schemas: Mapping[str, Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    params = loop_node.get("params") if isinstance(loop_node, Mapping) else None
    if not isinstance(params, Mapping):
        return None
    body = params.get("body_subgraph")
    if not isinstance(body, Mapping):
        return None

    body_nodes = [bn for bn in body.get("nodes", []) or [] if isinstance(bn, Mapping)]
    body_ids = [bn.get("id") for bn in body_nodes if isinstance(bn.get("id"), str)]
    exit_node = body.get("exit") if isinstance(body.get("exit"), str) else None
    from_node = exit_node if exit_node in body_ids else (body_ids[0] if body_ids else None)
    if not from_node:
        return None

    target_node = next((bn for bn in body_nodes if bn.get("id") == from_node), None)
    field_name = "status"
    if isinstance(target_node, Mapping):
        action_id = target_node.get("action_id")
        schema = action_schemas.get(action_id, {}) if isinstance(action_id, str) else {}
        props = (
            schema.get("output_schema", {}).get("properties")
            if isinstance(schema.get("output_schema"), Mapping)
            else None
        )
        if isinstance(props, Mapping):
            field_name = next((k for k in props.keys() if isinstance(k, str)), field_name)

    return {
        "items": f"{{{{ result_of.{from_node}.{field_name} }}}}",
    }


def ensure_loop_items_fields(
    *,
    exports: Mapping[str, Any],
    loop_node: Mapping[str, Any],
    action_schemas: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Placeholder pass-through for loop exports."""

    return dict(exports)
