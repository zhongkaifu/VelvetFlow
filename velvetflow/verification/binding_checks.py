# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Binding and schema validation helpers."""

import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from velvetflow.loop_dsl import build_loop_output_schema
from velvetflow.models import ALLOWED_PARAM_AGGREGATORS, ValidationError
from velvetflow.reference_utils import (
    canonicalize_template_placeholders,
    normalize_reference_path,
    parse_field_path,
)

_TEMPLATE_REF_PATTERN = re.compile(
    r"\{\{\s*([^{}]+?)\s*\}\}|\$\{\{\s*([^{}]+?)\s*\}\}|\$\{\s*([^{}]+?)\s*\}"
)


def _index_actions_by_id(action_registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {a["action_id"]: a for a in action_registry}


def _schema_from_out_params_schema(
    out_params_schema: Mapping[str, Any] | None,
) -> Optional[Mapping[str, Any]]:
    """Build a JSON schema from node-level ``out_params_schema`` definitions."""

    if not isinstance(out_params_schema, Mapping) or not out_params_schema:
        return None

    # If the schema already looks complete, return it as-is.
    if any(key in out_params_schema for key in ("type", "properties", "$schema")):
        return out_params_schema

    properties: Dict[str, Any] = {}
    for key, value in out_params_schema.items():
        if isinstance(value, Mapping):
            properties[key] = value
        else:
            # Best-effort wrapper when only scalar types are provided.
            properties[key] = {"type": value} if isinstance(value, str) else {}

    return {"type": "object", "properties": properties}


def _get_node_output_schema(
    node: Mapping[str, Any] | None, actions_by_id: Mapping[str, Mapping[str, Any]]
) -> Optional[Mapping[str, Any]]:
    if isinstance(node, Mapping):
        node_schema = _schema_from_out_params_schema(node.get("out_params_schema"))
        if isinstance(node_schema, Mapping):
            return node_schema

        action_id = node.get("action_id")
        if isinstance(action_id, str):
            action_def = actions_by_id.get(action_id)
            if isinstance(action_def, Mapping):
                output_schema = action_def.get("output_schema")
                if isinstance(output_schema, Mapping):
                    return output_schema

    return None


def _maybe_decode_binding_string(raw: str) -> Optional[Any]:
    """Attempt to parse a JSON-like binding stored as string."""

    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text.startswith("{"):
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, Mapping) and "__from__" in parsed:
        return parsed

    return None


def _iter_empty_param_fields(obj: Any, path_prefix: str = "params") -> Iterable[str]:
    """Yield parameter paths whose values are empty strings."""

    if isinstance(obj, str):
        if obj.strip() == "":
            yield path_prefix
        return

    if isinstance(obj, Mapping):
        for key, value in obj.items():
            next_prefix = f"{path_prefix}.{key}" if path_prefix else str(key)
            yield from _iter_empty_param_fields(value, next_prefix)
        return

    if isinstance(obj, Sequence) and not isinstance(obj, (bytes, str)):
        for idx, item in enumerate(obj):
            next_prefix = f"{path_prefix}[{idx}]" if path_prefix else f"[{idx}]"
            yield from _iter_empty_param_fields(item, next_prefix)


def _iter_template_references(text: str) -> Iterable[str]:
    """Yield templated reference paths (``{{foo.bar}}`` or ``${foo.bar}``) from text."""

    if not isinstance(text, str):
        return []

    text = canonicalize_template_placeholders(text)

    return (
        (match.group(1) or match.group(2) or match.group(3) or "").strip()
        for match in _TEMPLATE_REF_PATTERN.finditer(text)
        if match.group(1) or match.group(2) or match.group(3)
    )


def _collect_param_bindings(obj: Any, prefix: str = "") -> List[Dict[str, Any]]:
    """Collect bindings that carry a __from__ reference for lightweight checks."""

    bindings: List[Dict[str, Any]] = []

    if isinstance(obj, Mapping):
        if "__from__" in obj:
            binding: Dict[str, Any] = {"path": prefix or "params", "source": obj.get("__from__")}
            if "__agg__" in obj:
                binding["agg"] = obj.get("__agg__")
            binding["binding"] = obj
            bindings.append(binding)
        for key, value in obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else str(key)
            bindings.extend(_collect_param_bindings(value, new_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            bindings.extend(_collect_param_bindings(value, new_prefix))
    elif isinstance(obj, str):
        decoded = _maybe_decode_binding_string(obj)
        if decoded:
            bindings.extend(_collect_param_bindings(decoded, prefix))

    return bindings


def validate_param_binding(binding: Any) -> Optional[str]:
    """Validate the shape of a single parameter binding."""

    if not isinstance(binding, Mapping):
        return "参数绑定必须是对象。"

    if "__from__" not in binding:
        return "缺少 __from__ 字段"

    source_path = binding["__from__"]
    if isinstance(source_path, list):
        if not source_path:
            return "__from__ 不应为空数组"
        if any(not isinstance(item, str) for item in source_path):
            return "__from__ 数组元素必须是字符串"
    elif not isinstance(source_path, str):
        return "__from__ 必须是字符串或字符串数组"

    allowed_aggs = set(ALLOWED_PARAM_AGGREGATORS)
    agg = binding.get("__agg__", "identity")
    if agg not in allowed_aggs:
        return (
            f"__agg__ 不支持值 {agg}，可选值：{', '.join(ALLOWED_PARAM_AGGREGATORS)}"
        )

    if agg == "count_if":
        if "field" not in binding or "op" not in binding or "value" not in binding:
            return "count_if 需要提供 field/op/value"

    if agg == "pipeline":
        steps = binding.get("steps")
        if not isinstance(steps, list) or not steps:
            return "pipeline 需要非空 steps 数组"
        for idx, step in enumerate(steps):
            if not isinstance(step, Mapping):
                return f"pipeline.steps[{idx}] 必须是对象"
            op = step.get("op")
            if op not in {"filter", "map", "format_join", "sort", "limit"}:
                return f"pipeline.steps[{idx}].op 不支持值 {op}"

    return None


def _check_output_path_against_schema(
    source_path: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Optional[Mapping[str, str]] = None,
    *,
    context_node_id: Optional[str] = None,
) -> Optional[str]:
    source_path = normalize_reference_path(source_path)
    if not isinstance(source_path, str):
        return f"source/__from__ 应该是字符串，但收到类型: {type(source_path)}"

    try:
        parts = parse_field_path(source_path)
    except Exception:
        return f"source/__from__ 不是合法的路径字符串: {source_path}"

    if len(parts) < 2 or parts[0] != "result_of":
        return None

    node_id = parts[1]
    rest_path = parts[2:]

    loop_body_parents = loop_body_parents or {}

    if node_id not in nodes_by_id:
        return f"节点 '{node_id}' 不存在，无法引用其输出"

    target = nodes_by_id[node_id]
    parent_loop = loop_body_parents.get(node_id)
    if parent_loop and rest_path and rest_path[0] != "exports":
        return f"loop body 节点 '{node_id}' 只能通过 exports 暴露输出"

    context_parent_loop = (
        loop_body_parents.get(context_node_id)
        if context_node_id and isinstance(loop_body_parents, Mapping)
        else None
    )

    if target.get("type") == "loop":
        loop_params = target.get("params") if isinstance(target, Mapping) else {}
        loop_schema = build_loop_output_schema(loop_params or {})
        if not loop_schema:
            return f"loop 节点 '{node_id}' 缺少 exports/out_schema 等输出定义，无法引用"

        if context_parent_loop and context_parent_loop == node_id and rest_path:
            if rest_path[0] == "exports":
                return (
                    f"loop body 节点 '{context_node_id}' 不可直接引用所属 loop "
                    f"'{node_id}' 的 exports，请改为引用 body_subgraph 节点的输出"
                )

        effective_path = rest_path[1:] if rest_path and rest_path[0] == "exports" else rest_path
        return _schema_path_error(
            loop_schema, _normalize_field_tokens(list(effective_path))
        )

    if target.get("type") == "action":
        schema = _get_node_output_schema(target, actions_by_id) or {}
        if not rest_path:
            return None
        return _schema_path_error(schema, list(rest_path))

    if rest_path:
        pretty_path = ".".join(str(p) for p in rest_path)
        return (
            f"节点 '{node_id}' 类型为 {target.get('type')}，无法引用其输出路径 '{pretty_path}'"
        )

    return None


def _get_array_item_schema_from_output(
    source: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Optional[Mapping[str, str]] = None,
    *,
    skip_path_check: bool = False,
    context_node_id: Optional[str] = None,
) -> Optional[Mapping[str, Any]]:
    normalized_source = normalize_reference_path(source)
    if not skip_path_check:
        err = _check_output_path_against_schema(
            normalized_source,
            nodes_by_id,
            actions_by_id,
            loop_body_parents,
            context_node_id=context_node_id,
        )
        if err:
            return None

    try:
        parts = parse_field_path(normalized_source)
    except Exception:
        return None

    if len(parts) < 2 or parts[0] != "result_of":
        return None

    node_id = parts[1]
    first_field = None
    for token in parts[2:]:
        if isinstance(token, str):
            first_field = token
            break
    node = nodes_by_id.get(node_id)
    node_type = node.get("type") if node else None

    if node_type == "loop":
        loop_params = (node or {}).get("params") or {}
        item_alias = loop_params.get("item_alias")
        source = loop_params.get("source")
        if isinstance(item_alias, str):
            body_subgraph = loop_params.get("body_subgraph")
            body_nodes = body_subgraph.get("nodes") if isinstance(body_subgraph, Mapping) else []
            body_node_map = {bn.get("id"): bn for bn in body_nodes if isinstance(bn, Mapping)}

            exports = loop_params.get("exports") if isinstance(loop_params, Mapping) else None
            items_spec = exports.get("items") if isinstance(exports, Mapping) else None
            from_node = items_spec.get("from_node") if isinstance(items_spec, Mapping) else None
            fields = items_spec.get("fields") if isinstance(items_spec, Mapping) else None

            body_params = node.get("params") if isinstance(node, Mapping) else {}
            loop_params = body_params if isinstance(body_params, Mapping) else {}
            body_nodes = loop_params.get("body_subgraph", {}).get("nodes", [])
            body_node_map = {bn.get("id"): bn for bn in body_nodes if isinstance(bn, Mapping)}

            if (
                isinstance(from_node, str)
                and isinstance(fields, list)
                and from_node in body_node_map
            ):
                items_spec = (body_params.get("exports") or {}).get("items")
                items_schema = _get_loop_items_schema_from_exports(
                    items_spec, node, nodes_by_id, actions_by_id
                )
                if items_schema:
                    return items_schema

        schema = build_loop_output_schema(loop_params) or {}
        props = schema.get("properties") or {}
        return (props.get(first_field) or {}).get("items") if first_field in props else None

    schema_obj = _get_node_output_schema(node, actions_by_id) or {}
    schema = schema_obj.get("properties", {}) if isinstance(schema_obj, Mapping) else {}
    if first_field not in schema:
        return None

    return (schema.get(first_field) or {}).get("items")


def _get_output_schema_at_path(
    source: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Optional[Mapping[str, str]] = None,
) -> Optional[Mapping[str, Any]]:
    normalized_source = normalize_reference_path(source)
    try:
        parts = parse_field_path(normalized_source)
    except Exception:
        return None

    if len(parts) < 2 or parts[0] != "result_of":
        return None

    node_id = parts[1]
    rest_path = parts[2:]
    node = nodes_by_id.get(node_id)
    node_type = node.get("type") if node else None

    if node_type == "loop":
        params = node.get("params") if isinstance(node, Mapping) else {}
        loop_schema = build_loop_output_schema(params or {})
        exports = params.get("exports") if isinstance(params, Mapping) else {}
        items_spec = exports.get("items") if isinstance(exports, Mapping) else None
        items_schema = _get_loop_items_schema_from_exports(
            items_spec, node, nodes_by_id, actions_by_id
        )
        if not loop_schema:
            return None

        if items_schema:
            props = loop_schema.setdefault("properties", {})
            items_container = props.get("items") if isinstance(props, Mapping) else {}
            new_items = {"type": "array", "items": items_schema}
            if isinstance(items_container, Mapping):
                merged = dict(items_container)
                merged.update(new_items)
                props["items"] = merged
            else:
                props["items"] = new_items

        effective_path = rest_path[1:] if rest_path and rest_path[0] == "exports" else rest_path
        normalized_path = _normalize_field_tokens(list(effective_path))
        if normalized_path and normalized_path[-1] in {"length", "count"}:
            container_schema = _walk_schema_with_tokens(loop_schema, normalized_path[:-1])
            properties = (
                container_schema.get("properties")
                if isinstance(container_schema, Mapping)
                and isinstance(container_schema.get("properties"), Mapping)
                else {}
            )
            if normalized_path[-1] not in properties and isinstance(container_schema, Mapping):
                container_type = container_schema.get("type")
                if normalized_path[-1] == "length" and container_type == "array":
                    return {"type": "integer"}
                if normalized_path[-1] == "count" and container_type in {"array", "object", None}:
                    return {"type": "integer"}

        return _walk_schema_with_tokens(loop_schema, normalized_path)

    schema_obj = _get_node_output_schema(node, actions_by_id) or {}
    normalized_path = _normalize_field_tokens(list(rest_path))
    if normalized_path and normalized_path[-1] in {"length", "count"}:
        container_schema = _walk_schema_with_tokens(schema_obj, normalized_path[:-1])
        properties = (
            container_schema.get("properties")
            if isinstance(container_schema, Mapping)
            and isinstance(container_schema.get("properties"), Mapping)
            else {}
        )
        if normalized_path[-1] not in properties and isinstance(container_schema, Mapping):
            container_type = container_schema.get("type")
            if normalized_path[-1] == "length" and container_type == "array":
                return {"type": "integer"}
            if normalized_path[-1] == "count" and container_type in {"array", "object", None}:
                return {"type": "integer"}

    return _walk_schema_with_tokens(schema_obj, normalized_path)


def _get_field_schema_from_item(
    item_schema: Optional[Mapping[str, Any]], field: str
) -> Optional[Mapping[str, Any]]:
    """Extract the schema definition for a field inside an array item schema."""

    if not item_schema:
        return None

    try:
        parts = parse_field_path(field)
    except Exception:
        return None

    current: Mapping[str, Any] = item_schema
    for token in parts:
        typ = current.get("type")

        if isinstance(token, int):
            if typ != "array":
                return None
            current = current.get("items") or {}
            continue

        if typ == "array":
            current = current.get("items") or {}

        if typ in {None, "object"}:
            props = current.get("properties") or {}
            if token not in props:
                return None
            current = props[token]
            continue

        return None

    return current


def _is_numeric_schema_type(schema_type: Any) -> bool:
    """Return True if the schema type represents a numeric value."""

    if isinstance(schema_type, list):
        return any(t in {"number", "integer"} for t in schema_type)

    return schema_type in {"number", "integer"}


def _get_field_schema(schema: Mapping[str, Any], field: str) -> Optional[Mapping[str, Any]]:
    """Return the schema of a dotted field path inside an object/array schema."""

    if not isinstance(schema, Mapping) or not isinstance(field, str):
        return None

    current: Mapping[str, Any] = schema
    try:
        parts = parse_field_path(field)
    except Exception:
        return None

    for part in parts:
        if not isinstance(current, Mapping):
            return None

        current_type = current.get("type")
        if isinstance(part, int):
            if current_type != "array":
                return None
            current = current.get("items") or {}
            continue

        if current_type == "array":
            current = current.get("items") or {}

        if current_type in {None, "object"}:
            props = current.get("properties") or {}
            if part not in props:
                return None
            current = props.get(part) or {}
            continue

        return None

    return current


def _get_builtin_field_schema(schema: Mapping[str, Any], field: str) -> Optional[Mapping[str, Any]]:
    if not isinstance(schema, Mapping):
        return None

    field = str(field)
    typ = schema.get("type")
    properties = schema.get("properties") if isinstance(schema.get("properties"), Mapping) else {}

    # Prefer explicit schema definitions over builtin fallbacks when the field is defined.
    if field in properties:
        return None

    if field == "length" and typ == "array":
        return {"type": "integer"}

    if field == "count" and typ in {"array", "object", None}:
        return {"type": "integer"}

    if field == "id" and typ == "object":
        return {"type": "string"}

    return None


def _walk_schema_with_tokens(schema: Mapping[str, Any], fields: List[Any]) -> Optional[Mapping[str, Any]]:
    if not isinstance(schema, Mapping):
        return None

    if not fields:
        return schema

    builtin_schema = _get_builtin_field_schema(schema, fields[0])
    if builtin_schema is not None:
        return builtin_schema

    current: Mapping[str, Any] = schema
    for name in fields:
        typ = current.get("type")

        if isinstance(name, int):
            if typ != "array":
                return None
            current = current.get("items") or {}
            continue

        if name == "*":
            if typ != "array":
                return None
            current = current.get("items") or {}
            continue

        if typ == "array":
            current = current.get("items") or {}
            typ = current.get("type")

        if typ == "object" or typ is None:
            props = current.get("properties") or {}
            if name not in props:
                return None
            current = props[name]
            continue

        return None

    return current


def _normalize_field_tokens(fields: List[Any]) -> List[Any]:
    normalized_fields = []
    for token in fields:
        if isinstance(token, str) and token.isdigit():
            normalized_fields.append(int(token))
        else:
            normalized_fields.append(token)
    return normalized_fields


def _schema_path_error(schema: Mapping[str, Any], fields: List[Any]) -> Optional[str]:
    """Check whether a dotted/Indexed field path exists in a JSON schema."""

    if not isinstance(schema, Mapping):
        return "output_schema 不是对象，无法校验字段路径。"

    normalized_fields = _normalize_field_tokens(fields)

    current: Mapping[str, Any] = schema
    idx = 0
    while idx < len(normalized_fields):
        name = normalized_fields[idx]
        typ = current.get("type")

        if isinstance(name, int):
            if typ != "array":
                return f"字段路径 '{'.'.join(map(str, normalized_fields))}' 与 schema 类型 '{typ}' 不匹配（期望 array）。"
            current = current.get("items") or {}
            idx += 1
            continue

        if name == "*":
            if typ != "array":
                return f"字段路径 '{'.'.join(map(str, normalized_fields))}' 与 schema 类型 '{typ}' 不匹配（期望 array）。"
            current = current.get("items") or {}
            idx += 1
            continue

        builtin_schema = _get_builtin_field_schema(current, name)
        if builtin_schema is not None:
            current = builtin_schema
            idx += 1
            continue

        if typ == "array":
            current = current.get("items") or {}
            continue

        if typ == "object" or typ is None:
            props = current.get("properties") or {}
            if name not in props:
                return f"字段 '{name}' 不存在，已知字段有: {list(props.keys())}"
            current = props[name]
            idx += 1
            continue

        if idx == len(normalized_fields) - 1:
            return None

        return f"字段路径 '{'.'.join(map(str, normalized_fields))}' 与 schema 类型 '{typ}' 不匹配（期望 object/array）。"

    return None


def _suggest_numeric_subfield(schema: Mapping[str, Any], prefix: str = "") -> Optional[str]:
    typ = schema.get("type")

    if typ in {"number", "integer"}:
        return prefix or ""

    if typ == "array":
        return _suggest_numeric_subfield(schema.get("items") or {}, prefix)

    if typ == "object" or typ is None:
        props = schema.get("properties") or {}
        for name, subschema in props.items():
            candidate = _suggest_numeric_subfield(
                subschema,
                f"{prefix}.{name}" if prefix else name,
            )
            if candidate:
                return candidate

    return None


def _get_loop_items_schema_from_exports(
    items_spec: Any,
    loop_node: Mapping[str, Any],
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    actions_by_id: Mapping[str, Mapping[str, Any]],
) -> Optional[Mapping[str, Any]]:
    """Derive the loop items array schema using the referenced action's output schema."""

    if not isinstance(items_spec, Mapping):
        return None

    from_node_id = items_spec.get("from_node")
    fields = items_spec.get("fields")
    if not isinstance(from_node_id, str) or not isinstance(fields, list):
        return None

    body_nodes = (
        (loop_node.get("params") or {}).get("body_subgraph") or {}
    ).get("nodes", [])
    src_node = next(
        (n for n in body_nodes if isinstance(n, Mapping) and n.get("id") == from_node_id),
        None,
    )

    output_schema = _get_node_output_schema(src_node, actions_by_id)

    item_props: Dict[str, Any] = {}
    for fld in fields:
        if not isinstance(fld, str):
            continue
        field_schema = _get_field_schema(output_schema, fld) if output_schema else None
        item_props[fld] = field_schema or {}

    return {"type": "object", "properties": item_props} if item_props else None


def _check_array_item_field(
    source: str,
    field: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Optional[Mapping[str, str]] = None,
    alias_schemas: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    context_node_id: Optional[str] = None,
) -> Optional[str]:
    normalized_source = normalize_reference_path(source)

    alias_schema = (alias_schemas or {}).get(normalized_source)
    if alias_schema:
        return _schema_path_error(alias_schema, [field])

    path_error = _check_output_path_against_schema(
        normalized_source,
        nodes_by_id,
        actions_by_id,
        loop_body_parents,
        context_node_id=context_node_id,
    )
    if path_error:
        return path_error

    if field == "length":
        container_schema = _get_output_schema_at_path(
            normalized_source, nodes_by_id, actions_by_id, loop_body_parents
        )
        if container_schema is None:
            item_schema = _get_array_item_schema_from_output(
                normalized_source,
                nodes_by_id,
                actions_by_id,
                loop_body_parents,
                skip_path_check=True,
                context_node_id=context_node_id,
            )
            if item_schema is not None:
                container_schema = {"type": "array", "items": item_schema}

        if _get_builtin_field_schema(container_schema or {}, field) is not None:
            return None

    item_schema = _get_array_item_schema_from_output(
        normalized_source,
        nodes_by_id,
        actions_by_id,
        loop_body_parents,
        skip_path_check=True,
        context_node_id=context_node_id,
    )
    if not item_schema:
        return f"路径 '{normalized_source}' 不是数组输出，无法访问字段 '{field}'。"

    return _schema_path_error(item_schema, [field])


__all__ = [
    "_check_array_item_field",
    "_check_output_path_against_schema",
    "_collect_param_bindings",
    "_get_array_item_schema_from_output",
    "_get_field_schema",
    "_get_field_schema_from_item",
    "_get_loop_items_schema_from_exports",
    "_get_node_output_schema",
    "_get_output_schema_at_path",
    "_index_actions_by_id",
    "_iter_empty_param_fields",
    "_iter_template_references",
    "_maybe_decode_binding_string",
    "_schema_path_error",
    "_suggest_numeric_subfield",
    "_walk_schema_with_tokens",
    "validate_param_binding",
]
