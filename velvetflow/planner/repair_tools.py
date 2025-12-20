# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Deterministic repair tools exposed to the LLM for tool-calling."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from velvetflow.loop_dsl import iter_workflow_and_loop_body_nodes
from velvetflow.verification.binding_checks import _get_array_item_schema_from_output
from velvetflow.planner.structure import _fallback_loop_exports
from velvetflow.logging_utils import log_event, log_info, log_warn
from velvetflow.models import ValidationError, Workflow
from velvetflow.reference_utils import normalize_reference_path, parse_field_path


def _workflow_fingerprint(workflow: Mapping[str, Any]) -> str:
    payload = json.dumps(workflow, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _schema_default_value(field_schema: Mapping[str, Any]) -> Any:
    if not isinstance(field_schema, Mapping):
        return None

    if "default" in field_schema:
        return field_schema.get("default")

    schema_type = field_schema.get("type")
    if isinstance(schema_type, list):
        schema_type = schema_type[0] if schema_type else None

    if schema_type == "string":
        return ""
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        return []
    if schema_type == "object":
        return {}

    enum_values = field_schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    return None


def _format_tokens(tokens: Sequence[Any]) -> str:
    path = ""
    for token in tokens:
        if isinstance(token, int):
            path += f"[{token}]"
        else:
            path += ("." if path else "") + str(token)
    return path


def _schema_supports_tokens(schema: Mapping[str, Any] | None, tokens: Sequence[Any]) -> bool:
    current: Mapping[str, Any] | None = schema
    for token in tokens:
        if not isinstance(current, Mapping):
            return False

        if isinstance(token, int):
            items_schema = current.get("items")
            if not isinstance(items_schema, Mapping):
                return False
            current = items_schema
            continue

        properties = current.get("properties")
        if not isinstance(properties, Mapping):
            return False
        if token not in properties:
            return False
        current = properties.get(token)

    return True


def _coerce_value_to_schema_type(value: Any, field_schema: Mapping[str, Any]) -> Optional[Any]:
    if not isinstance(field_schema, Mapping):
        return None

    schema_type = field_schema.get("type")
    if isinstance(schema_type, list):
        schema_type = schema_type[0] if schema_type else None

    if schema_type == "string" and not isinstance(value, str):
        return str(value)

    if schema_type == "integer":
        if isinstance(value, bool):
            return int(value)
        if not isinstance(value, int):
            try:
                return int(value)
            except Exception:  # noqa: BLE001
                return None

    if schema_type == "number":
        if isinstance(value, bool):
            return float(value)
        if not isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:  # noqa: BLE001
                return None

    if schema_type == "boolean" and not isinstance(value, bool):
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
        return bool(value)

    if schema_type == "array" and not isinstance(value, list):
        return [value]

    if schema_type == "object" and not isinstance(value, Mapping):
        return {"value": value}

    return None


def _index_actions_by_id(action_registry: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {a.get("action_id"): dict(a) for a in action_registry if a.get("action_id")}


def _collect_related_errors(
    validation_errors: Iterable[ValidationError], node_id: str | None
) -> List[Dict[str, Any]]:
    related: List[Dict[str, Any]] = []
    for err in validation_errors:
        if node_id and err.node_id and err.node_id != node_id:
            continue
        related.append({
            "code": err.code,
            "node_id": err.node_id,
            "field": err.field,
            "message": err.message,
        })
    return related


def _split_path(path: str) -> List[str | int]:
    parts: List[str | int] = []
    for segment in path.split("."):
        if not segment:
            continue
        while segment:
            if "[" in segment:
                before, remainder = segment.split("[", 1)
                if before:
                    parts.append(before)
                idx_str, after = remainder.split("]", 1)
                parts.append(int(idx_str))
                segment = after
                if segment.startswith("."):
                    segment = segment[1:]
            else:
                parts.append(segment)
                segment = ""
    return parts


def _set_nested_field(container: MutableMapping[str, Any], path: str, value: Any) -> None:
    segments = _split_path(path)
    current: Any = container
    for idx, seg in enumerate(segments):
        is_last = idx == len(segments) - 1
        if isinstance(seg, int):
            if not isinstance(current, list):
                raise ValueError(f"路径 {path} 需要列表，但当前为 {type(current).__name__}")
            while len(current) <= seg:
                current.append({})
            if is_last:
                current[seg] = value
            else:
                nxt = current[seg]
                if not isinstance(nxt, (dict, list)):
                    nxt = {}
                    current[seg] = nxt
                current = nxt
        else:
            if not isinstance(current, MutableMapping):
                raise ValueError(f"路径 {path} 需要对象，但当前为 {type(current).__name__}")
            if is_last:
                current[seg] = value
            else:
                nxt = current.get(seg)
                if not isinstance(nxt, (dict, list)):
                    nxt = [] if isinstance(segments[idx + 1], int) else {}
                    current[seg] = nxt
                current = nxt


def _apply_local_repairs_for_unknown_params(
    current_workflow: Workflow,
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
) -> Optional[Workflow]:
    """Remove parameters that are not defined in the action schema."""

    actions_by_id = _index_actions_by_id(action_registry)
    workflow_dict = current_workflow.model_dump(by_alias=True)
    nodes: List[Dict[str, Any]] = [
        n for n in workflow_dict.get("nodes", []) if isinstance(n, Mapping)
    ]
    nodes_by_id: Dict[str, Dict[str, Any]] = {n.get("id"): n for n in nodes}

    changed = False

    for err in validation_errors:
        if err.code != "UNKNOWN_PARAM" or not err.node_id or not err.field:
            continue

        node = nodes_by_id.get(err.node_id)
        if not node or node.get("type") != "action":
            continue

        action_id = node.get("action_id")
        action_def = actions_by_id.get(action_id)
        if not action_def:
            continue

        arg_schema = action_def.get("arg_schema") or {}
        additional_allowed = bool(arg_schema.get("additionalProperties")) if isinstance(arg_schema, Mapping) else False
        if additional_allowed:
            continue

        params = node.get("params")
        if not isinstance(params, dict):
            continue

        if err.field in params:
            params.pop(err.field, None)
            changed = True

    if not changed:
        return None

    try:
        return Workflow.model_validate(workflow_dict)
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] 移除未定义参数时验证失败：{exc}")
        return None


def fill_loop_exports_defaults(
    workflow: Mapping[str, Any], *, action_registry: Iterable[Mapping[str, Any]]
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Ensure loop nodes carry minimally valid exports to avoid trivial LLM repairs."""

    patched = copy.deepcopy(workflow)
    actions_by_id = _index_actions_by_id(action_registry)
    nodes = patched.get("nodes") if isinstance(patched.get("nodes"), list) else []

    changed = False
    updated_nodes: List[Dict[str, Any]] = []

    for node in nodes:
        if not isinstance(node, MutableMapping) or node.get("type") != "loop":
            updated_nodes.append(node)
            continue

        params = node.get("params") if isinstance(node.get("params"), MutableMapping) else {}
        body = params.get("body_subgraph") if isinstance(params.get("body_subgraph"), Mapping) else {}
        body_nodes = [bn for bn in body.get("nodes", []) or [] if isinstance(bn, Mapping) and bn.get("id")]
        body_ids = [bn.get("id") for bn in body_nodes if isinstance(bn.get("id"), str)]

        exports = params.get("exports") if isinstance(params.get("exports"), Mapping) else None
        fallback = _fallback_loop_exports(node, actions_by_id)

        if not isinstance(exports, Mapping) or not exports:
            if fallback:
                params = dict(params)
                params["exports"] = fallback
                node = dict(node)
                node["params"] = params
                changed = True
            updated_nodes.append(node)
            continue

        cleaned_exports = {
            key: value
            for key, value in exports.items()
            if isinstance(key, str) and isinstance(value, str) and value.strip()
        }

        if not cleaned_exports and fallback:
            cleaned_exports = fallback
            changed = True

        if cleaned_exports != exports:
            params = dict(params)
            params["exports"] = cleaned_exports
            node = dict(node)
            node["params"] = params
            changed = True

        updated_nodes.append(node)

    if changed:
        patched["nodes"] = updated_nodes

    return patched, {"applied": changed, "reason": None if changed else "无修复项"}


def normalize_binding_paths(workflow: Mapping[str, Any]) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Normalize __from__ bindings that are wrapped in templates or whitespace."""

    patched = copy.deepcopy(workflow)
    changed = False

    def _walk(obj: Any) -> Any:
        nonlocal changed
        if isinstance(obj, Mapping):
            new_obj = obj if isinstance(obj, MutableMapping) else dict(obj)
            if "__from__" in obj and isinstance(obj.get("__from__"), str):
                normalized = normalize_reference_path(obj.get("__from__"))
                if normalized != obj.get("__from__"):
                    new_obj["__from__"] = normalized
                    changed = True
            for key, value in obj.items():
                new_obj[key] = _walk(value)
            return new_obj
        if isinstance(obj, list):
            return [_walk(item) for item in obj]
        return obj

    patched = _walk(patched)
    return patched, {"applied": changed, "reason": None if changed else "未发现需要规范化的绑定"}


def align_loop_body_alias_references(
    workflow: Mapping[str, Any], *, action_registry: Iterable[Mapping[str, Any]]
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Rewrite loop body references that accidentally use a stale alias name.

    If a loop body node references an unknown object (e.g. ``employee_temp``)
    but the referenced fields match the current ``item_alias`` schema, the
    reference is rewritten to use the active alias.
    """

    patched = copy.deepcopy(workflow)
    nodes = patched.get("nodes") if isinstance(patched.get("nodes"), list) else []

    actions_by_id = _index_actions_by_id(action_registry)
    nodes_by_id = {
        n.get("id"): n
        for n in iter_workflow_and_loop_body_nodes(patched)
        if isinstance(n, Mapping)
    }

    replacements: List[Dict[str, str]] = []
    changed = False

    for node in nodes:
        if not isinstance(node, MutableMapping) or node.get("type") != "loop":
            continue

        params = node.get("params") if isinstance(node.get("params"), MutableMapping) else {}
        item_alias = params.get("item_alias") if isinstance(params, Mapping) else None
        if not isinstance(item_alias, str) or not item_alias:
            continue

        source = params.get("source")
        source_path = source.get("__from__") if isinstance(source, Mapping) else source
        if not isinstance(source_path, str):
            continue

        item_schema = _get_array_item_schema_from_output(
            source_path, nodes_by_id, actions_by_id, skip_path_check=True
        )

        body = params.get("body_subgraph") if isinstance(params, Mapping) else {}
        body_nodes = body.get("nodes") if isinstance(body, Mapping) else []

        for body_node in body_nodes or []:
            if not isinstance(body_node, MutableMapping):
                continue

            params_obj = body_node.get("params") if isinstance(body_node.get("params"), MutableMapping) else None
            if not isinstance(params_obj, MutableMapping):
                continue

            def _walk(obj: Any) -> Any:
                nonlocal changed
                if isinstance(obj, Mapping):
                    new_obj = obj if isinstance(obj, MutableMapping) else dict(obj)
                    if "__from__" in obj and isinstance(obj.get("__from__"), str):
                        new_from = _walk(obj.get("__from__"))
                        if new_from != obj.get("__from__"):
                            new_obj["__from__"] = new_from
                    else:
                        for key, value in obj.items():
                            new_obj[key] = _walk(value)
                    return new_obj

                if isinstance(obj, list):
                    return [_walk(item) for item in obj]

                if isinstance(obj, str):
                    normalized = normalize_reference_path(obj)
                    if item_alias in {normalized, obj}:
                        return obj
                    if "." not in normalized and "[" not in normalized:
                        return obj
                    try:
                        tokens = parse_field_path(normalized)
                    except Exception:
                        return obj

                    if (
                        not tokens
                        or tokens[0] in {item_alias, "result_of", "params", "vars", "context"}
                        or len(tokens) < 2
                    ):
                        return obj

                    candidate_tokens = [item_alias, *tokens[1:]]
                    if not _schema_supports_tokens(item_schema, candidate_tokens[1:]):
                        return obj

                    new_path = _format_tokens(candidate_tokens)
                    replacements.append({"from": normalized, "to": new_path})
                    changed = True
                    return new_path

                return obj

            new_params = _walk(params_obj)
            if new_params is not params_obj:
                body_node["params"] = new_params

    return patched, {
        "applied": changed,
        "replacements": replacements,
        "reason": None if changed else "未发现需对齐到 item_alias 的引用",
    }


def fix_missing_loop_exports_items(
    workflow: Mapping[str, Any],
    errors: Sequence[ValidationError],
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Insert missing ``exports`` segments for loop references."""

    patched = copy.deepcopy(workflow)
    replacements: List[Dict[str, str]] = []

    nodes_by_id = {
        node.get("id"): node
        for node in iter_workflow_and_loop_body_nodes(patched)
        if isinstance(node.get("id"), str)
    }

    def _format_path(parts: List[Any]) -> str:
        path = ""
        for token in parts:
            if isinstance(token, int):
                path += f"[{token}]"
            else:
                path += ("." if path else "") + str(token)
        return path

    for err in errors:
        if err.code != "SCHEMA_MISMATCH" or not err.message:
            continue

        for match in re.findall(r"'(result_of[^']+)'", err.message):
            normalized = normalize_reference_path(match)
            try:
                tokens = parse_field_path(normalized)
            except Exception:
                continue

            if len(tokens) < 3 or tokens[0] != "result_of" or tokens[2] == "exports":
                continue

            loop_id = tokens[1]
            loop_node = nodes_by_id.get(loop_id)
            if not loop_node or loop_node.get("type") != "loop":
                continue

            params = loop_node.get("params") if isinstance(loop_node.get("params"), Mapping) else {}
            exports = params.get("exports") if isinstance(params, Mapping) else None
            if not isinstance(exports, Mapping):
                continue

            export_keys = {key for key in exports.keys() if isinstance(key, str)}
            if tokens[2] not in export_keys:
                continue

            corrected_parts = [*tokens[:2], "exports", *tokens[2:]]
            corrected = _format_path(corrected_parts)
            replacements.append({"from": normalized, "to": corrected})
            patched, _ = replace_reference_paths(patched, old=normalized, new=corrected)

    return patched, {"applied": bool(replacements), "replacements": replacements}


def replace_reference_paths(
    workflow: Mapping[str, Any],
    *,
    old: str,
    new: str,
    include_edges: bool = True,
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Rewrite node/field references across the workflow in a deterministic way.

    This helper is intended for validation errors that involve multiple nodes,
    such as missing edge endpoints or parameter bindings pointing to an invalid
    ``result_of.<node>`` path. It performs a best-effort search-and-replace on
    reference-like string values while leaving node definitions untouched.
    """

    patched = copy.deepcopy(workflow)
    normalized_old = normalize_reference_path(old)
    normalized_new = normalize_reference_path(new)

    changed = False
    binding_updates = 0
    edge_updates = 0

    def _walk(obj: Any, parent_key: str | None = None) -> Any:
        nonlocal changed, binding_updates, edge_updates

        if isinstance(obj, Mapping):
            new_obj = obj if isinstance(obj, MutableMapping) else dict(obj)
            for key, value in list(new_obj.items()):
                if key == "id":
                    continue
                new_obj[key] = _walk(value, key)
            return new_obj

        if isinstance(obj, list):
            return [_walk(item, parent_key) for item in obj]

        if isinstance(obj, str):
            normalized_val = normalize_reference_path(obj)

            if parent_key in {"from", "to"} and include_edges:
                if normalized_val == normalized_old:
                    edge_updates += 1
                    changed = True
                    return normalized_new

            if normalized_val == normalized_old:
                binding_updates += 1
                changed = True
                return normalized_new

            prefix = f"result_of.{normalized_old}"
            if normalized_val.startswith(prefix):
                suffix = normalized_val[len(prefix) :]
                binding_updates += 1
                changed = True
                return f"result_of.{normalized_new}{suffix}"

        return obj

    patched = _walk(patched)
    summary = {
        "applied": changed,
        "binding_updates": binding_updates,
        "edge_updates": edge_updates,
        "reason": None if changed else "未发现可替换的引用路径",
    }
    return patched, summary


def drop_invalid_references(
    workflow: Mapping[str, Any], *, remove_edges: bool = True
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    """Remove bindings and edges that point to nonexistent nodes or invalid paths."""

    patched = copy.deepcopy(workflow)
    nodes = patched.get("nodes") if isinstance(patched.get("nodes"), list) else []
    node_ids = {n.get("id") for n in nodes if isinstance(n, Mapping) and n.get("id")}

    removed_bindings: List[Dict[str, Any]] = []
    removed_edges: List[Dict[str, Any]] = []
    removed_branches: List[Dict[str, Any]] = []
    changed = False

    def _clean(obj: Any, path: str = "params") -> Any:
        nonlocal changed

        if isinstance(obj, Mapping):
            new_obj = obj if isinstance(obj, MutableMapping) else dict(obj)
            if "__from__" in obj and isinstance(obj.get("__from__"), str):
                raw = obj.get("__from__")
                normalized = normalize_reference_path(raw)
                try:
                    parts = parse_field_path(normalized)
                except Exception:
                    removed_bindings.append({"path": path, "source": raw, "reason": "非法路径"})
                    new_obj.pop("__from__", None)
                    changed = True
                else:
                    if len(parts) >= 2 and parts[0] == "result_of":
                        ref_node = parts[1]
                        if ref_node not in node_ids:
                            removed_bindings.append(
                                {
                                    "path": path,
                                    "source": raw,
                                    "reason": f"节点 {ref_node} 不存在",
                                }
                            )
                            new_obj.pop("__from__", None)
                            changed = True
            for key, value in obj.items():
                child_path = f"{path}.{key}" if path else str(key)
                new_obj[key] = _clean(value, child_path)
            return new_obj

        if isinstance(obj, list):
            return [_clean(value, f"{path}[{idx}]") for idx, value in enumerate(obj)]

        return obj

    patched = _clean(patched)

    for node in nodes:
        if not isinstance(node, MutableMapping):
            continue
        ntype = node.get("type")
        if ntype == "condition":
            for branch_field in ("true_to_node", "false_to_node"):
                target = node.get(branch_field)
                if isinstance(target, str) and target not in node_ids:
                    removed_branches.append(
                        {
                            "node_id": node.get("id"),
                            "field": branch_field,
                            "target": target,
                        }
                    )
                    node[branch_field] = None
                    changed = True
        elif ntype == "switch":
            cases = node.get("cases") if isinstance(node.get("cases"), list) else []
            cleaned_cases = []
            for idx, case in enumerate(cases):
                if not isinstance(case, Mapping):
                    continue
                target = case.get("to_node")
                if isinstance(target, str) and target not in node_ids:
                    removed_branches.append(
                        {
                            "node_id": node.get("id"),
                            "field": f"cases[{idx}].to_node",
                            "target": target,
                        }
                    )
                    changed = True
                    continue
                cleaned_cases.append(case)
            if cleaned_cases or "cases" in node:
                node["cases"] = cleaned_cases
            if "default_to_node" in node:
                default_target = node.get("default_to_node")
                if isinstance(default_target, str) and default_target not in node_ids:
                    removed_branches.append(
                        {
                            "node_id": node.get("id"),
                            "field": "default_to_node",
                            "target": default_target,
                        }
                    )
                    node["default_to_node"] = None
                    changed = True

    if remove_edges:
        edges = []
        for edge in patched.get("edges", []) or []:
            if not isinstance(edge, Mapping):
                continue
            frm = edge.get("from")
            to = edge.get("to")
            if frm not in node_ids or to not in node_ids:
                removed_edges.append({"from": frm, "to": to})
                changed = True
                continue
            edges.append(edge)
        if edges or "edges" in patched:
            patched["edges"] = edges

    summary = {
        "applied": changed,
        "removed_bindings": removed_bindings,
        "removed_edges": removed_edges,
        "removed_branches": removed_branches,
        "reason": None if changed else "未发现指向缺失节点的引用",
    }
    return patched, summary


def repair_loop_body_references(
    workflow: Mapping[str, Any], node_id: str
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    patched = copy.deepcopy(workflow)
    target = next(
        (n for n in patched.get("nodes", []) if isinstance(n, Mapping) and n.get("id") == node_id),
        None,
    )
    if not isinstance(target, Mapping):
        return patched, {"applied": False, "reason": f"节点 {node_id} 未找到"}

    params = target.get("params")
    if not isinstance(params, MutableMapping):
        return patched, {"applied": False, "reason": f"节点 {node_id} 缺少 params"}

    body = params.get("body_subgraph")
    if not isinstance(body, MutableMapping):
        return patched, {"applied": False, "reason": "body_subgraph 不是对象"}

    nodes: List[Mapping[str, Any]] = [
        bn for bn in body.get("nodes", []) if isinstance(bn, Mapping) and bn.get("id")
    ]
    changed = False

    # 移除无效的 entry/exit/edges 字段，loop.body_subgraph 只需要 nodes
    for key in ["entry", "exit", "edges"]:
        if key in body:
            body.pop(key, None)
            changed = True

    return patched, {"applied": changed, "reason": None if changed else "无可修复项"}


def fill_action_required_params(
    workflow: Mapping[str, Any], *, node_id: str, action_registry: Iterable[Mapping[str, Any]]
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    patched = copy.deepcopy(workflow)
    nodes = patched.get("nodes", [])
    target = next((n for n in nodes if isinstance(n, Mapping) and n.get("id") == node_id), None)
    if not isinstance(target, MutableMapping):
        return patched, {"applied": False, "reason": f"节点 {node_id} 未找到"}

    if target.get("type") != "action":
        return patched, {"applied": False, "reason": "仅支持 action 节点"}

    action_id = target.get("action_id")
    actions_by_id = _index_actions_by_id(action_registry)
    action_def = actions_by_id.get(action_id)
    if not action_def:
        return patched, {"applied": False, "reason": "action_id 未注册"}

    arg_schema = action_def.get("arg_schema") or {}
    required_fields = arg_schema.get("required") if isinstance(arg_schema, Mapping) else []
    properties = arg_schema.get("properties") if isinstance(arg_schema, Mapping) else {}
    if not isinstance(properties, Mapping):
        return patched, {"applied": False, "reason": "缺少 arg_schema.properties"}

    params = target.get("params") if isinstance(target.get("params"), MutableMapping) else {}
    if "params" not in target:
        target["params"] = params

    changed = False
    coerced_fields: List[str] = []
    filled_fields: List[str] = []

    for field in required_fields or []:
        schema = properties.get(field)
        if field not in params:
            params[field] = _schema_default_value(schema or {})
            changed = True
            filled_fields.append(field)
        else:
            coerced = _coerce_value_to_schema_type(params[field], schema or {})
            if coerced is not None and coerced != params[field]:
                params[field] = coerced
                changed = True
                coerced_fields.append(field)

    summary = {
        "applied": changed,
        "filled_fields": filled_fields,
        "coerced_fields": coerced_fields,
        "reason": None if changed else "未发现缺失字段",
    }
    return patched, summary


def update_node_field(
    workflow: Mapping[str, Any], *, node_id: str, field_path: str, value: Any
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    patched = copy.deepcopy(workflow)
    node = next(
        (n for n in patched.get("nodes", []) if isinstance(n, MutableMapping) and n.get("id") == node_id),
        None,
    )
    if not isinstance(node, MutableMapping):
        return patched, {"applied": False, "reason": f"节点 {node_id} 未找到"}

    try:
        _set_nested_field(node, field_path, value)
        return patched, {"applied": True, "reason": None}
    except Exception as exc:  # noqa: BLE001
        return patched, {"applied": False, "reason": str(exc)}


def apply_repair_tool(
    *,
    tool_name: str,
    args: Mapping[str, Any],
    workflow: Mapping[str, Any],
    validation_errors: Iterable[ValidationError],
    action_registry: Iterable[Mapping[str, Any]],
) -> Tuple[Mapping[str, Any], Dict[str, Any]]:
    before_hash = _workflow_fingerprint(workflow)
    node_id = args.get("node_id") if isinstance(args, Mapping) else None

    if tool_name == "fix_loop_body_references":
        patched, summary = repair_loop_body_references(
            workflow, node_id=node_id or ""
        )
    elif tool_name == "fill_action_required_params":
        patched, summary = fill_action_required_params(
            workflow, node_id=node_id or "", action_registry=action_registry
        )
    elif tool_name == "update_node_field":
        patched, summary = update_node_field(
            workflow,
            node_id=node_id or "",
            field_path=str(args.get("field_path", "")),
            value=args.get("value"),
        )
    elif tool_name == "normalize_binding_paths":
        patched, summary = normalize_binding_paths(workflow)
    elif tool_name == "replace_reference_paths":
        old = args.get("old") if isinstance(args, Mapping) else None
        new = args.get("new") if isinstance(args, Mapping) else None
        if not old or not new:
            patched, summary = workflow, {
                "applied": False,
                "reason": "old/new 参数不能为空",
            }
        else:
            patched, summary = replace_reference_paths(
                workflow,
                old=str(old),
                new=str(new),
                include_edges=bool(args.get("include_edges", True)),
            )
    elif tool_name == "drop_invalid_references":
        patched, summary = drop_invalid_references(
            workflow, remove_edges=bool(args.get("remove_edges", True))
        )
    else:
        patched, summary = workflow, {"applied": False, "reason": f"未知工具 {tool_name}"}

    after_hash = _workflow_fingerprint(patched)
    related_errors = _collect_related_errors(validation_errors, node_id)

    log_event(
        "repair_tool_call",
        {
            "tool": tool_name,
            "args": args,
            "node_id": node_id,
            "workflow_before": before_hash,
            "workflow_after": after_hash,
            "related_errors": related_errors,
            "summary": summary,
        },
    )

    if summary.get("applied"):
        log_info(
            f"[RepairTool] {tool_name} 已应用到节点 {node_id or '<unknown>'}，变更摘要: {summary}"
        )
    else:
        log_warn(f"[RepairTool] {tool_name} 未修改 workflow：{summary.get('reason')}")

    return patched, summary

__all__ = [
    "fill_loop_exports_defaults",
    "normalize_binding_paths",
    "apply_repair_tool",
    "fill_action_required_params",
    "_apply_local_repairs_for_unknown_params",
    "normalize_binding_paths",
    "fix_missing_loop_exports_items",
    "replace_reference_paths",
    "drop_invalid_references",
    "fix_loop_body_references",
    "align_loop_body_alias_references",
    "update_node_field",
]
