# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Node-level validation logic for workflows."""
import re
from typing import Any, Dict, List, Mapping, Optional

from velvetflow.jinja_utils import validate_jinja_expr
from velvetflow.models import ValidationError
from velvetflow.reference_utils import normalize_reference_path, parse_field_path
from velvetflow.loop_dsl import loop_body_has_action

from .binding_checks import (
    _check_array_item_field,
    _check_output_path_against_schema,
    _get_array_item_schema_from_output,
    _get_field_schema,
    _get_field_schema_from_item,
    _get_node_output_schema,
    _get_output_schema_at_path,
    _iter_empty_param_fields,
    _iter_template_references,
    _maybe_decode_binding_string,
    _project_schema_through_agg,
    _schema_path_error,
    _suggest_numeric_subfield,
    _walk_schema_with_tokens,
    validate_param_binding,
)


_SELECT_ATTR_PATTERN = re.compile(
    r"(?P<source>result_of\.[A-Za-z0-9_.-]+)\s*\|\s*selectattr\(\s*['\"](?P<field>[A-Za-z0-9_.-]+)['\"]"
)
_MAP_ATTR_PATTERN = re.compile(
    r"(?P<source>result_of\.[A-Za-z0-9_.-]+)\s*\|\s*map\(\s*attribute\s*=\s*['\"](?P<field>[A-Za-z0-9_.-]+)['\"]"
)


def _strip_jinja_filters(expr: str) -> str:
    """Return the reference path before any top-level Jinja filter pipeline."""

    in_single = False
    in_double = False
    paren_depth = 0

    for idx, ch in enumerate(expr):
        if ch == "'" and not in_double and (idx == 0 or expr[idx - 1] != "\\"):
            in_single = not in_single
            continue
        if ch == '"' and not in_single and (idx == 0 or expr[idx - 1] != "\\"):
            in_double = not in_double
            continue
        if ch == "(" and not in_single and not in_double:
            paren_depth += 1
            continue
        if ch == ")" and not in_single and not in_double and paren_depth > 0:
            paren_depth -= 1
            continue
        if ch == "|" and not in_single and not in_double and paren_depth == 0:
            return expr[:idx].strip()

    return expr.strip()


def _validate_jinja_expression(expr: str, *, node_id: str | None, field: str, errors: List[ValidationError]) -> None:
    try:
        validate_jinja_expr(expr, path=field)
    except Exception as exc:  # pragma: no cover - defensive for unexpected parser errors
        errors.append(
            ValidationError(
                code="INVALID_JINJA_EXPRESSION",
                node_id=node_id,
                field=field,
                message=f"Jinja expression could not be parsed: {exc}",
            )
        )


def _schema_contains_field(schema: Mapping[str, Any] | None, field: str) -> bool:
    if not isinstance(schema, Mapping) or not field:
        return False

    parts = [token for token in field.split(".") if token]
    current: Mapping[str, Any] | None = schema
    for token in parts:
        if not isinstance(current, Mapping):
            return False
        props = current.get("properties") if isinstance(current.get("properties"), Mapping) else None
        if not props or token not in props:
            return False
        next_schema = props.get(token)
        if isinstance(next_schema, Mapping) and next_schema.get("type") == "array":
            current = next_schema.get("items") if isinstance(next_schema.get("items"), Mapping) else None
        else:
            current = next_schema if isinstance(next_schema, Mapping) else None
    return True


def _collect_collection_attribute_refs(expr: str) -> List[Dict[str, str]]:
    checks: List[Dict[str, str]] = []
    for pattern in (_SELECT_ATTR_PATTERN, _MAP_ATTR_PATTERN):
        for match in pattern.finditer(expr):
            source = match.group("source")
            field = match.group("field")
            if source and field:
                checks.append({"source": source, "field": field})
    return checks

CONDITION_PARAM_FIELDS = {"expression"}

LOOP_PARAM_FIELDS = {
    "loop_kind",
    "source",
    "condition",
    "item_alias",
    "body_subgraph",
    "exports",
}


def _filter_params_by_supported_fields(
    *,
    node: Mapping[str, Any],
    actions_by_id: Mapping[str, Mapping[str, Any]],
) -> List[str]:
    params = node.get("params")
    if not isinstance(params, Mapping):
        return []

    node_type = node.get("type")
    allowed_fields: set[str] | None = None

    if node_type == "condition":
        allowed_fields = set(CONDITION_PARAM_FIELDS)
    elif node_type == "loop":
        allowed_fields = set(LOOP_PARAM_FIELDS)
    elif node_type == "action":
        action_id = node.get("action_id")
        action_def = actions_by_id.get(action_id) if isinstance(action_id, str) else None
        properties = (action_def or {}).get("arg_schema", {}).get("properties")
        if isinstance(properties, Mapping):
            allowed_fields = set(properties.keys())

    if not allowed_fields:
        return []

    removed = [key for key in params if key not in allowed_fields]
    if removed:
        node_params = {k: v for k, v in params.items() if k in allowed_fields}
        node["params"] = node_params

    return removed


def _is_self_reference_path(path: Any, node_id: str | None) -> bool:
    """Return True if ``path`` points to ``result_of.<node_id>`` (self-cycle)."""

    if not isinstance(path, str) or not node_id:
        return False

    normalized = normalize_reference_path(path)
    if not isinstance(normalized, str):
        return False

    try:
        parts = parse_field_path(normalized)
    except Exception:
        return False

    return len(parts) >= 2 and parts[0] == "result_of" and parts[1] == node_id


def _resolve_condition_schema(
    source_path: str,
    field: str | None,
    nodes_by_id: Mapping[str, Mapping[str, Any]],
    actions_by_id: Mapping[str, Mapping[str, Any]],
    loop_body_parents: Mapping[str, str],
    alias_schemas: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    """Return the schema of the condition target (source + optional field)."""

    normalized_source = normalize_reference_path(source_path)
    field_path = field if isinstance(field, str) and field else None

    alias_schema = (alias_schemas or {}).get(normalized_source)
    if isinstance(alias_schema, Mapping):
        if field_path:
            try:
                tokens = parse_field_path(field_path)
            except Exception:
                return None
            return _walk_schema_with_tokens(alias_schema, tokens)
        return alias_schema

    if not isinstance(normalized_source, str):
        return None

    combined_path = (
        f"{normalized_source}.{field_path}" if field_path else normalized_source
    )
    return _get_output_schema_at_path(
        combined_path, nodes_by_id, actions_by_id, loop_body_parents
    )


def _strip_illegal_exports(node: Mapping[str, Any]) -> bool:
    """Remove ``params.exports`` from non-loop nodes to avoid repeated errors."""

    if node.get("type") == "loop":
        return False

    params = node.get("params")
    if not isinstance(params, Mapping) or "exports" not in params:
        return False

    new_params = {k: v for k, v in params.items() if k != "exports"}
    node["params"] = new_params
    return True


def _validate_nodes_recursive(
    nodes: List[Mapping[str, Any]],
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
    loop_body_parents: Mapping[str, str],
    errors: List[ValidationError],
    alias_schemas: Optional[Dict[str, Mapping[str, Any]]] = None,
):
    """Validate nodes (and nested loop body nodes) against planner rules."""

    node_ids = set(nodes_by_id.keys())

    for n in nodes or []:
        if not isinstance(n, Mapping):
            continue

        if hasattr(errors, "set_context"):
            errors.set_context({"node": n})

        nid = n.get("id")
        ntype = n.get("type")
        action_id = n.get("action_id")
        params = n.get("params", {})

        def _flag_self_reference(field_path: str, ref: str) -> None:
            errors.append(
                ValidationError(
                    code="SELF_REFERENCE",
                    node_id=nid,
                    field=field_path,
                    message=(
                        f"Field '{field_path}' on node '{nid}' self-references {ref}, which creates a cycle. "
                        "Share this context with the LLM and use the available tools to point to an upstream output "
                        "or split the node."
                    ),
                )
            )

        if ntype == "action" and (not isinstance(params, Mapping) or len(params) == 0):
            errors.append(
                ValidationError(
                    code="EMPTY_PARAMS",
                    node_id=nid,
                    field="params",
                    message=(
                        "The action node has empty params; ask the LLM to analyze the cause and fill required fields or bindings using the tools."
                    ),
                )
            )

        empty_fields = list(_iter_empty_param_fields(params))
        for path in empty_fields:
            context_parts = []
            if ntype:
                context_parts.append(f"type={ntype}")
            if action_id:
                context_parts.append(f"action_id={action_id}")
            context_hint = f"（{', '.join(context_parts)}）" if context_parts else ""

            errors.append(
                ValidationError(
                    code="EMPTY_PARAM_VALUE",
                    node_id=nid,
                    field=path,
                    message=(
                        f"Parameter '{path}' on node '{nid}' is empty; fill the real value or binding based on context {context_hint}. "
                        "Use the input/output schema and tools to repair it."
                    ),
                )
            )

        # exports is only allowed on loop nodes
        if "exports" in params and ntype != "loop":
            errors.append(
                ValidationError(
                    code="INVALID_SCHEMA",
                    node_id=nid,
                    field="exports",
                    message="'exports' is only allowed on loop nodes to expose subgraph results.",
                )
            )

        # 1) action node
        if ntype == "action" and action_id:
            action_def = actions_by_id.get(action_id)
            if not action_def:
                errors.append(
                    ValidationError(
                        code="UNKNOWN_ACTION_ID",
                        node_id=nid,
                        field="action_id",
                        message=f"Node '{nid}' uses action_id '{action_id}' which is not in the Action Registry.",
                    )
                )
            else:
                schema = action_def.get("arg_schema") or {}
                required_fields = (schema.get("required") or []) if isinstance(schema, dict) else []
                properties = schema.get("properties") if isinstance(schema, Mapping) else None
                allow_additional = bool(schema.get("additionalProperties")) if isinstance(schema, Mapping) else False

                if not isinstance(params, dict) or len(params) == 0:
                    if required_fields:
                        for field in required_fields:
                            errors.append(
                                ValidationError(
                                    code="MISSING_REQUIRED_PARAM",
                                    node_id=nid,
                                    field=field,
                                    message=(
                                        f"Action node '{nid}' has empty params, but action '{action_id}' requires field '{field}'."
                                    ),
                                )
                            )
                else:
                    for field in required_fields:
                        if field not in params:
                            errors.append(
                                ValidationError(
                                    code="MISSING_REQUIRED_PARAM",
                                    node_id=nid,
                                    field=field,
                                    message=(
                                        f"Action node '{nid}' params are missing required field '{field}' (action_id='{action_id}')."
                                    ),
                                )
                            )

                    if isinstance(properties, Mapping) and not allow_additional:
                        unknown_fields = [k for k in params if k not in properties]
                        for field in unknown_fields:
                            errors.append(
                                ValidationError(
                                    code="UNKNOWN_PARAM",
                                    node_id=nid,
                                    field=field,
                                    message=(
                                        f"Parameter '{field}' on action node '{nid}' is not defined in action '{action_id}' arg_schema."
                                    ),
                                )
                            )

            # Static validation for binding DSL
            def _walk_params_for_from(obj: Any, path_prefix: str = ""):
                if isinstance(obj, dict):
                    if "__from__" in obj:
                        schema_err = validate_param_binding(obj)
                        if schema_err:
                            field_label = path_prefix or "params"
                            if path_prefix and not path_prefix.startswith("params"):
                                field_label = f"params.{path_prefix}"
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field=field_label,
                                    message=f"Parameter binding ({path_prefix or '<root>'}) on action node '{nid}' is invalid: {schema_err}",
                                )
                            )

                        source = obj.get("__from__")
                        sources: List[str] = []

                        if isinstance(source, str):
                            sources = [source]
                        elif isinstance(source, list):
                            for idx, item in enumerate(source):
                                if isinstance(item, str):
                                    sources.append(item)
                                else:
                                    errors.append(
                                        ValidationError(
                                            code="INVALID_SCHEMA",
                                            node_id=nid,
                                            field=path_prefix or "params",
                                            message=(
                                                f"Parameter binding ({path_prefix or '<root>'}) on action node '{nid}' "
                                                f"has invalid __from__[{idx}] type; expected string."
                                            ),
                                        )
                                    )

                        for src_idx, src in enumerate(sources):
                            if isinstance(src, str) and src != normalize_reference_path(src):
                                # Templated references are resolved at runtime; skip strict checks.
                                continue
                            if _is_self_reference_path(src, nid):
                                field_label = path_prefix or "params"
                                if path_prefix and not path_prefix.startswith("params"):
                                    field_label = f"params.{path_prefix}"
                                _flag_self_reference(field_label, src)
                            schema_err = _check_output_path_against_schema(
                                src,
                                nodes_by_id,
                                actions_by_id,
                                loop_body_parents,
                                context_node_id=nid,
                            )
                            if schema_err:
                                suffix = f"[{src_idx}]" if len(sources) > 1 else ""
                                field_label = path_prefix or "params"
                                if path_prefix and not path_prefix.startswith("params"):
                                    field_label = f"params.{path_prefix}"
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=field_label,
                                        message=(
                                            f"Parameter binding ({path_prefix or '<root>'}) on action node '{nid}' has an invalid reference"
                                            f"{suffix}: {schema_err}"
                                        ),
                                    )
                                )

                        agg_spec = obj.get("__agg__")
                        agg_op = agg_spec.get("op") if isinstance(agg_spec, Mapping) else agg_spec
                        if agg_op == "pipeline":
                            steps = obj.get("steps", [])
                            if isinstance(steps, list):
                                for idx, step in enumerate(steps):
                                    if not isinstance(step, Mapping):
                                        continue
                                    if step.get("op") == "filter":
                                        fld = step.get("field")
                                        for src_idx, src in enumerate(sources or [source]):
                                            if not isinstance(src, str):
                                                continue
                                            item_err = _check_array_item_field(
                                                src,
                                                fld,
                                                nodes_by_id,
                                                actions_by_id,
                                                loop_body_parents,
                                                alias_schemas,
                                            )
                                            if item_err:
                                                suffix = f"[{src_idx}]" if len(sources) > 1 else ""
                                                errors.append(
                                                    ValidationError(
                                                        code="SCHEMA_MISMATCH",
                                                        node_id=nid,
                                                        field=f"{path_prefix or 'params'}.pipeline.steps[{idx}].field",
                                                        message=(
                                                            f"Parameter binding ({path_prefix or '<root>'}) on action node '{nid}' has pipeline.steps[{idx}].field='{fld}'"
                                                            f" is invalid{suffix}: {item_err}"
                                                        ),
                                                    )
                                                )

                    for k, v in list(obj.items()):
                        new_prefix = f"{path_prefix}.{k}" if path_prefix else k
                        _walk_params_for_from(v, new_prefix)
                elif isinstance(obj, list):
                    for idx, v in enumerate(obj):
                        new_prefix = f"{path_prefix}[{idx}]"
                        _walk_params_for_from(v, new_prefix)
                elif isinstance(obj, str):
                    decoded = _maybe_decode_binding_string(obj)
                    if decoded:
                        _walk_params_for_from(decoded, path_prefix)

            _walk_params_for_from(params)

            def _walk_params_for_templates(obj: Any, path_prefix: str = "") -> None:
                if isinstance(obj, str):
                    for ref in _iter_template_references(obj):
                        _validate_jinja_expression(ref, node_id=nid, field=path_prefix or "params", errors=errors)

                        ref_head = _strip_jinja_filters(ref)
                        if not ref_head:
                            continue
                        ref_path = normalize_reference_path(ref_head)
                        try:
                            ref_parts = parse_field_path(ref_path)
                        except Exception:
                            continue

                        missing_target_node = False
                        if ref_path.startswith("result_of."):
                            target_node = None
                            if ref_parts and len(ref_parts) >= 2 and isinstance(ref_parts[1], str):
                                target_node = ref_parts[1]

                            if target_node and target_node not in nodes_by_id:
                                missing_target_node = True
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field=path_prefix,
                                        message=(
                                            f"Template reference '{ref}' on action node '{nid}' is invalid: "
                                            f"Referenced node '{target_node}' does not exist."
                                        ),
                                    )
                                )

                            if missing_target_node:
                                continue

                        if _is_self_reference_path(ref_path, nid):
                            field_label = path_prefix or "params"
                            if path_prefix and not path_prefix.startswith("params"):
                                field_label = f"params.{path_prefix}"
                            _flag_self_reference(field_label, ref_path)

                        if ref_parts and isinstance(ref_parts[0], str):
                            loop_ctx_root = ref_parts[0]
                            loop_node = nodes_by_id.get(loop_ctx_root)
                            if isinstance(loop_node, Mapping) and loop_node.get("type") == "loop":
                                loop_params = loop_node.get("params") if isinstance(loop_node.get("params"), Mapping) else {}
                                loop_item_alias = loop_params.get("item_alias") if isinstance(loop_params.get("item_alias"), str) else None
                                if len(ref_parts) >= 2 and isinstance(ref_parts[1], str):
                                    root_field = ref_parts[1]
                                    allowed_loop_fields = {"index", "size", "accumulator"}
                                    if loop_item_alias:
                                        allowed_loop_fields.add(loop_item_alias)
                                    # Allow using item directly only when item_alias is missing or equals 'item'
                                    if root_field == "item" and loop_item_alias and loop_item_alias != "item":
                                        errors.append(
                                            ValidationError(
                                                code="SCHEMA_MISMATCH",
                                                node_id=nid,
                                                field=path_prefix,
                                                message=(
                                                    f"Template reference '{ref}' on node '{nid}' is invalid: loop node '{loop_ctx_root}' currently exposes item_alias '{loop_item_alias}',"
                                                    "Use the alias instead of '.item' to access loop elements."
                                                ),
                                            )
                                        )
                                        continue
                                    if isinstance(root_field, str) and root_field not in allowed_loop_fields and not (root_field == "item" and loop_item_alias in {None, "item"}):
                                        errors.append(
                                            ValidationError(
                                                code="SCHEMA_MISMATCH",
                                                node_id=nid,
                                                field=path_prefix,
                                                message=(
                                                    f"Template reference '{ref}' on node '{nid}' is invalid: loop node '{loop_ctx_root}' context only exposes {', '.join(sorted(allowed_loop_fields | {'item'}))},"
                                                    f"Field '{root_field}' was not found."
                                                ),
                                            )
                                        )
                                        continue

                        schema_err = None
                        if ref_parts:
                            alias = ref_parts[0]
                            if alias_schemas and alias in alias_schemas:
                                schema_err = _schema_path_error(alias_schemas[alias], ref_parts[1:])
                            else:
                                schema_err = _check_output_path_against_schema(
                                    ref_path,
                                    nodes_by_id,
                                    actions_by_id,
                                    loop_body_parents,
                                    context_node_id=nid,
                                )

                        if schema_err:
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field=path_prefix,
                                    message=(
                                        f"Template reference '{ref}' on action node '{nid}' is invalid: {schema_err}"
                                    ),
                                )
                            )
                elif isinstance(obj, Mapping):
                    for key, value in list(obj.items()):
                        new_prefix = f"{path_prefix}.{key}" if path_prefix else str(key)
                        _walk_params_for_templates(value, new_prefix)
                elif isinstance(obj, list):
                    for idx, value in enumerate(obj):
                        new_prefix = f"{path_prefix}[{idx}]"
                        _walk_params_for_templates(value, new_prefix)

            _walk_params_for_templates(params)

        # 2) condition node
        if ntype == "condition":
            true_to_node = n.get("true_to_node")
            false_to_node = n.get("false_to_node")
            true_branch_empty = true_to_node is None or (
                isinstance(true_to_node, str) and not true_to_node.strip()
            )
            false_branch_empty = false_to_node is None or (
                isinstance(false_to_node, str) and not false_to_node.strip()
            )
            if true_branch_empty and false_branch_empty:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="true_to_node/false_to_node",
                        message=(
                            "condition node true_to_node and false_to_node cannot both be empty."
                            "Ask the LLM to analyze and use the tools to fix this error."
                        ),
                    )
                )


            expr_val = params.get("expression") if isinstance(params, Mapping) else None
            if not isinstance(expr_val, str) or not expr_val.strip():
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="expression",
                        message=(
                            "condition node requires a Jinja expression that returns a boolean value."
                            "Use params.expression to specify the logic; other fields (kind/source/field/threshold, etc.) are deprecated."
                        ),
                    )
                )
            else:
                expression_checks = _collect_collection_attribute_refs(expr_val)
                # If the expression is wrapped in a template (e.g., {{ ... }}), also check inner references
                for templated_ref in _iter_template_references(expr_val):
                    expression_checks.extend(_collect_collection_attribute_refs(templated_ref))

                if expression_checks:
                    for ref in expression_checks:
                        source = normalize_reference_path(ref.get("source", ""))
                        field = ref.get("field")
                        item_schema = _get_array_item_schema_from_output(
                            source,
                            nodes_by_id,
                            actions_by_id,
                            loop_body_parents,
                            context_node_id=nid,
                        )
                        if not item_schema:
                            continue

                        if not _schema_contains_field(item_schema, field or ""):
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field="expression",
                                    message=(
                                        f"The condition expression references attribute '{field}' of {source}, which is not declared in the upstream output schema."
                                        "Adjust the fields exposed in exports or modify the expression to match available fields."
                                    ),
                                )
                            )
        # 3) loop node
        if ntype == "loop":
            body_graph = params.get("body_subgraph") if isinstance(params, Mapping) else None
            body_exports = body_graph.get("exports") if isinstance(body_graph, Mapping) else None
            body_nodes = body_graph.get("nodes") if isinstance(body_graph, Mapping) else None
            if "exports" in params and not isinstance(body_graph, Mapping):
                errors.append(
                    ValidationError(
                        code="INVALID_SCHEMA",
                        node_id=nid,
                        field="exports",
                        message="loop nodes must provide body_subgraph when defining exports.",
                    )
                )
            if isinstance(body_exports, Mapping):
                errors.append(
                    ValidationError(
                        code="INVALID_SCHEMA",
                        node_id=nid,
                        field="body_subgraph.exports",
                        message="loop.exports should be defined in params.exports; remove it from body_subgraph.",
                    )
                )

            if isinstance(body_nodes, list) and body_nodes and not loop_body_has_action(body_graph):
                errors.append(
                    ValidationError(
                        code="INVALID_LOOP_BODY",
                        node_id=nid,
                        field="body_subgraph.nodes",
                        message=(
                            "The loop node's body_subgraph needs at least one action node."
                            "Use planning/repair tools to add executable steps."
                        ),
                    )
                )

            loop_kind = (params or {}).get("loop_kind")
            allowed_loop_kinds = {"foreach", "for_each", "while"}
            if not loop_kind:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="loop_kind",
                        message=f"loop node '{nid}' requires params.loop_kind to be non-empty.",
                    )
                )
            elif loop_kind not in allowed_loop_kinds:
                errors.append(
                    ValidationError(
                        code="SCHEMA_MISMATCH",
                        node_id=nid,
                        field="loop_kind",
                        message=f"loop node '{nid}' uses unsupported loop_kind='{loop_kind}'.",
                    )
                )

            source = params.get("source")
            item_alias = params.get("item_alias")
            if not source:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="source",
                        message=f"loop node '{nid}' requires a source field to specify loop input.",
                    )
                )
            elif isinstance(source, str):
                if _is_self_reference_path(source, nid):
                    _flag_self_reference("source", source)
                src_err = _check_output_path_against_schema(
                    source,
                    nodes_by_id,
                    actions_by_id,
                    loop_body_parents,
                    context_node_id=nid,
                )
                if src_err:
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=nid,
                            field="source",
                            message=f"loop node '{nid}' has an invalid source: {src_err}",
                        )
                    )
                else:
                    normalized_source = normalize_reference_path(source)
                    container_schema = _get_output_schema_at_path(
                        normalized_source,
                        nodes_by_id,
                        actions_by_id,
                        loop_body_parents,
                    )
                    actual_type = (
                        container_schema.get("type")
                        if isinstance(container_schema, Mapping)
                        else None
                    )
                    if actual_type not in {"array"}:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="source",
                                message=(
                                    f"loop node '{nid}' source should reference an array/sequence"
                                    f", but resolved type is {actual_type or 'unknown'}, path: {normalized_source}"
                                ),
                            )
                        )

                    if isinstance(item_alias, str):
                        nested_schema = _get_array_item_schema_from_output(
                            normalized_source,
                            nodes_by_id,
                            actions_by_id,
                            loop_body_parents,
                            context_node_id=nid,
                        )

                        if nested_schema:
                            alias_schemas = dict(alias_schemas or {})
                            alias_schemas[normalize_reference_path(item_alias)] = (
                                nested_schema
                            )
            elif isinstance(source, Mapping):
                src_err = validate_param_binding(source)
                if src_err:
                    errors.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=nid,
                            field="source",
                            message=f"loop node '{nid}' has an invalid source: {src_err}",
                        )
                    )
                elif isinstance(source.get("__from__"), str):
                    normalized_source = normalize_reference_path(source["__from__"])
                    if _is_self_reference_path(normalized_source, nid):
                        _flag_self_reference("source", normalized_source)
                    schema_err = _check_output_path_against_schema(
                        normalized_source,
                        nodes_by_id,
                        actions_by_id,
                        loop_body_parents,
                        context_node_id=nid,
                    )
                    if schema_err:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="source",
                                message=f"loop node '{nid}' has an invalid source reference: {schema_err}",
                            )
                        )
                    else:
                        container_schema = _get_output_schema_at_path(
                            normalized_source,
                            nodes_by_id,
                            actions_by_id,
                            loop_body_parents,
                        )
                        actual_type = (
                            container_schema.get("type")
                            if isinstance(container_schema, Mapping)
                            else None
                        )
                        if actual_type not in {"array"}:
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field="source",
                                    message=(
                                        f"loop node '{nid}' source should reference an array/sequence"
                                        f", but resolved type is {actual_type or 'unknown'}, path: {normalized_source}"
                                    ),
                                )
                            )

                        nested_schema = _get_array_item_schema_from_output(
                            normalized_source,
                            nodes_by_id,
                            actions_by_id,
                            loop_body_parents,
                            context_node_id=nid,
                        )
                        if nested_schema:
                            alias_schemas = dict(alias_schemas or {})
                            alias_schemas[normalize_reference_path(item_alias or "")] = nested_schema
            else:
                errors.append(
                    ValidationError(
                        code="INVALID_SCHEMA",
                        node_id=nid,
                        field="source",
                        message="loop node source must be a string or binding object.",
                    )
                )

            if not isinstance(item_alias, str):
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="item_alias",
                        message=f"loop node '{nid}' requires item_alias to reference the current element in the loop body.",
                    )
                )

            if loop_kind == "while" and not params.get("condition"):
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="condition",
                        message=f"loop node '{nid}' requires condition for a while loop.",
                    )
                )

            exports = params.get("exports")
            if exports is not None and not isinstance(exports, Mapping):
                errors.append(
                    ValidationError(
                        code="SCHEMA_MISMATCH",
                        node_id=nid,
                        field="exports",
                        message=f"loop node '{nid}' exports must be an object.",
                    )
                )
            elif isinstance(exports, Mapping):
                body_node_ids: set[str] = set()
                body_nodes = (params.get("body_subgraph") or {}).get("nodes", [])
                for bn in body_nodes:
                    if isinstance(bn, Mapping) and isinstance(bn.get("id"), str):
                        body_node_ids.add(bn.get("id"))
                for key, value in exports.items():
                    if not isinstance(key, str):
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="exports",
                                message=f"loop node '{nid}' exports keys must be strings.",
                            )
                        )
                        continue
                    if not isinstance(value, str) or not value.strip():
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field=f"exports.{key}",
                                message=f"loop node '{nid}' exports.{key} must be a non-empty Jinja expression string.",
                            )
                        )
                        continue
                    refs = list(_iter_template_references(value))
                    if not refs:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field=f"exports.{key}",
                                message=f"loop node '{nid}' exports.{key} must reference output fields from body_subgraph nodes.",
                            )
                        )
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
                        if isinstance(ref_node, str) and ref_node in body_node_ids:
                            has_body_ref = True
                        else:
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field=f"exports.{key}",
                                    message=(
                                        f"loop node '{nid}' exports.{key} may only reference outputs of nodes within body_subgraph."
                                    ),
                                )
                            )
                            break
                    if not has_body_ref:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field=f"exports.{key}",
                                message=f"loop node '{nid}' exports.{key} must reference output fields from body_subgraph nodes.",
                            )
                        )
                        continue
                    expr = normalize_reference_path(value)
                    _validate_jinja_expression(expr, node_id=nid, field=f"exports.{key}", errors=errors)

            # Recursively validate body_subgraph
            body_graph = params.get("body_subgraph") if isinstance(params, Mapping) else None
            body_nodes = body_graph.get("nodes") if isinstance(body_graph, Mapping) else None
            if isinstance(body_nodes, list) and body_nodes:
                extended_nodes_by_id = dict(nodes_by_id)
                loop_alias_schemas = dict(alias_schemas or {})
                if isinstance(source, str) and isinstance(item_alias, str):
                    item_schema = _get_array_item_schema_from_output(
                        source, nodes_by_id, actions_by_id, loop_body_parents
                    )
                    if item_schema:
                        loop_alias_schemas[item_alias] = item_schema

                for body_node in body_nodes:
                    if isinstance(body_node, Mapping) and isinstance(body_node.get("id"), str):
                        extended_nodes_by_id[body_node["id"]] = body_node
                _validate_nodes_recursive(
                    body_nodes,
                    extended_nodes_by_id,
                    actions_by_id,
                    loop_body_parents,
                    errors,
                    loop_alias_schemas,
                )

        # 4) parallel node
        if ntype == "parallel":
            branches = params.get("branches") if isinstance(params, dict) else None
            if not isinstance(branches, list) or not branches:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="branches",
                        message=f"parallel node '{nid}' requires a non-empty branches list.",
                    )
                )

        if hasattr(errors, "set_context"):
            errors.set_context(None)


def _is_numeric_schema_type(schema_type: Any) -> bool:
    """Return True if the schema type represents a numeric value."""

    if isinstance(schema_type, list):
        return any(t in {"number", "integer"} for t in schema_type)

    return schema_type in {"number", "integer"}


def _is_array_schema_type(schema_type: Any) -> bool:
    """Return True if the schema type represents an array."""

    if isinstance(schema_type, list):
        return "array" in schema_type

    return schema_type == "array"


def _is_array_or_string_schema_type(schema_type: Any) -> bool:
    """Return True if the schema type represents an array or string."""

    if isinstance(schema_type, list):
        return any(t in {"array", "string"} for t in schema_type)

    return schema_type in {"array", "string"}


__all__ = [
    "CONDITION_PARAM_FIELDS",
    "LOOP_PARAM_FIELDS",
    "_filter_params_by_supported_fields",
    "_is_numeric_schema_type",
    "_is_array_schema_type",
    "_is_array_or_string_schema_type",
    "_strip_illegal_exports",
    "_validate_nodes_recursive",
]
