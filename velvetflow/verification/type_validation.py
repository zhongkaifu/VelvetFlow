# velvetflow/verification/type_validation.py
from __future__ import annotations
import re
from typing import List, Dict, Any, Iterable

from velvetflow.models import ValidationError

from velvetflow.type_system import (
    TypeEnvironment,
    TypeRef,
    infer_type_from_from_binding,
    infer_type_from_agg,
    infer_type_from_loop,
    WorkflowTypeValidationError,
)


def _action_id(action_def: Dict[str, Any]) -> str:
    return action_def.get("action_id") or action_def.get("id")


def _matches_json_type(value: Any, schema_type: Any) -> bool:
    types: Iterable[Any] = schema_type if isinstance(schema_type, list) else [schema_type]

    for t in types:
        if t is None:
            return True
        if t == "string" and isinstance(value, str):
            return True
        if t == "boolean" and isinstance(value, bool):
            return True
        if t == "integer" and isinstance(value, int) and not isinstance(value, bool):
            return True
        if t == "number" and isinstance(value, (int, float)) and not isinstance(value, bool):
            return True
        if t == "array" and isinstance(value, list):
            return True
        if t == "object" and isinstance(value, dict):
            return True
    return False


def _value_type_label(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def build_node_output_type(action_def: Dict[str, Any]) -> TypeRef:
    output_schema = action_def.get("output_schema") or {"type": "object"}
    return TypeRef(
        json_schema=output_schema,
        source=f"action:{_action_id(action_def)}:output"
    )


def get_param_expected_type(action_def: Dict[str, Any], param_name: str) -> TypeRef:
    input_schema = action_def.get("arg_schema") or action_def.get("input_schema") or {}
    props = input_schema.get("properties") or {}
    if param_name not in props:
        return None
    return TypeRef(
        json_schema=props[param_name],
        source=f"action:{_action_id(action_def)}:input.{param_name}"
    )


def check_param_type(
    env: TypeEnvironment,
    node_id: str,
    action_def: Dict[str, Any],
    param_name: str,
    param_val: Any,
    errors: List[str],
):
    expected = get_param_expected_type(action_def, param_name)
    if expected is None:
        errors.append(
            f"[SchemaError] node '{node_id}' param '{param_name}' not in action input_schema"
        )
        return

    # __from__
    if isinstance(param_val, dict) and "__from__" in param_val:
        src = infer_type_from_from_binding(env, param_val["__from__"])
        if not src:
            errors.append(
                f"[TypeError] node '{node_id}' param '{param_name}' cannot infer __from__ type"
            )
            return
        if not src.is_compatible_with(expected):
            errors.append(
                f"[TypeError] node '{node_id}' param '{param_name}' expects "
                f"{expected.json_schema.get('type')}, got {src.json_schema.get('type')}"
            )
        return

    # __agg__
    if isinstance(param_val, dict) and "__agg__" in param_val:
        try:
            out = infer_type_from_agg(env, param_val["__agg__"])
        except Exception as e:
            errors.append(
                f"[TypeError] node '{node_id}' param '{param_name}' agg error: {e}"
            )
            return
        if not out.is_compatible_with(expected):
            errors.append(
                f"[TypeError] node '{node_id}' param '{param_name}' expects "
                f"{expected.json_schema.get('type')}, agg type {out.json_schema.get('type')}"
            )
        return

    # loop
    if isinstance(param_val, dict) and "loop" in param_val:
        try:
            out = infer_type_from_loop(env, param_val["loop"])
        except Exception as e:
            errors.append(
                f"[TypeError] node '{node_id}' param '{param_name}' loop error: {e}"
            )
            return
        if not out.is_compatible_with(expected):
            errors.append(
                f"[TypeError] node '{node_id}' param '{param_name}' expects "
                f"{expected.json_schema.get('type')}, loop type {out.json_schema.get('type')}"
            )
        return

    expected_type = expected.json_schema.get("type")
    if expected_type is not None and not _matches_json_type(param_val, expected_type):
        errors.append(
            f"[TypeError] node '{node_id}' param '{param_name}' expects "
            f"{expected_type}, got {_value_type_label(param_val)}"
        )


def validate_workflow_types(workflow, action_registry) -> None:
    errors: List[str] = []
    env = TypeEnvironment(entries={})

    # (1) 记录所有 node.output 类型
    for node in workflow.nodes:
        action_def = action_registry.get(node.action_id)
        if not action_def:
            errors.append(
                f"[ActionError] node '{node.id}' uses unknown action '{node.action_id}'"
            )
            continue
        output_type = build_node_output_type(action_def)
        env.set(f"{node.id}.output", output_type)

    # (2) 检查所有参数绑定
    for node in workflow.nodes:
        action_def = action_registry.get(node.action_id)
        if not action_def:
            continue
        params = node.params or {}
        for k, v in params.items():
            check_param_type(env, node.id, action_def, k, v, errors)

    if errors:
        raise WorkflowTypeValidationError(errors)


def convert_type_errors(type_errors: List[str]) -> List[ValidationError]:
    converted: List[ValidationError] = []

    for msg in type_errors:
        node_id = None
        field = None

        node_match = re.search(r"node '([^']+)'", msg)
        if node_match:
            node_id = node_match.group(1)

        param_match = re.search(r"param '([^']+)'", msg)
        if param_match:
            field = f"params.{param_match.group(1)}"

        converted.append(
            ValidationError(
                code="SCHEMA_MISMATCH",
                node_id=node_id,
                field=field,
                message=msg,
            )
        )

    return converted
