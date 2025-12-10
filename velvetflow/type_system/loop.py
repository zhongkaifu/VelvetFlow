# velvetflow/type_system/loop.py

from __future__ import annotations
from typing import Dict, Any

from .types import TypeCheckError, TypeRef, TypeEnvironment
from .infer import infer_type_from_from_binding


def parse_type_string_to_schema(type_str: str) -> Dict[str, Any]:
    type_str = type_str.strip()

    if type_str.startswith("list<") and type_str.endswith(">"):
        inner = type_str[5:-1].strip()
        return {
            "type": "array",
            "items": parse_type_string_to_schema(inner),
        }

    if type_str in {"string", "number", "integer", "boolean"}:
        return {"type": type_str}

    return {"type": "object"}


def infer_type_from_loop(env: TypeEnvironment, loop_expr: Dict[str, Any]) -> TypeRef:
    input_expr = loop_expr.get("input")
    if not input_expr:
        raise TypeCheckError("loop.input missing")

    if "__from__" not in input_expr:
        raise TypeCheckError("loop.input only supports __from__ binding")

    input_type = infer_type_from_from_binding(env, input_expr["__from__"])
    if not input_type:
        raise TypeCheckError(f"loop.input __from__ '{input_expr['__from__']}' cannot infer type")

    if input_type.json_schema.get("type") != "array":
        raise TypeCheckError("loop.input must be array")

    merge_expr = loop_expr.get("merge")
    if not merge_expr:
        raise TypeCheckError("loop.merge missing")

    output_type_str = merge_expr.get("output_type")
    if not output_type_str:
        raise TypeCheckError("loop.merge.output_type missing")

    out_schema = parse_type_string_to_schema(output_type_str)
    return TypeRef(json_schema=out_schema, source=f"loop({input_expr['__from__']})")
