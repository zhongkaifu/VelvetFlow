# velvetflow/type_system/agg.py
from __future__ import annotations
from typing import Dict, Any

from .types import TypeCheckError, TypeRef, TypeEnvironment
from .infer import infer_type_from_from_binding


def infer_type_from_agg(env: TypeEnvironment, agg_expr: Dict[str, Any]) -> TypeRef:
    op = agg_expr.get("op")
    if not op:
        raise TypeCheckError("agg.op missing")

    input_expr = agg_expr.get("input")
    if not input_expr:
        raise TypeCheckError("agg.input missing")

    if "__from__" not in input_expr:
        raise TypeCheckError("agg.input only supports __from__")

    input_type = infer_type_from_from_binding(env, input_expr["__from__"])
    if not input_type:
        raise TypeCheckError(f"agg.input __from__ '{input_expr['__from__']}' cannot infer type")

    if input_type.json_schema.get("type") != "array":
        raise TypeCheckError(f"agg '{op}' expects array input")

    element_schema = input_type.json_schema.get("items") or {}
    element_type = TypeRef(json_schema=element_schema, source=f"{input_type.source}[]")

    # filter_map
    if op == "filter_map":
        map_expr = agg_expr.get("map")
        if not map_expr:
            raise TypeCheckError("filter_map missing 'map'")

        field = map_expr.get("field")
        if not field:
            raise TypeCheckError("filter_map.map missing 'field'")

        props = element_schema.get("properties") or {}
        if field not in props:
            raise TypeCheckError(
                f"filter_map.map.field '{field}' not found in element schema"
            )

        return TypeRef(
            json_schema={"type": "array", "items": props[field]},
            source=f"agg:filter_map({input_type.source} -> {field})"
        )

    raise TypeCheckError(f"unsupported agg op '{op}'")
