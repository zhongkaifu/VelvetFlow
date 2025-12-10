# velvetflow/type_system/infer.py
from __future__ import annotations
from typing import Dict, Any, List, Optional

from .types import TypeCheckError, TypeRef, TypeEnvironment


def resolve_path_in_schema(base_type: TypeRef, path_parts: List[str]) -> Optional[TypeRef]:
    schema = dict(base_type.json_schema)
    for part in path_parts:
        if schema.get("type") == "object":
            props = schema.get("properties") or {}
            if part not in props:
                return None
            schema = props[part]
        elif schema.get("type") == "array" and part == "[]":
            schema = schema.get("items") or {}
        else:
            return None

    return TypeRef(json_schema=schema, source=f"{base_type.source}.{'.'.join(path_parts)}")


def infer_type_from_from_binding(
    env: TypeEnvironment,
    from_expr: str,
) -> Optional[TypeRef]:
    parts = from_expr.split(".")
    if len(parts) < 2:
        return None

    base_key = ".".join(parts[:2])
    base_type = env.get(base_key)
    if not base_type:
        return None

    sub_parts = parts[2:]
    if not sub_parts:
        return base_type

    return resolve_path_in_schema(base_type, sub_parts)
