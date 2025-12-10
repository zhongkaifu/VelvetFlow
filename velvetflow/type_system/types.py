# velvetflow/type_system/types.py
from __future__ import annotations
from typing import Any, Dict, Optional, List
from pydantic import BaseModel


class TypeCheckError(Exception):
    pass


class WorkflowTypeValidationError(Exception):
    def __init__(self, errors: List[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


class TypeRef(BaseModel):
    json_schema: Dict[str, Any]
    source: Optional[str] = None

    def is_compatible_with(self, other: "TypeRef") -> bool:
        t1 = self.json_schema.get("type")
        t2 = other.json_schema.get("type")

        if t1 is None or t2 is None:
            return True

        if isinstance(t1, list) or isinstance(t2, list):
            s1 = set(t1 if isinstance(t1, list) else [t1])
            s2 = set(t2 if isinstance(t2, list) else [t2])
            return len(s1 & s2) > 0

        if t1 != t2:
            return False

        if t1 == "array":
            i1 = self.json_schema.get("items") or {}
            i2 = other.json_schema.get("items") or {}
            return TypeRef(json_schema=i1).is_compatible_with(TypeRef(json_schema=i2))

        return True


class TypeEnvironment(BaseModel):
    entries: Dict[str, TypeRef] = {}

    def set(self, key: str, type_ref: TypeRef) -> None:
        self.entries[key] = type_ref

    def get(self, key: str) -> Optional[TypeRef]:
        return self.entries.get(key)
