"""Grammar-aware semantic checks for the workflow DSL."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, Optional, Sequence

from velvetflow.dsl_spec import BindingLexer
from velvetflow.loop_dsl import build_loop_output_schema, index_loop_body_nodes
from velvetflow.models import ValidationError
from velvetflow.reference_utils import normalize_reference_path


@dataclass
class SemanticIssue:
    field_path: str
    message: str
    node_id: str | None = None


class WorkflowSemanticAnalyzer:
    """Run lexical+semantic analysis on bindings and node signatures."""

    def __init__(self, action_registry: Sequence[Mapping[str, Any]]):
        self.actions_by_id = {str(a.get("action_id")): dict(a) for a in action_registry}
        self._lexer = BindingLexer()

    # Public API ---------------------------------------------------------
    def analyze(self, workflow: Mapping[str, Any]) -> List[ValidationError]:
        issues: List[ValidationError] = []
        nodes = workflow.get("nodes") if isinstance(workflow, Mapping) else None
        if not isinstance(nodes, list):
            return issues

        nodes_by_id = {n.get("id"): n for n in nodes if isinstance(n, Mapping)}
        loop_body_parents = index_loop_body_nodes(workflow)

        for node in nodes_by_id.values():
            node_id = node.get("id") if isinstance(node, Mapping) else None
            action_id = node.get("action_id") if isinstance(node, Mapping) else None
            params = node.get("params") if isinstance(node, Mapping) else None
            arg_schema = self._arg_schema(action_id)

            for path, binding in self._iter_bindings(params):
                src = binding.get("__from__") if isinstance(binding, Mapping) else None
                agg = binding.get("__agg__") if isinstance(binding, Mapping) else None
                normalized_src = self._lexer.normalize(src) if isinstance(src, str) else src

                if isinstance(normalized_src, str):
                    # 1) lexical validation
                    try:
                        self._lexer.tokenize(normalized_src)
                    except ValueError as exc:
                        issues.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=node_id,
                                field=path,
                                message=str(exc),
                            )
                        )
                        continue

                # 2) semantic type alignment
                source_schema = self._resolve_source_schema(
                    normalized_src, nodes_by_id, loop_body_parents
                )
                target_schema = self._resolve_target_schema(arg_schema, path)
                issue = self._check_type_compatibility(
                    source_schema=source_schema,
                    target_schema=target_schema,
                    agg=agg,
                    source_path=normalized_src,
                )
                if issue:
                    issues.append(
                        ValidationError(
                            code="SCHEMA_MISMATCH",
                            node_id=node_id,
                            field=path,
                            message=issue,
                        )
                    )

        return issues

    # Binding traversal --------------------------------------------------
    def _iter_bindings(
        self, params: Any, prefix: str | None = None
    ) -> Iterable[tuple[str, Mapping[str, Any]]]:
        if isinstance(params, Mapping):
            if "__from__" in params:
                yield prefix or "params", params
            for key, value in params.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from self._iter_bindings(value, next_prefix)
        elif isinstance(params, list):
            for idx, item in enumerate(params):
                next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                yield from self._iter_bindings(item, next_prefix)

    # Schema resolution --------------------------------------------------
    def _arg_schema(self, action_id: str | None) -> Mapping[str, Any] | None:
        if not action_id:
            return None
        action_def = self.actions_by_id.get(action_id)
        schema = action_def.get("arg_schema") if isinstance(action_def, Mapping) else None
        return schema if isinstance(schema, Mapping) else None

    def _resolve_source_schema(
        self,
        source_path: str | None,
        nodes_by_id: Mapping[str, Mapping[str, Any]],
        loop_body_parents: Mapping[str, str],
    ) -> Optional[Mapping[str, Any]]:
        if not isinstance(source_path, str):
            return None

        source_path = normalize_reference_path(source_path)
        parts = source_path.split(".")
        if len(parts) < 2:
            return None
        if parts[0] == "params":
            # For params.<field> bindings, we cannot infer schema here.
            return None
        if parts[0] != "result_of":
            return None

        node_id, rest = parts[1], parts[2:]
        node = nodes_by_id.get(node_id)
        if not isinstance(node, Mapping):
            return None

        if node.get("type") == "loop":
            loop_schema = build_loop_output_schema(node.get("params") or {})
            return self._descend_schema(loop_schema, rest)

        action_id = node.get("action_id")
        action_def = self.actions_by_id.get(action_id)
        output_schema = action_def.get("output_schema") if isinstance(action_def, Mapping) else None
        arg_schema = action_def.get("arg_schema") if isinstance(action_def, Mapping) else None

        if rest and rest[0] == "params":
            return self._descend_schema(arg_schema, rest[1:])

        return self._descend_schema(output_schema, rest)

    def _resolve_target_schema(
        self, arg_schema: Mapping[str, Any] | None, field_path: str
    ) -> Optional[Mapping[str, Any]]:
        if not arg_schema:
            return None
        # field_path examples: "params.foo" or "params.items[0].bar"
        parts = [p for p in field_path.replace("[", ".").replace("]", "").split(".") if p]
        # drop leading "params" if present
        if parts and parts[0] == "params":
            parts = parts[1:]
        return self._descend_schema(arg_schema, parts)

    def _descend_schema(
        self, schema: Mapping[str, Any] | None, path: Sequence[str]
    ) -> Optional[Mapping[str, Any]]:
        if not isinstance(schema, Mapping):
            return None

        current: Mapping[str, Any] | None = schema
        for field in path:
            if not isinstance(current, Mapping):
                return None
            typ = current.get("type")
            if typ == "array":
                current = current.get("items") if isinstance(current.get("items"), Mapping) else None
                continue
            if typ == "object":
                props = current.get("properties") if isinstance(current.get("properties"), Mapping) else None
                current = props.get(field) if props else None
                continue
            # scalar - cannot go deeper
            return None
        return current

    # Type checks -------------------------------------------------------
    def _json_type(self, schema: Mapping[str, Any] | None) -> str | None:
        if not isinstance(schema, Mapping):
            return None
        typ = schema.get("type")
        if isinstance(typ, list):
            for candidate in typ:
                if candidate != "null":
                    return candidate
        if isinstance(typ, str):
            return typ
        return None

    def _check_type_compatibility(
        self,
        source_schema: Mapping[str, Any] | None,
        target_schema: Mapping[str, Any] | None,
        agg: str | None,
        source_path: str | None,
    ) -> Optional[str]:
        source_type = self._json_type(source_schema)
        target_type = self._json_type(target_schema)

        # Aggregation requirements
        if agg in {"count", "count_if", "filter_map", "pipeline"}:
            if source_type and source_type != "array":
                return f"聚合操作 {agg} 仅支持数组类型来源（当前为 {source_type!r}）。"
        if agg == "format_join":
            if source_type and source_type not in {"array", "string"}:
                return "format_join 仅支持 array 或 string 类型输入。"

        # Skip if missing schema information
        if not source_type or not target_type:
            return None

        if agg in {"count", "count_if"}:
            # count returns integer and targets often expect integer/number
            if target_type not in {"integer", "number"}:
                return f"count 结果类型为数字，但目标参数期望 {target_type!r}。"
            return None

        if source_type == target_type:
            return None

        return (
            f"绑定来源 {source_path!r} 的类型 {source_type!r} 与目标参数期望的类型 "
            f"{target_type!r} 不兼容。"
        )


__all__ = ["WorkflowSemanticAnalyzer", "SemanticIssue"]
