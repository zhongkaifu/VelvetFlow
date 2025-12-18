"""Condition evaluation helpers for the executor."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Union

from velvetflow.aggregation import (
    JinjaExprValidationError,
    eval_jinja_expression,
    validate_jinja_expression,
)
from velvetflow.bindings import BindingContext
from velvetflow.logging_utils import log_json, log_warn
from velvetflow.reference_utils import normalize_reference_path, parse_field_path
from velvetflow.jinja_utils import render_jinja_template


class ConditionEvaluationMixin:
    def _condition_jinja_context(self, ctx: BindingContext) -> Dict[str, Any]:
        def _jinja_get(path: str) -> Any:
            normalized = normalize_reference_path(path)
            qualified = ctx._qualify_context_path(normalized)
            return ctx.get_value(qualified)

        base_ctx: Dict[str, Any] = {
            "result_of": ctx.results,
            "loop": ctx.loop_ctx,
            "loop_ctx": ctx.loop_ctx,
            "loop_id": ctx.loop_id,
            "get": _jinja_get,
            "system": {"date": datetime.now().date().isoformat()},
        }

        if isinstance(ctx.loop_ctx, Mapping):
            for key, value in ctx.loop_ctx.items():
                if key not in base_ctx:
                    base_ctx[key] = value

        return base_ctx

    def _render_condition_value(self, value: Any, ctx: BindingContext) -> Any:
        if isinstance(value, dict):
            return {k: self._render_condition_value(v, ctx) for k, v in value.items()}

        if isinstance(value, list):
            return [self._render_condition_value(v, ctx) for v in value]

        if isinstance(value, str):
            context = self._condition_jinja_context(ctx)
            text = value.strip()
            try:
                if text.startswith("{{") and text.endswith("}}"):  # pure expression, preserve type
                    inner = text[2:-2].strip()
                    return eval_jinja_expression(inner, context)

                if "{{" in text or "{%" in text:
                    rendered = render_jinja_template(text, context)
                    # If render returns the original string, fall back to eval for expression-only
                    if isinstance(rendered, str) and rendered != value and rendered.strip() != text:
                        return rendered
                    validate_jinja_expression(text, path="condition.expression")
                    return eval_jinja_expression(text, context)

                validate_jinja_expression(text, path="condition.expression")
                return eval_jinja_expression(text, context)
            except Exception:
                return value

        return value

    def _resolve_condition_source(self, source: Any, ctx: BindingContext) -> Any:
        """Resolve condition/switch sources which may be bindings or paths."""

        if isinstance(source, list):
            return [self._resolve_condition_source(item, ctx) for item in source]

        if isinstance(source, dict) and "__from__" in source:
            return ctx.resolve_binding(source)

        if isinstance(source, str):
            rendered = self._render_condition_value(source, ctx)
            if rendered is not source:
                return rendered
            if source in ctx.results:
                return ctx.results[source]

            if source.startswith("result_of."):
                return ctx.get_value(source)

            try:
                return ctx.get_value(f"result_of.{source}")
            except Exception:
                return ctx.get_value(source)

        return source

    def _get_field_value(self, obj: Any, field: Optional[str]) -> Any:
        if field is None or field == "":
            return obj

        def _apply_builtin_field(value: Any, name: str) -> Any:
            if name in {"length", "count"}:
                try:
                    return len(value)
                except Exception:
                    return None
            return None

        try:
            parts = parse_field_path(field)
        except Exception:
            return None

        current: Any = obj
        for p in parts:
            if isinstance(p, int):
                if isinstance(current, list) and 0 <= p < len(current):
                    current = current[p]
                    continue
                return None

            if isinstance(current, Mapping):
                if p in current:
                    current = current.get(p)
                    continue

            builtin_val = _apply_builtin_field(current, p)
            if builtin_val is not None:
                current = builtin_val
                continue

            if isinstance(current, list):
                extracted = []
                for item in current:
                    if isinstance(item, Mapping) and p in item:
                        extracted.append(item.get(p))
                if extracted:
                    current = extracted
                    continue
                return None

            if isinstance(current, Mapping):
                current = current.get(p)
                continue

            return None
        return current

    def _collect_exit_results(
        self, exit_node_def: Any, binding_ctx: BindingContext
    ) -> Optional[Union[Any, Dict[str, Any]]]:
        """Normalize exit node definitions and collect available results."""

        if not exit_node_def:
            return None

        if isinstance(exit_node_def, str):
            node_ids = [exit_node_def]
        elif isinstance(exit_node_def, list):
            node_ids = [nid for nid in exit_node_def if isinstance(nid, str)]
            if not node_ids:
                return None
        else:
            log_warn(f"[loop] exit 节点定义类型不支持: {type(exit_node_def)}")
            return None

        collected = {
            nid: binding_ctx.results.get(nid) for nid in node_ids if nid in binding_ctx.results
        }
        if not collected:
            return None

        if len(collected) == 1 and isinstance(exit_node_def, str):
            return next(iter(collected.values()))
        return collected

    def _normalize_to_jinja_expr(self, raw: Any) -> str:
        if isinstance(raw, str):
            text = raw.strip()
            if text.startswith("{{") and text.endswith("}}"):  # already pure Jinja expression
                return text[2:-2].strip()
            if text.startswith("{%") and text.endswith("%}"):
                return text[2:-2].strip()
            return text

        return repr(raw)

    def _strip_wrapped_expression(self, expr: Optional[str]) -> Optional[str]:
        if not isinstance(expr, str):
            return None
        text = expr.strip()
        if text.startswith("{{") and text.endswith("}}"):  # unwrap template delimiters
            return text[2:-2].strip()
        return text if text else None

    def _eval_condition(
        self, node: Dict[str, Any], ctx: BindingContext, *, include_debug: bool = False
    ) -> Any:
        raw_params = node.get("params") or {}
        params = raw_params if isinstance(raw_params, Mapping) else {"expression": raw_params}
        expr_raw = params.get("expression") if isinstance(params, Mapping) else None
        expression = self._strip_wrapped_expression(expr_raw) if expr_raw is not None else None
        if not expression or not isinstance(expression, str):
            log_warn("[condition] 缺少可执行的 Jinja 表达式，返回 False")
            return (False, {"resolved_value": None, "values": None}) if include_debug else False

        ctx_vars = self._condition_jinja_context(ctx)
        try:
            validate_jinja_expression(expression, path="condition.expression")
            raw_result = eval_jinja_expression(expression, ctx_vars)
            result = bool(raw_result)
        except Exception as exc:
            log_warn(f"[condition] 表达式执行失败: {exc}")
            return (False, {"resolved_value": None, "values": None}) if include_debug else False

        if include_debug:
            log_json(
                "[condition] 调试信息",
                {
                    "expression": expression,
                    "result": result,
                    "expression_value": raw_result,
                    "context_keys": list(ctx_vars.keys()),
                },
            )
            return result, {"resolved_value": result, "values": None}

        return result
    def _eval_switch(
        self, node: Mapping[str, Any], ctx: BindingContext, *, include_debug: bool = False
    ) -> tuple[Optional[str], Dict[str, Any]]:
        params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
        source = params.get("source")
        field = params.get("field") if isinstance(params.get("field"), str) else None

        try:
            raw_value = self._resolve_condition_source(source, ctx)
        except Exception as exc:  # noqa: BLE001
            log_warn(f"[switch] 无法解析 source {source!r}: {exc}")
            raw_value = None

        resolved_value = self._get_field_value(raw_value, field) if field else raw_value

        matched_case: Any = None
        matched_to: Optional[str] = None
        cases = node.get("cases") if isinstance(node.get("cases"), list) else []
        for case in cases:
            if not isinstance(case, Mapping):
                continue
            match_val = case.get("match") if "match" in case else case.get("value")
            match_values: List[Any] = match_val if isinstance(match_val, list) else [match_val]
            match_field = case.get("field") if isinstance(case.get("field"), str) else None
            if match_field:
                try:
                    match_values = [self._get_field_value(v, match_field) for v in match_values]
                except Exception:
                    log_warn(f"[switch] 解析 case 字段 '{match_field}' 失败，跳过该 case")
                    continue

            if resolved_value in match_values:
                matched_case = match_val
                to_node = case.get("to_node")
                matched_to = to_node if isinstance(to_node, str) else None
                break

        if matched_to is None and "default_to_node" in node:
            default_to = node.get("default_to_node")
            matched_to = default_to if isinstance(default_to, str) else None

        payload: Dict[str, Any] = {
            "switch_value": resolved_value,
            "matched_case": matched_case,
        }
        if include_debug:
            payload["raw_value"] = raw_value
        return matched_to, payload
