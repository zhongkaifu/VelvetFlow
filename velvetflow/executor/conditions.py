"""Condition evaluation helpers for the executor."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Union

from velvetflow.bindings import BindingContext
from velvetflow.logging_utils import log_json, log_warn
from velvetflow.reference_utils import parse_field_path


class ConditionEvaluationMixin:
    def _resolve_condition_source(self, source: Any, ctx: BindingContext) -> Any:
        """Resolve condition source which may be a binding dict or a path string."""

        if isinstance(source, list):
            return [self._resolve_condition_source(item, ctx) for item in source]

        if isinstance(source, dict) and "__from__" in source:
            return ctx.resolve_binding(source)

        if isinstance(source, str):
            if source in ctx.results:
                return ctx.results[source]

            if source.startswith("result_of."):
                return ctx.get_value(source)

            try:
                return ctx.get_value(f"result_of.{source}")
            except Exception:
                return ctx.get_value(source)

        return source

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

    def _extract_export_values(self, source: Any, field: Optional[str]) -> List[Any]:
        if field in {"length", "count"}:
            value = self._get_field_value(source, field)
            return [value]

        if isinstance(source, list):
            values: List[Any] = []
            for item in source:
                values.append(self._get_field_value(item, field) if field else item)
            return values

        if isinstance(source, Mapping):
            return [self._get_field_value(source, field)] if field else [source]

        return [self._get_field_value(source, field)] if field else [source]

    def _eval_condition(
        self, node: Dict[str, Any], ctx: BindingContext, *, include_debug: bool = False
    ) -> Any:
        import re

        params = node.get("params") or {}
        kind = params.get("kind")
        if not kind:
            log_warn("[condition] 未指定 kind，默认 False")
            return False if not include_debug else (False, {"resolved_value": None, "values": None})

        def _return(result: Any, resolved_value: Any, values: Optional[list] = None) -> Any:
            if include_debug:
                return result, {"resolved_value": resolved_value, "values": values}
            return result

        def _log_condition_debug(
            field: Optional[str], resolved_value: Any, condition: Any, structure: Any
        ) -> None:
            log_json(
                f"[condition:{kind}] 调试信息",
                {
                    "field": field,
                    "resolved_value": resolved_value,
                    "condition": condition,
                    "structure": structure,
                },
            )

        def _safe_get_source() -> Any:
            source = params.get("source")
            try:
                return self._resolve_condition_source(source, ctx)
            except Exception as e:
                log_warn(
                    f"[condition:{kind}] source 路径 '{source}' 无法从 context 读取: {e}，返回 False"
                )
                return None

        data = _safe_get_source()
        field_path = params.get("field") if isinstance(params.get("field"), str) else None
        target_data = self._get_field_value(data, field_path) if field_path else data

        if kind == "list_not_empty":
            condition = {"check": "len(value) > 0", "type": "list"}
            if not isinstance(target_data, list):
                condition["reason"] = "source_not_list"
                log_warn("[condition:list_not_empty] source 不是 list，返回 False")
                result = False
            else:
                result = len(target_data) > 0
            _log_condition_debug(field_path, target_data, condition, params)
            return _return(result, target_data)

        if kind == "is_empty":
            condition = {"check": "value is None or len(value) == 0"}
            if data is None:
                result = True
            elif isinstance(data, (list, dict, str)):
                result = len(data) == 0
            else:
                result = False
            _log_condition_debug(None, data, condition, params)
            return _return(result, data)

        if kind in {"not_empty", "is_not_empty"}:
            condition = {"check": "value is not None and (len(value) > 0 if sized else True)"}
            if data is None:
                result = False
            elif isinstance(data, (list, dict, str)):
                result = len(data) > 0
            else:
                result = True
            _log_condition_debug(None, data, condition, params)
            return _return(result, data)

        def _extract_values(val: Any, field: Optional[str]) -> List[Any]:
            if val is None:
                return []

            if field in {"length", "count"}:
                extracted = self._get_field_value(val, field)
                return [] if extracted is None else [extracted]

            if isinstance(val, list):
                extracted: List[Any] = []
                for item in val:
                    extracted.append(self._get_field_value(item, field) if field else item)
                return extracted

            if isinstance(val, dict):
                return [self._get_field_value(val, field)] if field else [val]

            return [self._get_field_value(val, field)] if field else [val]

        if kind == "any_greater_than":
            field = params.get("field")
            threshold = params.get("threshold")
            values = _extract_values(data, field)

            def _is_gt(v: Any) -> bool:
                if v is None:
                    return False
                try:
                    return v > threshold
                except Exception as exc:
                    raise TypeError(
                        f"[condition:any_greater_than] 值 {v!r} 无法与阈值 {threshold!r} 比较"
                    ) from exc

            result = any(_is_gt(v) for v in values)
            condition = {"check": ">", "threshold": threshold, "values": values}
            _log_condition_debug(field, data, condition, params)
            return _return(result, data, values)

        if kind == "any_less_than":
            field = params.get("field")
            threshold = params.get("threshold")
            values = _extract_values(data, field)

            def _is_lt(v: Any) -> bool:
                if v is None:
                    return False
                try:
                    return v < threshold
                except Exception as exc:
                    raise TypeError(
                        f"[condition:any_less_than] 值 {v!r} 无法与阈值 {threshold!r} 比较"
                    ) from exc

            result = any(_is_lt(v) for v in values)
            condition = {"check": "<", "threshold": threshold, "values": values}
            _log_condition_debug(field, data, condition, params)
            return _return(result, data, values)

        if kind == "contains":
            target = params.get("target")
            if target is None:
                log_warn("[condition:contains] 未提供 target，返回 False")
                condition = {"check": "target in value", "reason": "missing_target"}
                _log_condition_debug(params.get("field"), data, condition, params)
                return _return(False, data)

            if isinstance(data, list):
                result = target in data
                condition = {"check": "target in list", "target": target}
                _log_condition_debug(None, data, condition, params)
                return _return(result, data)

            if data is None:
                condition = {"check": "target in None", "target": target, "reason": "data_none"}
                _log_condition_debug(None, data, condition, params)
                return _return(False, data)

            field = params.get("field")
            if isinstance(data, dict) and field:
                try:
                    v = self._get_field_value(data, field)
                    result = target in v if isinstance(v, (list, str)) else False
                    condition = {
                        "check": "target in dict[field]",
                        "target": target,
                        "value": v,
                    }
                    _log_condition_debug(field, data, condition, params)
                    return _return(result, data)
                except Exception:
                    log_warn("[condition:contains] 从字典提取 field 失败，返回 False")
                    condition = {
                        "check": "target in dict[field]",
                        "target": target,
                        "reason": "field_extraction_failed",
                    }
                    _log_condition_debug(field, data, condition, params)
                    return _return(False, data)

            if isinstance(data, dict):
                log_warn("[condition:contains] 未提供 field，且 source 是字典，返回 False")
                condition = {
                    "check": "target in dict (no field)",
                    "target": target,
                    "reason": "missing_field",
                }
                _log_condition_debug(field, data, condition, params)
                return _return(False, data)

            if isinstance(data, str):
                result = target in data
                condition = {"check": "target in string", "target": target}
                _log_condition_debug(None, data, condition, params)
                return _return(result, data)

            log_warn("[condition:contains] source 不是列表/字典/字符串，返回 False")
            condition = {
                "check": "unsupported source type",
                "target": target,
                "reason": f"source_type_{type(data)}",
            }
            _log_condition_debug(field, data, condition, params)
            return _return(False, data)

        if kind == "compare":
            op = params.get("op") or params.get("operator") or "=="
            target = params.get("value")
            field = params.get("field")
            values = _extract_values(data, field)

            def _do_compare(v: Any) -> bool:
                try:
                    if op == ">":
                        return v is not None and v > target
                    if op == ">=":
                        return v is not None and v >= target
                    if op == "<":
                        return v is not None and v < target
                    if op == "<=":
                        return v is not None and v <= target
                    if op == "!=":
                        return v != target
                    if op == "in":
                        try:
                            return v in target  # type: ignore[operator]
                        except Exception:
                            return False
                    if op == "not_in":
                        try:
                            return v not in target  # type: ignore[operator]
                        except Exception:
                            return False
                    return v == target
                except Exception as exc:
                    raise TypeError(
                        f"[condition:compare] 值 {v!r} 无法使用 op '{op}' 与目标 {target!r} 比较"
                    ) from exc

            result = any(_do_compare(v) for v in values)
            condition = {
                "check": f"any(value {op} target)",
                "operator": op,
                "target": target,
                "values": values,
            }
            _log_condition_debug(field, data, condition, params)
            return _return(result, data, values)

        if kind == "expression":
            expr = params.get("expression")
            if expr is None:
                log_warn("[condition:expression] 未提供 expression，返回 False")
                condition = {"check": "expression is None"}
                _log_condition_debug(params.get("field"), data, condition, params)
                return _return(False, data)

            expr_str = str(expr)
            field = params.get("field")
            values = _extract_values(data, field)
            safe_globals = {
                "__builtins__": {},
                "len": len,
                "sum": sum,
                "min": min,
                "max": max,
                "any": any,
                "all": all,
                "abs": abs,
            }

            def _eval_expr(v: Any) -> bool:
                try:
                    return bool(
                        eval(
                            expr_str,
                            safe_globals,
                            {"value": v, "values": values, "data": data},
                        )
                    )
                except Exception as exc:
                    log_warn(f"[condition:expression] 执行表达式失败: {exc}")
                    return False

            targets = values if field else [data]
            result = any(_eval_expr(v) for v in targets)
            condition = {
                "check": expr_str,
                "values": values,
                "data": data,
            }
            _log_condition_debug(field, data, condition, params)
            return _return(result, data, values)

        if kind == "regex_match":
            pattern = params.get("pattern")
            if pattern is None:
                log_warn("[condition:regex_match] 未提供 pattern，返回 False")
                condition = {"check": "regex match", "reason": "missing_pattern"}
                _log_condition_debug(None, data, condition, params)
                return _return(False, data)
            try:
                matched = False
                if isinstance(data, str):
                    matched = re.search(pattern, data) is not None
                elif isinstance(data, list):
                    matched = any(isinstance(v, str) and re.search(pattern, v) for v in data)
                condition = {"check": "regex match", "pattern": pattern, "matched": matched}
                _log_condition_debug(None, data, condition, params)
                return _return(matched, data)
            except Exception:
                condition = {
                    "check": "regex match",
                    "pattern": pattern,
                    "reason": "exception",
                }
                _log_condition_debug(None, data, condition, params)
                log_warn(f"[condition:regex_match] 处理正则 '{pattern}' 时发生异常")
                return _return(False, data)

        if kind == "max_in_range":
            field = params.get("field")
            min_v = params.get("min")
            max_v = params.get("max")
            values = _extract_values(data, field)
            matched_bands: List[Any] = []
            try:
                bands = sorted(
                    [b for b in params.get("bands", []) if isinstance(b, dict)],
                    key=lambda x: x.get("max"),
                )
            except Exception:
                bands = []
            for band in bands:
                try:
                    if all(k in band for k in ("label", "min", "max")):
                        in_range = any(band["min"] <= v <= band["max"] for v in values)
                        matched_bands.append((band, in_range))
                except Exception:
                    continue

            if matched_bands:
                for band, in_range in matched_bands:
                    if in_range:
                        result = band.get("label")
                        condition = {
                            "check": "value <= band.max",
                            "bands": bands,
                            "matched_band": band,
                        }
                        _log_condition_debug(None, data, condition, params)
                        return _return(result, data, values)
                try:
                    result = bands[-1].get("label") if isinstance(bands[-1], dict) else None
                except Exception:
                    result = None
                condition = {
                    "check": "fallback_last_band",
                    "bands": bands,
                }
                _log_condition_debug(None, data, condition, params)
                return _return(result, data, values)

            log_warn("[condition:max_in_range] 未提供合法的 bands，返回 False")
            condition = {
                "check": "value in range",
                "field": field,
                "min": min_v,
                "max": max_v,
                "values": values,
                "reason": "invalid_bands",
            }
            _log_condition_debug(field, data, condition, params)
            return _return(False, data, values)

        log_warn(f"[condition] 未知 kind={kind}，默认 False")
        return _return(False, data)

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
            match_val = case.get("value")
            match_values: List[Any] = match_val if isinstance(match_val, list) else [match_val]
            match_field = case.get("field") if isinstance(case.get("field"), str) else None
            if match_field:
                try:
                    match_values = [self._get_field_value(v, match_field) for v in match_values]
                except Exception:
                    log_warn(f"[switch] 解析 case 字段 '{match_field}' 失败，跳过该 case")
                    continue

            if resolved_value in match_values:
                matched_case = case
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
