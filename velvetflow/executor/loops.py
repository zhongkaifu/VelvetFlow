"""Loop execution helpers used by the executor."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Mapping, Optional

from velvetflow.bindings import BindingContext
from velvetflow.jinja_utils import render_jinja_string_constants
from velvetflow.logging_utils import log_json, log_warn
from velvetflow.models import Node, Workflow


class LoopExecutionMixin:
    def _extract_export_values(self, raw_source: Any, field: Optional[str]) -> List[Any]:
        """Extract values from a loop export source for aggregate calculations."""

        # Normalize a single mapping/list/scalar into a flat list for aggregation.
        if isinstance(raw_source, list):
            values = [self._get_field_value(item, field) for item in raw_source]
        elif isinstance(raw_source, Mapping):
            values = [self._get_field_value(raw_source, field)]
        else:
            values = [raw_source]

        return [v for v in values if v is not None]

    def _apply_loop_exports(
        self,
        items_spec: Optional[Mapping[str, Any]],
        aggregates_spec: Optional[List[Mapping[str, Any]]],
        results: Mapping[str, Any],
        items_output: List[Dict[str, Any]],
        aggregates_output: Dict[str, Any],
        avg_state: Dict[str, Dict[str, float]],
        default_from_node: Optional[str],
    ) -> None:
        normalized_items_spec: Optional[Mapping[str, Any]] = None

        if isinstance(items_spec, list):
            normalized_items_spec = {"fields": [f for f in items_spec if isinstance(f, str)]}
        elif isinstance(items_spec, Mapping):
            normalized_items_spec = dict(items_spec)

        if isinstance(normalized_items_spec, Mapping):
            from_node = normalized_items_spec.get("from_node") or default_from_node
            fields = normalized_items_spec.get("fields") if isinstance(normalized_items_spec.get("fields"), list) else []
            list_field = (
                normalized_items_spec.get("list_field") if isinstance(normalized_items_spec.get("list_field"), str) else None
            )

            def _build_record(obj: Any) -> Dict[str, Any]:
                if isinstance(obj, Mapping):
                    return {f: self._get_field_value(obj, f) for f in fields if isinstance(f, str)}
                return {f: None for f in fields if isinstance(f, str)}

            if isinstance(from_node, str):
                source_res = results.get(from_node)
                handled_items = False

                if isinstance(source_res, Mapping) and list_field:
                    list_data = self._get_field_value(source_res, list_field)
                    if isinstance(list_data, list):
                        for element in list_data:
                            items_output.append(_build_record(element))
                        handled_items = True

                if not handled_items and isinstance(source_res, list):
                    for element in source_res:
                        items_output.append(_build_record(element))
                    handled_items = True

                if not handled_items:
                    if isinstance(source_res, Mapping):
                        items_output.append(_build_record(source_res))
                    elif source_res is not None:
                        items_output.append(_build_record(None))

        if not isinstance(aggregates_spec, list):
            return

        for agg in aggregates_spec:
            if not isinstance(agg, Mapping):
                continue
            name = agg.get("name")
            expr = agg.get("expr") or {}
            from_node = agg.get("from_node")
            if not isinstance(name, str) or not isinstance(from_node, str):
                continue

            expr_kind = expr.get("kind") if isinstance(expr, Mapping) else None
            kind = (expr_kind or agg.get("kind") or "").lower()
            field = expr.get("field") if isinstance(expr, Mapping) else None
            if field is None and isinstance(agg.get("field"), str):
                field = agg.get("field")

            if from_node not in results:
                continue

            raw_source = results.get(from_node)
            values = self._extract_export_values(raw_source, field if isinstance(field, str) else None)

            log_json(
                "[loop.exports.aggregates] 输入",
                {
                    "name": name,
                    "kind": kind,
                    "from_node": from_node,
                    "field": field,
                    "raw_result": raw_source,
                    "extracted_values": values,
                    "current_output": aggregates_output.get(name),
                    "avg_state": avg_state.get(name),
                },
            )

            if kind == "count":
                base = aggregates_output.get(name, 0) or 0
                increment = len(values)
                aggregates_output[name] = base + increment
                log_json(
                    "[loop.exports.aggregates] count 计算结果",
                    {
                        "name": name,
                        "values_length": increment,
                        "base": base,
                        "result": aggregates_output[name],
                    },
                )
                continue

            if kind == "count_if":
                op = expr.get("op", "==")
                target = expr.get("value")
                count = aggregates_output.get(name, 0) or 0

                def _match(v: Any) -> bool:
                    try:
                        if op == ">" and v is not None:
                            return v > target
                        if op == ">=" and v is not None:
                            return v >= target
                        if op == "<" and v is not None:
                            return v < target
                        if op == "<=" and v is not None:
                            return v <= target
                        if op == "!=":
                            return v != target
                        return v == target
                    except Exception:
                        return False

                for v in values:
                    if _match(v):
                        count += 1
                aggregates_output[name] = count
                log_json(
                    "[loop.exports.aggregates] count_if 计算结果",
                    {
                        "name": name,
                        "op": op,
                        "target": target,
                        "matched_values": [v for v in values if _match(v)],
                        "count_before": count - sum(1 for v in values if _match(v)),
                        "count_after": count,
                    },
                )
                continue

            numeric_values: List[float] = []
            for v in values:
                try:
                    if v is None:
                        continue
                    numeric_values.append(float(v))
                except Exception:
                    continue

            log_json(
                "[loop.exports.aggregates] 数值化输入",
                {
                    "name": name,
                    "kind": kind,
                    "numeric_values": numeric_values,
                    "previous_output": aggregates_output.get(name),
                    "avg_state": avg_state.get(name),
                },
            )

            if not numeric_values and kind not in {"max", "min"}:
                continue

            if kind == "max":
                current = aggregates_output.get(name)
                max_val = current
                for v in numeric_values:
                    if max_val is None or v > max_val:
                        max_val = v
                aggregates_output[name] = max_val
                log_json(
                    "[loop.exports.aggregates] max 计算结果",
                    {"name": name, "previous": current, "values": numeric_values, "result": max_val},
                )
            elif kind == "min":
                current = aggregates_output.get(name)
                min_val = current
                for v in numeric_values:
                    if min_val is None or v < min_val:
                        min_val = v
                aggregates_output[name] = min_val
                log_json(
                    "[loop.exports.aggregates] min 计算结果",
                    {"name": name, "previous": current, "values": numeric_values, "result": min_val},
                )
            elif kind == "sum":
                base = aggregates_output.get(name, 0)
                aggregates_output[name] = base + sum(numeric_values)
                log_json(
                    "[loop.exports.aggregates] sum 计算结果",
                    {"name": name, "base": base, "values": numeric_values, "result": aggregates_output[name]},
                )
            elif kind == "avg":
                state = avg_state.setdefault(name, {"sum": 0.0, "count": 0})
                state["sum"] += sum(numeric_values)
                state["count"] += len(numeric_values)
                if state["count"] > 0:
                    aggregates_output[name] = state["sum"] / state["count"]
                    log_json(
                        "[loop.exports.aggregates] avg 计算结果",
                        {
                            "name": name,
                            "values": numeric_values,
                            "state_sum": state["sum"],
                            "state_count": state["count"],
                            "result": aggregates_output.get(name),
                        },
                    )

    def _clear_loop_body_results(self, results: Dict[str, Any], body_node_ids: List[str]) -> None:
        """Remove intermediate results produced by loop body nodes."""

        for nid in body_node_ids:
            results.pop(nid, None)

    def _infer_default_export_source(
        self, body_nodes: List[Any], node_models: Mapping[str, Node]
    ) -> Optional[str]:
        """Pick the last action node in the loop body as the default export source."""

        body_nodes = render_jinja_string_constants(body_nodes)
        last_action: Optional[str] = None
        for n in body_nodes:
            if not isinstance(n, Mapping):
                continue
            nid = n.get("id")
            if not isinstance(nid, str):
                continue
            model = node_models.get(nid)
            ntype = model.type if model else n.get("type")
            if isinstance(ntype, str) and ntype == "action":
                last_action = nid
        return last_action

    def _execute_loop_node(self, node: Dict[str, Any], binding_ctx: BindingContext) -> Dict[str, Any]:
        params = render_jinja_string_constants(node.get("params") or {})
        loop_kind = params.get("loop_kind", "for_each")
        body_graph = render_jinja_string_constants(params.get("body_subgraph") or {})
        body_nodes = body_graph.get("nodes") or []
        entry = body_graph.get("entry")
        exit_node = body_graph.get("exit")

        def _collect_body_node_ids(nodes: List[Any]) -> List[str]:
            collected: List[str] = []
            for n in nodes:
                if not isinstance(n, Mapping):
                    continue
                nid = n.get("id")
                if isinstance(nid, str):
                    collected.append(nid)
                if n.get("type") == "loop":
                    params = n.get("params") or {}
                    nested_body = params.get("body_subgraph")
                    if isinstance(nested_body, Mapping):
                        nested_nodes = nested_body.get("nodes") or []
                    elif isinstance(nested_body, list):
                        nested_nodes = nested_body
                    else:
                        nested_nodes = []
                    collected.extend(_collect_body_node_ids(list(nested_nodes)))
            return collected

        body_node_ids = _collect_body_node_ids(list(body_nodes))

        extra_node_models: Dict[str, Node] = {}
        try:
            sub_wf = Workflow.model_validate({"workflow_name": "loop_body", "nodes": body_nodes})
            extra_node_models = {n.id: n for n in sub_wf.nodes}
        except Exception:
            extra_node_models = {}

        default_export_from_node = self._infer_default_export_source(body_nodes, extra_node_models)

        accumulator = params.get("accumulator") or {}
        max_iterations = params.get("max_iterations") or 100
        iterations: List[Dict[str, Any]] = []
        body_exports = body_graph.get("exports") if isinstance(body_graph, Mapping) else None
        exports = params.get("exports") if isinstance(params, Mapping) else None
        if not isinstance(exports, Mapping) and isinstance(body_exports, Mapping):
            exports = body_exports
        items_spec = exports.get("items") if isinstance(exports, Mapping) else None
        aggregates_spec = exports.get("aggregates") if isinstance(exports, Mapping) else None
        export_items: List[Dict[str, Any]] = []
        aggregates_output: Dict[str, Any] = {}
        avg_state: Dict[str, Dict[str, float]] = {}

        if loop_kind == "for_each":
            source = params.get("source")
            data = self._resolve_condition_source(source, binding_ctx)
            is_sequence = isinstance(data, Sequence) and not isinstance(
                data, (str, bytes, bytearray)
            )
            if not is_sequence:
                log_warn(
                    "[loop] for_each 的 source 不是 list/sequence "
                    f"(实际类型: {type(data).__name__}), 跳过执行"
                )
                return {
                    "status": "skipped",
                    "reason": "source_not_list",
                    "actual_type": type(data).__name__,
                }

            size = len(data)
            for idx, item in enumerate(data):
                if idx >= max_iterations:
                    log_warn("[loop] 达到 max_iterations 上限，提前结束循环")
                    break
                self._clear_loop_body_results(binding_ctx.results, body_node_ids)
                loop_ctx = {"index": idx, "item": item, "size": size, "accumulator": accumulator}
                item_alias = params.get("item_alias")
                if isinstance(item_alias, str) and item_alias:
                    loop_ctx[item_alias] = item
                loop_binding = BindingContext(
                    self.workflow,
                    binding_ctx.results,
                    extra_nodes=extra_node_models,
                    loop_ctx=loop_ctx,
                    loop_id=node.get("id"),
                )
                self._execute_graph(
                    {
                        "workflow_name": f"{node.get('id')}_iter_{idx}",
                        "description": body_graph.get("description", ""),
                        "nodes": body_nodes,
                    },
                    loop_binding,
                    start_nodes=[entry] if entry else None,
                )

                iter_result: Dict[str, Any] = {"index": idx, "item": item}
                exit_result = self._collect_exit_results(exit_node, binding_ctx)
                if exit_result is not None:
                    iter_result["exit_result"] = exit_result
                iterations.append(iter_result)

                self._apply_loop_exports(
                    items_spec,
                    aggregates_spec if isinstance(aggregates_spec, list) else None,
                    binding_ctx.results,
                    export_items,
                    aggregates_output,
                    avg_state,
                    default_export_from_node,
                )
                self._clear_loop_body_results(binding_ctx.results, body_node_ids)

            return {
                "status": "loop_completed",
                "loop_kind": loop_kind,
                "iterations": iterations,
                "accumulator": accumulator,
                "items": export_items,
                "aggregates": aggregates_output,
                "exports": {
                    "items": export_items,
                    "aggregates": aggregates_output,
                },
            }

        if loop_kind == "while":
            cond_def = params.get("condition") or {}
            iteration = 0
            while iteration < max_iterations:
                cond_value = self._eval_condition({"params": cond_def}, binding_ctx)
                if not cond_value:
                    break
                self._clear_loop_body_results(binding_ctx.results, body_node_ids)
                loop_ctx = {
                    "index": iteration,
                    "size": max_iterations,
                    "accumulator": accumulator,
                }
                loop_binding = BindingContext(
                    self.workflow,
                    binding_ctx.results,
                    extra_nodes=extra_node_models,
                    loop_ctx=loop_ctx,
                    loop_id=node.get("id"),
                )
                self._execute_graph(
                    {
                        "workflow_name": f"{node.get('id')}_while_{iteration}",
                        "description": body_graph.get("description", ""),
                        "nodes": body_nodes,
                    },
                    loop_binding,
                    start_nodes=[entry] if entry else None,
                )

                iter_result: Dict[str, Any] = {"index": iteration}
                exit_result = self._collect_exit_results(exit_node, binding_ctx)
                if exit_result is not None:
                    iter_result["exit_result"] = exit_result
                iterations.append(iter_result)
                iteration += 1

                self._apply_loop_exports(
                    items_spec,
                    aggregates_spec if isinstance(aggregates_spec, list) else None,
                    binding_ctx.results,
                    export_items,
                    aggregates_output,
                    avg_state,
                    default_export_from_node,
                )
                self._clear_loop_body_results(binding_ctx.results, body_node_ids)

            return {
                "status": "loop_completed",
                "loop_kind": loop_kind,
                "iterations": iterations,
                "accumulator": accumulator,
                "items": export_items,
                "aggregates": aggregates_output,
                "exports": {
                    "items": export_items,
                    "aggregates": aggregates_output,
                },
            }

        log_warn(f"[loop] 未知 loop_kind={loop_kind}，跳过执行")
        return {"status": "skipped", "reason": "unknown_loop_kind"}
