"""Loop execution helpers used by the executor."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Dict, List, Mapping, Optional

from jinja2 import UndefinedError

from velvetflow.aggregation import eval_jinja_expression
from velvetflow.bindings import BindingContext
from velvetflow.jinja_utils import render_jinja_string_constants
from velvetflow.logging_utils import log_warn
from velvetflow.models import Node, Workflow
from velvetflow.reference_utils import canonicalize_template_placeholders, normalize_reference_path


class LoopExecutionMixin:
    def _is_missing_export_reference(self, exc: Exception) -> bool:
        if isinstance(exc, (AttributeError, KeyError, UndefinedError)):
            return True
        message = str(exc)
        return "has no attribute" in message or "is undefined" in message

    def _apply_loop_exports(
        self,
        exports_spec: Optional[Mapping[str, Any]],
        exports_output: Dict[str, List[Any]],
        loop_binding: Optional[BindingContext],
    ) -> None:
        if not isinstance(exports_spec, Mapping) or loop_binding is None:
            return

        ctx = loop_binding.build_jinja_context()
        for key, template in exports_spec.items():
            if not isinstance(key, str):
                continue
            resolved_val = template
            if isinstance(template, str):
                normalized = canonicalize_template_placeholders(template)
                expr = normalize_reference_path(normalized)
                try:
                    resolved_val = eval_jinja_expression(expr, ctx)
                except Exception as exc:  # noqa: BLE001
                    if self._is_missing_export_reference(exc):
                        resolved_val = None
                        continue
                    log_warn(f"[loop.exports] 无法解析 {key} 的表达式: {exc}")
                    resolved_val = None

            if resolved_val is None:
                continue

            existing = exports_output.setdefault(key, [])
            if isinstance(resolved_val, list):
                existing.extend(resolved_val)
            else:
                existing.append(resolved_val)

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

        accumulator = params.get("accumulator") or {}
        max_iterations = params.get("max_iterations") or 100
        iterations: List[Dict[str, Any]] = []
        body_exports = body_graph.get("exports") if isinstance(body_graph, Mapping) else None
        exports = params.get("exports") if isinstance(params, Mapping) else None
        if not isinstance(exports, Mapping) and isinstance(body_exports, Mapping):
            exports = body_exports
        exports_spec = exports if isinstance(exports, Mapping) else {}
        exports_output: Dict[str, List[Any]] = {
            key: []
            for key in exports_spec.keys()
            if isinstance(key, str)
        }

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
                    exports_spec,
                    exports_output,
                    loop_binding,
                )
                self._clear_loop_body_results(binding_ctx.results, body_node_ids)

            return {
                "status": "loop_completed",
                "loop_kind": loop_kind,
                "iterations": iterations,
                "accumulator": accumulator,
                "exports": exports_output,
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
                    exports_spec,
                    exports_output,
                    loop_binding,
                )
                self._clear_loop_body_results(binding_ctx.results, body_node_ids)

            return {
                "status": "loop_completed",
                "loop_kind": loop_kind,
                "iterations": iterations,
                "accumulator": accumulator,
                "exports": exports_output,
            }

        log_warn(f"[loop] 未知 loop_kind={loop_kind}，跳过执行")
        return {"status": "skipped", "reason": "unknown_loop_kind"}
