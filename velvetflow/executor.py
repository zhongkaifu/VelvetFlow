# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Workflow execution utilities."""
import copy
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Set, Union

from velvetflow.bindings import BindingContext, eval_node_params
from velvetflow.action_registry import get_action_by_id
from velvetflow.logging_utils import (
    TraceContext,
    child_span,
    log_debug,
    log_event,
    log_info,
    log_json,
    log_kv,
    log_section,
    log_success,
    log_warn,
    use_trace_context,
)
from velvetflow.reference_utils import (
    canonicalize_template_placeholders,
    parse_field_path,
)
from velvetflow.models import Node, ValidationError, Workflow, infer_edges_from_bindings
from tools import get_registered_tool

# ===================== 16. 执行器 =====================

SimulationData = Dict[str, Any]


def load_simulation_data(path: str) -> SimulationData:
    """Load simulated action results from an external JSON file."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"simulation data file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError("simulation data must be a JSON object keyed by action_id")

    return data


class DynamicActionExecutor:
    """Execute a validated Workflow with optional simulation data.

    该执行器假定传入的 `Workflow` 已经过 Planner 层校验与修复，所有 action_id
    均可在 Action Registry 中找到。核心职责是在拓扑排序的顺序下依次解析
    参数绑定、调用动作（或根据 `simulation_data` 直接返回模拟结果），并将每个
    节点的输出存入 `BindingContext`。外部调用者不需要理解内部的绑定协议，
    只需关注最终 `run()` 返回的 `results` 映射：键为节点 id，值为动作执行
    （或模拟）返回的任意 JSON 兼容对象。

    参数
    ----
    workflow:
        已通过 Pydantic 校验的工作流对象，仅需声明 `nodes`；拓扑关系会基于
        参数绑定自动推导。
    simulations:
        * `None`：正常执行真实动作；
        * `Mapping`：以 action_id 为键的模拟结果；
        * `str`：指向包含上述结构的 JSON 文件路径，便于在离线环境重放。
    user_role:
        当前调用者身份，在绑定/权限控制逻辑中传递给 `BindingContext`。
    """
    def __init__(
        self,
        workflow: Workflow,
        simulations: Union[str, Mapping[str, Any], None] = None,
        user_role: str = "user",
        trace_context: TraceContext | None = None,
        run_manager: Any | None = None,
    ):
        if not isinstance(workflow, Workflow):
            raise ValueError(f"workflow 必须是 Workflow 对象，当前类型为 {type(workflow)}")

        workflow_dict = workflow.model_dump(by_alias=True)
        if not isinstance(workflow_dict.get("nodes"), list):
            raise ValueError("workflow 缺少合法的 'nodes' 列表。")

        self.workflow = workflow
        self.workflow_dict = workflow_dict
        self.node_models = {n.id: n for n in workflow.nodes}
        self.nodes = {n["id"]: n for n in workflow_dict["nodes"]}
        self.user_role = user_role
        self.trace_context = trace_context
        self.run_manager = run_manager

        if isinstance(simulations, str):
            self.simulation_data: SimulationData = load_simulation_data(simulations)
        elif simulations is None:
            self.simulation_data = {}
        elif isinstance(simulations, Mapping):
            self.simulation_data = dict(simulations)
        else:
            raise ValueError("simulations 必须是 None、字典或 JSON 文件路径")

        self._validate_registered_actions()

    def _validate_registered_actions(self) -> None:
        """提前阻断未注册的业务动作，避免执行阶段才发现问题。

        任何 action 节点缺失 action_id，或 action_id 不存在于 Registry，都会直接抛出
        ValueError，提醒调用方在构建 workflow 阶段修复或补全动作注册。
        """

        unknown: List[str] = []
        for node in self.workflow_dict.get("nodes", []):
            if node.get("type") != "action":
                continue

            nid = node.get("id") or "<unknown>"
            aid = node.get("action_id") or "<missing>"
            if not node.get("action_id"):
                unknown.append(f"{nid}: 缺少 action_id")
                continue

            if not get_action_by_id(aid):
                unknown.append(f"{nid}: action_id='{aid}' 未注册")

        if unknown:
            details = "; ".join(unknown)
            raise ValueError(
                "workflow 中存在未注册或缺失的 action_id，请在构建阶段修复: " + details
            )

    def _derive_edges(self, workflow: Workflow) -> List[Dict[str, Any]]:
        """Rebuild implicit edges from the latest node bindings.

        返回值中的 ``from``/``to`` 字段每次调用都会重新从节点参数引用和条件
        分支推导，保持与当前 nodes 一致，不依赖任何声明式缓存。
        """

        # 所有连线信息都由绑定实时推导，不再消费/缓存声明式 edges 字段。
        return infer_edges_from_bindings(workflow.nodes)

    def _find_start_nodes(self, nodes: Mapping[str, Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
        """Locate start nodes by combining explicit标记与“无入度”节点。

        无论是否声明 start 节点，都将所有未作为 `to` 端出现的节点纳入起点，这样
        任何独立子图都可以从自身的入度为 0 节点开始执行，避免因缺失 start 节点
        而导致的不可达警告。
        """

        all_ids = set(nodes.keys())
        to_ids = set()
        from_ids = set()
        for e in edges:
            if not isinstance(e, Mapping):
                continue
            to_ids.add(e.get("to"))
            from_ids.add(e.get("from"))

        inbound_free = list(all_ids - to_ids)

        starts = [nid for nid, n in nodes.items() if n.get("type") == "start"]
        condition_starts = [
            nid for nid, n in nodes.items() if n.get("type") == "condition" and nid in inbound_free
        ]

        outbound_roots = [nid for nid in inbound_free if nid in from_ids]

        if starts:
            return list(dict.fromkeys(starts + condition_starts + outbound_roots))

        # 若没有显式 start 节点，则退化为无入度节点集合以保证流程仍可执行。
        merged = list(dict.fromkeys(condition_starts + outbound_roots + inbound_free))
        return merged

    def _topological_sort(self, workflow: Workflow) -> List[Node]:
        """Return execution order while validating reachability constraints.

        * 若存在环（例如 loop 子图），会回退为声明顺序并提示警告；
        * 若没有入度为 0 的节点，会回退为声明顺序并提示警告；
        * 若部分节点无法从起点到达，同样抛出 `DISCONNECTED_GRAPH`。
        """

        nodes = workflow.nodes
        edges = self._derive_edges(workflow)

        indegree: Dict[str, int] = {n.id: 0 for n in nodes}
        adjacency: Dict[str, List[str]] = {n.id: [] for n in nodes}
        node_lookup: Dict[str, Node] = {n.id: n for n in nodes}
        for e in edges:
            if not isinstance(e, Mapping):
                continue
            frm = e.get("from")
            to = e.get("to")
            if not frm or not to or frm not in indegree or to not in indegree:
                continue
            indegree[to] += 1
            adjacency[frm].append(to)

        queue: List[str] = [nid for nid, deg in indegree.items() if deg == 0]
        ordered: List[str] = []

        # 若未发现入度为 0 的节点，说明存在环或入度计算异常，直接退回声明顺序
        if not queue and nodes:
            log_warn("未找到入度为 0 的节点，按声明顺序执行（可能存在环或隐式依赖）。")
            ordered = [n.id for n in nodes]
        else:
            while queue:
                nid = queue.pop(0)
                ordered.append(nid)
                for nxt in adjacency.get(nid, []):
                    indegree[nxt] -= 1
                    if indegree[nxt] == 0:
                        queue.append(nxt)

            if len(ordered) != len(nodes):
                log_warn("检测到工作流包含环（例如 loop 子图），按声明顺序执行。")
                ordered = [n.id for n in nodes]

        nodes_dump = {n.id: n.model_dump(by_alias=True) for n in nodes}
        start_nodes = self._find_start_nodes(nodes_dump, edges)
        reachable: Set[str] = set()
        if start_nodes:
            queue = list(start_nodes)
            while queue:
                nid = queue.pop(0)
                if nid in reachable:
                    continue
                reachable.add(nid)
                for nxt in adjacency.get(nid, []):
                    if nxt not in reachable:
                        queue.append(nxt)

        if nodes and not start_nodes:
            raise ValueError(
                ValidationError(
                    code="DISCONNECTED_GRAPH",
                    node_id=None,
                    field=None,
                    message="存在无法从 start 节点到达的节点。",
                )
            )

        if nodes and len(reachable) != len(nodes):
            log_warn("存在无法从 start 节点到达的节点，将跳过这些节点。")

        if reachable:
            ordered = [nid for nid in ordered if nid in reachable]

        return [node_lookup[nid] for nid in ordered]

    def _next_nodes(
        self,
        edges: List[Dict[str, Any]],
        nid: str,
        cond_value: Any = None,
        nodes_data: Optional[Mapping[str, Any]] = None,
    ) -> List[str]:
        res: List[str] = []
        cond_label: Optional[str]
        if isinstance(cond_value, bool):
            cond_label = "true" if cond_value else "false"
        elif cond_value is None:
            cond_label = None
        else:
            cond_label = str(cond_value)

        node_lookup = nodes_data if isinstance(nodes_data, Mapping) else self.nodes
        node_def = node_lookup.get(nid, {}) if isinstance(node_lookup.get(nid, {}), Mapping) else {}
        if node_def.get("type") == "condition" and isinstance(cond_value, bool):
            branch_key = "true_to_node" if cond_value else "false_to_node"
            if branch_key in node_def:
                branch_target = node_def.get(branch_key)
                if branch_target in {None, "null"}:
                    return []
                if isinstance(branch_target, str):
                    return [branch_target]
                return []
            # Explicit true/false bindings take precedence over declarative edges;
            # when missing, fall back to edge-based routing for backward compatibility.
            #
            # Without this guard, legacy edges in serialized workflows would override
            # the structured true_to_node/false_to_node intent (including explicit
            # nulls that terminate a branch).
            return []

        for e in edges:
            if not isinstance(e, Mapping):
                continue
            frm = e.get("from")
            if frm != nid:
                continue
            cond = e.get("condition")
            if cond is None:
                res.append(e.get("to"))
            elif cond_label is not None and cond == cond_label:
                res.append(e.get("to"))
        return [r for r in res if r not in {None, "null"}]

    def _render_template(self, value: Any, params: Mapping[str, Any]) -> Any:
        if isinstance(value, str):
            normalized_value = canonicalize_template_placeholders(value)
            pattern = r"\{\{\s*([^{}]+)\s*\}\}"

            def _replace(match: re.Match[str]) -> str:
                key = match.group(1)
                return str(params.get(key, match.group(0)))

            return re.sub(pattern, _replace, normalized_value)

        if isinstance(value, list):
            return [self._render_template(v, params) for v in value]

        if isinstance(value, dict):
            return {k: self._render_template(v, params) for k, v in value.items()}

        return value

    def _simulate_action(self, action_id: str, resolved_params: Dict[str, Any]) -> Dict[str, Any]:
        """Return deterministic simulation payload for a given action.

        如果传入的 `simulation_data` 包含对应 action_id 的配置，会优先根据
        `result` 模板渲染输出；否则返回标记为 `status="simulated"` 的占位结果。
        该返回结构会直接写入 `BindingContext.results`，保持与真实执行结果的键
        兼容性。
        """

        payload = self.simulation_data.get(action_id)
        if not isinstance(payload, dict):
            return {
                "status": "simulated",
                "action_id": action_id,
                "params_used": resolved_params,
            }

        defaults = payload.get("defaults") if isinstance(payload.get("defaults"), dict) else {}
        params_for_render = {**defaults, **resolved_params}

        template = payload.get("result")
        if template is None:
            return {
                "status": "simulated",
                "action_id": action_id,
                "params_used": resolved_params,
            }

        return copy.deepcopy(self._render_template(template, params_for_render))

    def _should_simulate(self, action_id: str) -> bool:
        """Check if the current action_id has an explicit simulation entry."""

        return action_id in self.simulation_data

    def _invoke_tool(
        self, tool_name: str, resolved_params: Dict[str, Any], action_id: str
    ) -> Dict[str, Any]:
        """Execute a registered tool and normalize its output."""

        tool = get_registered_tool(tool_name)
        if not tool:
            log_warn(f"[action:{action_id}] tool '{tool_name}' not registered")
            return {
                "status": "tool_not_registered",
                "tool_name": tool_name,
                "action_id": action_id,
            }

        log_info(f"-> 调用工具: {tool_name} with params={resolved_params}")
        try:
            output = tool(**resolved_params)
            log_debug(f"<- 工具返回: {output}")
        except Exception as exc:  # pragma: no cover - runtime safety
            log_warn(f"[action:{action_id}] tool '{tool_name}' 执行失败: {exc}")
            return {
                "status": "tool_error",
                "tool_name": tool_name,
                "action_id": action_id,
                "error": str(exc),
            }

        if isinstance(output, dict):
            return output
        return {"value": output, "tool_name": tool_name}

    def _resolve_condition_source(self, source: Any, ctx: BindingContext) -> Any:
        """Resolve condition source which may be a binding dict or a path string."""

        if isinstance(source, list):
            return [self._resolve_condition_source(item, ctx) for item in source]

        if isinstance(source, dict) and "__from__" in source:
            return ctx.resolve_binding(source)

        if isinstance(source, str):
            # 兼容部分工作流直接使用节点 id 作为 source 的写法，默认视为
            # "result_of.<node_id>"，避免因缺失前缀导致条件评估失败。
            if source in ctx.results:
                return ctx.results[source]

            if source.startswith("result_of."):
                return ctx.get_value(source)

            # 如果缺失 "result_of." 前缀，尝试补全后解析；解析失败则回退到
            # 直接解析原始路径以保留原有行为。
            try:
                return ctx.get_value(f"result_of.{source}")
            except Exception:
                return ctx.get_value(source)

        return source

    def _collect_exit_results(
        self, exit_node_def: Any, binding_ctx: BindingContext
    ) -> Optional[Union[Any, Dict[str, Any]]]:
        """Normalize exit node definitions and collect available results.

        The loop "exit" configuration may be a single node id or a list of node ids.
        This helper safely gathers the corresponding results from the binding context
        without assuming hashability of the configuration value. When only one node id
        is provided, the raw result value is returned to keep the downstream prompt
        contract consistent with planner 的约定；否则返回 `{node_id: result}` 映射。
        """

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

    def _apply_loop_exports(
        self,
        items_spec: Optional[Mapping[str, Any]],
        aggregates_spec: Optional[List[Mapping[str, Any]]],
        results: Mapping[str, Any],
        items_output: List[Dict[str, Any]],
        aggregates_output: Dict[str, Any],
        avg_state: Dict[str, Dict[str, float]],
    ) -> None:
        if isinstance(items_spec, Mapping):
            from_node = items_spec.get("from_node")
            fields = items_spec.get("fields") if isinstance(items_spec.get("fields"), list) else []
            list_field = items_spec.get("list_field") if isinstance(items_spec.get("list_field"), str) else None

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

            kind = expr.get("kind")
            field = expr.get("field") if isinstance(expr, Mapping) else None
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
        """Remove intermediate results produced by loop body nodes.

        Each iteration must start from a clean slate to avoid leaking state
        between iterations. Body node outputs should only be exposed through
        loop exports, so we drop them from the shared results map once they
        have been consumed.
        """

        for nid in body_node_ids:
            results.pop(nid, None)

    def _eval_condition(
        self, node: Dict[str, Any], ctx: BindingContext, *, include_debug: bool = False
    ) -> Any:
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
            """Log debug info for condition evaluation."""

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

            # If the source is already a list, normalize each element.
            if isinstance(val, list):
                extracted: List[Any] = []
                for item in val:
                    extracted.append(self._get_field_value(item, field) if field else item)
                return extracted

            # For dict sources, allow pulling a specific field.
            if isinstance(val, dict):
                return [self._get_field_value(val, field)] if field else [val]

            # Fallback to treating the value itself as the comparable payload.
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
            condition = {
                "check": "any(value > threshold)",
                "threshold": threshold,
                "values": values,
            }
            _log_condition_debug(field, data, condition, params)
            return _return(result, data, values)

        if kind == "all_less_than":
            field = params.get("field")
            threshold = params.get("threshold")
            values = [v for v in _extract_values(data, field) if v is not None]
            if not values:
                condition = {
                    "check": "all(value < threshold)",
                    "threshold": threshold,
                    "values": values,
                    "reason": "no_values",
                }
                _log_condition_debug(field, data, condition, params)
                return _return(False, data, values)

            def _is_lt(v: Any) -> bool:
                try:
                    return v < threshold
                except Exception as exc:
                    raise TypeError(
                        f"[condition:all_less_than] 值 {v!r} 无法与阈值 {threshold!r} 比较"
                    ) from exc

            result = all(_is_lt(v) for v in values)
            condition = {
                "check": "all(value < threshold)",
                "threshold": threshold,
                "values": values,
            }
            _log_condition_debug(field, data, condition, params)
            return _return(result, data, values)

        if kind == "equals":
            value = params.get("value")
            result = data == value
            condition = {"check": "value == target", "target": value}
            _log_condition_debug(None, data, condition, params)
            return _return(result, data)

        if kind == "not_equals":
            value = params.get("value")
            result = data != value
            condition = {"check": "value != target", "target": value}
            _log_condition_debug(None, data, condition, params)
            return _return(result, data)

        if kind == "greater_than":
            threshold = params.get("threshold")
            values = _extract_values(data, params.get("field"))

            def _is_gt(v: Any) -> bool:
                if v is None:
                    return False
                try:
                    return v > threshold
                except Exception as exc:
                    raise TypeError(
                        f"[condition:greater_than] 值 {v!r} 无法与阈值 {threshold!r} 比较"
                    ) from exc

            result = any(_is_gt(v) for v in values)
            condition = {
                "check": "any(value > threshold)",
                "threshold": threshold,
                "values": values,
            }
            _log_condition_debug(params.get("field"), data, condition, params)
            return _return(result, data, values)

        if kind == "less_than":
            threshold = params.get("threshold")
            values = _extract_values(data, params.get("field"))

            def _is_lt(v: Any) -> bool:
                if v is None:
                    return False
                try:
                    return v < threshold
                except Exception as exc:
                    raise TypeError(
                        f"[condition:less_than] 值 {v!r} 无法与阈值 {threshold!r} 比较"
                    ) from exc

            result = any(_is_lt(v) for v in values)
            condition = {
                "check": "any(value < threshold)",
                "threshold": threshold,
                "values": values,
            }
            _log_condition_debug(params.get("field"), data, condition, params)
            return _return(result, data, values)

        if kind == "between":
            min_v = params.get("min")
            max_v = params.get("max")
            values = [v for v in _extract_values(data, params.get("field")) if v is not None]

            def _in_range(v: Any) -> bool:
                try:
                    return v >= min_v and v <= max_v
                except Exception as exc:
                    raise TypeError(
                        f"[condition:between] 值 {v!r} 无法与区间 [{min_v!r}, {max_v!r}] 比较"
                    ) from exc

            result = any(_in_range(v) for v in values)
            condition = {
                "check": "any(min <= value <= max)",
                "min": min_v,
                "max": max_v,
                "values": values,
            }
            _log_condition_debug(params.get("field"), data, condition, params)
            return _return(result, data, values)

        if kind == "contains":
            target = params.get("value")
            field = params.get("field")
            if isinstance(data, list):
                if field:
                    try:
                        values = [self._get_field_value(item, field) for item in data]
                        result = target in values
                        condition = {
                            "check": "target in list[field]",
                            "target": target,
                            "values": values,
                        }
                        _log_condition_debug(field, data, condition, params)
                        return _return(result, data, values)
                    except Exception:
                        log_warn("[condition:contains] 从列表元素提取 field 失败，返回 False")
                        condition = {
                            "check": "target in list[field]",
                            "target": target,
                            "reason": "field_extraction_failed",
                        }
                        _log_condition_debug(field, data, condition, params)
                        return _return(False, data)

                log_warn("[condition:contains] 未提供 field，且 source 是列表，返回 False")
                condition = {
                    "check": "target in list (no field)",
                    "target": target,
                    "reason": "missing_field",
                }
                _log_condition_debug(field, data, condition, params)
                return _return(False, data)

            if isinstance(data, dict):
                if field:
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
                import re

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
    def _execute_loop_node(self, node: Dict[str, Any], binding_ctx: BindingContext) -> Dict[str, Any]:
        params = node.get("params") or {}
        loop_kind = params.get("loop_kind", "for_each")
        body_graph = params.get("body_subgraph") or {}
        body_nodes = body_graph.get("nodes") or []
        entry = body_graph.get("entry")
        exit_node = body_graph.get("exit")

        body_node_ids: List[str] = []
        for n in body_nodes:
            if isinstance(n, Mapping):
                nid = n.get("id")
                if isinstance(nid, str):
                    body_node_ids.append(nid)

        extra_node_models: Dict[str, Node] = {}
        try:
            sub_wf = Workflow.model_validate({"workflow_name": "loop_body", "nodes": body_nodes})
            extra_node_models = {n.id: n for n in sub_wf.nodes}
        except Exception:
            extra_node_models = {}

        accumulator = params.get("accumulator") or {}
        max_iterations = params.get("max_iterations") or 100
        iterations: List[Dict[str, Any]] = []
        body_exports = (
            body_graph.get("exports") if isinstance(body_graph, Mapping) else None
        )
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
            if not isinstance(data, list):
                log_warn("[loop] for_each 的 source 不是 list，跳过执行")
                return {"status": "skipped", "reason": "source_not_list"}

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

    def _record_node_metrics(self, result: Any) -> None:
        if not self.run_manager:
            return

        success = True
        if isinstance(result, Mapping):
            status = result.get("status")
            if status and str(status).lower() not in {"ok", "success", "allowed"}:
                success = False
        self.run_manager.metrics.record_node(success=success)

    def _execute_graph(
        self,
        workflow_dict: Dict[str, Any],
        binding_ctx: BindingContext,
        start_nodes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        workflow = Workflow.model_validate(workflow_dict)
        nodes_data = {n["id"]: n for n in workflow.model_dump(by_alias=True)["nodes"]}
        sorted_nodes = self._topological_sort(workflow)
        edges = self._derive_edges(workflow)

        graph_label = workflow_dict.get("workflow_name", "")
        log_section("执行工作流", graph_label)
        log_kv("描述", workflow_dict.get("description", ""))
        log_kv("模拟用户角色", self.user_role)

        if start_nodes is None:
            start_nodes = self._find_start_nodes(nodes_data, edges)

        indegree = {nid: 0 for nid in nodes_data}
        for e in edges:
            if not isinstance(e, Mapping):
                continue
            frm = e.get("from")
            to = e.get("to")
            if frm in indegree and to in indegree:
                indegree[to] += 1

        zero_indegree = [nid for nid, deg in indegree.items() if deg == 0]
        start_nodes = list(dict.fromkeys((start_nodes or []) + zero_indegree))
        if not start_nodes and sorted_nodes:
            log_warn("未找到 start 节点，将从任意一个节点开始（仅 demo）。")
            start_nodes = [sorted_nodes[0].id]

        visited: Set[str] = set()
        reachable: Set[str] = set(start_nodes)
        results = binding_ctx.results

        pending: List[Node] = list(sorted_nodes)
        while pending:
            progress = False
            remaining: List[Node] = []
            for node_model in pending:
                nid = node_model.id
                if nid in visited or (reachable and nid not in reachable):
                    remaining.append(node_model)
                    continue
                visited.add(nid)
                progress = True

                node = nodes_data[nid]
                ntype = node.get("type")
                action_id = node.get("action_id")
                display_name = node.get("display_name") or action_id or ntype
                params = node.get("params", {})

                log_event(
                    "node_start",
                    {
                        "node_id": nid,
                        "type": ntype,
                        "display_name": display_name,
                        "action_id": action_id,
                        "params": params,
                    },
                    node_id=nid,
                    action_id=action_id,
                )
                log_info(f"[Node {nid}] type={ntype}, display_name={display_name}, action_id={action_id}")
                if params:
                    log_json("raw params", params)

                if ntype == "action" and action_id:
                    resolved_params = eval_node_params(node_model, binding_ctx)
                    log_json("resolved params", resolved_params)

                    action = get_action_by_id(action_id)
                    if not action:
                        log_warn(f"未在 Registry 中找到 action_id={action_id}")
                        result = {"status": "no_action_impl"}
                    else:
                        allowed_roles = action.get("allowed_roles") or []
                        if allowed_roles and self.user_role not in allowed_roles:
                            log_warn(
                                f"[FORBIDDEN] 当前角色 '{self.user_role}' 无权执行该动作，允许角色：{allowed_roles}"
                            )
                            result = {
                                "status": "forbidden",
                                "reason": "role_not_allowed",
                                "required_roles": allowed_roles,
                                "actor_role": self.user_role,
                            }
                        else:
                            log_info(
                                f"-> 执行业务动作: {action['name']} (domain={action['domain']})"
                            )
                            log_debug(f"-> 描述: {action['description']}")
                            if self._should_simulate(action_id):
                                result = self._simulate_action(action_id, resolved_params)
                            elif tool_name := action.get("tool_name"):
                                result = self._invoke_tool(tool_name, resolved_params, action_id)
                            else:
                                log_warn(
                                    f"[action:{action_id}] 未找到工具映射，返回占位结果"
                                )
                                result = {
                                    "status": "no_tool_mapping",
                                    "action_id": action_id,
                                    "description": action.get("description"),
                                }

                    payload = result.copy() if isinstance(result, dict) else {"value": result}
                    payload["params"] = resolved_params
                    results[nid] = payload
                    self._record_node_metrics(payload)
                    next_ids = self._next_nodes(
                        self._derive_edges(workflow), nid, nodes_data=nodes_data
                    )
                    for nxt in next_ids:
                        if nxt not in visited:
                            reachable.add(nxt)
                    log_event(
                        "node_end",
                        {
                            "node_id": nid,
                            "type": ntype,
                            "action_id": action_id,
                            "resolved_params": resolved_params,
                            "result": result,
                            "next_nodes": next_ids,
                        },
                        node_id=nid,
                        action_id=action_id,
                    )
                    continue

                if ntype == "condition":
                    cond_eval = self._eval_condition(node, binding_ctx, include_debug=True)
                    if isinstance(cond_eval, tuple) and len(cond_eval) == 2:
                        cond_value, cond_debug = cond_eval
                    else:
                        cond_value, cond_debug = cond_eval, None
                    log_info(f"[condition] 结果: {cond_value}")
                    payload: Dict[str, Any] = {"condition_result": cond_value}
                    if isinstance(cond_debug, Mapping):
                        resolved_value = cond_debug.get("resolved_value")
                        if resolved_value is not None:
                            payload["resolved_value"] = resolved_value
                            if isinstance(resolved_value, Mapping):
                                for k, v in resolved_value.items():
                                    payload.setdefault(k, v)
                        values = cond_debug.get("values")
                        if values is not None:
                            payload["evaluated_values"] = values
                    results[nid] = payload
                    self._record_node_metrics(payload)
                    next_ids = self._next_nodes(
                        self._derive_edges(workflow),
                        nid,
                        cond_value=cond_value,
                        nodes_data=nodes_data,
                    )
                    for nxt in next_ids:
                        if nxt not in visited:
                            reachable.add(nxt)
                    log_event(
                        "node_end",
                        {
                            "node_id": nid,
                            "type": ntype,
                            "condition_result": cond_value,
                            "next_nodes": next_ids,
                        },
                        node_id=nid,
                    )
                    continue

                if ntype == "loop":
                    loop_result = self._execute_loop_node(node, binding_ctx)
                    results[nid] = loop_result
                    self._record_node_metrics(loop_result)
                    log_event(
                        "node_end",
                        {
                            "node_id": nid,
                            "type": ntype,
                            "result": loop_result,
                        },
                        node_id=nid,
                        action_id=action_id,
                    )
                    next_ids = self._next_nodes(
                        self._derive_edges(workflow), nid, nodes_data=nodes_data
                    )
                    for nxt in next_ids:
                        if nxt not in visited:
                            reachable.add(nxt)
                    log_event(
                        "node_end",
                        {
                            "node_id": nid,
                            "type": ntype,
                            "next_nodes": next_ids,
                            "result": results.get(nid),
                        },
                        node_id=nid,
                        action_id=action_id,
                    )
                    continue

                next_ids = self._next_nodes(
                    self._derive_edges(workflow), nid, nodes_data=nodes_data
                )
                for nxt in next_ids:
                    if nxt not in visited:
                        reachable.add(nxt)

                log_event(
                    "node_end",
                    {
                        "node_id": nid,
                        "type": ntype,
                        "next_nodes": next_ids,
                        "result": results.get(nid),
                    },
                    node_id=nid,
                )
                self._record_node_metrics(results.get(nid))

            pending = remaining
            if not progress:
                break

        return results

    def run_workflow(self, trace_context: TraceContext | None = None):
        binding_ctx = BindingContext(self.workflow, {})
        if self.run_manager:
            if not self.run_manager.workflow_name:
                self.run_manager.workflow_name = self.workflow.workflow_name
            self.run_manager.metrics.extra.setdefault(
                "declared_nodes", len(self.workflow.nodes)
            )
        ctx = trace_context or self.trace_context
        if ctx:
            with use_trace_context(ctx):
                with child_span("executor_run") as span_ctx:
                    log_event(
                        "executor_start",
                        {
                            "workflow_name": self.workflow.workflow_name,
                            "node_count": len(self.workflow.nodes),
                        },
                        context=span_ctx,
                    )
                    results = self._execute_graph(self.workflow_dict, binding_ctx)
                    if self.run_manager:
                        self.run_manager.metrics.extra["result_nodes"] = list(results.keys())
                    log_event(
                        "executor_finished",
                        {
                            "workflow_name": self.workflow.workflow_name,
                            "result_nodes": list(results.keys()),
                        },
                        context=span_ctx,
                    )
                    return results

        results = self._execute_graph(self.workflow_dict, binding_ctx)
        if self.run_manager:
            self.run_manager.metrics.extra["result_nodes"] = list(results.keys())
        return results

    def run(self, trace_context: TraceContext | None = None):
        return self.run_workflow(trace_context=trace_context)


