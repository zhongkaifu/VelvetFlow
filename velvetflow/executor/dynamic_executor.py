"""Dynamic workflow executor implementation."""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set, Union

from velvetflow.action_registry import get_action_by_id
from velvetflow.bindings import BindingContext, eval_node_params
from velvetflow.logging_utils import (
    TraceContext,
    child_span,
    log_debug,
    log_event,
    log_info,
    log_json,
    log_kv,
    log_section,
    log_warn,
    use_trace_context,
)
from velvetflow.models import Node, Workflow

from .actions import ActionExecutionMixin
from .conditions import ConditionEvaluationMixin
from .graph import GraphTraversalMixin
from .loops import LoopExecutionMixin
from .rendering import TemplateRendererMixin
from .simulation import SimulationData, load_simulation_data


class DynamicActionExecutor(
    ActionExecutionMixin,
    LoopExecutionMixin,
    ConditionEvaluationMixin,
    TemplateRendererMixin,
    GraphTraversalMixin,
):
    """Execute a validated Workflow with optional simulation data."""

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
        """提前阻断未注册的业务动作，避免执行阶段才发现问题。"""

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

                if ntype == "switch":
                    matched_to, switch_payload = self._eval_switch(
                        node, binding_ctx, include_debug=True
                    )
                    if switch_payload:
                        results[nid] = switch_payload
                        self._record_node_metrics(switch_payload)
                    if matched_to:
                        next_ids = [matched_to]
                    else:
                        next_ids = self._next_nodes(
                            self._derive_edges(workflow),
                            nid,
                            cond_value=switch_payload.get("matched_case"),
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
                            "switch_value": switch_payload.get("switch_value"),
                            "matched_case": switch_payload.get("matched_case"),
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
