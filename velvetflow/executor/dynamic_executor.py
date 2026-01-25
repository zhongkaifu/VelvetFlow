"""Dynamic workflow executor implementation."""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Mapping, Optional, Set, Union

from velvetflow.action_registry import BUSINESS_ACTIONS, get_action_by_id
from velvetflow.bindings import BindingContext, eval_node_params
from velvetflow.config import OPENAI_MODEL
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
from velvetflow.models import Node, Workflow, infer_depends_on_from_edges

from openai import BadRequestError, OpenAI
from tools.registry import get_registered_tool

from .async_runtime import (
    ExecutionCheckpoint,
    GLOBAL_ASYNC_RESULT_STORE,
    WorkflowSuspension,
)
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

    def _build_reasoning_prompt(
        self,
        *,
        task_prompt: str | None,
        context: Any,
        expected_output_format: Any,
    ) -> str:
        prompt_parts: List[str] = []
        if task_prompt:
            prompt_parts.append(f"Task:\n{task_prompt}")
        if context is not None:
            if isinstance(context, Mapping) or isinstance(context, list):
                context_text = json.dumps(context, ensure_ascii=False, indent=2)
            else:
                context_text = str(context)
            prompt_parts.append(f"Context:\n{context_text}")
        if expected_output_format is not None:
            if isinstance(expected_output_format, Mapping) or isinstance(expected_output_format, list):
                expected_text = json.dumps(expected_output_format, ensure_ascii=False, indent=2)
            else:
                expected_text = str(expected_output_format)
            prompt_parts.append(f"Expected Output Format:\n{expected_text}")
        return "\n\n".join(prompt_parts).strip()

    def _resolve_reasoning_toolset(self, toolset: Any) -> List[Dict[str, Any]]:
        if not isinstance(toolset, list):
            return []

        actions: List[Dict[str, Any]] = []
        for item in toolset:
            if isinstance(item, Mapping):
                actions.append(dict(item))
                continue
            if not isinstance(item, str):
                continue
            action_def = get_action_by_id(item)
            if action_def:
                actions.append(action_def)
                continue
            matched = next(
                (
                    action
                    for action in BUSINESS_ACTIONS
                    if action.get("tool_name") == item
                ),
                None,
            )
            if matched:
                actions.append(matched)
            else:
                log_warn(f"[reasoning] toolset item '{item}' not found in action registry")
        return actions

    def _build_reasoning_tool_schema(self, action: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        parameters = action.get("arg_schema") or action.get("params_schema")
        if not isinstance(parameters, Mapping):
            return None
        action_id = action.get("action_id") or action.get("tool_name") or "tool"
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", action.get("tool_name") or action_id)
        description = action.get("description") or action.get("name") or action_id
        return {
            "type": "function",
            "function": {
                "name": safe_name,
                "description": description,
                "parameters": parameters,
            },
        }

    def _execute_reasoning_tool(
        self, *, action: Mapping[str, Any], args: Mapping[str, Any], call_name: str
    ) -> Dict[str, Any]:
        tool_fn_name = action.get("tool_name") or action.get("action_id") or call_name
        registered_tool = get_registered_tool(tool_fn_name)
        if not registered_tool:
            return {
                "status": "unavailable",
                "action_id": action.get("action_id"),
                "message": f"Business tool '{tool_fn_name}' is not registered; returning arguments.",
                "echo": args,
            }
        try:
            output = registered_tool(**args)
            return {
                "status": "ok",
                "action_id": action.get("action_id"),
                "output": output,
            }
        except Exception as exc:  # pragma: no cover - runtime errors are surfaced to LLM
            return {
                "status": "error",
                "action_id": action.get("action_id"),
                "message": str(exc),
            }

    def _run_reasoning_llm(
        self,
        *,
        system_prompt: str | None,
        prompt_text: str,
        tools: List[Dict[str, Any]],
        max_tokens: int = 512,
        max_rounds: int = 3,
    ) -> Dict[str, Any]:
        client = OpenAI()
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ]
        tools_spec: List[Dict[str, Any]] = []
        tool_lookup: Dict[str, Mapping[str, Any]] = {}
        for action in tools:
            schema = self._build_reasoning_tool_schema(action)
            if not schema:
                continue
            tool_name = schema["function"]["name"]
            tools_spec.append(schema)
            tool_lookup[tool_name] = action

        forced_tool_choice: str | None = None
        for round_idx in range(1, max_rounds + 1):
            request_kwargs = {
                "model": OPENAI_MODEL,
                "messages": messages,
                "tools": tools_spec or None,
                "tool_choice": forced_tool_choice if tools_spec and forced_tool_choice else ("auto" if tools_spec else None),
                "max_completion_tokens": max_tokens,
            }
            try:
                response = client.chat.completions.create(**request_kwargs)
            except BadRequestError as exc:
                message = str(exc)
                if "max_completion_tokens" in message:
                    request_kwargs.pop("max_completion_tokens", None)
                    request_kwargs["max_tokens"] = max_tokens
                    response = client.chat.completions.create(**request_kwargs)
                else:
                    raise

            if not response.choices:
                raise RuntimeError("reasoning LLM returned no choices")

            choice = response.choices[0]
            message = choice.message
            assistant_msg = {"role": "assistant", "content": message.content or ""}
            if message.tool_calls:
                assistant_msg["tool_calls"] = message.tool_calls
            messages.append(assistant_msg)
            log_info(
                "[reasoning] round=%s assistant response received (finish_reason=%s)",
                round_idx,
                choice.finish_reason,
            )

            if not assistant_msg.get("content") and not message.tool_calls:
                messages.append(
                    {
                        "role": "user",
                        "content": "Your previous response had no content. Please provide a concise response.",
                    }
                )
                if tools_spec:
                    forced_tool_choice = "none"
                continue

            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                forced_tool_choice = None
                for tc in tool_calls:
                    func_name = tc.function.name
                    raw_args = tc.function.arguments
                    try:
                        parsed_args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        parsed_args = {"__raw__": raw_args or ""}

                    action_def = tool_lookup.get(func_name)
                    if not action_def:
                        tool_result = {
                            "status": "error",
                            "message": f"Unknown business tool invocation: {func_name}",
                            "echo": parsed_args,
                        }
                    else:
                        tool_result = self._execute_reasoning_tool(
                            action=action_def, args=parsed_args, call_name=func_name
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": json.dumps(tool_result, ensure_ascii=False),
                        }
                    )
                    log_info("[reasoning] round=%s tool_call=%s", round_idx, func_name)
                    log_json("[reasoning] tool_args", parsed_args)
                    log_json("[reasoning] tool_result", tool_result)
                continue

            final_content = (message.content or "").strip()
            if not final_content:
                continue

            try:
                parsed_content = json.loads(final_content)
                results = parsed_content if isinstance(parsed_content, dict) else {"answer": parsed_content}
            except json.JSONDecodeError:
                results = {"answer": final_content}
            return {"status": "ok", "results": results, "messages": messages}

        return {
            "status": "error",
            "results": {"message": "reasoning LLM could not produce an answer within the allowed rounds"},
            "messages": messages,
        }

    def _build_checkpoint(
        self,
        workflow_dict: Dict[str, Any],
        binding_ctx: BindingContext,
        visited: Set[str],
        reachable: Set[str],
        blocked: Set[str],
        sorted_nodes: List[Node],
    ) -> ExecutionCheckpoint:
        pending_ids = [n.id for n in sorted_nodes if n.id not in visited]
        return ExecutionCheckpoint(
            workflow_dict=workflow_dict,
            binding_snapshot=binding_ctx.snapshot(),
            visited=list(visited),
            reachable=list(reachable),
            pending_ids=pending_ids,
            blocked=list(blocked),
        )

    def _execute_graph(
        self,
        workflow_dict: Dict[str, Any],
        binding_ctx: BindingContext,
        start_nodes: Optional[List[str]] = None,
        checkpoint: ExecutionCheckpoint | None = None,
    ) -> Union[Dict[str, Any], WorkflowSuspension]:
        workflow = Workflow.model_validate(workflow_dict)
        nodes_data = {n["id"]: n for n in workflow.model_dump(by_alias=True)["nodes"]}
        sorted_nodes = self._topological_sort(workflow)
        edges = self._derive_edges(workflow)
        depends_map = infer_depends_on_from_edges(nodes_data.values(), edges)

        graph_label = workflow_dict.get("workflow_name", "")
        log_section("执行工作流", graph_label)
        log_kv("描述", workflow_dict.get("description", ""))
        log_kv("模拟用户角色", self.user_role)

        if checkpoint:
            visited: Set[str] = set(checkpoint.visited)
            reachable: Set[str] = set(checkpoint.reachable)
            blocked: Set[str] = set(checkpoint.blocked)
            pending: List[Node] = [
                self.node_models[nid]
                for nid in checkpoint.pending_ids
                if nid in self.node_models
            ]
        else:
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
            start_nodes = list(dict.fromkeys(start_nodes)) if start_nodes else []
            if not start_nodes:
                start_nodes = zero_indegree
            if not start_nodes and sorted_nodes:
                log_warn("未找到入口节点，将从任意一个节点开始（仅 demo）。")
                start_nodes = [sorted_nodes[0].id]

            visited = set()
            reachable = set(start_nodes)
            blocked: Set[str] = set()
            pending = list(sorted_nodes)

        results = binding_ctx.results

        while pending:
            progress = False
            remaining: List[Node] = []
            for node_model in pending:
                nid = node_model.id
                deps = depends_map.get(nid, [])
                if (
                    nid in visited
                    or nid in blocked
                    or (reachable and nid not in reachable)
                    or any(dep not in visited for dep in deps)
                ):
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
                            tool_name = action.get("tool_name")
                            if self._should_simulate(action_id):
                                result = self._simulate_action(action_id, resolved_params)
                            elif tool_name:
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
                    next_ids = self._next_nodes(edges, nid, nodes_data=nodes_data)
                    for nxt in next_ids:
                        if nxt not in visited:
                            reachable.add(nxt)
                    if isinstance(payload, Mapping) and payload.get("status") == "async_pending":
                        checkpoint = self._build_checkpoint(
                            workflow_dict,
                            binding_ctx,
                            visited,
                            reachable,
                            blocked,
                            sorted_nodes,
                        )
                        suspension = WorkflowSuspension(
                            checkpoint=checkpoint,
                            node_id=nid,
                            request_id=str(payload.get("request_id", "")),
                            tool_name=tool_name or action_id or "",
                        )
                        log_event(
                            "workflow_suspended",
                            {
                                "node_id": nid,
                                "request_id": suspension.request_id,
                                "tool_name": suspension.tool_name,
                            },
                            node_id=nid,
                            action_id=action_id,
                        )
                        return suspension
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

                if ntype == "reasoning":
                    resolved_params = eval_node_params(node_model, binding_ctx)
                    log_json("resolved params", resolved_params)

                    system_prompt = (
                        resolved_params.get("system_prompt")
                        if isinstance(resolved_params.get("system_prompt"), str)
                        else None
                    )
                    task_prompt = (
                        resolved_params.get("task_prompt")
                        if isinstance(resolved_params.get("task_prompt"), str)
                        else None
                    )
                    context = resolved_params.get("context")
                    expected_output_format = resolved_params.get("expected_output_format")
                    toolset = resolved_params.get("toolset")

                    prompt_text = self._build_reasoning_prompt(
                        task_prompt=task_prompt,
                        context=context,
                        expected_output_format=expected_output_format,
                    )
                    if not prompt_text and task_prompt:
                        prompt_text = task_prompt
                    if not prompt_text:
                        prompt_text = "Follow the expected output format and return the result."

                    tools_for_reasoning = self._resolve_reasoning_toolset(toolset)
                    result = self._run_reasoning_llm(
                        system_prompt=system_prompt,
                        prompt_text=prompt_text,
                        tools=tools_for_reasoning,
                    )

                    result_payload = result.get("results") if isinstance(result, Mapping) else None
                    if isinstance(result_payload, Mapping):
                        results[nid] = dict(result_payload)
                    else:
                        results[nid] = {"output": result_payload}

                    self._record_node_metrics(result)
                    next_ids = self._next_nodes(edges, nid, nodes_data=nodes_data)
                    for nxt in next_ids:
                        if nxt not in visited:
                            reachable.add(nxt)
                    log_event(
                        "node_end",
                        {
                            "node_id": nid,
                            "type": ntype,
                            "resolved_params": resolved_params,
                            "result": result,
                            "next_nodes": next_ids,
                        },
                        node_id=nid,
                        action_id=action_id,
                    )
                    continue

                if ntype == "data":
                    resolved_params = eval_node_params(node_model, binding_ctx)
                    log_json("resolved params", resolved_params)

                    payload = {
                        "schema": resolved_params.get("schema"),
                        "dataset": resolved_params.get("dataset"),
                    }
                    results[nid] = payload
                    self._record_node_metrics(payload)
                    next_ids = self._next_nodes(edges, nid, nodes_data=nodes_data)
                    for nxt in next_ids:
                        if nxt not in visited:
                            reachable.add(nxt)
                    log_event(
                        "node_end",
                        {
                            "node_id": nid,
                            "type": ntype,
                            "resolved_params": resolved_params,
                            "result": payload,
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
                        edges,
                        nid,
                        cond_value=cond_value,
                        nodes_data=nodes_data,
                    )
                    inactive_branch = (
                        node.get("false_to_node")
                        if cond_value is True
                        else node.get("true_to_node")
                    )
                    if isinstance(inactive_branch, str):
                        blocked_nodes = self._collect_downstream_nodes(
                            edges, inactive_branch
                        )
                        blocked.update(blocked_nodes)
                        reachable.difference_update(blocked_nodes)
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
                            edges,
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
                        edges, nid, nodes_data=nodes_data
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
                    edges, nid, nodes_data=nodes_data
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
                        self.run_manager.metrics.extra["result_nodes"] = list(
                            binding_ctx.results.keys()
                        )
                    log_event(
                        "executor_finished",
                        {
                            "workflow_name": self.workflow.workflow_name,
                            "result_nodes": list(binding_ctx.results.keys()),
                        },
                        context=span_ctx,
                    )
                    return results

        results = self._execute_graph(self.workflow_dict, binding_ctx)
        if self.run_manager:
            self.run_manager.metrics.extra["result_nodes"] = list(
                binding_ctx.results.keys()
            )
        return results

    def resume_from_suspension(
        self,
        suspension: WorkflowSuspension,
        tool_result: Any | None = None,
        trace_context: TraceContext | None = None,
    ) -> Union[Dict[str, Any], WorkflowSuspension]:
        """Resume workflow execution once an async tool finishes."""

        checkpoint = suspension.checkpoint
        binding_ctx = BindingContext.from_snapshot(
            self.workflow, checkpoint.binding_snapshot
        )
        previous_payload = binding_ctx.results.get(suspension.node_id, {})

        if tool_result is None:
            tool_result = GLOBAL_ASYNC_RESULT_STORE.pop_result(
                suspension.request_id
            )
            if tool_result is None:
                raise ValueError(
                    f"未找到异步请求 {suspension.request_id} 的结果，无法恢复执行"
                )

        payload: Dict[str, Any] = (
            tool_result.copy() if isinstance(tool_result, dict) else {"value": tool_result}
        )
        if isinstance(previous_payload, Mapping) and "params" in previous_payload:
            payload.setdefault("params", previous_payload.get("params"))
        payload.setdefault("request_id", suspension.request_id)
        original_status = payload.get("status")
        payload["status"] = "async_resolved"
        if original_status and original_status != payload["status"]:
            payload.setdefault("tool_status", original_status)

        binding_ctx.results[suspension.node_id] = payload

        ctx = trace_context or self.trace_context
        if ctx:
            with use_trace_context(ctx):
                return self._execute_graph(
                    checkpoint.workflow_dict, binding_ctx, checkpoint=checkpoint
                )

        return self._execute_graph(
            checkpoint.workflow_dict, binding_ctx, checkpoint=checkpoint
        )

    def run(self, trace_context: TraceContext | None = None):
        return self.run_workflow(trace_context=trace_context)
