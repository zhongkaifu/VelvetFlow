"""Workflow execution utilities."""
import copy
import json
import os
import re
from typing import Any, Dict, List, Mapping, Optional, Set, Union

from velvetflow.bindings import BindingContext, eval_node_params
from velvetflow.action_registry import get_action_by_id
from velvetflow.logging_utils import (
    log_debug,
    log_event,
    log_info,
    log_json,
    log_kv,
    log_section,
    log_success,
    log_warn,
)
from velvetflow.models import Node, ValidationError, Workflow

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
    def __init__(
        self,
        workflow: Workflow,
        simulations: Union[str, Mapping[str, Any], None] = None,
        user_role: str = "user",
    ):
        if not isinstance(workflow, Workflow):
            raise ValueError(f"workflow 必须是 Workflow 对象，当前类型为 {type(workflow)}")

        workflow_dict = workflow.model_dump(by_alias=True)
        if not isinstance(workflow_dict.get("nodes"), list):
            raise ValueError("workflow 缺少合法的 'nodes' 列表。")
        if not isinstance(workflow_dict.get("edges"), list):
            raise ValueError("workflow 缺少合法的 'edges' 列表。")

        self.workflow = workflow
        self.workflow_dict = workflow_dict
        self.node_models = {n.id: n for n in workflow.nodes}
        self.nodes = {n["id"]: n for n in workflow_dict["nodes"]}
        self.edges = workflow_dict["edges"]
        self.user_role = user_role

        if isinstance(simulations, str):
            self.simulation_data: SimulationData = load_simulation_data(simulations)
        elif simulations is None:
            self.simulation_data = {}
        elif isinstance(simulations, Mapping):
            self.simulation_data = dict(simulations)
        else:
            raise ValueError("simulations 必须是 None、字典或 JSON 文件路径")

    def _find_start_nodes(self, nodes: Mapping[str, Dict[str, Any]], edges: List[Dict[str, Any]]) -> List[str]:
        starts = [nid for nid, n in nodes.items() if n.get("type") == "start"]
        if not starts:
            all_ids = set(nodes.keys())
            to_ids = {e["to"] for e in edges}
            starts = list(all_ids - to_ids)
        return starts

    def _topological_sort(self, workflow: Workflow) -> List[Node]:
        nodes = workflow.nodes
        edges = workflow.edges

        indegree: Dict[str, int] = {n.id: 0 for n in nodes}
        adjacency: Dict[str, List[str]] = {n.id: [] for n in nodes}
        node_lookup: Dict[str, Node] = {n.id: n for n in nodes}
        for e in edges:
            indegree[e.to_node] += 1
            adjacency[e.from_node].append(e.to_node)

        queue: List[str] = [nid for nid, deg in indegree.items() if deg == 0]
        if not queue and nodes:
            raise ValueError(
                ValidationError(
                    code="DISCONNECTED_GRAPH",
                    node_id=None,
                    field=None,
                    message="未找到入度为 0 的节点，无法确定拓扑起点。",
                )
            )

        ordered: List[str] = []
        while queue:
            nid = queue.pop(0)
            ordered.append(nid)
            for nxt in adjacency.get(nid, []):
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    queue.append(nxt)

        if len(ordered) != len(nodes):
            log_warn("可能是 LLM 生成了有环的工作流")
            raise ValueError(
                ValidationError(
                    code="CYCLE_DETECTED",
                    node_id=None,
                    field=None,
                    message="工作流中存在环，无法拓扑排序。",
                )
            )

        edges_dump = [e.model_dump(by_alias=True) for e in edges]
        nodes_dump = {n.id: n.model_dump(by_alias=True) for n in nodes}
        start_nodes = self._find_start_nodes(nodes_dump, edges_dump)
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

        if nodes and (not start_nodes or len(reachable) != len(nodes)):
            raise ValueError(
                ValidationError(
                    code="DISCONNECTED_GRAPH",
                    node_id=None,
                    field=None,
                    message="存在无法从 start 节点到达的节点。",
                )
            )

        return [node_lookup[nid] for nid in ordered]

    def _next_nodes(self, edges: List[Dict[str, Any]], nid: str, cond_value: Any = None) -> List[str]:
        res: List[str] = []
        cond_label: Optional[str]
        if isinstance(cond_value, bool):
            cond_label = "true" if cond_value else "false"
        elif cond_value is None:
            cond_label = None
        else:
            cond_label = str(cond_value)

        for e in edges:
            if e.get("from") != nid:
                continue
            cond = e.get("condition")
            if cond is None:
                res.append(e.get("to"))
            elif cond_label is not None and cond == cond_label:
                res.append(e.get("to"))
        return [r for r in res if r]

    def _render_template(self, value: Any, params: Mapping[str, Any]) -> Any:
        if isinstance(value, str):
            pattern = r"\{\{([^{}]+)\}\}"

            def _replace(match: re.Match[str]) -> str:
                key = match.group(1)
                return str(params.get(key, match.group(0)))

            return re.sub(pattern, _replace, value)

        if isinstance(value, list):
            return [self._render_template(v, params) for v in value]

        if isinstance(value, dict):
            return {k: self._render_template(v, params) for k, v in value.items()}

        return value

    def _simulate_action(self, action_id: str, resolved_params: Dict[str, Any]) -> Dict[str, Any]:
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

    def _resolve_condition_source(self, source: Any, ctx: BindingContext) -> Any:
        """Resolve condition source which may be a binding dict or a path string."""

        if isinstance(source, dict) and "__from__" in source:
            return ctx.resolve_binding(source)

        if isinstance(source, str):
            return ctx.get_value(source)

        return source

    def _eval_condition(self, node: Dict[str, Any], ctx: BindingContext) -> Any:
        params = node.get("params") or {}
        kind = params.get("kind")
        if not kind:
            log_warn("[condition] 未指定 kind，默认 False")
            return False

        def _safe_get_source() -> Any:
            source = params.get("source")
            try:
                return self._resolve_condition_source(source, ctx)
            except Exception as e:
                log_warn(f"[condition:{kind}] source 路径 '{source}' 无法从 context 读取: {e}，返回 False")
                return None

        data = _safe_get_source()

        if kind == "list_not_empty":
            if not isinstance(data, list):
                log_warn("[condition:list_not_empty] source 不是 list，返回 False")
                return False
            return len(data) > 0

        if kind == "is_empty":
            if data is None:
                return True
            if isinstance(data, (list, dict, str)):
                return len(data) == 0
            return False

        if kind == "not_empty":
            if data is None:
                return False
            if isinstance(data, (list, dict, str)):
                return len(data) > 0
            return True

        if kind == "any_greater_than":
            field = params.get("field")
            threshold = params.get("threshold")
            if not isinstance(data, list):
                log_warn("[condition:any_greater_than] source 不是 list，返回 False")
                return False
            return any((item.get(field, 0) > threshold) for item in data if isinstance(item, dict))

        if kind == "all_less_than":
            field = params.get("field")
            threshold = params.get("threshold")
            if not isinstance(data, list):
                log_warn("[condition:all_less_than] source 不是 list，返回 False")
                return False
            filtered = [item for item in data if isinstance(item, dict) and field in item]
            return bool(filtered) and all((item.get(field, 0) < threshold) for item in filtered)

        if kind == "equals":
            value = params.get("value")
            return data == value

        if kind == "not_equals":
            value = params.get("value")
            return data != value

        if kind == "greater_than":
            threshold = params.get("threshold")
            try:
                return data is not None and data > threshold
            except Exception:
                return False

        if kind == "less_than":
            threshold = params.get("threshold")
            try:
                return data is not None and data < threshold
            except Exception:
                return False

        if kind == "between":
            min_v = params.get("min")
            max_v = params.get("max")
            try:
                return data is not None and data >= min_v and data <= max_v
            except Exception:
                return False

        if kind == "contains":
            field = params.get("field")
            value = params.get("value")

            def _match_target(target: Any) -> bool:
                if target is None:
                    return False
                target_str = target if isinstance(target, str) else str(target)
                return str(value) in target_str

            if isinstance(data, list):
                if field is None:
                    log_warn("[condition:contains] 未提供 field，且 source 是列表，返回 False")
                    return False
                for item in data:
                    if isinstance(item, dict):
                        if _match_target(item.get(field)):
                            return True
                    elif _match_target(item):
                        return True
                return False

            if isinstance(data, dict):
                if field is None:
                    log_warn("[condition:contains] 未提供 field，且 source 是字典，返回 False")
                    return False
                return _match_target(data.get(field))

            if isinstance(data, str):
                return _match_target(data)

            log_warn("[condition:contains] source 不是列表/字典/字符串，返回 False")
            return False

        if kind == "multi_band":
            bands = params.get("bands") or []
            if data is None or not bands:
                return None
            try:
                value = float(data)
            except Exception:
                return None
            for band in bands:
                label = band.get("label")
                max_v = band.get("max")
                try:
                    if max_v is not None and value <= max_v:
                        return label
                except Exception:
                    continue
            return bands[-1].get("label") if isinstance(bands[-1], dict) else None

        log_warn(f"[condition] 未知 kind={kind}，默认 False")
        return False

    def _execute_loop_node(self, node: Dict[str, Any], binding_ctx: BindingContext) -> Dict[str, Any]:
        params = node.get("params") or {}
        loop_kind = params.get("loop_kind", "for_each")
        body_graph = params.get("body_subgraph") or {}
        body_nodes = body_graph.get("nodes") or []
        body_edges = body_graph.get("edges") or []
        entry = body_graph.get("entry")
        exit_node = body_graph.get("exit")

        extra_node_models: Dict[str, Node] = {}
        try:
            sub_wf = Workflow.model_validate(
                {"workflow_name": "loop_body", "nodes": body_nodes, "edges": body_edges}
            )
            extra_node_models = {n.id: n for n in sub_wf.nodes}
        except Exception:
            extra_node_models = {}

        accumulator = params.get("accumulator") or {}
        max_iterations = params.get("max_iterations") or 100
        iterations: List[Dict[str, Any]] = []

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
                loop_ctx = {"index": idx, "item": item, "size": size, "accumulator": accumulator}
                loop_binding = BindingContext(
                    self.workflow, binding_ctx.results, extra_nodes=extra_node_models, loop_ctx=loop_ctx
                )
                self._execute_graph(
                    {
                        "workflow_name": f"{node.get('id')}_iter_{idx}",
                        "description": body_graph.get("description", ""),
                        "nodes": body_nodes,
                        "edges": body_edges,
                    },
                    loop_binding,
                    start_nodes=[entry] if entry else None,
                )

                iter_result: Dict[str, Any] = {"index": idx, "item": item}
                if exit_node and exit_node in binding_ctx.results:
                    iter_result["exit_result"] = binding_ctx.results.get(exit_node)
                iterations.append(iter_result)

            return {
                "status": "loop_completed",
                "loop_kind": loop_kind,
                "iterations": iterations,
                "accumulator": accumulator,
            }

        if loop_kind == "while":
            cond_def = params.get("condition") or {}
            iteration = 0
            while iteration < max_iterations:
                cond_value = self._eval_condition({"params": cond_def}, binding_ctx)
                if not cond_value:
                    break
                loop_ctx = {
                    "index": iteration,
                    "size": max_iterations,
                    "accumulator": accumulator,
                }
                loop_binding = BindingContext(
                    self.workflow, binding_ctx.results, extra_nodes=extra_node_models, loop_ctx=loop_ctx
                )
                self._execute_graph(
                    {
                        "workflow_name": f"{node.get('id')}_while_{iteration}",
                        "description": body_graph.get("description", ""),
                        "nodes": body_nodes,
                        "edges": body_edges,
                    },
                    loop_binding,
                    start_nodes=[entry] if entry else None,
                )

                iter_result: Dict[str, Any] = {"index": iteration}
                if exit_node and exit_node in binding_ctx.results:
                    iter_result["exit_result"] = binding_ctx.results.get(exit_node)
                iterations.append(iter_result)
                iteration += 1

            return {
                "status": "loop_completed",
                "loop_kind": loop_kind,
                "iterations": iterations,
                "accumulator": accumulator,
            }

        log_warn(f"[loop] 未知 loop_kind={loop_kind}，跳过执行")
        return {"status": "skipped", "reason": "unknown_loop_kind"}

    def _execute_graph(
        self,
        workflow_dict: Dict[str, Any],
        binding_ctx: BindingContext,
        start_nodes: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        workflow = Workflow.model_validate(workflow_dict)
        nodes_data = {n["id"]: n for n in workflow.model_dump(by_alias=True)["nodes"]}
        edges = workflow.model_dump(by_alias=True)["edges"]
        sorted_nodes = self._topological_sort(workflow)

        graph_label = workflow_dict.get("workflow_name", "")
        log_section("执行工作流", graph_label)
        log_kv("描述", workflow_dict.get("description", ""))
        log_kv("模拟用户角色", self.user_role)

        if start_nodes is None:
            start_nodes = self._find_start_nodes(nodes_data, edges)
        if not start_nodes and sorted_nodes:
            log_warn("未找到 start 节点，将从任意一个节点开始（仅 demo）。")
            start_nodes = [sorted_nodes[0].id]

        visited: Set[str] = set()
        reachable: Set[str] = set(start_nodes)
        results = binding_ctx.results

        for node_model in sorted_nodes:
            nid = node_model.id
            if nid in visited or (reachable and nid not in reachable):
                continue
            visited.add(nid)

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
                        result = self._simulate_action(action_id, resolved_params)

                payload = result.copy() if isinstance(result, dict) else {"value": result}
                payload["params"] = resolved_params
                results[nid] = payload
                log_event(
                    "node_end",
                    {
                        "node_id": nid,
                        "type": ntype,
                        "action_id": action_id,
                        "resolved_params": resolved_params,
                        "result": result,
                    },
                )

            elif ntype == "condition":
                cond_value = self._eval_condition(node, binding_ctx)
                log_info(f"[condition] 结果: {cond_value}")
                results[nid] = {"condition_result": cond_value}
                next_ids = self._next_nodes(edges, nid, cond_value=cond_value)
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
                )
                continue

            elif ntype == "loop":
                loop_result = self._execute_loop_node(node, binding_ctx)
                results[nid] = loop_result
                log_event(
                    "node_end",
                    {
                        "node_id": nid,
                        "type": ntype,
                        "result": loop_result,
                    },
                )
                next_ids = self._next_nodes(edges, nid)
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
                )
                continue

            next_ids = self._next_nodes(edges, nid)
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
            )

        log_success("执行结束")
        return results

    def run_workflow(self):
        binding_ctx = BindingContext(self.workflow, {})
        self._execute_graph(self.workflow_dict, binding_ctx)

    def run(self):
        self.run_workflow()


