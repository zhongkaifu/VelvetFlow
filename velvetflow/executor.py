"""Workflow execution utilities."""
import copy
import json
import os
import re
from typing import Any, Dict, List, Mapping, Union

from velvetflow.bindings import BindingContext, eval_node_params
from velvetflow.action_registry import get_action_by_id
from velvetflow.models import Workflow

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
        self, workflow: Workflow, simulations: Union[str, Mapping[str, Any], None] = None
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

        if isinstance(simulations, str):
            self.simulation_data: SimulationData = load_simulation_data(simulations)
        elif simulations is None:
            self.simulation_data = {}
        elif isinstance(simulations, Mapping):
            self.simulation_data = dict(simulations)
        else:
            raise ValueError("simulations 必须是 None、字典或 JSON 文件路径")

    def find_start_nodes(self) -> List[str]:
        starts = [nid for nid, n in self.nodes.items() if n.get("type") == "start"]
        if not starts:
            all_ids = set(self.nodes.keys())
            to_ids = {e["to"] for e in self.edges}
            starts = list(all_ids - to_ids)
        return starts

    def next_nodes(self, nid: str, cond_value: bool = True) -> List[str]:
        res = []
        for e in self.edges:
            if e["from"] != nid:
                continue
            cond = e.get("condition")
            if cond is None:
                res.append(e["to"])
            else:
                if cond == "true" and cond_value:
                    res.append(e["to"])
                elif cond == "false" and not cond_value:
                    res.append(e["to"])
        return res

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

    def _eval_condition(self, node: Dict[str, Any], ctx: BindingContext) -> bool:
        params = node.get("params") or {}
        kind = params.get("kind")
        if not kind:
            print("  [condition] 未指定 kind，默认 False")
            return False

        if kind == "any_greater_than":
            source = params["source"]
            field = params["field"]
            threshold = params["threshold"]
            try:
                data = ctx.get_value(source)
            except Exception as e:
                print(f"  [condition:any_greater_than] source 路径 '{source}' 无法从 context 读取: {e}，返回 False")
                return False
            if not isinstance(data, list):
                print("  [condition:any_greater_than] source 不是 list，返回 False")
                return False
            return any((item.get(field, 0) > threshold) for item in data if isinstance(item, dict))

        if kind == "equals":
            source = params["source"]
            value = params["value"]
            try:
                data = ctx.get_value(source)
            except Exception as e:
                print(f"  [condition:equals] source 路径 '{source}' 无法从 context 读取: {e}，返回 False")
                return False
            return data == value

        print(f"  [condition] 未知 kind={kind}，默认 False")
        return False

    def run(self):
        print("\n==== 执行工作流 ====")
        print("名称：", self.workflow_dict.get("workflow_name"))
        print("描述：", self.workflow_dict.get("description", ""))
        print("================================\n")

        start_nodes = self.find_start_nodes()
        if not start_nodes:
            print("未找到 start 节点，将从任意一个节点开始（仅 demo）。")
            start_nodes = [next(iter(self.nodes))]

        visited = set()
        queue = list(start_nodes)
        results: Dict[str, Any] = {}
        binding_ctx = BindingContext(self.workflow, results)

        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)

            node = self.nodes[nid]
            ntype = node.get("type")
            action_id = node.get("action_id")
            display_name = node.get("display_name") or action_id or ntype
            params = node.get("params", {})

            print(f"[Node {nid}] type={ntype}, display_name={display_name}, action_id={action_id}")
            if params:
                print("  raw params =", json.dumps(params, ensure_ascii=False))

            if ntype == "action" and action_id:
                resolved_params = eval_node_params(self.node_models[nid], binding_ctx)
                print("  resolved params =", json.dumps(resolved_params, ensure_ascii=False))

                action = get_action_by_id(action_id)
                if not action:
                    print(f"  [WARN] 未在 Registry 中找到 action_id={action_id}")
                    result = {"status": "no_action_impl"}
                else:
                    print(f"  -> 执行业务动作: {action['name']} (domain={action['domain']})")
                    print(f"  -> 描述: {action['description']}")
                    result = self._simulate_action(action_id, resolved_params)

                results[nid] = result

            elif ntype == "condition":
                cond_value = self._eval_condition(node, binding_ctx)
                print(f"  [condition] 结果: {cond_value}")
                next_ids = self.next_nodes(nid, cond_value=cond_value)
                for nxt in next_ids:
                    if nxt not in visited:
                        queue.append(nxt)
                continue

            next_ids = self.next_nodes(nid)
            for nxt in next_ids:
                if nxt not in visited:
                    queue.append(nxt)

        print("\n==== 执行结束 ====\n")


