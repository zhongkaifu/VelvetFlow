"""Workflow execution utilities."""
import copy
import json
import os
import re
from typing import Any, Dict, List, Mapping, Union

from velvetflow.action_registry import get_action_by_id

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
    def __init__(self, workflow: Dict[str, Any], simulations: Union[str, Mapping[str, Any], None] = None):
        if not isinstance(workflow, dict):
            raise ValueError(f"workflow 必须是 dict，当前类型为 {type(workflow)}")
        if not isinstance(workflow.get("nodes"), list):
            raise ValueError("workflow 缺少合法的 'nodes' 列表。")
        if not isinstance(workflow.get("edges"), list):
            raise ValueError("workflow 缺少合法的 'edges' 列表。")

        self.workflow = workflow
        self.nodes = {n["id"]: n for n in workflow["nodes"]}
        self.edges = workflow["edges"]

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

    def _get_from_context(self, context: Dict[str, Any], path: str):
        if not path:
            raise KeyError("空路径")

        parts = path.split(".")

        if parts[0] == "result_of" and len(parts) >= 2:
            first_key = f"result_of.{parts[1]}"
            if first_key not in context:
                raise KeyError(first_key)
            cur: Any = context[first_key]
            rest = parts[2:]
        else:
            if parts[0] not in context:
                raise KeyError(parts[0])
            cur = context[parts[0]]
            rest = parts[1:]

        for p in rest:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
                continue

            # When the current value is a list of dicts, allow selecting a
            # sub-field from every element. This prevents parameter resolving
            # from failing on paths like ``result_of.xxx.data.employee_id``
            # where ``data`` is a list of objects.
            if isinstance(cur, list):
                extracted = []
                for item in cur:
                    if isinstance(item, dict) and p in item:
                        extracted.append(item[p])
                if extracted:
                    cur = extracted
                    continue
                raise KeyError(p)

            raise KeyError(p)

        return cur

    def _resolve_param_value(self, value: Any, context: Dict[str, Any]) -> Any:
        if not isinstance(value, dict) or "__from__" not in value:
            return value

        src_path = value["__from__"]
        data = self._get_from_context(context, src_path)
        agg = value.get("__agg__", "identity")

        if agg == "identity":
            return data

        if agg == "count_if":
            if not isinstance(data, list):
                return 0
            field = value.get("field") or value.get("filter_field")
            op = value.get("op") or value.get("filter_op") or "=="
            target = value.get("value") or value.get("filter_value")
            count = 0
            for item in data:
                # Support both list-of-dicts and list-of-scalars
                if isinstance(item, dict):
                    v = item.get(field)
                else:
                    v = item

                try:
                    if op == ">" and v is not None and v > target:
                        count += 1
                    elif op == ">=" and v is not None and v >= target:
                        count += 1
                    elif op == "<" and v is not None and v < target:
                        count += 1
                    elif op == "<=" and v is not None and v <= target:
                        count += 1
                    elif op == "==" and v == target:
                        count += 1
                except TypeError:
                    # If comparison fails (e.g., string vs number), skip the item
                    continue
            return count

        if agg == "filter_map":
            filter_field = value.get("filter_field")
            filter_op = value.get("filter_op", "==")
            filter_value = value.get("filter_value")
            map_field = value.get("map_field")
            fmt = value.get("format", "{value}")
            sep = value.get("sep", "\n")

            pipeline_value = {
                "__from__": src_path,
                "__agg__": "pipeline",
                "steps": [
                    {
                        "op": "filter",
                        "field": filter_field,
                        "cmp": filter_op,
                        "value": filter_value,
                    },
                    {
                        "op": "map",
                        "field": map_field,
                    },
                    {
                        "op": "format_join",
                        "format": fmt,
                        "sep": sep,
                    },
                ],
            }
            return self._resolve_param_value(pipeline_value, context)

        if agg == "pipeline":
            steps = value.get("steps") or []
            current = data

            for step in steps:
                if not isinstance(step, dict):
                    continue
                op = step.get("op")

                if op == "filter":
                    if not isinstance(current, list):
                        continue
                    field = step.get("field") or step.get("filter_field")
                    cmp_op = step.get("cmp") or step.get("filter_op") or "=="
                    cmp_val = step.get("value") or step.get("filter_value")
                    filtered = []
                    for item in current:
                        if not isinstance(item, dict):
                            continue
                        v = item.get(field)
                        passed = False
                        if cmp_op == ">" and v is not None and v > cmp_val:
                            passed = True
                        elif cmp_op == ">=" and v is not None and v >= cmp_val:
                            passed = True
                        elif cmp_op == "<" and v is not None and v < cmp_val:
                            passed = True
                        elif cmp_op == "<=" and v is not None and v <= cmp_val:
                            passed = True
                        elif cmp_op == "==" and v == cmp_val:
                            passed = True
                        elif cmp_op in (None, "", "always"):
                            passed = True
                        if passed:
                            filtered.append(item)
                    current = filtered

                elif op == "map":
                    if not isinstance(current, list):
                        continue
                    field = step.get("field")
                    mapped = []
                    for item in current:
                        if isinstance(item, dict):
                            mapped.append(item.get(field))
                        else:
                            mapped.append(item)
                    current = mapped

                elif op == "format_join":
                    fmt = step.get("format", "{value}")
                    sep = step.get("sep", "\n")
                    if not isinstance(current, list):
                        return fmt.replace("{value}", str(current))
                    lines: List[str] = []
                    for v in current:
                        lines.append(fmt.replace("{value}", str(v)))
                    return sep.join(lines)

            return current

        return value

    def _resolve_params(self, params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        resolved = {}
        for k, v in params.items():
            if isinstance(v, dict) and "__from__" in v:
                try:
                    resolved_value = self._resolve_param_value(v, context)
                except Exception as e:
                    print(f"  [param-resolver] 解析参数 {k} 失败: {e}，使用原值")
                    resolved[k] = v
                else:
                    resolved[k] = resolved_value
            else:
                resolved[k] = v
        return resolved

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

    def _eval_condition(self, node: Dict[str, Any], context: Dict[str, Any]) -> bool:
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
                data = self._get_from_context(context, source)
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
                data = self._get_from_context(context, source)
            except Exception as e:
                print(f"  [condition:equals] source 路径 '{source}' 无法从 context 读取: {e}，返回 False")
                return False
            return data == value

        print(f"  [condition] 未知 kind={kind}，默认 False")
        return False

    def run(self):
        print("\n==== 执行工作流 ====")
        print("名称：", self.workflow.get("workflow_name"))
        print("描述：", self.workflow.get("description", ""))
        print("================================\n")

        start_nodes = self.find_start_nodes()
        if not start_nodes:
            print("未找到 start 节点，将从任意一个节点开始（仅 demo）。")
            start_nodes = [next(iter(self.nodes))]

        visited = set()
        queue = list(start_nodes)
        context: Dict[str, Any] = {}

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
                resolved_params = self._resolve_params(params, context)
                print("  resolved params =", json.dumps(resolved_params, ensure_ascii=False))

                action = get_action_by_id(action_id)
                if not action:
                    print(f"  [WARN] 未在 Registry 中找到 action_id={action_id}")
                    result = {"status": "no_action_impl"}
                else:
                    print(f"  -> 执行业务动作: {action['name']} (domain={action['domain']})")
                    print(f"  -> 描述: {action['description']}")
                    result = self._simulate_action(action_id, resolved_params)

                context[f"result_of.{nid}"] = result

            elif ntype == "condition":
                cond_value = self._eval_condition(node, context)
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


