"""LLM-powered parameter completion utilities."""

import copy
import json
import os
from collections import deque
from typing import Any, Dict, List

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_debug, log_error
from velvetflow.models import Node, Workflow
from velvetflow.planner.relations import get_upstream_nodes


def _find_start_nodes_for_params(workflow: Workflow) -> List[str]:
    starts = [n.id for n in workflow.nodes if n.type == "start"]
    if starts:
        return starts

    to_ids = {e.to_node for e in workflow.edges}
    candidates = [n.id for n in workflow.nodes if n.id not in to_ids]
    if candidates:
        return candidates

    return [workflow.nodes[0].id] if workflow.nodes else []


def _traverse_order(workflow: Workflow) -> List[str]:
    """按 start -> downstream 的顺序遍历，保证上游节点先被处理。"""

    adj: Dict[str, List[str]] = {}
    for e in workflow.edges:
        adj.setdefault(e.from_node, []).append(e.to_node)

    visited: set[str] = set()
    order: List[str] = []
    dq: deque[str] = deque(_find_start_nodes_for_params(workflow))

    while dq:
        nid = dq.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        order.append(nid)
        for nxt in adj.get(nid, []):
            if nxt not in visited:
                dq.append(nxt)

    for node in workflow.nodes:
        if node.id not in visited:
            order.append(node.id)

    return order


def fill_params_with_llm(
    workflow_skeleton: Dict[str, Any],
    action_registry: List[Dict[str, Any]],
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    action_schemas = {}
    for a in action_registry:
        aid = a["action_id"]
        action_schemas[aid] = {
            "name": a.get("name", ""),
            "description": a.get("description", ""),
            "domain": a.get("domain", ""),
            "arg_schema": a.get("arg_schema"),
            "output_schema": a.get("output_schema"),
        }

    system_prompt = (
        "你是一个工作流参数补全助手。一次只处理一个节点，请根据给定的 arg_schema 补齐当前节点的 params。\n"
        "当某个字段需要引用其他节点的输出时，必须使用数据绑定 DSL，并且只能引用提供的 allowed_node_ids 中的节点。\n"
        "你只能从以下节点中读取上下游结果：result_of.<node_id>.<field>...，其中 node_id 必须来自 allowed_node_ids，field 必须存在于该节点的 output_schema。\n"
        "当引用循环节点时，只能使用 loop 节点的 exports（如 result_of.<loop_id>.items / result_of.<loop_id>.aggregates.xxx），禁止直接引用 loop body 的节点。\n"
        "start/end 节点可以保持 params 为空。\n"
        "返回 JSON：{\"id\": <当前节点 id>, \"params\": { ...补全后的参数... }}，不要加代码块标记。\n\n"
        "【重要说明：示例仅为模式，不代表具体业务】\n"
        "示例（字段名仅示意）：\n"
        "- 直接引用：{\"__from__\": \"result_of.some_node.items\", \"__agg__\": \"identity\"}\n"
        "- 列表计数：{\"__from__\": \"result_of.some_node.items\", \"__agg__\": \"count\"}\n"
        "- 条件计数：{\"__from__\": \"result_of.some_node.items\", \"__agg__\": \"count_if\", \"field\": \"value\", \"op\": \">\", \"value\": 10}\n"
        "- 直接格式化并拼接：{\"__from__\": \"result_of.some_node.items\", \"__agg__\": \"format_join\", \"format\": \"{name}: {score}\", \"sep\": "
        "\\n\"}\n"
        "- pipeline：{\"__from__\": \"result_of.list_node.items\", \"__agg__\": \"pipeline\", \"steps\": [{\"op\": \"filter\", \"field\": \"score\", \"cmp\": \">\", \"value\": 0.8}, {\"op\": \"format_join\", \"field\": \"id\", \"format\": \"ID={value} 异常\", \"sep\": \"\\n\"}]}\n"
        "示例中的节点名/字段名只是格式说明，实际必须使用 payload 中的节点信息和 output_schema。"
    )

    workflow = Workflow.model_validate(workflow_skeleton)
    nodes_by_id: Dict[str, Node] = {n.id: n for n in workflow.nodes}
    filled_params: Dict[str, Dict[str, Any]] = {
        n.id: copy.deepcopy(n.params) for n in workflow.nodes
    }

    for node_id in _traverse_order(workflow):
        node = nodes_by_id[node_id]
        upstream_nodes = get_upstream_nodes(workflow, node_id)
        allowed_node_ids = [n.id for n in upstream_nodes]

        upstream_context = []
        for n in upstream_nodes:
            action_schema = action_schemas.get(n.action_id, {}) if n.action_id else {}
            upstream_context.append(
                {
                    "id": n.id,
                    "type": n.type,
                    "action_id": n.action_id,
                    "output_schema": action_schema.get("output_schema"),
                    "params": filled_params.get(n.id, n.params),
                }
            )

        target_action_schema = action_schemas.get(node.action_id, {}) if node.action_id else {}
        user_payload = {
            "target_node": {
                "id": node.id,
                "type": node.type,
                "action_id": node.action_id,
                "display_name": node.display_name,
                "existing_params": filled_params[node.id],
            },
            "arg_schema": target_action_schema.get("arg_schema"),
            "allowed_node_ids": allowed_node_ids,
            "allowed_upstream_nodes": upstream_context,
        }

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            temperature=0.1,
        )

        content = resp.choices[0].message.content or ""
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                first_line, rest = text.split("\n", 1)
                if first_line.strip().lower().startswith("json"):
                    text = rest

        decoder = json.JSONDecoder()
        node_result: Any
        try:
            node_result, _ = decoder.raw_decode(text)
        except json.JSONDecodeError:
            # 某些模型可能在 JSON 前后附带额外文本，尝试截取第一个 JSON 对象以提升鲁棒性。
            first_curly = text.find("{")
            if first_curly >= 0:
                try:
                    node_result, _ = decoder.raw_decode(text[first_curly:])
                except json.JSONDecodeError:
                    log_error("[fill_params_with_llm] 无法解析模型返回 JSON")
                    log_debug(content)
                    raise
            else:
                log_error("[fill_params_with_llm] 无法解析模型返回 JSON")
                log_debug(content)
                raise

        if isinstance(node_result, dict):
            params = node_result.get("params", {})
            if isinstance(params, dict):
                filled_params[node.id] = params

    completed_nodes = []
    for node in workflow.nodes:
        data = node.model_dump(by_alias=True)
        data["params"] = filled_params.get(node.id, node.params)
        completed_nodes.append(data)

    completed_workflow = {
        "workflow_name": workflow.workflow_name,
        "description": workflow.description,
        "nodes": completed_nodes,
        "edges": [e.model_dump(by_alias=True) for e in workflow.edges],
    }

    return completed_workflow


__all__ = ["fill_params_with_llm"]
