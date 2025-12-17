"""LLM-driven workflow execution summarization and requirement alignment helpers."""

from __future__ import annotations

import json
from collections import defaultdict, deque
from typing import Any, Dict, List, Mapping, Sequence

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    child_span,
    log_debug,
    log_error,
    log_llm_message,
    log_llm_usage,
)
from velvetflow.models import infer_edges_from_bindings


def _build_action_lookup(action_registry: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for action in action_registry:
        if not isinstance(action, Mapping):
            continue
        action_id = action.get("action_id")
        if not isinstance(action_id, str):
            continue
        lookup[action_id] = {
            "name": action.get("name", ""),
            "description": action.get("description", ""),
            "domain": action.get("domain", ""),
        }
    return lookup


def _derive_edges(nodes: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    inferred_edges = infer_edges_from_bindings(nodes)
    manual_edges: List[Dict[str, Any]] = []
    node_ids = {node.get("id") for node in nodes if isinstance(node, Mapping)}

    for node in nodes:
        if not isinstance(node, Mapping):
            continue
        node_id = node.get("id")
        if not isinstance(node_id, str):
            continue
        depends_on = node.get("depends_on") if isinstance(node.get("depends_on"), list) else []
        for dep in depends_on:
            if isinstance(dep, str) and dep in node_ids and dep != node_id:
                manual_edges.append({"from": dep, "to": node_id, "condition": None})

    seen_pairs = {(edge.get("from"), edge.get("to")) for edge in inferred_edges}
    for edge in manual_edges:
        pair = (edge["from"], edge["to"])
        if pair not in seen_pairs:
            inferred_edges.append(edge)
            seen_pairs.add(pair)

    return inferred_edges


def _topological_order(nodes: List[Mapping[str, Any]]) -> List[str]:
    """Return a stable execution order derived from depends_on and inferred edges."""

    node_ids: List[str] = []
    node_index: Dict[str, int] = {}
    for idx, node in enumerate(nodes):
        if isinstance(node, Mapping) and isinstance(node.get("id"), str):
            node_ids.append(node["id"])
            node_index[node["id"]] = idx

    edges = _derive_edges(nodes)
    indegree: Dict[str, int] = defaultdict(int)
    graph: Dict[str, List[str]] = defaultdict(list)

    for edge in edges:
        if not isinstance(edge, Mapping):
            continue
        frm = edge.get("from") if "from" in edge else edge.get("from_node")
        to = edge.get("to") if "to" in edge else edge.get("to_node")
        if not isinstance(frm, str) or not isinstance(to, str):
            continue
        if frm not in node_index or to not in node_index:
            continue
        graph[frm].append(to)
        indegree[to] += 1
        indegree.setdefault(frm, indegree.get(frm, 0))

    ordered = []
    queue = deque(sorted([nid for nid in node_ids if indegree.get(nid, 0) == 0], key=node_index.get))
    while queue:
        nid = queue.popleft()
        ordered.append(nid)
        for neighbor in sorted(graph.get(nid, []), key=node_index.get):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) < len(node_ids):
        remaining = [nid for nid in node_ids if nid not in ordered]
        ordered.extend(remaining)

    return ordered


def _build_execution_outline(
    workflow: Mapping[str, Any],
    action_registry: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    nodes = workflow.get("nodes") if isinstance(workflow.get("nodes"), list) else []
    execution_order = _topological_order(nodes)
    actions = _build_action_lookup(action_registry)
    node_lookup = {node.get("id"): node for node in nodes if isinstance(node, Mapping)}

    steps: List[Dict[str, Any]] = []
    for node_id in execution_order:
        node = node_lookup.get(node_id, {})
        action_id = node.get("action_id") if isinstance(node, Mapping) else None
        params = node.get("params") if isinstance(node, Mapping) else {}
        depends_on = node.get("depends_on") if isinstance(node, Mapping) else []
        step: Dict[str, Any] = {
            "node_id": node_id,
            "type": node.get("type"),
            "display_name": node.get("display_name"),
            "action_id": action_id,
            "depends_on": depends_on if isinstance(depends_on, list) else [],
            "params": params if isinstance(params, Mapping) else {},
            "action_info": actions.get(action_id, {}),
        }
        if isinstance(node, Mapping) and node.get("type") == "loop":
            params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
            body_graph = params.get("body_subgraph") if isinstance(params, Mapping) else {}
            body_nodes = body_graph.get("nodes") if isinstance(body_graph, Mapping) else []
            step["body_execution_order"] = _topological_order(body_nodes) if isinstance(body_nodes, list) else []
            step["body_nodes"] = body_nodes
            step["exports"] = params.get("exports") if isinstance(params, Mapping) else {}
        steps.append(step)

    return {
        "execution_order": execution_order,
        "steps": steps,
        "inferred_edges": _derive_edges(nodes),
    }


def summarize_workflow_execution(
    workflow: Mapping[str, Any],
    action_registry: Sequence[Mapping[str, Any]],
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    """Simulate workflow execution order and ask LLM to summarize accomplished tasks."""

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    outline = _build_execution_outline(workflow, action_registry)
    payload = {
        "workflow": workflow,
        "execution_order": outline["execution_order"],
        "steps": outline["steps"],
        "inferred_edges": outline["inferred_edges"],
    }

    system_prompt = (
        "你是一个工作流执行模拟器，需要根据节点的依赖关系顺序描述工作流完成的任务。\n"
        "输入提供：\n"
        "1) workflow JSON（包含 nodes），\n"
        "2) 自动推导出的 execution_order（必须严格按此顺序模拟执行），\n"
        "3) 每个节点关联的业务动作描述（action_info）。\n"
        "请依次模拟每个节点及其调用的业务动作，解释该步骤完成了什么，再汇总成整体任务描述。\n"
        "输出 JSON，对应结构：\n"
        "{\n"
        '  "overall_description": "整体上工作流完成的任务",\n'
        '  "step_summaries": [\n'
        '    {"node_id": "a1", "description": "该节点完成了什么"}\n'
        "  ]\n"
        "}\n"
        "不要返回多余字段，也不要使用代码块标记。"
    )

    with child_span("workflow_execution_summary"):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
        )

    log_llm_usage(model, getattr(resp, "usage", None), operation="workflow_summary")
    if not resp.choices:
        raise RuntimeError("summarize_workflow_execution 未返回任何候选消息")

    message = resp.choices[0].message
    log_llm_message(model, message, operation="workflow_summary")

    content = message.content or ""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                text = rest

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        log_error("[summarize_workflow_execution] 无法解析 JSON，返回空摘要")
        log_debug(content)
        parsed = {"overall_description": "", "step_summaries": []}

    parsed.setdefault("execution_order", outline["execution_order"])
    return parsed


def assess_requirement_alignment(
    requirement: str,
    execution_summary: Mapping[str, Any],
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    """Judge whether the execution summary matches the user's natural language requirement."""

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    system_prompt = (
        "你是一个工作流需求对齐审查员。\n"
        "给定用户需求（requirement）以及按照执行顺序得到的 workflow 摘要（execution_summary），\n"
        "请判断 workflow 完成的任务是否覆盖需求。如果存在缺口，请列出差异点。\n"
        "输出 JSON：\n"
        "{\n"
        '  "is_aligned": true/false,\n'
        '  "missing_points": ["缺少的能力1", "缺少的能力2"],\n'
        '  "analysis": "简要分析差异"\n'
        "}\n"
        "不要添加额外字段或代码块。"
    )
    payload = {
        "requirement": requirement,
        "execution_summary": execution_summary,
    }

    with child_span("workflow_requirement_alignment"):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.1,
        )

    log_llm_usage(model, getattr(resp, "usage", None), operation="workflow_alignment")
    if not resp.choices:
        raise RuntimeError("assess_requirement_alignment 未返回任何候选消息")

    message = resp.choices[0].message
    log_llm_message(model, message, operation="workflow_alignment")

    content = message.content or ""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                text = rest

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        log_error("[assess_requirement_alignment] 无法解析 JSON，返回对齐失败结果")
        log_debug(content)
        return {"is_aligned": False, "missing_points": [], "analysis": ""}


__all__ = ["summarize_workflow_execution", "assess_requirement_alignment"]
