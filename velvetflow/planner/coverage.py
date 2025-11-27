"""Coverage analysis and structure refinement driven by LLM."""

import copy
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    child_span,
    log_debug,
    log_error,
    log_info,
    log_llm_usage,
    log_warn,
)

COVERAGE_EDIT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_workflow_meta",
            "description": "更新 workflow 的名称和描述。",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["workflow_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_or_update_node",
            "description": "新增或更新一个节点，若 id 已存在则覆盖。",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["start", "end", "action", "condition", "loop", "parallel"],
                    },
                    "action_id": {"type": "string", "nullable": True},
                    "display_name": {"type": "string", "nullable": True},
                    "params": {
                        "type": "object",
                        "description": "节点参数，可部分覆盖已有参数。",
                        "additionalProperties": True,
                        "nullable": True,
                    },
                },
                "required": ["id", "type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_or_update_edge",
            "description": "新增或更新一条边，依据 from/to 唯一定位。",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_node": {"type": "string"},
                    "to_node": {"type": "string"},
                    "condition": {"type": "string", "nullable": True},
                },
                "required": ["from_node", "to_node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_edit",
            "description": "当覆盖度改进完成时调用，结束本轮编辑。",
            "parameters": {
                "type": "object",
                "properties": {
                    "ready": {"type": "boolean", "default": True},
                    "notes": {"type": "string"},
                },
                "required": ["ready"],
            },
        },
    },
]


def check_requirement_coverage_with_llm(
    nl_requirement: str,
    workflow: Dict[str, Any],
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    """
    让 LLM 审核：当前 workflow 是否完全覆盖 nl_requirement。
    返回结构示例：
    {
      "is_covered": true/false,
      "missing_points": ["...", "..."],
      "analysis": "..."
    }
    """

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system_prompt = (
        "你是一个严谨的工作流需求覆盖度审查员。\n"
        "给定：\n"
        "1) 用户的自然语言需求 nl_requirement\n"
        "2) 当前的 workflow（workflow_name/description/nodes/edges）\n\n"
        "你的任务：\n"
        "1. 先把 nl_requirement 中的关键子需求拆分成若干个“原子能力”，例如：\n"
        "   - 某个触发方式（定时 / 事件 / 手动等）\n"
        "   - 若干个数据读取 / 查询步骤\n"
        "   - 若干个过滤 / 条件判断步骤\n"
        "   - 若干个聚合 / 统计 / 总结步骤\n"
        "   - 若干个对外动作（通知、写入数据库、调用外部系统等）\n"
        "   这里的例子仅用于说明“拆分粒度”，不要把具体业务词带入其它任务。\n"
        "2. 再逐项检查当前 workflow 是否对这些原子能力都有完整的支持：\n"
        "   - 是否有对应的节点；\n"
        "   - 节点之间的连接顺序是否合理；\n"
        "   - 是否存在明显缺失（例如：需求中提到“通知”，但 workflow 中完全没有任何通知/写入相关节点）。\n"
        "3. 如果完全覆盖，则 is_covered=true，missing_points 列表为空。\n"
        "4. 如果有任何一条需求没有被覆盖或只被部分覆盖，则 is_covered=false，\n"
        "   并在 missing_points 中用简短中文列出缺失点（例如：“缺少对特定条件的过滤”、“缺少结果汇总后发送给用户的步骤”等）。\n\n"
        "输出格式（非常重要）：\n"
        "返回一个 JSON 对象，形如：\n"
        "{\n"
        "  \"is_covered\": true/false,\n"
        "  \"missing_points\": [\"...\", \"...\"],\n"
        "  \"analysis\": \"详细分析\"\n"
        "}\n"
        "不要添加额外字段，不要输出代码块标记。"
    )

    payload = {
        "nl_requirement": nl_requirement,
        "workflow": workflow,
    }

    with child_span("coverage_check_llm"):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.1,
        )
    log_llm_usage(model, getattr(resp, "usage", None), operation="coverage_check")
    if not resp.choices:
        raise RuntimeError("check_requirement_coverage_with_llm 未返回任何候选消息")

    content = resp.choices[0].message.content or ""
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
        log_error("[check_requirement_coverage_with_llm] 无法解析 JSON，返回空结果")
        log_debug(content)
        return {"is_covered": False, "missing_points": [], "analysis": ""}


def refine_workflow_structure_with_llm(
    nl_requirement: str,
    current_workflow: Dict[str, Any],
    missing_points: List[str],
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    """让 LLM 使用工具调用直接编辑 workflow，以覆盖缺失点。"""

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system_prompt = (
        "你是一个工作流覆盖度修补助手。\n"
        "已知：\n"
        "1) 用户需求 nl_requirement\n"
        "2) 当前 workflow（nodes/edges）\n"
        "3) 覆盖度检查中发现的 missing_points\n\n"
        "请通过工具调用直接编辑 workflow：\n"
        "- 必须围绕 missing_points 增补节点或连线，确保需求被覆盖。\n"
        "- 如需调整已有节点，可重写同 id 的节点。\n"
        "- 只输出工具调用，不要返回自然语言方案。\n"
        "- 当你认为已覆盖所有缺失点时调用 finalize_edit。"
    )

    payload = {
        "nl_requirement": nl_requirement,
        "current_workflow": current_workflow,
        "missing_points": missing_points,
    }

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    working_workflow = _normalize_workflow(current_workflow)
    finalized = False

    for round_idx in range(6):
        with child_span("coverage_refine_llm"):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=COVERAGE_EDIT_TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
        log_llm_usage(model, getattr(resp, "usage", None), operation="coverage_refine")
        if not resp.choices:
            raise RuntimeError("refine_workflow_structure_with_llm 未返回任何候选消息")

        msg = resp.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            }
        )

        if not msg.tool_calls:
            log_warn("[Coverage] 本轮没有生成工具调用，提前结束覆盖度修补。")
            break

        for tc in msg.tool_calls:
            func_name = tc.function.name
            raw_args = tc.function.arguments
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                log_error(f"[Coverage] 解析工具参数失败: {raw_args}")
                args = {}
                continue

            log_info(f"[Coverage] 调用工具: {func_name}({args})")

            if func_name == "set_workflow_meta":
                _apply_meta(working_workflow, args.get("workflow_name", ""), args.get("description"))
                tool_result = {"status": "ok", "type": "meta_set"}

            elif func_name == "add_or_update_node":
                updated_node = _apply_node_edit(
                    working_workflow,
                    node_id=args.get("id", ""),
                    node_type=args.get("type"),
                    action_id=args.get("action_id"),
                    display_name=args.get("display_name"),
                    params=args.get("params"),
                )
                tool_result = {"status": "ok", "type": "node_upserted", "node": updated_node}

            elif func_name == "add_or_update_edge":
                updated_edge = _apply_edge_edit(
                    working_workflow,
                    from_node=args.get("from_node", ""),
                    to_node=args.get("to_node", ""),
                    condition=args.get("condition"),
                )
                tool_result = {"status": "ok", "type": "edge_upserted", "edge": updated_edge}

            elif func_name == "finalize_edit":
                finalized = True
                tool_result = {"status": "ok", "type": "finalized", "notes": args.get("notes")}

            else:
                tool_result = {"status": "error", "message": f"未知工具 {func_name}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )

        if finalized:
            break

    if finalized:
        log_info("[Coverage] 已完成覆盖度修补，返回修改后的 workflow。")
    else:
        log_warn("[Coverage] 未收到 finalize_edit，将使用当前工作流版本。")

    return working_workflow


def _normalize_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(workflow) if isinstance(workflow, dict) else {}
    normalized.setdefault("workflow_name", "unnamed_workflow")
    normalized.setdefault("description", "")
    normalized.setdefault("nodes", [])
    normalized.setdefault("edges", [])
    if not isinstance(normalized["nodes"], list):
        normalized["nodes"] = []
    if not isinstance(normalized["edges"], list):
        normalized["edges"] = []
    return normalized


def _apply_meta(workflow: Dict[str, Any], name: str, description: Optional[str]):
    if name:
        workflow["workflow_name"] = name
    if description is not None:
        workflow["description"] = description


def _apply_node_edit(
    workflow: Dict[str, Any],
    node_id: str,
    node_type: Optional[str],
    action_id: Optional[str],
    display_name: Optional[str],
    params: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not node_id or not node_type:
        return {}

    nodes: List[Dict[str, Any]] = workflow.get("nodes", [])
    existing = next((n for n in nodes if n.get("id") == node_id), None)
    merged_params = params or {}
    if existing and isinstance(existing.get("params"), dict):
        merged = dict(existing.get("params") or {})
        merged.update(merged_params)
        merged_params = merged

    new_node = {
        "id": node_id,
        "type": node_type,
        "action_id": action_id,
        "display_name": display_name,
        "params": merged_params,
    }

    if existing:
        for idx, node in enumerate(nodes):
            if node.get("id") == node_id:
                nodes[idx] = new_node
                break
    else:
        nodes.append(new_node)

    workflow["nodes"] = nodes
    return new_node


def _apply_edge_edit(
    workflow: Dict[str, Any],
    from_node: str,
    to_node: str,
    condition: Optional[str],
) -> Dict[str, Any]:
    if not from_node or not to_node:
        return {}

    edges: List[Dict[str, Any]] = workflow.get("edges", [])
    updated_edge = {"from": from_node, "to": to_node, "condition": condition}

    for idx, edge in enumerate(edges):
        if edge.get("from") == from_node and edge.get("to") == to_node:
            edges[idx] = updated_edge
            workflow["edges"] = edges
            return updated_edge

    edges.append(updated_edge)
    workflow["edges"] = edges
    return updated_edge


__all__ = ["check_requirement_coverage_with_llm", "refine_workflow_structure_with_llm"]
