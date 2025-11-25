"""LLM helpers for synthesizing missing edges."""

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_debug, log_error


def synthesize_edges_with_llm(
    nodes: List[Dict[str, Any]],
    nl_requirement: str,
    model: str = OPENAI_MODEL,
) -> List[Dict[str, Any]]:
    """
    当第一阶段没有生成任何 edges 时，让 LLM 根据节点列表和需求补一份 edges。
    """

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    node_brief = [
        {
            "id": n["id"],
            "type": n.get("type"),
            "action_id": n.get("action_id"),
            "display_name": n.get("display_name"),
        }
        for n in nodes
    ]

    system_prompt = (
        "你是一个工作流接线助手。\n"
        "现在有一组已经确定的节点 nodes（每个节点有 id/type/action_id/display_name），\n"
        "但是 edges 为空。\n"
        "你的任务是：\n"
        "1. 根据用户的自然语言需求和这些节点的含义，推理它们的执行顺序和分支结构。\n"
        "2. 生成一组 edges，每条 edge 形如：\n"
        "   {\"from\": \"节点ID\", \"to\": \"节点ID\", \"condition\": \"true/false 或 null\"}\n"
        "3. 整体必须是一个有向无环图（DAG），通常从 type='start' 节点开始，到 type='end' 节点结束。\n"
        "4. 如果存在条件节点(type='condition')，请用 edge.condition 表示 true/false 分支，\n"
        "   无条件顺序执行时 condition 用 null。\n"
        "5. 返回的 JSON 必须是：{\"edges\": [ ... ]}，不要包含其它字段，也不要加代码块标记。"
    )

    user_payload = {
        "nl_requirement": nl_requirement,
        "nodes": node_brief,
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.2,
    )

    content = resp.choices[0].message.content or ""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                text = rest

    try:
        obj = json.loads(text)
        edges = obj.get("edges", [])
        if not isinstance(edges, list):
            raise ValueError("edges 不是 list")
        for e in edges:
            if "from" not in e or "to" not in e:
                raise ValueError("edge 缺少 from/to")
        return edges
    except Exception as e:
        log_error("[synthesize_edges_with_llm] 无法解析/使用 LLM 返回的 edges")
        log_debug(f"错误详情: {e}")
        log_debug(f"原始内容：{content}")
        return []


__all__ = ["synthesize_edges_with_llm"]
