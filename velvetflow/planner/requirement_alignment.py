# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""LLM-driven requirement alignment checks for workflows."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_llm_message, log_llm_usage


def check_missing_requirements(
    workflow_raw: Mapping[str, Any],
    requirement_plan: Mapping[str, Any],
    model: str = OPENAI_MODEL,
) -> List[Dict[str, Any]]:
    """Return unmet requirements after aligning the workflow with decomposed needs."""

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    requirements = requirement_plan.get("requirements")
    if not isinstance(requirements, list) or not requirements:
        return []

    system_prompt = (
        "你是一名工作流需求对齐检查助手。\n"
        "输入包含拆解后的用户需求清单与 workflow JSON。\n"
        "请逐条检查每个需求是否在 workflow 中有对应的节点或输出覆盖。\n"
        "输出必须是严格 JSON 格式，不要包含额外解释或代码块。\n"
        "返回结构：{\"missing_requirements\": [\n"
        "  {\"index\": 0, \"requirement\": {\"description\": \"...\", \"intent\": \"...\","
        " \"inputs\": [...], \"constraints\": [...]}, \"reason\": \"未找到对应节点/输出\"}\n"
        "]}\n"
        "如果全部满足，请返回 missing_requirements 为空数组。"
    )

    payload = {
        "requirements": requirements,
        "assumptions": requirement_plan.get("assumptions", []),
        "workflow": workflow_raw,
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.2,
    )

    log_llm_usage(model, getattr(resp, "usage", None), operation="align_requirements")

    if not resp.choices:
        raise RuntimeError("对齐检查未返回任何候选消息")

    message = resp.choices[0].message
    log_llm_message(model, message, operation="align_requirements")

    content = (message.content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        if "\n" in content:
            first_line, rest = content.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                content = rest

    parsed = json.loads(content)
    missing = parsed.get("missing_requirements") if isinstance(parsed, dict) else None
    if not isinstance(missing, list):
        return []
    return [item for item in missing if isinstance(item, dict)]


__all__ = ["check_missing_requirements"]
