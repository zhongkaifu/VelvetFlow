# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Coverage analysis helpers driven by LLM."""

import copy
import json
import os
from typing import Any, Dict

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    child_span,
    log_debug,
    log_error,
    log_llm_message,
    log_llm_usage,
)


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
        "2) 当前的 workflow（workflow_name/description/nodes，其中节点的输入输出绑定隐式定义了执行顺序）\n\n"
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
        "workflow": _normalize_workflow(workflow),
    }

    with child_span("coverage_check_llm"):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
    log_llm_usage(model, getattr(resp, "usage", None), operation="coverage_check")
    if not resp.choices:
        raise RuntimeError("check_requirement_coverage_with_llm 未返回任何候选消息")

    message = resp.choices[0].message
    log_llm_message(model, message, operation="coverage_check")

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
        log_error("[check_requirement_coverage_with_llm] 无法解析 JSON，返回空结果")
        log_debug(content)
        return {"is_covered": False, "missing_points": [], "analysis": ""}


def _normalize_workflow(workflow: Dict[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(workflow) if isinstance(workflow, dict) else {}
    normalized.setdefault("workflow_name", "unnamed_workflow")
    normalized.setdefault("description", "")
    normalized.setdefault("nodes", [])
    nodes = normalized.get("nodes") if isinstance(normalized.get("nodes"), list) else []
    normalized["nodes"] = nodes
    normalized.pop("edges", None)
    return normalized


__all__ = ["check_requirement_coverage_with_llm"]
