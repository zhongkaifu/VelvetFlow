"""Coverage analysis and structure refinement driven by LLM."""

import json
from typing import Any, Dict, List

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_debug, log_error, log_warn


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

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
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
    """让 LLM 根据缺失点对 workflow 结构进行增量改进。"""

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system_prompt = (
        "你是一个严谨的工作流架构师。现在你将看到：\n"
        "1) 用户需求 nl_requirement\n"
        "2) 当前 workflow（nodes/edges）\n"
        "3) 在覆盖度检查中发现的缺失点 missing_points\n"
        "请基于这些信息，增量地改进 workflow 结构，保证缺失点被覆盖。\n"
        "输出 JSON：{\"nodes\": [...], \"edges\": [...]}，不要加代码块标记。\n"
        "注意：\n"
        "- 你可以新增节点、修改节点参数、调整 edges。\n"
        "- 确保整个结构是有向无环图（DAG）。\n"
    )

    payload = {
        "nl_requirement": nl_requirement,
        "current_workflow": current_workflow,
        "missing_points": missing_points,
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
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
        refined = json.loads(text)
    except json.JSONDecodeError:
        log_error("[refine_workflow_structure_with_llm] 无法解析 JSON，使用当前结构回退")
        log_debug(content)
        return current_workflow

    if not isinstance(refined, dict) or not isinstance(refined.get("nodes"), list) or not isinstance(refined.get("edges"), list):
        log_warn("[refine_workflow_structure_with_llm] LLM 返回的结构不完整，回退到 current_workflow。")
        return current_workflow

    return refined


__all__ = ["check_requirement_coverage_with_llm", "refine_workflow_structure_with_llm"]
