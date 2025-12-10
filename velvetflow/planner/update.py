# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""LLM-driven workflow updater based on a new requirement description."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_llm_message, log_llm_usage
from velvetflow.models import ValidationError


def _build_action_schemas(action_registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    action_schemas: Dict[str, Dict[str, Any]] = {}
    for action in action_registry:
        action_id = action.get("action_id")
        if not action_id:
            continue
        action_schemas[action_id] = {
            "name": action.get("name", ""),
            "description": action.get("description", ""),
            "domain": action.get("domain", ""),
            "arg_schema": action.get("arg_schema"),
            "output_schema": action.get("output_schema"),
        }
    return action_schemas


def update_workflow_with_llm(
    workflow_raw: Mapping[str, Any],
    requirement: str,
    action_registry: List[Dict[str, Any]],
    model: str = OPENAI_MODEL,
    validation_errors: Optional[Sequence[ValidationError]] = None,
) -> Dict[str, Any]:
    """Update or repair a workflow according to a new natural language requirement.

    When ``validation_errors`` are provided, the LLM is expected to fix the
    listed problems while keeping the requirement satisfied.
    """

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    action_schemas = _build_action_schemas(action_registry)

    system_prompt = (
        "你是一名资深的工作流设计师，负责修改现有的 workflow JSON 以满足新的业务需求。\n"
        "【Workflow DSL 语法与语义（务必遵守）】\n"
        "- workflow = {workflow_name, description, nodes: []}，只能返回合法 JSON（edges 会由系统基于节点绑定自动推导，不需要生成）。\n"
        "- node 基本结构：{id, type, display_name, params, action_id?, out_params_schema?, loop/subgraph/branches?}。\n"
        "  type 仅允许 start/action/condition/loop/parallel/end/exit。start/exit/end 不需要 params/out_params_schema。\n"
        "  action 节点必须填写 action_id（来自动作库）与 params；只有 action 节点允许 out_params_schema。\n"
        "  condition 节点需包含 kind/source/field/op/value 以及 true_to_node/false_to_node（字符串或 null）。\n"
        "  loop 节点包含 loop_kind/iter/source/body_subgraph/exports，循环外部只能引用 exports.items 或 exports.aggregates。\n"
        "  parallel 节点的 branches 为非空数组，每个元素包含 id/entry_node/sub_graph_nodes。\n"
        "- params 内部可使用绑定 DSL：{\"__from__\": \"result_of.<node_id>.<field_path>\", \"__agg__\": <identity/count/...>}，"
        "  其中 <node_id> 必须存在且字段需与上游 output_schema 或 loop.exports 对齐。\n\n"
        "输入包含自然语言需求、当前 workflow 以及动作库的 schema。当前 workflow 往往是用户提供的 DSL 模版，除非需求明确要求替换，否则应视为起点并在其上做增量修改。\n\n"
        "约束与建议：\n"
        "1) 优先复用现有节点，保持未提及的结构与模版一致；如需新增节点请保持 id 唯一，并确保入口出口连通。\n"
        "2) 所有 action_id 必须出现在动作库中，并且 params 必须符合对应 arg_schema；必要时修正 display_name 以反映新需求。\n"
        "3) condition/loop/parallel 节点需要补齐必填字段：condition 必须提供 true_to_node/false_to_node（指向真实存在的下游节点，或设置为 null 表示该分支结束），并补齐 op/source/value 等；loop 需包含 iter/body_subgraph/exports 等，引用路径必须合法。\n"
        "4) edges 将由系统基于节点引用自动推导，不需要手动增删；入口节点通常是 type=start，出口节点通常是 type=end。\n"
        "5) 所有绑定表达式 (__from__/__agg__) 必须与上游 action 的 output_schema 或 loop.exports 对齐，避免访问不存在的字段；__agg__ 必须使用 identity/count/count_if/join/format_join/filter_map/pipeline 之一，其他取值视为非法并会在校验报告中列出。需要条件过滤时请把 __agg__.condition 或 pipeline.steps[].condition 写成 Mini-Expression AST（JSON），支持 const/var、op=and/or/not/exists/==/!=/>/>=/</<=/in 等结构，严禁再写自然语言条件。\n"
        "6) 若上一步校验未通过，会给出 validation_errors，请务必根据问题逐条修复；忽略任何一条错误会导致下一轮直接失败。\n"
        "7) 修复回合有限，请一次性修完所有校验问题后再输出结果。\n"
        "8) 返回结果必须是完整的 JSON 对象，不要包含多余的注释或代码块标记。"
    )

    errors_payload = None
    if validation_errors:
        errors_payload = [
            {
                "code": err.code,
                "node_id": err.node_id,
                "field": err.field,
                "message": err.message,
            }
            for err in validation_errors
        ]

    user_payload: Dict[str, Any] = {
        "requirement": requirement,
        "current_workflow": workflow_raw,
        "action_schemas": action_schemas,
        "validation_errors": errors_payload,
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.2,
    )

    log_llm_usage(model, getattr(resp, "usage", None), operation="update_workflow")

    if not resp.choices:
        raise RuntimeError("update_workflow_with_llm 未返回任何候选消息")

    message = resp.choices[0].message
    log_llm_message(model, message, operation="update_workflow")

    content = message.content or ""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                text = rest

    return json.loads(text)


__all__ = ["update_workflow_with_llm"]
