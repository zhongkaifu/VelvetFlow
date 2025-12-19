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
        "- node 基本结构：{id, type, display_name, params, depends_on, action_id?, out_params_schema?, loop/subgraph/branches?}。\n"
        "  type 仅允许 action/condition/loop/parallel。无需 start/end/exit 节点。\n"
        "  action 节点必须填写 action_id（来自动作库）与 params；只有 action 节点允许 out_params_schema。\n"
        "  condition 节点的 params 仅包含 expression（单个返回布尔值的 Jinja 表达式）；true_to_node/false_to_node 必须是节点顶层字段（字符串或 null），不得放入 params。\n"
        "  loop 节点包含 loop_kind/iter/source/body_subgraph/exports，循环外部只能引用 exports.items 或 exports.aggregates。\n"
        "  引用 loop.exports 结构时必须指向 exports 内部的字段或子结构，不能只写 result_of.<loop_id>.exports；body_subgraph 只需要 nodes 数组，不要提供 entry/exit/edges。\n"
        "  parallel 节点的 branches 为非空数组，每个元素包含 id/entry_node/sub_graph_nodes。\n"
        "- params 必须直接使用 Jinja 表达式引用上游结果（如 {{ result_of.<node_id>.<field_path> }} 或 {{ loop.item.xxx }}），不再允许旧的 __from__/__agg__ DSL。"
        "  其中 <node_id> 必须存在且字段需与上游 output_schema 或 loop.exports 对齐。\n\n"
        "输入包含自然语言需求、当前 workflow 以及动作库的 schema。当前 workflow 往往是用户提供的 DSL 模版，除非需求明确要求替换，否则应视为起点并在其上做增量修改。\n\n"
        "约束与建议：\n"
        "1) 优先复用现有节点，保持未提及的结构与模版一致；如需新增节点请保持 id 唯一，并确保入口出口连通。\n"
        "2) 所有 action_id 必须出现在动作库中，并且 params 必须符合对应 arg_schema；必要时修正 display_name 以反映新需求。\n"
        "3) condition/loop/parallel 节点需要补齐必填字段：condition 必须提供 true_to_node/false_to_node（顶层字段，指向真实存在的下游节点，或设置为 null 表示该分支结束）且 params.expression 为布尔 Jinja 表达式；loop 需包含 iter/body_subgraph/exports 等，引用路径必须合法。\n"
        "4) 所有节点应维护 depends_on（字符串数组）来声明依赖；如果节点被 condition.true_to_node/false_to_node 指向，必须将该 condition 节点加入目标节点的 depends_on，以确保执行顺序受依赖控制。\n"
        "5) edges 将由系统基于节点引用自动推导，不需要手动增删；无需额外的 start/end 占位节点。\n"
        "6) 所有 Jinja 绑定表达式必须与上游 action 的 output_schema 或 loop.exports 对齐，避免访问不存在的字段；聚合/过滤/拼接逻辑请直接写在 Jinja 表达式或过滤器中，严禁输出 __agg__/__from__ 结构。\n"
        "7) 若上一步校验未通过，会给出 validation_errors，请务必根据问题逐条修复；忽略任何一条错误会导致下一轮直接失败。\n"
        "8) 修复回合有限，请一次性修完所有校验问题后再输出结果。\n"
        "9) 返回结果必须是完整的 JSON 对象，不要包含多余的注释或代码块标记。"
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
