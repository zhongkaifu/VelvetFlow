"""LLM-driven repair helpers used when validation fails."""

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_debug, log_error, log_info, log_success, log_warn
from velvetflow.models import Node, PydanticValidationError, ValidationError, Workflow
from velvetflow.planner.action_guard import ensure_registered_actions


def _convert_pydantic_errors(
    workflow_raw: Any, error: PydanticValidationError
) -> List[ValidationError]:
    """Map Pydantic validation errors to generic ValidationError objects."""

    nodes = []
    if isinstance(workflow_raw, dict):
        nodes = workflow_raw.get("nodes") or []

    def _node_id_from_index(index: int) -> Optional[str]:
        if 0 <= index < len(nodes):
            node = nodes[index]
            if isinstance(node, dict):
                return node.get("id")
            if isinstance(node, Node):
                return node.id
        return None

    validation_errors: List[ValidationError] = []
    for err in error.errors():
        loc = err.get("loc", ()) or ()
        msg = err.get("msg", "")

        node_id: Optional[str] = None
        field: Optional[str] = None

        if loc:
            if loc[0] == "nodes" and len(loc) >= 2 and isinstance(loc[1], int):
                node_id = _node_id_from_index(loc[1])
                if len(loc) >= 3:
                    field = str(loc[2])
            elif loc[0] == "edges" and len(loc) >= 2 and isinstance(loc[1], int):
                if len(loc) >= 3 and isinstance(loc[-1], str):
                    field = str(loc[-1])
                else:
                    field = "edges"
            else:
                field = ".".join(str(part) for part in loc)

        validation_errors.append(
            ValidationError(
                code="INVALID_SCHEMA",
                node_id=node_id,
                field=field,
                message=msg,
            )
        )

    return validation_errors


def _make_failure_validation_error(message: str) -> ValidationError:
    return ValidationError(
        code="INVALID_SCHEMA", node_id=None, field=None, message=message
    )


def _repair_with_llm_and_fallback(
    *,
    broken_workflow: Dict[str, Any],
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
    search_service,
    reason: str,
) -> Workflow:
    log_info(f"[AutoRepair] {reason}，将错误上下文提交给 LLM 尝试修复。")
    try:
        repaired_raw = repair_workflow_with_llm(
            broken_workflow=broken_workflow,
            validation_errors=validation_errors,
            action_registry=action_registry,
            model=OPENAI_MODEL,
        )
        repaired = Workflow.model_validate(repaired_raw)
        repaired = ensure_registered_actions(
            repaired,
            action_registry=action_registry,
            search_service=search_service,
        )
        if not isinstance(repaired, Workflow):
            repaired = Workflow.model_validate(repaired)
        log_success("[AutoRepair] LLM 修复成功，继续构建 workflow。")
        return repaired
    except Exception as err:  # noqa: BLE001
        log_warn(f"[AutoRepair] LLM 修复失败：{err}")
        try:
            fallback = Workflow.model_validate(broken_workflow)
            fallback = ensure_registered_actions(
                fallback,
                action_registry=action_registry,
                search_service=search_service,
            )
            if not isinstance(fallback, Workflow):
                fallback = Workflow.model_validate(fallback)
            log_warn("[AutoRepair] 使用未修复的结构作为回退，继续后续流程。")
            return fallback
        except Exception as inner_err:  # noqa: BLE001
            log_error(
                f"[AutoRepair] 回退到原始结构失败，将返回空的 fallback workflow：{inner_err}"
            )
            return Workflow(workflow_name="fallback_workflow", nodes=[], edges=[])


def repair_workflow_with_llm(
    broken_workflow: Dict[str, Any],
    validation_errors: List[ValidationError],
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
        "你是一个工作流修复助手。\n"
        "当前有一个 workflow JSON 和一组结构化校验错误 validation_errors。\n"
        "validation_errors 是 JSON 数组，元素包含 code/node_id/field/message。\n"
        "这些错误来自：\n"
        "- action 参数缺失或不符合 arg_schema\n"
        "- condition 条件不完整\n"
        "- source/__from__ 路径引用了不存在的节点\n"
        "- source/__from__ 路径与上游 action 的 output_schema 不匹配\n"
        "- source/__from__ 指向的数组元素 schema 中不存在某个字段\n\n"
        "- workflow 结构不符合 DSL schema（例如节点 type 非法）\n\n"
        "总体目标：在“尽量不改变工作流整体结构”的前提下，修复这些错误，使 workflow 通过静态校验。\n\n"
        "具体要求（很重要，请严格遵守）：\n"
        "1. 结构保持稳定：\n"
        "   - 不要增加或删除节点；\n"
        "   - 不要随意增加或删除 edges；\n"
        "   - 只能在必要时局部调整 edge.condition（true/false/null），一般情况下保持 edges 原样。\n\n"
        "2. action 节点修复优先级：\n"
        "   - 首先根据 action_schemas[action_id].arg_schema 补齐 params 里缺失的必填字段，或修正错误类型；\n"
        "   - 如果 action_id 本身是合的（存在于 action_schemas 中），优先“修 params”，不要改 action_id；\n"
        "   - 只有当 validation_errors 明确指出 action_id 不存在时，才考虑把 action_id 改成一个更合理的候选，"
        "     并同步更新该节点的 params 使之符合新的 arg_schema。\n\n"
        "3. condition 节点修复：\n"
        "   - 确保 kind/source/field/value 等必填字段齐全且类型正确；\n"
        "   - source 可以是字符串路径，也可以是 {\"__from__\": ...} 对象；\n"
        "   - 当 source 指向数组输出时，field 需要存在于元素 schema 中。\n\n"
        "4. loop 节点：确保 loop_kind（for_each/while）和 source 字段合法，并使用 exports 暴露 body 结果，下游只能引用 result_of.<loop_id>.items / aggregates。\n"
        "5. parallel 节点：branches 必须是非空数组。\n\n"
        "6. 参数绑定 DSL 修复（__from__ 及其聚合逻辑）：\n"
        "   - 对于 {\"__from__\": \"result_of.xxx.data\", \"__agg__\": \"...\", ...}：\n"
        "       - 检查 __from__ 路径是否合法、与 output_schema 对齐；\n"
        "       - 检查 count_if/filter_map/pipeline 中的 field/filter_field/map_field 是否存在于数组元素 schema 中；\n"
        "   - 当错误涉及这些字段时，优先只改字段名（根据元素 schema 的 properties），保持聚合逻辑不变。\n\n"
        "   - 常见错误：loop.exports.items.fields 只有包装字段，但条件需要访问内部子字段，请将 field 改成 <字段>.<子字段>。\n\n"
        "7. 修改范围尽量最小化：\n"
        "   - 当有多种修复方式时，优先选择改动最小、语义最接近原意的方案（如只改一个字段名，而不是重写整个 params）。\n\n"
        "8. 输出要求：\n"
        "   - 保持顶层结构：workflow_name/description/nodes/edges 不变（仅节点内部内容可调整）；\n"
        "   - 节点的 id/type 不变；\n"
        "   - 返回修复后的 workflow JSON，只返回 JSON 对象本身，不要包含代码块标记。"
    )

    user_payload = {
        "workflow": broken_workflow,
        "validation_errors": [asdict(e) for e in validation_errors],
        "action_schemas": action_schemas,
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

    try:
        repaired_workflow = json.loads(text)
    except json.JSONDecodeError:
        log_error("[repair_workflow_with_llm] 无法解析模型返回 JSON")
        log_debug(content)
        raise

    return repaired_workflow


__all__ = [
    "_convert_pydantic_errors",
    "_make_failure_validation_error",
    "_repair_with_llm_and_fallback",
    "repair_workflow_with_llm",
]
