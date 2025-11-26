"""LLM-driven repair helpers used when validation fails."""

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    child_span,
    log_debug,
    log_error,
    log_info,
    log_llm_reasoning,
    log_llm_tool_call,
    log_llm_usage,
    log_success,
    log_warn,
)
from velvetflow.models import Node, PydanticValidationError, ValidationError, Workflow
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.edit_session import WorkflowEditingSession
from velvetflow.planner.tools import WORKFLOW_EDIT_TOOLS, WORKFLOW_VALIDATION_TOOLS
from velvetflow.verification import (
    precheck_loop_body_graphs,
    run_lightweight_static_rules,
    validate_completed_workflow,
)


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


def _collect_validation_errors(
    workflow_raw: Dict[str, Any], action_registry: List[Dict[str, Any]]
) -> List[ValidationError]:
    """Run verification against the current workflow snapshot."""

    precheck_errors = precheck_loop_body_graphs(workflow_raw)
    if precheck_errors:
        return precheck_errors

    try:
        workflow = Workflow.model_validate(workflow_raw)
    except PydanticValidationError as err:
        return _convert_pydantic_errors(workflow_raw, err)

    static_errors = run_lightweight_static_rules(
        workflow.model_dump(by_alias=True), action_registry=action_registry
    )
    if static_errors:
        return static_errors

    return validate_completed_workflow(
        workflow.model_dump(by_alias=True), action_registry=action_registry
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
    editor = WorkflowEditingSession(
        broken_workflow,
        validation_fn=lambda wf: _collect_validation_errors(
            wf, action_registry=action_registry
        ),
    )

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
        "你是一个工作流修复助手，所有修改必须通过工具调用完成。\n"
        "可用工具：update_node/update_node_params/update_edge/add_edge/remove_edge/validate_workflow/submit_workflow。\n"
        "给定当前 workflow、校验失败列表 validation_errors（code/node_id/field/message）以及 action_schemas，请逐条修复。\n"
        "严格要求：\n"
        "- 每次批量修改后调用 validate_workflow，查看最新错误列表，再继续修复。\n"
        "- 默认保持 nodes/edges 数量不变，除非错误信息明确需要调整边。\n"
        "- 优先修复 params/condition/__from__ 等字段，使之符合 arg_schema 和输出 schema。\n"
        "- 当 action_id 合法时优先保留；只有错误指出未知 action_id 时才更新 action_id 并补齐 params。\n"
        "- loop/parallel/condition 节点需要补齐必填字段，引用 loop 结果时只能使用 result_of.<loop_id>.items/aggregates。\n"
        "完成后请调用 validate_workflow 确认无错误，再调用 submit_workflow 返回结果，不要输出自然语言。"
    )

    user_payload = {
        "workflow": broken_workflow,
        "validation_errors": [asdict(e) for e in validation_errors],
        "action_schemas": action_schemas,
    }

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
    ]

    for round_idx in range(6):
        with child_span(f"repair_llm_round_{round_idx}"):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=WORKFLOW_EDIT_TOOLS + WORKFLOW_VALIDATION_TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
        log_llm_usage(model, getattr(resp, "usage", None), operation="repair_workflow")

        msg = resp.choices[0].message
        messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})

        log_llm_reasoning(
            operation="repair_workflow",
            round_idx=round_idx,
            content=msg.content,
            metadata={"validation_error_count": len(validation_errors)},
        )

        if msg.tool_calls:
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    log_error(f"[repair_workflow_with_llm] 工具参数解析失败: {tc.function.arguments}")
                    args = {}

                result = editor.handle_tool_call(tc.function.name, args)
                log_llm_tool_call(
                    operation="repair_workflow",
                    round_idx=round_idx,
                    tool_name=tc.function.name,
                    tool_call_id=tc.id,
                    arguments=args,
                    result=result,
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

                if result.get("type") == "final":
                    return editor.workflow
        else:
            content = msg.content or ""
            text = content.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if "\n" in text:
                    first_line, rest = text.split("\n", 1)
                    if first_line.strip().lower().startswith("json"):
                        text = rest

            try:
                editor.workflow = json.loads(text)
                return editor.workflow
            except json.JSONDecodeError:
                log_error("[repair_workflow_with_llm] 模型未返回工具调用且无法解析 JSON，继续下一轮。")
                log_debug(content)

    return editor.workflow


__all__ = [
    "_convert_pydantic_errors",
    "_make_failure_validation_error",
    "_collect_validation_errors",
    "_repair_with_llm_and_fallback",
    "repair_workflow_with_llm",
]
