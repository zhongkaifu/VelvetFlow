# velvetflow/verification/llm_repair.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import json

from velvetflow.type_system import WorkflowTypeValidationError
from velvetflow.verification.type_validation import validate_workflow_types
from velvetflow.verification.type_repair import (
    RepairAction,
    RepairResult,
    attempt_local_type_repairs,
)


@runtime_checkable
class LLMClientProtocol(Protocol):
    """
    你自己的 LLM client 只要实现这个协议就可以接入。
    例如：
        class OpenAILLMClient:
            def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
                ...
    """
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        ...


@dataclass
class LLMRepairConfig:
    """
    LLM 修复的一些参数，可按需扩展。
    """
    max_rounds: int = 2           # 最多几轮 LLM 修复
    include_failed_errors: bool = True  # 每轮失败后是否把错误信息再喂回 LLM
    pretty_workflow_json: bool = True   # 传给 LLM 的 workflow 是否做 pretty-print


@dataclass
class LLMRepairResult:
    """
    LLM 修复结束后的结果：
    - workflow: 修复后的 workflow 对象（可能就是原始对象，如果没修）
    - rounds: 实际使用了几轮 LLM 修复
    - success: 是否最终通过类型校验
    - last_error: 如果失败，最后一次校验的错误（字符串）
    - local_repair: 之前本地规则修复的结果（方便上层统一日志/调试）
    """
    workflow: Any
    rounds: int
    success: bool
    last_error: Optional[str]
    local_repair: Optional[RepairResult] = None


# ============================
# prompt & payload 构造
# ============================

def _serialize_workflow_to_json(workflow: Any, pretty: bool = True) -> str:
    """
    将 Workflow 对象转成 JSON 字符串。假定是 Pydantic 模型或带 .dict() 方法。
    """
    if hasattr(workflow, "model_dump"):
        data = workflow.model_dump()
    elif hasattr(workflow, "dict"):
        data = workflow.dict()
    else:
        # 最后兜底：如果本身就是 dict，就直接用
        if isinstance(workflow, dict):
            data = workflow
        else:
            # 不认识的类型，试试 asdict / __dict__
            try:
                data = workflow.__dict__
            except Exception as e:
                raise TypeError(f"Cannot serialize workflow to JSON: {e}")

    if pretty:
        return json.dumps(data, indent=2, ensure_ascii=False)
    return json.dumps(data, ensure_ascii=False)


def _build_llm_repair_system_prompt() -> str:
    """
    提供给 LLM 的 system prompt。
    核心要求：
      - 你是一个“类型安全工作流修复器”
      - 不能更改语义，只能修复为了匹配类型系统的绑定/结构
    """
    return (
        "You are an expert workflow compiler and type checker. "
        "You receive a JSON representation of a workflow DSL, a list of type errors, "
        "and some local repair suggestions.\n\n"
        "Your task:\n"
        "1. Modify the workflow JSON to fix the type errors.\n"
        "2. Prefer minimal changes that preserve the original business intent.\n"
        "3. Use the provided repair suggestions when they make sense.\n"
        "4. You MUST return ONLY a JSON object with a single top-level key 'workflow', "
        "   whose value is the full repaired workflow JSON.\n"
        "5. Do NOT include any explanations, comments, or extra keys.\n"
    )


def _build_llm_repair_user_prompt(
    workflow_json: str,
    type_errors: List[str],
    suggestions: List[RepairAction],
) -> str:
    """
    构造 user prompt，把：
      - 当前 workflow（JSON）
      - 类型错误
      - 本地修复建议（含候选字段等）
    全部打包给 LLM。
    """
    payload: Dict[str, Any] = {
        "workflow": json.loads(workflow_json),
        "type_errors": type_errors,
        "local_repair_suggestions": [],
    }

    for s in suggestions:
        payload["local_repair_suggestions"].append(
            {
                "node_id": s.node_id,
                "param_name": s.param_name,
                "kind": s.kind,
                "description": s.description,
                "old_value": s.old_value,
                "new_value": s.new_value,
            }
        )

    # 直接把 payload 作为 JSON 给 LLM，当作“输入上下文”
    # 这样 LLM 只需在此基础上改 workflow 字段即可
    return json.dumps(payload, indent=2, ensure_ascii=False)


def _extract_workflow_from_llm_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    LLM 约定：返回 {"workflow": {...}}。
    这里做一层防御式检查。
    """
    if not isinstance(response, dict):
        raise ValueError("LLM response is not a JSON object")

    if "workflow" not in response:
        # 有些模型可能把 JSON 当字符串包了一层，可以尝试解析一次
        # （如果你用的是严格 JSON mode，可以把这段删了）
        if len(response) == 1:
            only_val = next(iter(response.values()))
            if isinstance(only_val, str):
                try:
                    inner = json.loads(only_val)
                    if "workflow" in inner:
                        return inner["workflow"]
                except Exception:
                    pass
        raise ValueError("LLM response does not contain top-level key 'workflow'")

    return response["workflow"]
    

# ============================
# 顶层：基于 LLM 的修复循环
# ============================

def llm_based_repair_workflow(
    *,
    workflow: Any,
    action_registry: Any,
    type_errors: List[str],
    suggestions: List[RepairAction],
    llm_client: LLMClientProtocol,
    config: Optional[LLMRepairConfig] = None,
) -> LLMRepairResult:
    """
    对给定 workflow 做多轮 LLM 自动修复，直到：
      - 通过 validate_workflow_types，或
      - 达到 max_rounds

    参数：
      workflow        当前 workflow 对象（Pydantic/Dataclass/dict 都可）
      action_registry ActionRegistry 实例（用于 validate_workflow_types）
      type_errors     上一轮类型校验的错误列表（字符串）
      suggestions     本地规则修复产生的 RepairAction（is_suggestion=True 的那批）
      llm_client      实现了 LLMClientProtocol 的 client
      config          LLMRepairConfig 配置

    返回：
      LLMRepairResult（包含最终 workflow、是否成功、使用了几轮等）
    """
    from velvetflow.verification.type_validation import validate_workflow_types

    if config is None:
        config = LLMRepairConfig()

    current_workflow = workflow
    current_errors = list(type_errors)
    rounds = 0
    last_error_text: Optional[str] = None

    system_prompt = _build_llm_repair_system_prompt()

    while rounds < config.max_rounds:
        rounds += 1

        # 1) 序列化当前 workflow
        wf_json = _serialize_workflow_to_json(
            current_workflow, pretty=config.pretty_workflow_json
        )

        # 2) 构造 user prompt payload
        user_payload_errors = current_errors if config.include_failed_errors else type_errors
        user_prompt = _build_llm_repair_user_prompt(
            workflow_json=wf_json,
            type_errors=user_payload_errors,
            suggestions=suggestions,
        )

        # 3) 调用 LLM（要求返回 JSON）
        try:
            llm_response = llm_client.complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as e:
            # 调用失败，无法继续修复
            last_error_text = f"LLM call failed: {e}"
            break

        # 4) 从 LLM 响应中提取 workflow JSON
        try:
            repaired_workflow_dict = _extract_workflow_from_llm_response(llm_response)
        except Exception as e:
            last_error_text = f"LLM response parsing failed: {e}"
            break

        # 5) 把 dict 转回 Workflow 对象
        #    注意：这里需要根据你的实际 Workflow 模型来构造
        workflow_cls = type(current_workflow)
        try:
            if hasattr(workflow_cls, "model_validate"):
                # Pydantic v2
                repaired_workflow = workflow_cls.model_validate(repaired_workflow_dict)
            elif hasattr(workflow_cls, "parse_obj"):
                # Pydantic v1
                repaired_workflow = workflow_cls.parse_obj(repaired_workflow_dict)
            else:
                # 如果不是 Pydantic，你可以自定义 from_dict 之类的构造方法
                repaired_workflow = workflow_cls(**repaired_workflow_dict)
        except Exception as e:
            last_error_text = f"Failed to construct Workflow from LLM result: {e}"
            current_workflow = repaired_workflow_dict  # 作为调试信息保留
            break

        # 6) 类型校验
        try:
            validate_workflow_types(repaired_workflow, action_registry)
            # ✅ 通过类型校验，成功
            return LLMRepairResult(
                workflow=repaired_workflow,
                rounds=rounds,
                success=True,
                last_error=None,
                local_repair=None,  # 上层可以自己填（如果想一起返回）
            )
        except WorkflowTypeValidationError as e:
            # 记录错误，用于下一轮 prompt
            current_workflow = repaired_workflow
            current_errors = list(e.errors)
            last_error_text = "\n".join(e.errors)
            # 再继续下一轮（如果还没超过 max_rounds）

    # 如果走到这里，说明超过 max_rounds 或中途失败
    return LLMRepairResult(
        workflow=current_workflow,
        rounds=rounds,
        success=False,
        last_error=last_error_text,
        local_repair=None,
    )


def repair_workflow_types_with_local_and_llm(
    *,
    workflow: Any,
    action_registry: Any,
    llm_client: LLMClientProtocol,
    config: Optional[LLMRepairConfig] = None,
) -> LLMRepairResult:
    """
    Run type validation with layered repairs: rule-based first, LLM fallback.

    1) 调用 ``validate_workflow_types``。若通过，直接返回。
    2) 若有错误，先用 ``attempt_local_type_repairs`` 自动修复常见问题。
    3) 如果仍有错误，再调用 ``llm_based_repair_workflow``，把剩余错误
       和本地修复的 suggestions 作为提示。
    4) 最后再次运行 ``validate_workflow_types``，确保返回的 workflow 类型安全。
    """

    if config is None:
        config = LLMRepairConfig()

    try:
        validate_workflow_types(workflow, action_registry)
        return LLMRepairResult(
            workflow=workflow,
            rounds=0,
            success=True,
            last_error=None,
            local_repair=None,
        )
    except WorkflowTypeValidationError as exc:
        initial_errors = list(exc.errors)

    local_repair = attempt_local_type_repairs(workflow, action_registry)
    if not local_repair.remaining_errors:
        # 本地修复已解决全部问题，做一次最终校验
        validate_workflow_types(workflow, action_registry)
        return LLMRepairResult(
            workflow=workflow,
            rounds=0,
            success=True,
            last_error=None,
            local_repair=local_repair,
        )

    llm_result = llm_based_repair_workflow(
        workflow=workflow,
        action_registry=action_registry,
        type_errors=local_repair.remaining_errors or initial_errors,
        suggestions=local_repair.suggestions,
        llm_client=llm_client,
        config=config,
    )

    llm_result.local_repair = local_repair

    if llm_result.success:
        # 再次确认类型安全
        try:
            validate_workflow_types(llm_result.workflow, action_registry)
        except WorkflowTypeValidationError as exc:  # pragma: no cover - defensive
            return LLMRepairResult(
                workflow=llm_result.workflow,
                rounds=llm_result.rounds,
                success=False,
                last_error="\n".join(exc.errors),
                local_repair=local_repair,
            )

    return llm_result
