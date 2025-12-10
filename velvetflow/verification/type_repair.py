# velvetflow/verification/type_repair.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from velvetflow.type_system import (
    TypeEnvironment,
    TypeRef,
    infer_type_from_from_binding,
)
from velvetflow.verification.type_validation import (
    build_node_output_type,
    get_param_expected_type,
)
from velvetflow.type_system.types import WorkflowTypeValidationError


@dataclass
class RepairAction:
    """
    描述一次自动修复操作（或建议）。
    """
    node_id: str
    param_name: str
    old_value: Any
    new_value: Any
    kind: str
    description: str
    # 如果是「建议」，而非已自动生效，可以 is_suggestion=True
    is_suggestion: bool = False


@dataclass
class RepairResult:
    """
    一轮修复后的结果：
    - applied: 真正改动了 workflow 的修复
    - suggestions: 仅作为建议的修复（需要 LLM/人工选）
    - remaining_errors: 仍然存在的类型错误（字符串版，便于直接日志/传给 LLM）
    """
    applied: List[RepairAction]
    suggestions: List[RepairAction]
    remaining_errors: List[str]


# ==========
# 核心修复规则
# ==========

def _find_object_field_candidates_for_scalar(
    src_type: TypeRef,
    expected_type: TypeRef,
) -> List[Tuple[str, TypeRef]]:
    """
    当 src_type 是 object，expected 是 primitive（string/number/...）时，
    在 src_type.properties 里找出所有兼容字段，返回 (field_name, field_type) 列表。
    """
    schema = src_type.json_schema
    if schema.get("type") != "object":
        return []

    props = schema.get("properties") or {}
    candidates: List[Tuple[str, TypeRef]] = []
    for field_name, field_schema in props.items():
        field_type = TypeRef(json_schema=field_schema, source=f"{src_type.source}.{field_name}")
        if field_type.is_compatible_with(expected_type):
            candidates.append((field_name, field_type))

    return candidates


def _try_fix_param_object_to_scalar(
    *,
    env: TypeEnvironment,
    node,
    action_def: Dict[str, Any],
    param_name: str,
    param_val: Dict[str, Any],
) -> Tuple[Optional[RepairAction], Optional[RepairAction]]:
    """
    修复策略 R1：
    - param 是 {"__from__": "..."}，源类型是 object，期望是 scalar
    - 尝试在 object 的 properties 中找一个类型兼容的字段，自动补 .field_name

    返回：
    - (applied_repair, suggestion_repair)
      二者最多其一非空：
      - 如果有唯一候选 → applied_repair
      - 如果有多个候选 → suggestion_repair（仅建议，不动 workflow）
    """
    expected_type = get_param_expected_type(action_def, param_name)
    if expected_type is None:
        return None, None

    if not (isinstance(param_val, dict) and "__from__" in param_val):
        return None, None

    from_expr = param_val["__from__"]
    src_type = infer_type_from_from_binding(env, from_expr)
    if not src_type:
        return None, None

    src_t = src_type.json_schema.get("type")
    dst_t = expected_type.json_schema.get("type")

    # 只处理“object -> scalar”这种情况
    scalar_types = {"string", "number", "integer", "boolean"}
    if src_t != "object" or dst_t not in scalar_types:
        return None, None

    candidates = _find_object_field_candidates_for_scalar(src_type, expected_type)
    if not candidates:
        return None, None

    old_value = param_val

    # 唯一候选 -> 自动修
    if len(candidates) == 1:
        field_name, _ = candidates[0]
        new_from = f"{from_expr}.{field_name}"
        new_value = {"__from__": new_from}
        return (
            RepairAction(
                node_id=node.id,
                param_name=param_name,
                old_value=old_value,
                new_value=new_value,
                kind="object_to_scalar_field_autofix",
                description=(
                    f"param '{param_name}' expects scalar {dst_t}, "
                    f"but __from__='{from_expr}' is object; "
                    f"auto-select field '{field_name}' → '{new_from}'."
                ),
            ),
            None,
        )

    # 多个候选 -> 生成 suggestion，不自动改
    suggestion = RepairAction(
        node_id=node.id,
        param_name=param_name,
        old_value=old_value,
        new_value={
            "candidates": [
                {
                    "field": field_name,
                    "from_expr": f"{from_expr}.{field_name}",
                    "field_schema": field_type.json_schema,
                }
                for field_name, field_type in candidates
            ]
        },
        kind="object_to_scalar_field_suggestion",
        description=(
            f"param '{param_name}' expects scalar {dst_t}, "
            f"__from__='{from_expr}' is object; "
            f"found {len(candidates)} candidate fields, please choose one."
        ),
        is_suggestion=True,
    )
    return None, suggestion


# ==========
# 顶层：基于 类型错误 做一轮自动修复
# ==========

def _build_type_environment_for_workflow(workflow, action_registry) -> TypeEnvironment:
    """
    和 validate_workflow_types 的第一步一致：
    为每个 node 构建 node.output 的类型环境。
    """
    env = TypeEnvironment(entries={})
    for node in workflow.nodes:
        action_def = action_registry.get(node.action_id)
        if not action_def:
            continue
        out_type = build_node_output_type(action_def)
        env.set(f"{node.id}.output", out_type)
    return env


def attempt_local_type_repairs(
    workflow,
    action_registry,
) -> RepairResult:
    """
    对 workflow 做一轮“局部自动修复”。

    流程：
      1. 先运行类型校验（validate_workflow_types），拿到错误列表
      2. 基于 TypeEnvironment 和 Action 定义，对每个 node/param 尝试局部修复：
         - 当前只实现 R1：object -> scalar 自动选择字段
      3. 应用修复（in-place 改 workflow），再跑一遍类型校验
      4. 返回 RepairResult（applied / suggestions / remaining_errors）

    注意：本函数不会抛 WorkflowTypeValidationError，
          所有错误会记录在 remaining_errors 里。
    """
    from velvetflow.verification.type_validation import validate_workflow_types

    applied: List[RepairAction] = []
    suggestions: List[RepairAction] = []
    remaining_errors: List[str] = []

    # 第一次类型校验，看看有哪些问题
    try:
        validate_workflow_types(workflow, action_registry)
        # 无错误，直接返回空修复
        return RepairResult(applied=applied, suggestions=suggestions, remaining_errors=[])
    except WorkflowTypeValidationError as e:
        # 先保留原始错误文本（方便 debug / 传给上层 LLM）
        remaining_errors = list(e.errors)

    # 构建类型环境
    env = _build_type_environment_for_workflow(workflow, action_registry)

    # 遍历所有 node/param，尝试规则型修复
    for node in workflow.nodes:
        action_def = action_registry.get(node.action_id)
        if not action_def:
            continue

        params: Dict[str, Any] = node.params or {}
        for param_name, param_val in list(params.items()):
            # 只对 dict 类型的绑定做修复尝试
            if not isinstance(param_val, dict):
                continue

            # 规则 R1：object -> scalar 补字段
            auto_fix, suggestion = _try_fix_param_object_to_scalar(
                env=env,
                node=node,
                action_def=action_def,
                param_name=param_name,
                param_val=param_val,
            )
            if auto_fix:
                # 应用修复
                node.params[param_name] = auto_fix.new_value
                applied.append(auto_fix)
                continue
            if suggestion:
                suggestions.append(suggestion)

    # 应用修复后，再跑一遍类型校验，看看还剩什么问题
    try:
        validate_workflow_types(workflow, action_registry)
        # 能通过就清空剩余错误
        remaining_errors = []
    except WorkflowTypeValidationError as e:
        remaining_errors = list(e.errors)

    return RepairResult(
        applied=applied,
        suggestions=suggestions,
        remaining_errors=remaining_errors,
    )
