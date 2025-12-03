# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""LLM-driven repair helpers used when validation fails."""

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    child_span,
    log_debug,
    log_error,
    log_info,
    log_llm_usage,
    log_success,
    log_warn,
)
from velvetflow.models import Node, PydanticValidationError, ValidationError, Workflow
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.repair_tools import REPAIR_TOOLS, apply_repair_tool
from velvetflow.verification import generate_repair_suggestions, validate_completed_workflow


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


def _summarize_validation_errors_for_llm(
    errors: Sequence[ValidationError],
    *,
    workflow: Mapping[str, Any] | None = None,
    action_registry: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Convert validation errors to an LLM-friendly, human-readable summary."""

    if not errors:
        return "未提供可用的错误信息。"

    lines: List[str] = [
        "请逐条修复下列校验错误（必须全部清零才能继续，系统只有有限的自动修复轮次）："
    ]
    loop_hints: List[str] = []
    schema_hints: List[str] = []
    repair_prompts: List[str] = []

    action_map: Dict[str, Mapping[str, Any]] = {}
    node_to_action: Dict[str, str] = {}
    if action_registry:
        action_map = {
            str(a.get("action_id")): a
            for a in action_registry
            if isinstance(a, Mapping) and a.get("action_id")
        }

    if workflow and isinstance(workflow, Mapping):
        for n in workflow.get("nodes", []) or []:
            if not isinstance(n, Mapping):
                continue
            node_id = n.get("id")
            action_id = n.get("action_id") if n.get("type") == "action" else None
            if isinstance(node_id, str) and isinstance(action_id, str):
                node_to_action[node_id] = action_id
    for idx, err in enumerate(errors, start=1):
        locations = []
        if err.node_id:
            locations.append(f"节点 {err.node_id}")
        if err.field:
            locations.append(f"字段 {err.field}")

        location = f"（{', '.join(locations)}）" if locations else ""
        lines.append(f"{idx}. [{err.code}]{location}：{err.message}")

        # 提供针对 loop 节点的额外修复提示，帮助模型快速定位核心问题
        if err.field and (
            "loop_kind" in err.field
            or err.field.startswith("exports.items")
            or err.field in {"source", "item_alias"}
        ):
            loop_hints.append(
                "- loop 节点需同时包含 params.loop_kind（for_each/while）、params.source、"
                "params.item_alias；exports.items.fields 只能包含 body_subgraph 或 source 元素 schema 中存在的字段"
            )

        if err.node_id and err.field:
            action_id = node_to_action.get(err.node_id)
            action_def = action_map.get(action_id) if action_id else None
            properties = (
                action_def.get("arg_schema", {}).get("properties")
                if isinstance(action_def, Mapping)
                else None
            )
            if isinstance(properties, Mapping) and err.field in properties:
                field_schema = properties.get(err.field)
                if isinstance(field_schema, Mapping):
                    expected_type = field_schema.get("type") or field_schema.get("anyOf")
                    schema_hints.append(
                        f"- 动作 {action_id} 的字段 {err.field} 期望类型/结构：{expected_type}，请按 schema 修正。"
                    )

        prompt = _ERROR_TYPE_PROMPTS.get(err.code)
        if prompt:
            repair_prompts.append(f"- [{err.code}] {prompt}")

    if schema_hints:
        lines.append("")
        lines.append("Schema 提示（按提示修改可减少重复校验）：")
        lines.extend(sorted(set(schema_hints)))

    if loop_hints:
        lines.append("")
        lines.append("Loop 修复提示（补齐必填字段/字段名需与 source schema 对齐）：")
        lines.extend(sorted(set(loop_hints)))

    if repair_prompts:
        lines.append("")
        lines.append("错误特定提示（按类型引导 LLM 思考与修复）：")
        lines.extend(sorted(set(repair_prompts)))

    return "\n".join(lines)


_ERROR_TYPE_PROMPTS: Dict[str, str] = {
    "MISSING_REQUIRED_PARAM": "补齐 params.<field>，优先绑定上游符合 arg_schema 的输出或使用 schema 默认值。",
    "UNKNOWN_ACTION_ID": "替换为 Action Registry 中存在的 action_id，保持节点含义最接近并同步校正 params。",
    "UNKNOWN_PARAM": "移除或重命名 params 中未在 arg_schema 声明的字段，使其与 schema 对齐。",
    "DISCONNECTED_GRAPH": "连接无入度/出度节点，确保从 start 或入参到每个节点都有可达路径。",
    "INVALID_EDGE": "修复边的 from/to 以引用存在的节点，并保持条件字段合法。",
    "SCHEMA_MISMATCH": "调整绑定字段或聚合方式，使类型与上游 output_schema/arg_schema 兼容。",
    "INVALID_SCHEMA": "按 DSL 结构补齐/修正字段类型（nodes/edges/params），避免 JSON 结构缺失。",
    "INVALID_LOOP_BODY": "补齐 loop.body_subgraph 的 entry/exit/nodes/edges，并确保 exports.items 字段匹配 body 输出。",
    "STATIC_RULES_SUMMARY": "遵循静态规则摘要中的提示逐项修复，优先处理连通性和 schema 不匹配。",
    "EMPTY_PARAM_VALUE": "为空的参数应填入非空值或绑定，避免空字符串/空对象。",
    "EMPTY_PARAMS": "params 不能是空对象，为必填字段提供值或引用，并使用工具完成不确定的填充。",
    "SELF_REFERENCE": "移除节点对自身 result_of.<node> 的引用，改用上游输出或拆分节点并通过工具应用修改。",
    "SYNTAX_ERROR": "修正 DSL 语法（括号、逗号、引号等），使解析器能生成 AST。",
    "GRAMMAR_VIOLATION": "遵循文法约束调整节点/字段位置，按期望的 token/产生式重新组织。",
}


def apply_rule_based_repairs(
    workflow_raw: Mapping[str, Any],
    action_registry: Sequence[Mapping[str, Any]],
    validation_errors: Sequence[ValidationError],
) -> Tuple[Dict[str, Any], List[ValidationError]]:
    """Run deterministic repairs before falling back to LLM fixes.

    The repair logic relies on parser-backed AST cloning plus schema-driven templates
    and constraint solving to patch common issues. Remaining validation errors are
    returned so callers can decide whether an LLM hand-off is still required.
    """

    patched_workflow, _suggestions = generate_repair_suggestions(
        workflow_raw, action_registry, errors=validation_errors
    )
    try:
        remaining_errors = validate_completed_workflow(patched_workflow, list(action_registry))
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] 规则修复后重跑校验失败：{exc}")
        remaining_errors = list(validation_errors)

    return patched_workflow, remaining_errors


def _safe_repair_invalid_loop_body(workflow_raw: Mapping[str, Any]) -> Workflow:
    """Best-effort patch for loop body graphs with missing nodes.

    当 loop.body_subgraph 的 edges/entry/exit 引用缺失节点时，校验会抛出
    异常并导致 AutoRepair 的回退逻辑失败。这里先补齐缺失节点（默认使用
    ``end`` 类型）后再做校验，确保至少可以返回结构化的 fallback workflow。
    """

    workflow_dict: Dict[str, Any] = dict(workflow_raw)
    nodes = workflow_dict.get("nodes") if isinstance(workflow_dict.get("nodes"), list) else []

    for node in nodes:
        if not isinstance(node, Mapping) or node.get("type") != "loop":
            continue

        params = node.get("params") if isinstance(node.get("params"), dict) else {}
        body = params.get("body_subgraph") if isinstance(params, Mapping) else None
        if not isinstance(body, dict):
            continue

        body_nodes = body.get("nodes") if isinstance(body.get("nodes"), list) else []
        body_ids = {bn.get("id") for bn in body_nodes if isinstance(bn, Mapping)}

        def _append_missing(node_id: str):
            if node_id in body_ids:
                return
            placeholder = {"id": node_id, "type": "end"}
            body_nodes.append(placeholder)
            body_ids.add(node_id)

        entry = body.get("entry")
        if isinstance(entry, str):
            _append_missing(entry)

        exit_node = body.get("exit")
        if isinstance(exit_node, str):
            _append_missing(exit_node)

        for edge in body.get("edges") or []:
            if not isinstance(edge, Mapping):
                continue
            frm = edge.get("from")
            to = edge.get("to")
            if isinstance(frm, str):
                _append_missing(frm)
            if isinstance(to, str):
                _append_missing(to)

        body["nodes"] = body_nodes
        params["body_subgraph"] = body
        node["params"] = params

    workflow_dict["nodes"] = nodes
    return Workflow.model_validate(workflow_dict)


def _repair_with_llm_and_fallback(
    *,
    broken_workflow: Dict[str, Any],
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
    search_service,
    reason: str,
) -> Workflow:
    log_info(f"[AutoRepair] {reason}，将错误上下文提交给 LLM 尝试修复。")

    patched_workflow, remaining_errors = apply_rule_based_repairs(
        broken_workflow, action_registry, validation_errors
    )
    if not remaining_errors:
        log_success("[AutoRepair] 规则/模板修复已清零错误，无需进入 LLM。")
        return Workflow.model_validate(patched_workflow)

    if len(remaining_errors) < len(validation_errors):
        log_info(
            f"[AutoRepair] 规则修复降低错误数量：{len(validation_errors)} -> {len(remaining_errors)}"
        )

    broken_workflow = patched_workflow
    validation_errors = list(remaining_errors)

    error_summary = _summarize_validation_errors_for_llm(
        validation_errors,
        workflow=broken_workflow,
        action_registry=action_registry,
    )
    try:
        repaired_raw = repair_workflow_with_llm(
            broken_workflow=broken_workflow,
            validation_errors=validation_errors,
            action_registry=action_registry,
            error_summary=error_summary,
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
            fallback = _safe_repair_invalid_loop_body(broken_workflow)
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
            return Workflow(
                workflow_name="fallback_workflow", nodes=[], declared_edges=[]
            )


def repair_workflow_with_llm(
    broken_workflow: Dict[str, Any],
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
    error_summary: str | None = None,
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

    system_prompt = """
你是一个确定性的工作流修复系统（deterministic Workflow Repair System）。

当前输入包括：
- 完整的 workflow 文件；
- 完整的错误日志（不得截断）；
- 最近一次应用的 patch（如果有）；
- 复现输入。

你需要：
1. 找到确切的根因，并给出 Root-Cause → Failure 链条。
2. 生成 3 个候选修复策略，并详细说明为何选择最优策略。
3. 按以下结构输出精确补丁：
   {
     "patch": [{ "file": "...", "before": "...", "after": "..." }],
     "rationale": "..."
   }
4. 进行自我验证：说明为何修复有效、为何错误不会复现、为何不会影响其他 workflow 部分。

当前有一个 workflow JSON 和一组结构化校验错误 validation_errors。
validation_errors 是 JSON 数组，元素包含 code/node_id/field/message。
这些错误来自：
- action 参数缺失或不符合 arg_schema
- condition 条件不完整
- source/__from__ 路径引用了不存在的节点
- source/__from__ 路径与上游 action 的 output_schema 不匹配
- source/__from__ 指向的数组元素 schema 中不存在某个字段
- workflow 结构不符合 DSL schema（例如节点 type 非法）

总体目标：在“尽量不改变工作流整体结构”的前提下，修复这些错误，使 workflow 通过静态校验。

具体要求（很重要，请严格遵守）：
1. 结构保持稳定：
   - 不要增加或删除节点；
   - 不要随意增加或删除 edges；
   - 只能在必要时局部调整 edge.condition（true/false/null），一般情况下保持 edges 原样。

2. action 节点修复优先级：
   - 首先根据 action_schemas[action_id].arg_schema 补齐 params 里缺失的必填字段，或修正错误类型；
   - 如果 action_id 本身是合的（存在于 action_schemas 中），优先“修 params”，不要改 action_id；
   - 只有当 validation_errors 明确指出 action_id 不存在时，才考虑把 action_id 改成一个更合理的候选，
     并同步更新该节点的 params 使之符合新的 arg_schema。

3. condition 节点修复：
   - 确保 kind/source/field/value 等必填字段齐全且类型正确；
   - source 可以是字符串路径，也可以是 {"__from__": ...} 对象；
   - 当 source 指向数组输出时，field 需要存在于元素 schema 中。

4. loop 节点：确保 loop_kind（for_each/while）和 source 字段合法，并使用 exports 暴露 body 结果，下游只能引用 result_of.<loop_id>.items / aggregates。
5. parallel 节点：branches 必须是非空数组。

6. 参数绑定 DSL 修复（__from__ 及其聚合逻辑）：
   - 对于 {"__from__": "result_of.xxx.data", "__agg__": "...", ...}：
       - 检查 __from__ 路径是否合法、与 output_schema 对齐；
       - __agg__ 必须是 identity/count/count_if/join/format_join/filter_map/pipeline 之一，非法取值需要改成最接近语义的枚举值；
       - 检查 count_if/filter_map/pipeline 中的 field/filter_field/map_field 是否存在于数组元素 schema 中；
   - 当错误涉及这些字段时，优先只改字段名（根据元素 schema 的 properties），保持聚合逻辑不变。

   - 常见错误：loop.exports.items.fields 只有包装字段，但条件需要访问内部子字段，请将 field 改成 <字段>.<子字段>。

7. 修改范围尽量最小化：
   - 当有多种修复方式时，优先选择改动最小、语义最接近原意的方案（如只改一个字段名，而不是重写整个 params）。
   - 当 validation_error_summary 提供 Schema 提示或路径信息时，优先按提示矫正字段类型/结构，避免多轮重复犯错。

8. 修复回合有限且必须闭环：
   - 系统会在有限轮次内重新校验；如果你忽略任何一条 validation_error，流程将直接进入下一轮甚至终止。
   - 请逐条对照 validation_errors，把所有问题修到为 0 再输出结果，避免留存隐患。

9. 输出要求：
   - 保持顶层结构：workflow_name/description/nodes/edges 不变（仅节点内部内容可调整）；
   - 节点的 id/type 不变；
   - 返回修复后的 workflow JSON，只返回 JSON 对象本身，不要包含代码块标记。
10. 可用工具：当你需要结构化修改时，优先调用提供的工具（无 LLM 依赖、结果确定），用来修复 loop body 引用、补齐必填参数或写入指定字段。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "workflow": broken_workflow,
                    "validation_error_summary": error_summary,
                    "validation_errors": [asdict(e) for e in validation_errors],
                    "action_schemas": action_schemas,
                },
                ensure_ascii=False,
            ),
        },
    ]

    working_workflow: Dict[str, Any] = broken_workflow

    while True:
        with child_span("repair_llm"):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=REPAIR_TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
        log_llm_usage(model, getattr(resp, "usage", None), operation="repair_workflow")
        if not resp.choices:
            raise RuntimeError("repair_workflow_with_llm 未返回任何候选消息")

        msg = resp.choices[0].message
        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
            for tc in msg.tool_calls:
                func_name = tc.function.name
                raw_args = tc.function.arguments
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    log_error(f"[repair_workflow_with_llm] 无法解析工具参数: {raw_args}")
                    args = {}

                log_info(f"[repair_workflow_with_llm] 调用修复工具 {func_name}，参数: {args}")
                patched_workflow, summary = apply_repair_tool(
                    tool_name=func_name,
                    args=args,
                    workflow=working_workflow,
                    validation_errors=validation_errors,
                    action_registry=action_registry,
                )
                log_info(
                    "[repair_workflow_with_llm] 工具调用完成："
                    f"tool={func_name}, args={args}, summary={summary}"
                )
                if summary.get("applied"):
                    working_workflow = patched_workflow

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            {
                                "workflow": working_workflow,
                                "summary": summary,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
            continue

        content = msg.content or ""
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                first_line, rest = text.split("\n", 1)
                if first_line.strip().lower().startswith("json"):
                    text = rest

        if not text:
            log_warn("[repair_workflow_with_llm] LLM 返回空内容，使用最近一次工具输出。")
            return working_workflow

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
    "apply_rule_based_repairs",
    "repair_workflow_with_llm",
]
