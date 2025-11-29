"""LLM-driven repair helpers used when validation fails."""

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional

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
from velvetflow.planner.tool_catalog import get_llm_toolbox, list_tool_names
from velvetflow.planner.workflow_builder import WorkflowBuilder


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


def _builder_from_workflow(workflow: Mapping[str, Any]) -> WorkflowBuilder:
    builder = WorkflowBuilder()
    builder.load_workflow(workflow)
    return builder


def _apply_planner_tool_in_repair(
    *,
    builder: WorkflowBuilder,
    func_name: str,
    args: Mapping[str, Any],
    action_registry: List[Mapping[str, Any]],
    last_action_candidates: List[str],
):
    tool_result: Dict[str, Any]

    if func_name == "search_business_actions":
        query = args.get("query", "")
        top_k = int(args.get("top_k", 5))
        lowered = str(query).lower()
        candidates = []
        for action in action_registry:
            if not isinstance(action, Mapping):
                continue
            text = " ".join(
                str(action.get(field, "")) for field in ("name", "description", "domain")
            ).lower()
            if lowered in text:
                candidates.append(
                    {
                        "id": action.get("action_id"),
                        "name": action.get("name", ""),
                        "description": action.get("description", ""),
                        "category": action.get("domain") or "general",
                    }
                )

        candidates = [c for c in candidates if c.get("id")][:top_k]
        last_action_candidates[:] = [c["id"] for c in candidates]
        tool_result = {"status": "ok", "query": query, "candidates": candidates}

    elif func_name == "set_workflow_meta":
        builder.set_meta(args.get("workflow_name", ""), args.get("description"))
        tool_result = {"status": "ok", "type": "meta_set"}

    elif func_name == "add_node":
        node_id = args.get("id")
        node_type = args.get("type")
        action_id = args.get("action_id")
        display_name = args.get("display_name")
        params = args.get("params")

        if not isinstance(node_id, str) or not isinstance(node_type, str):
            tool_result = {"status": "error", "message": "add_node 需要提供字符串类型的 id/type。"}
        elif action_id and last_action_candidates and action_id not in last_action_candidates:
            tool_result = {
                "status": "error",
                "message": "action_id 必须是最近一次 search_business_actions 返回的 candidates.id 之一。",
                "allowed_action_ids": list(last_action_candidates),
            }
        else:
            builder.add_node(
                node_id,
                node_type,
                action_id if isinstance(action_id, str) else None,
                display_name if isinstance(display_name, str) else None,
                params if isinstance(params, Mapping) else {},
                args.get("true_to_node") if isinstance(args.get("true_to_node"), (str, type(None))) else None,
                args.get("false_to_node") if isinstance(args.get("false_to_node"), (str, type(None))) else None,
            )
            tool_result = {"status": "ok", "type": "node_added", "node_id": node_id}

    elif func_name == "update_node":
        node_id = args.get("id")
        updates = args.get("updates")
        if not isinstance(node_id, str):
            tool_result = {"status": "error", "message": "update_node 需要提供字符串类型的 id。"}
        elif not isinstance(updates, list):
            tool_result = {"status": "error", "message": "updates 必须是数组。"}
        else:
            builder.update_node(node_id, updates)
            tool_result = {"status": "ok", "type": "node_updated", "node_id": node_id}

    elif func_name == "remove_node":
        node_id = args.get("id")
        if not isinstance(node_id, str):
            tool_result = {"status": "error", "message": "remove_node 需要提供字符串类型的 id。"}
        else:
            builder.remove_node(node_id)
            tool_result = {"status": "ok", "type": "node_removed", "node_id": node_id}

    elif func_name == "finalize_workflow":
        tool_result = {"status": "ok", "type": "finalized", "workflow": builder.to_workflow()}

    else:
        tool_result = {"status": "error", "message": f"未知工具 {func_name}"}

    return tool_result


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
你是一个工作流修复助手。
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
       - 检查 count_if/filter_map/pipeline 中的 field/filter_field/map_field 是否存在于数组元素 schema 中；
   - 当错误涉及这些字段时，优先只改字段名（根据元素 schema 的 properties），保持聚合逻辑不变。

   - 常见错误：loop.exports.items.fields 只有包装字段，但条件需要访问内部子字段，请将 field 改成 <字段>.<子字段>。

7. 修改范围尽量最小化：
   - 当有多种修复方式时，优先选择改动最小、语义最接近原意的方案（如只改一个字段名，而不是重写整个 params）。

8. 输出要求：
   - 保持顶层结构：workflow_name/description/nodes/edges 不变（仅节点内部内容可调整）；
   - 节点的 id/type 不变；
   - 返回修复后的 workflow JSON，只返回 JSON 对象本身，不要包含代码块标记。
9. 可用工具：当你需要结构化修改时，优先调用提供的工具（无 LLM 依赖、结果确定），用来修复 loop body 引用、补齐必填参数或写入指定字段。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "workflow": broken_workflow,
                    "validation_errors": [asdict(e) for e in validation_errors],
                    "action_schemas": action_schemas,
                },
                ensure_ascii=False,
            ),
        },
    ]

    working_workflow: Dict[str, Any] = broken_workflow
    builder = _builder_from_workflow(working_workflow)
    last_action_candidates: List[str] = []
    tool_specs = get_llm_toolbox()
    repair_tool_names = list_tool_names(include_planner=False, include_repair=True)
    planner_tool_names = list_tool_names(include_planner=True, include_repair=False)

    while True:
        with child_span("repair_llm"):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=tool_specs,
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

                if func_name in repair_tool_names:
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
                        builder.load_workflow(working_workflow)

                    tool_payload = {"workflow": working_workflow, "summary": summary}

                elif func_name in planner_tool_names:
                    log_info(f"[repair_workflow_with_llm] 调用规划工具 {func_name}，参数: {args}")
                    tool_result = _apply_planner_tool_in_repair(
                        builder=builder,
                        func_name=func_name,
                        args=args,
                        action_registry=action_registry,
                        last_action_candidates=last_action_candidates,
                    )
                    working_workflow = builder.to_workflow()
                    tool_payload = {"workflow": working_workflow, "result": tool_result}

                else:
                    tool_payload = {
                        "workflow": working_workflow,
                        "summary": {"applied": False, "reason": f"未知工具 {func_name}"},
                    }

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(tool_payload, ensure_ascii=False),
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
    "repair_workflow_with_llm",
]
