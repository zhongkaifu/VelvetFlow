"""Structure planning logic for the planner."""

import copy
import json
import os
from typing import Any, Dict, List, Mapping, Optional

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    child_span,
    log_error,
    log_event,
    log_info,
    log_json,
    log_llm_usage,
    log_section,
    log_success,
    log_warn,
)
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.approval import detect_missing_approval_nodes
from velvetflow.planner.coverage import check_requirement_coverage_with_llm
from velvetflow.planner.tools import PLANNER_TOOLS
from velvetflow.planner.workflow_builder import (
    WorkflowBuilder,
    attach_condition_branches,
)
from velvetflow.search import HybridActionSearchService
from velvetflow.models import infer_edges_from_bindings

LOOP_EXPORT_EDIT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "edit_loop_exports",
            "description": "编辑 loop 节点的 exports（items/aggregates），请直接给出完整的 exports。",
            "parameters": {
                "type": "object",
                "properties": {
                    "exports": {
                        "type": "object",
                        "description": "完整的 exports 对象，包含 items/aggregates。",
                        "additionalProperties": True,
                    },
                    "items": {
                        "type": "object",
                        "description": "如果不提供 exports，也可以单独提供 items 段。",
                        "properties": {
                            "from_node": {"type": "string"},
                            "fields": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "mode": {"type": "string", "enum": ["collect", "first", "last"]},
                        },
                    },
                    "aggregates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "from_node": {"type": "string"},
                                "expr": {"type": "object", "additionalProperties": True},
                            },
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
    }
]


def _attach_inferred_edges(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild derived edges so LLMs can see the implicit wiring."""

    copied = copy.deepcopy(workflow)
    nodes = copied.get("nodes") if isinstance(copied.get("nodes"), list) else []
    copied["edges"] = infer_edges_from_bindings(nodes)
    return attach_condition_branches(copied)


def _build_action_schema_map(action_registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    action_schemas: Dict[str, Dict[str, Any]] = {}
    for action in action_registry:
        aid = action.get("action_id")
        if not aid:
            continue
        action_schemas[aid] = {
            "name": action.get("name", ""),
            "description": action.get("description", ""),
            "domain": action.get("domain", ""),
            "arg_schema": action.get("arg_schema"),
            "output_schema": action.get("output_schema"),
        }
    return action_schemas


def _extract_loop_body_context(
    loop_node: Mapping[str, Any], action_schemas: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    params = loop_node.get("params") if isinstance(loop_node, Mapping) else None
    body = params.get("body_subgraph") if isinstance(params, Mapping) else None
    if not isinstance(body, Mapping):
        return {"nodes": []}

    context_nodes = []
    for child in body.get("nodes", []) or []:
        if not isinstance(child, Mapping):
            continue
        action_id = child.get("action_id")
        schema = action_schemas.get(action_id, {}) if isinstance(action_id, str) else {}
        context_nodes.append(
            {
                "id": child.get("id"),
                "type": child.get("type"),
                "action_id": action_id,
                "display_name": child.get("display_name"),
                "output_schema": schema.get("output_schema"),
            }
        )

    return {"nodes": context_nodes}


def _validate_loop_exports(
    *, loop_node: Mapping[str, Any], exports: Mapping[str, Any]
) -> List[str]:
    params = loop_node.get("params") if isinstance(loop_node.get("params"), Mapping) else {}
    body = params.get("body_subgraph") if isinstance(params, Mapping) else {}
    body_nodes = [bn for bn in body.get("nodes", []) or [] if isinstance(bn, Mapping)]
    body_ids = {bn.get("id") for bn in body_nodes if isinstance(bn.get("id"), str)}

    errors: List[str] = []

    if not isinstance(exports, Mapping):
        return ["exports 必须是对象"]

    items = exports.get("items")
    if not isinstance(items, Mapping):
        errors.append("缺少 items 对象")
    else:
        from_node = items.get("from_node")
        if not isinstance(from_node, str) or from_node not in body_ids:
            errors.append("items.from_node 必须引用 body_subgraph.nodes 中的节点")

        fields = items.get("fields")
        if not (isinstance(fields, list) and [f for f in fields if isinstance(f, str)]):
            errors.append("items.fields 必须是非空字符串数组")

        mode = items.get("mode")
        if mode is not None and mode not in {"collect", "first", "last"}:
            errors.append("items.mode 仅支持 collect/first/last")

    aggregates = exports.get("aggregates")
    if aggregates is not None:
        if not isinstance(aggregates, list):
            errors.append("aggregates 必须是数组或省略")
        else:
            for idx, agg in enumerate(aggregates):
                if not isinstance(agg, Mapping):
                    errors.append(f"aggregates[{idx}] 必须是对象")
                    continue

                if not isinstance(agg.get("name"), str):
                    errors.append(f"aggregates[{idx}].name 必须是字符串")

                from_node = agg.get("from_node")
                if not isinstance(from_node, str) or from_node not in body_ids:
                    errors.append(
                        f"aggregates[{idx}].from_node 必须引用 body_subgraph.nodes 中的节点"
                    )

                expr = agg.get("expr")
                if not isinstance(expr, Mapping):
                    errors.append(f"aggregates[{idx}].expr 必须是对象")

    return errors


def _fallback_loop_exports(
    loop_node: Mapping[str, Any], action_schemas: Mapping[str, Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    params = loop_node.get("params") if isinstance(loop_node, Mapping) else None
    if not isinstance(params, Mapping):
        return None
    body = params.get("body_subgraph")
    if not isinstance(body, Mapping):
        return None

    body_nodes = [bn for bn in body.get("nodes", []) or [] if isinstance(bn, Mapping)]
    body_ids = [bn.get("id") for bn in body_nodes if isinstance(bn.get("id"), str)]
    exit_node = body.get("exit") if isinstance(body.get("exit"), str) else None
    from_node = exit_node if exit_node in body_ids else (body_ids[0] if body_ids else None)
    if not from_node:
        return None

    field_candidates: List[str] = []
    target_node = next((bn for bn in body_nodes if bn.get("id") == from_node), None)
    if isinstance(target_node, Mapping):
        action_id = target_node.get("action_id")
        schema = action_schemas.get(action_id, {}) if isinstance(action_id, str) else {}
        props = schema.get("output_schema", {}).get("properties") if isinstance(schema.get("output_schema"), Mapping) else None
        if isinstance(props, Mapping):
            field_candidates = [k for k in props.keys() if isinstance(k, str)]

    fields = field_candidates[:4] if field_candidates else ["status"]
    return {
        "items": {
            "from_node": from_node,
            "fields": fields,
            "mode": "collect",
        },
        "aggregates": [],
    }


def _ensure_loop_items_fields(
    *,
    exports: Mapping[str, Any],
    loop_node: Mapping[str, Any],
    action_schemas: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Ensure items.fields is a non-empty list.

    If the original exports already contains non-empty fields, it will be
    returned unchanged. Otherwise, we try to infer several representative
    fields from the referenced body node's output schema; fall back to a
    single "status" field if nothing is available.
    """

    items_spec = exports.get("items")
    if not isinstance(items_spec, Mapping):
        return dict(exports)

    fields = items_spec.get("fields") if isinstance(items_spec.get("fields"), list) else []
    normalized_fields = [f for f in fields if isinstance(f, str)]
    if normalized_fields:
        return exports

    params = loop_node.get("params") if isinstance(loop_node.get("params"), Mapping) else {}
    body = params.get("body_subgraph") if isinstance(params, Mapping) else {}
    body_nodes = [bn for bn in body.get("nodes", []) or [] if isinstance(bn, Mapping)]
    target_id = items_spec.get("from_node") if isinstance(items_spec.get("from_node"), str) else None
    target_node = next((bn for bn in body_nodes if bn.get("id") == target_id), None)

    fallback_fields: list[str] = []
    if isinstance(target_node, Mapping):
        action_id = target_node.get("action_id") if isinstance(target_node.get("action_id"), str) else None
        schema = action_schemas.get(action_id, {}) if isinstance(action_id, str) else {}
        props = schema.get("output_schema", {}).get("properties") if isinstance(schema.get("output_schema"), Mapping) else None
        if isinstance(props, Mapping):
            fallback_fields = [k for k in props.keys() if isinstance(k, str)]

    if not fallback_fields:
        fallback_fields = ["status"]

    new_items = dict(items_spec)
    new_items["fields"] = fallback_fields[:4]
    new_exports = dict(exports)
    new_exports["items"] = new_items
    return new_exports


def _synthesize_loop_exports_with_llm(
    *,
    client: OpenAI,
    model: str,
    nl_requirement: str,
    loop_node: Mapping[str, Any],
    action_schemas: Mapping[str, Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    body_context = _extract_loop_body_context(loop_node, action_schemas)
    system_prompt = (
        "你是一个专门为循环节点设计 exports 的助手。\n"
        "给定 loop 节点（含 body_subgraph）以及上游的自然语言需求，"
        "请输出符合 DSL 的 exports 结构，用于将循环子图的结果暴露给外部节点。\n"
        "要求：\n"
        "1) 只输出 JSON（不要代码块），格式可以是 {\"exports\": {...}} 或直接 exports 对象。\n"
        "2) items.from_node 必须引用 body_subgraph.nodes 中的节点，fields 需列出你希望暴露的字段。\n"
        "3) aggregates 是可选的 count_if/max/min/sum/avg 聚合，from_node 同样只能指向 body_subgraph 节点。\n"
        "4) 避免自然语言解释，使用结构化表达式，字段名优先依据节点 output_schema.properties。\n"
        "示例（仅示意，不要生搬硬套字段名）：\n"
        "{\n  \"items\": {\"from_node\": \"finish_employee\", \"fields\": [\"employee_id\", \"risk\"], \"mode\": \"collect\"},\n"
        " \"aggregates\": [{\"name\": \"high_risk_count\", \"from_node\": \"finish_employee\", \"expr\": {\"kind\": \"count_if\", \"field\": \"risk\", \"op\": \">\", \"value\": 0.8}}]\n}"
    )

    payload = {
        "nl_requirement": nl_requirement,
        "loop_node": loop_node,
        "loop_body": body_context,
        "hint": "选择具有完整输出的节点作为 items.from_node，字段来自其 output_schema.properties。",
    }

    with child_span("loop_exports_llm"):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
    log_llm_usage(model, getattr(resp, "usage", None), operation="synthesize_loop_exports")
    if not resp.choices:
        raise RuntimeError("_synthesize_loop_exports_with_llm 未返回任何候选消息")

    content = resp.choices[0].message.content or ""
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            first_line, rest = text.split("\n", 1)
            if first_line.strip().lower().startswith("json"):
                text = rest

    decoder = json.JSONDecoder()
    parsed: Any
    try:
        parsed, _ = decoder.raw_decode(text)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, Mapping):
        exports = parsed.get("exports") if "exports" in parsed else parsed
        if isinstance(exports, Mapping):
            return dict(exports)
    return None


def _extract_exports_from_tool_args(raw_args: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw_args, Mapping):
        return None

    if isinstance(raw_args.get("exports"), Mapping):
        return dict(raw_args["exports"])

    candidate: Dict[str, Any] = {}
    if isinstance(raw_args.get("items"), Mapping):
        candidate["items"] = raw_args["items"]
    if isinstance(raw_args.get("aggregates"), list):
        candidate["aggregates"] = raw_args["aggregates"]

    return candidate or None


def _plan_loop_exports_with_tools(
    *,
    client: OpenAI,
    model: str,
    nl_requirement: str,
    loop_node: Mapping[str, Any],
    action_schemas: Mapping[str, Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    body_context = _extract_loop_body_context(loop_node, action_schemas)
    system_prompt = (
        "你是一个专门为循环节点设计 exports 的助手。\n"
        "你可以调用工具 edit_loop_exports 直接给出 exports 对象。\n"
        "要求：items.from_node 必须引用 body_subgraph.nodes 中的节点（首选 exit），fields 需列出你希望暴露的字段。"
    )

    payload = {
        "nl_requirement": nl_requirement,
        "loop_node": loop_node,
        "loop_body": body_context,
        "hint": "使用工具输出完整 exports，对 aggregates 可选填 count_if/sum/avg 等表达式。",
    }

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    for round_idx in range(3):
        with child_span(f"loop_exports_tool_round_{round_idx}"):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=LOOP_EXPORT_EDIT_TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
        log_llm_usage(model, getattr(resp, "usage", None), operation="plan_loop_exports")
        if not resp.choices:
            raise RuntimeError("_plan_loop_exports_with_tools 未返回任何候选消息")

        msg = resp.choices[0].message
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": msg.tool_calls,
        })

        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_call_id = tc.id
                func_name = tc.function.name
                raw_args = tc.function.arguments
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    log_error(f"[Error] 解析 exports 工具参数失败: {raw_args}")
                    args = {}

                if func_name != "edit_loop_exports":
                    tool_result = {"status": "error", "message": f"未知工具 {func_name}"}
                else:
                    extracted = _extract_exports_from_tool_args(args)
                    if extracted:
                        validation_errors = _validate_loop_exports(
                            loop_node=loop_node, exports=extracted
                        )
                        if not validation_errors:
                            tool_result = {
                                "status": "ok",
                                "message": "已接收并通过校验的 exports",
                                "exports": extracted,
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": json.dumps(tool_result, ensure_ascii=False),
                                }
                            )
                            return extracted

                        tool_result = {
                            "status": "error",
                            "message": "exports 校验失败，请修正后重试",
                            "errors": validation_errors,
                        }
                    else:
                        tool_result = {"status": "error", "message": "未提供合法的 exports"}

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

            continue

        text = (msg.content or "").strip()
        if not text:
            continue
        decoder = json.JSONDecoder()
        try:
            parsed, _ = decoder.raw_decode(text)
        except json.JSONDecodeError:
            log_warn("[Planner] LLM 没有调用工具且返回内容无法解析为 JSON。")
            continue

        if isinstance(parsed, Mapping):
            exports = parsed.get("exports") if "exports" in parsed else parsed
            if isinstance(exports, Mapping):
                validation_errors = _validate_loop_exports(
                    loop_node=loop_node, exports=exports
                )
                if not validation_errors:
                    return dict(exports)
                log_warn(
                    "[Planner] LLM 文本返回的 exports 未通过校验，将继续尝试。"
                )

    return None


def _ensure_loop_exports_with_llm(
    *,
    workflow: Dict[str, Any],
    action_registry: List[Dict[str, Any]],
    nl_requirement: str,
    model: str,
) -> Dict[str, Any]:
    nodes = workflow.get("nodes", []) if isinstance(workflow, Mapping) else []
    loop_nodes = [n for n in nodes if isinstance(n, Mapping) and n.get("type") == "loop"]
    if not loop_nodes:
        return workflow

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    action_schemas = _build_action_schema_map(action_registry)

    new_nodes: List[Dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("type") != "loop":
            new_nodes.append(node)
            continue

        params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
        exports = params.get("exports") if isinstance(params, Mapping) else None
        if isinstance(exports, Mapping) and exports:
            ensured_exports = _ensure_loop_items_fields(
                exports=exports, loop_node=node, action_schemas=action_schemas
            )
            new_params = dict(params)
            new_params["exports"] = ensured_exports
            new_node = dict(node)
            new_node["params"] = new_params
            new_nodes.append(new_node)
            continue

        synthesized: Optional[Dict[str, Any]] = None
        planned = _plan_loop_exports_with_tools(
            client=client,
            model=model,
            nl_requirement=nl_requirement,
            loop_node=node,
            action_schemas=action_schemas,
        )
        used_tool = planned is not None
        if not planned:
            synthesized = _synthesize_loop_exports_with_llm(
                client=client,
                model=model,
                nl_requirement=nl_requirement,
                loop_node=node,
                action_schemas=action_schemas,
            )
            planned = synthesized

        if not planned:
            planned = _fallback_loop_exports(node, action_schemas) or {}
            log_warn(
                f"[Planner] LLM 未能生成 exports，loop 节点 {node.get('id')} 使用兜底 exports。"
            )
        elif used_tool:
            log_info(f"[Planner] LLM 工具已为 loop 节点 {node.get('id')} 编辑 exports。")
        elif synthesized is planned:
            log_info(f"[Planner] LLM 已为 loop 节点 {node.get('id')} 生成 exports。")

        ensured_exports = _ensure_loop_items_fields(
            exports=planned, loop_node=node, action_schemas=action_schemas
        )
        new_params = dict(params)
        new_params["exports"] = ensured_exports
        new_node = dict(node)
        new_node["params"] = new_params
        new_nodes.append(new_node)

    new_workflow = dict(workflow)
    new_workflow["nodes"] = new_nodes
    return new_workflow


def _prepare_skeleton_for_coverage(
    *,
    builder: WorkflowBuilder,
    action_registry: List[Dict[str, Any]],
    search_service: HybridActionSearchService,
) -> Dict[str, Any]:
    skeleton = _attach_inferred_edges(builder.to_workflow())
    skeleton = ensure_registered_actions(
        skeleton, action_registry=action_registry, search_service=search_service
    )
    return _attach_inferred_edges(skeleton)


def _run_coverage_check(
    *,
    nl_requirement: str,
    builder: WorkflowBuilder,
    action_registry: List[Dict[str, Any]],
    search_service: HybridActionSearchService,
) -> Dict[str, Any]:
    skeleton = _prepare_skeleton_for_coverage(
        builder=builder, action_registry=action_registry, search_service=search_service
    )

    coverage = check_requirement_coverage_with_llm(
        nl_requirement=nl_requirement,
        workflow=skeleton,
        model=OPENAI_MODEL,
    )
    approval_missing = detect_missing_approval_nodes(
        workflow=skeleton, action_registry=action_registry
    )
    if approval_missing:
        coverage.setdefault("missing_points", [])
        coverage["missing_points"].extend(approval_missing)
        coverage["is_covered"] = False

    log_event("coverage_check", {"coverage": coverage})
    log_json("覆盖度检查结果", coverage)
    return skeleton, coverage


def _build_coverage_feedback_message(
    *, coverage: Mapping[str, Any], workflow: Mapping[str, Any]
) -> str:
    missing_points = coverage.get("missing_points", []) or []
    analysis = coverage.get("analysis", "")
    return (
        "覆盖度检查未通过，请继续使用规划工具补充缺失点，并再次调用 finalize_workflow。\n"
        f"- missing_points: {json.dumps(missing_points, ensure_ascii=False)}\n"
        f"- analysis: {analysis}\n"
        "当前 workflow 供参考（含推导的 edges）：\n"
        f"{json.dumps(workflow, ensure_ascii=False)}"
    )


def plan_workflow_structure_with_llm(
    nl_requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    max_rounds: int = 10,
    max_coverage_refine_rounds: int = 2,
) -> Dict[str, Any]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    builder = WorkflowBuilder()
    last_action_candidates: List[str] = []

    system_prompt = (
        "你是一个通用业务工作流编排助手。\n"
        "系统中有一个 Action Registry，包含大量业务动作，你只能通过 search_business_actions 查询。\n"
        "构建方式：\n"
        "1) 使用 set_workflow_meta 设置工作流名称和描述。\n"
        "2) 当需要业务动作时，必须先用 search_business_actions 查询候选；add_node(type='action') 的 action_id 必须取自最近一次 candidates.id。\n"
        "3) 如需修改已创建节点（补充 display_name/params/分支指向等），请调用 update_node 并传入需要覆盖的字段列表。\n"
        "4) condition 节点必须显式提供 true_to_node 和 false_to_node，值可以是节点 id（继续执行）或 null（表示该分支结束）；通过节点 params 中的输入/输出引用表达依赖关系，不需要显式绘制 edges。\n"
        "5) 当结构完成时调用 finalize_workflow。\n\n"
        "【非常重要的原则】\n"
        "1. 所有示例（包括后续你在补参阶段看到的示例）都只是为说明“DSL 的写法”和“节点之间如何连线”，\n"
        "   不是实际的业务约束，不要在新任务里硬复用这些示例中的业务名或字段名。\n"
        "2. 你必须严格围绕当前对话中的自然语言需求来设计 workflow：\n"
        "   - 触发方式（定时 / 事件 / 手动）\n"
        "   - 数据查询/读取\n"
        "   - 筛选/过滤条件\n"
        "   - 聚合/统计/总结\n"
        "   - 通知 / 写入 / 落库 / 调用下游系统\n"
        "3. 不允许为了模仿示例，而在与当前任务无关的情况下引入“健康/体温/新闻/Nvidia/员工/HR”等具体词汇。\n\n"
        "4. 循环节点的内部数据只能通过 loop.exports 暴露给外部，下游引用循环结果时必须使用 result_of.<loop_id>.items（或 result_of.<loop_id>.exports.items）/ result_of.<loop_id>.aggregates.*，禁止直接引用 body 子图的节点。\n\n"
        "【覆盖度要求】\n"
        "你必须确保工作流结构能够完全覆盖用户自然语言需求中的每个子任务，而不是只覆盖前半部分：\n"
        "例如，如果需求包含：触发 + 查询 + 筛选 + 总结 + 通知，你不能只实现触发 + 查询，\n"
        "必须在结构里显式包含筛选、总结、通知等对应节点和数据流。\n"
        "调用 finalize_workflow 后系统会立即对照 nl_requirement 做覆盖度检查；如果发现 missing_points 会把缺失点和当前 workflow 反馈给你，请继续用规划工具修补后再次 finalize。"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": nl_requirement},
    ]

    finalized = False
    latest_skeleton: Dict[str, Any] = {}
    latest_coverage: Dict[str, Any] = {}
    coverage_retry = 0
    total_rounds = max_rounds + max_coverage_refine_rounds

    # ---------- 结构规划（多轮 tool-calling） ----------
    for round_idx in range(total_rounds):
        log_section(f"结构规划 Round {round_idx + 1}")
        with child_span("structure_planning_llm"):
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=messages,
                tools=PLANNER_TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
        log_llm_usage(OPENAI_MODEL, getattr(resp, "usage", None), operation="structure_planning")
        if not resp.choices:
            raise RuntimeError("plan_workflow_structure_with_llm 未返回任何候选消息")

        msg = resp.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": msg.tool_calls,
            }
        )

        if not msg.tool_calls:
            log_warn("[Planner] 本轮没有 tool_calls，提前结束。")
            break

        for tc in msg.tool_calls:
            func_name = tc.function.name
            raw_args = tc.function.arguments
            tool_call_id = tc.id
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                log_error(f"[Error] 解析工具参数失败: {raw_args}")
                args = {}

            log_info(f"[Planner] 调用工具: {func_name}({args})")

            if func_name == "search_business_actions":
                query = args.get("query", "")
                top_k = int(args.get("top_k", 5))
                actions_raw = search_service.search(query=query, top_k=top_k)
                candidates = [
                    {
                        "id": a.get("action_id"),
                        "name": a.get("name", ""),
                        "description": a.get("description", ""),
                        "category": a.get("domain") or "general",
                    }
                    for a in actions_raw
                    if a.get("action_id")
                ]
                last_action_candidates = [c["id"] for c in candidates]
                tool_result = {
                    "status": "ok",
                    "query": query,
                    "actions": actions_raw,
                    "candidates": candidates,
                }

            elif func_name == "set_workflow_meta":
                builder.set_meta(args.get("workflow_name", ""), args.get("description"))
                tool_result = {"status": "ok", "type": "meta_set"}

            elif func_name == "add_node":
                node_type = args["type"]
                action_id = args.get("action_id")
                true_to_node = args.get("true_to_node")
                false_to_node = args.get("false_to_node")

                if node_type == "condition":
                    missing_fields = [
                        name
                        for name in ("true_to_node", "false_to_node")
                        if name not in args
                    ]
                    non_str_fields = [
                        name
                        for name, value in (
                            ("true_to_node", true_to_node),
                            ("false_to_node", false_to_node),
                        )
                        if value is not None and not isinstance(value, str)
                    ]

                    if missing_fields or non_str_fields:
                        tool_result = {
                            "status": "error",
                            "message": (
                                "condition 节点需要提供 true_to_node/false_to_node 字段，值可为节点 id（继续执行）"
                                "或 null（表示该分支结束），非字符串/未提供会被拒绝。"
                            ),
                            "missing_fields": missing_fields,
                            "invalid_fields": non_str_fields,
                        }
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": json.dumps(tool_result, ensure_ascii=False),
                            }
                        )
                        continue

                if node_type == "action":
                    if not last_action_candidates:
                        tool_result = {
                            "status": "error",
                            "message": "action 节点必须在调用 search_business_actions 之后创建，请先查询候选动作。",
                        }
                    elif action_id not in last_action_candidates:
                        tool_result = {
                            "status": "error",
                            "message": "action_id 必须是最近一次 search_business_actions 返回的 candidates.id 之一。",
                            "allowed_action_ids": last_action_candidates,
                        }
                    else:
                        builder.add_node(
                            node_id=args["id"],
                            node_type=node_type,
                            action_id=action_id,
                            display_name=args.get("display_name"),
                            params=args.get("params") or {},
                            true_to_node=true_to_node if isinstance(true_to_node, str) else None,
                            false_to_node=false_to_node if isinstance(false_to_node, str) else None,
                        )
                        tool_result = {"status": "ok", "type": "node_added", "node_id": args["id"]}
                else:
                    builder.add_node(
                        node_id=args["id"],
                        node_type=node_type,
                        action_id=action_id,
                        display_name=args.get("display_name"),
                        params=args.get("params") or {},
                        true_to_node=true_to_node if isinstance(true_to_node, str) else None,
                        false_to_node=false_to_node if isinstance(false_to_node, str) else None,
                    )
                    tool_result = {"status": "ok", "type": "node_added", "node_id": args["id"]}

            elif func_name == "update_node":
                node_id = args.get("id")
                updates = args.get("updates")

                if not isinstance(node_id, str):
                    tool_result = {"status": "error", "message": "update_node 需要提供字符串类型的 id。"}
                elif node_id not in builder.nodes:
                    tool_result = {"status": "error", "message": f"节点 {node_id} 尚未创建，无法更新。"}
                elif not isinstance(updates, list):
                    tool_result = {
                        "status": "error",
                        "message": "updates 必须是 {op,key,value} 对象组成的数组。",
                    }
                else:
                    invalid_entries = []
                    invalid_branch_fields = []
                    invalid_ops = []
                    normalized_updates = []
                    for idx, entry in enumerate(updates):
                        if not isinstance(entry, Mapping):
                            invalid_entries.append(idx)
                            continue

                        op = entry.get("op", "modify")
                        if op not in {"add", "modify", "remove"}:
                            invalid_ops.append(idx)
                            continue

                        key = entry.get("key")
                        if not isinstance(key, str):
                            invalid_entries.append(idx)
                            continue

                        value = entry.get("value") if "value" in entry else None
                        if (
                            op != "remove"
                            and key in {"true_to_node", "false_to_node"}
                            and value is not None
                            and not isinstance(value, str)
                        ):
                            invalid_branch_fields.append(key)
                            continue

                        normalized_updates.append({"op": op, "key": key, "value": value})

                    node_type = builder.nodes.get(node_id, {}).get("type")
                    if invalid_entries:
                        tool_result = {
                            "status": "error",
                            "message": f"updates[{invalid_entries}] 不是合法的 {{op,key,value}} 对象。",
                        }
                    elif invalid_ops:
                        tool_result = {
                            "status": "error",
                            "message": f"updates[{invalid_ops}] 包含不支持的 op（仅支持 add/modify/remove）。",
                        }
                    elif invalid_branch_fields:
                        tool_result = {
                            "status": "error",
                            "message": "condition 的 true_to_node/false_to_node 只能是节点 id 或 null。",
                            "invalid_fields": invalid_branch_fields,
                        }
                    elif node_type == "action" and any(
                        entry.get("op", "modify") != "remove" and entry.get("key") == "action_id"
                        for entry in normalized_updates
                    ):
                        new_action_id = next(
                            entry.get("value")
                            for entry in normalized_updates
                            if entry.get("op", "modify") != "remove" and entry.get("key") == "action_id"
                        )
                        if not last_action_candidates:
                            tool_result = {
                                "status": "error",
                                "message": "更新 action_id 前请先调用 search_business_actions 以获取候选。",
                            }
                        elif new_action_id not in last_action_candidates:
                            tool_result = {
                                "status": "error",
                                "message": "action_id 必须是最近一次 search_business_actions 返回的 candidates.id 之一。",
                                "allowed_action_ids": last_action_candidates,
                            }
                        else:
                            builder.update_node(node_id, normalized_updates)
                            tool_result = {"status": "ok", "type": "node_updated", "node_id": node_id}
                    else:
                        builder.update_node(node_id, normalized_updates)
                        tool_result = {"status": "ok", "type": "node_updated", "node_id": node_id}

            elif func_name == "finalize_workflow":
                skeleton, coverage = _run_coverage_check(
                    nl_requirement=nl_requirement,
                    builder=builder,
                    action_registry=action_registry,
                    search_service=search_service,
                )
                latest_skeleton = skeleton
                latest_coverage = coverage
                is_covered = bool(coverage.get("is_covered", False))
                tool_result = {
                    "status": "ok" if is_covered else "needs_more_coverage",
                    "type": "finalized",
                    "notes": args.get("notes"),
                    "coverage": coverage,
                }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

                if is_covered:
                    finalized = True
                    log_success("[Planner] 覆盖度检查通过，结束结构规划。")
                else:
                    coverage_retry += 1
                    log_info("🔧 覆盖度检查未通过，将继续使用规划工具完善。")
                    feedback_message = _build_coverage_feedback_message(
                        coverage=coverage, workflow=skeleton
                    )
                    messages.append({"role": "system", "content": feedback_message})
                    if coverage_retry > max_coverage_refine_rounds:
                        log_warn("已达到覆盖度补全上限，仍有缺失点，结束规划阶段。")
                        finalized = True

                continue

            else:
                tool_result = {"status": "error", "message": f"未知工具 {func_name}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(tool_result, ensure_ascii=False),
                }
            )

        if finalized:
            break

    if not finalized:
        if latest_coverage and not latest_coverage.get("is_covered", False):
            log_warn("[Planner] 规划回合结束但覆盖度仍未通过，使用当前骨架继续后续阶段。")
        else:
            log_warn("[Planner] 未收到 finalize_workflow，使用当前骨架继续后续阶段。")

    if not finalized or not latest_skeleton:
        latest_skeleton = _prepare_skeleton_for_coverage(
            builder=builder, action_registry=action_registry, search_service=search_service
        )

    skeleton = _ensure_loop_exports_with_llm(
        workflow=latest_skeleton,
        action_registry=action_registry,
        nl_requirement=nl_requirement,
        model=OPENAI_MODEL,
    )

    return skeleton


__all__ = ["plan_workflow_structure_with_llm"]
