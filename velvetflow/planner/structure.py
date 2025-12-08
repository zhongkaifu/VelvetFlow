# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Structure planning logic for the planner."""

import copy
import json
import os
from typing import Any, Dict, List, Mapping, Optional

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    child_span,
    log_info,
    log_error,
    log_event,
    log_json,
    log_llm_message,
    log_llm_usage,
    log_section,
    log_success,
    log_warn,
    log_tool_call,
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


CONDITION_ALLOWED_KINDS = {
    "list_not_empty",
    "any_greater_than",
    "equals",
    "contains",
    "not_equals",
    "greater_than",
    "less_than",
    "between",
    "all_less_than",
    "is_empty",
    "not_empty",
    "is_not_empty",
    "multi_band",
    "compare",
}

CONDITION_PARAM_FIELDS = {
    "kind",
    "source",
    "field",
    "value",
    "threshold",
    "min",
    "max",
    "bands",
}

SWITCH_PARAM_FIELDS = {
    "source",
    "field",
}

LOOP_PARAM_FIELDS = {
    "loop_kind",
    "source",
    "condition",
    "item_alias",
    "body_subgraph",
    "exports",
}

ACTION_NODE_FIELDS = {
    "id",
    "type",
    "action_id",
    "display_name",
    "params",
    "out_params_schema",
    "parent_node_id",
}

CONDITION_NODE_FIELDS = {
    "id",
    "type",
    "display_name",
    "params",
    "true_to_node",
    "false_to_node",
    "parent_node_id",
}

SWITCH_NODE_FIELDS = {
    "id",
    "type",
    "display_name",
    "params",
    "cases",
    "default_to_node",
    "parent_node_id",
}

LOOP_NODE_FIELDS = {
    "id",
    "type",
    "display_name",
    "params",
    "parent_node_id",
}

def _attach_inferred_edges(workflow: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild derived edges so LLMs can see the implicit wiring."""

    copied = copy.deepcopy(workflow)
    nodes = copied.get("nodes") if isinstance(copied.get("nodes"), list) else []
    copied["edges"] = infer_edges_from_bindings(nodes)
    return attach_condition_branches(copied)


def _normalize_sub_graph_nodes(
    raw: Any, *, builder: WorkflowBuilder
) -> tuple[List[str], Optional[Dict[str, Any]]]:
    """Validate and normalize a list of node ids to attach to a loop body."""

    if raw is None:
        return [], None

    if not isinstance(raw, list):
        return [], {"message": "sub_graph_nodes 需要是节点 id 的数组。"}

    non_str_indices = [idx for idx, value in enumerate(raw) if not isinstance(value, str)]
    normalized = [value for value in raw if isinstance(value, str)]
    missing_nodes = [nid for nid in normalized if nid not in builder.nodes]

    if non_str_indices or missing_nodes:
        return [], {
            "message": "sub_graph_nodes 应为已创建节点的 id 字符串列表。",
            "invalid_indices": non_str_indices,
            "missing_nodes": missing_nodes,
        }

    return normalized, None


def _attach_sub_graph_nodes(builder: WorkflowBuilder, loop_id: str, node_ids: List[str]):
    """Mark the given nodes as belonging to the loop's body_subgraph."""

    for nid in node_ids:
        node = builder.nodes.get(nid)
        if isinstance(node, dict):
            node["parent_node_id"] = loop_id


def _filter_supported_params(
    *,
    node_type: str,
    params: Any,
    action_schemas: Mapping[str, Mapping[str, Any]],
    action_id: Optional[str] = None,
) -> tuple[Dict[str, Any], List[str]]:
    """Keep only supported param fields for the given node type.

    Returns the sanitized params dict and a list of removed field names.
    """

    if not isinstance(params, Mapping):
        return {}, []

    allowed_fields: Optional[set[str]] = None
    if node_type == "condition":
        allowed_fields = set(CONDITION_PARAM_FIELDS)
    elif node_type == "switch":
        allowed_fields = set(SWITCH_PARAM_FIELDS)
    elif node_type == "loop":
        allowed_fields = set(LOOP_PARAM_FIELDS)
    elif node_type == "action" and action_id:
        schema = action_schemas.get(action_id, {}) if isinstance(action_id, str) else {}
        properties = schema.get("arg_schema", {}).get("properties") if isinstance(schema.get("arg_schema"), Mapping) else None
        if isinstance(properties, Mapping):
            allowed_fields = set(properties.keys())

    if not allowed_fields:
        return dict(params), []

    cleaned: Dict[str, Any] = {k: v for k, v in params.items() if k in allowed_fields}
    removed = [k for k in params if k not in allowed_fields]

    if node_type == "loop" and "exports" in cleaned and isinstance(cleaned["exports"], Mapping):
        cleaned_exports: Dict[str, Any] = {}
        removed_exports: List[str] = []
        for key, value in cleaned["exports"].items():
            if key in {"items", "aggregates"}:
                cleaned_exports[key] = value
            else:
                removed_exports.append(key)
        cleaned["exports"] = cleaned_exports
        removed.extend([f"exports.{key}" for key in removed_exports])

    return cleaned, removed


def _sanitize_builder_node_params(
    builder: WorkflowBuilder, node_id: str, action_schemas: Mapping[str, Mapping[str, Any]]
) -> List[str]:
    node = builder.nodes.get(node_id)
    if not isinstance(node, Mapping):
        return []

    params = node.get("params") or {}
    cleaned, removed = _filter_supported_params(
        node_type=str(node.get("type")),
        params=params,
        action_schemas=action_schemas,
        action_id=node.get("action_id") if isinstance(node.get("action_id"), str) else None,
    )

    if removed:
        node["params"] = cleaned

    return removed


def _sanitize_builder_node_fields(builder: WorkflowBuilder, node_id: str) -> List[str]:
    node = builder.nodes.get(node_id)
    if not isinstance(node, Mapping):
        return []

    node_type = node.get("type")
    allowed_fields: Optional[set[str]] = None
    if node_type == "action":
        allowed_fields = set(ACTION_NODE_FIELDS)
    elif node_type == "condition":
        allowed_fields = set(CONDITION_NODE_FIELDS)
    elif node_type == "switch":
        allowed_fields = set(SWITCH_NODE_FIELDS)
    elif node_type == "loop":
        allowed_fields = set(LOOP_NODE_FIELDS)

    if not allowed_fields:
        return []

    removed_keys = [key for key in list(node.keys()) if key not in allowed_fields]
    for key in removed_keys:
        node.pop(key, None)

    return removed_keys


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
    body = params.get("body_subgraph") if isinstance(params, Mapping) else None
    if not isinstance(body, Mapping):
        body = {}

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


def _find_nodes_without_upstream(workflow: Mapping[str, Any]) -> List[Dict[str, Any]]:
    nodes = workflow.get("nodes") if isinstance(workflow.get("nodes"), list) else []
    inferred_edges = infer_edges_from_bindings(nodes)

    indegree = {}
    for node in nodes:
        if isinstance(node, Mapping) and isinstance(node.get("id"), str):
            indegree[node["id"]] = 0

    for edge in inferred_edges:
        if not isinstance(edge, Mapping):
            continue
        target = edge.get("to")
        if isinstance(target, str) and target in indegree:
            indegree[target] += 1

    dangling: List[Dict[str, Any]] = []
    for node in nodes:
        if not isinstance(node, Mapping):
            continue

        node_id = node.get("id")
        node_type = node.get("type")
        if not isinstance(node_id, str) or not isinstance(node_type, str):
            continue

        if node_type in {"start", "end", "exit"}:
            continue

        if indegree.get(node_id, 0) == 0:
            dangling.append(
                {
                    "id": node_id,
                    "type": node_type,
                    "action_id": node.get("action_id")
                    if isinstance(node.get("action_id"), str)
                    else None,
                    "display_name": node.get("display_name"),
                }
            )

    return dangling


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


def _build_dependency_feedback_message(
    *, workflow: Mapping[str, Any], nodes_without_upstream: List[Mapping[str, Any]]
) -> str:
    return (
        "检测到以下节点没有任何上游依赖（不包含 start/end/exit），"
        "请检查是否遗漏了对相关节点结果的引用或绑定。如果需要，请继续使用规划工具补充；"
        "如果确认这些节点应该独立存在，请在 finalize_workflow.notes 中简单说明原因。\n"
        f"- nodes_without_upstream: {json.dumps(nodes_without_upstream, ensure_ascii=False)}\n"
        "当前 workflow 供参考（含推导的 edges）：\n"
        f"{json.dumps(workflow, ensure_ascii=False)}"
    )
def plan_workflow_structure_with_llm(
    nl_requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    max_rounds: int = 10,
    max_coverage_refine_rounds: int = 2,
    max_dependency_refine_rounds: int = 1,
) -> Dict[str, Any]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    builder = WorkflowBuilder()
    action_schemas = _build_action_schema_map(action_registry)
    last_action_candidates: List[str] = []

    system_prompt = (
        "你是一个通用业务工作流编排助手。\n"
        "系统中有一个 Action Registry，包含大量业务动作，你只能通过 search_business_actions 查询。\n"
        "构建方式：\n"
        "1) 使用 set_workflow_meta 设置工作流名称和描述。\n"
        "2) 当需要业务动作时，必须先用 search_business_actions 查询候选；add_action_node 的 action_id 必须取自最近一次 candidates.id。\n"
        "3) 如需修改已创建节点（补充 display_name/params/分支指向/父节点等），请调用 update_action_node 或 update_condition_node 并传入需要覆盖的字段列表；调用后务必检查上下游关联节点是否也需要同步更新以保持一致性。\n"
        "4) condition 节点必须显式提供 true_to_node 和 false_to_node，值可以是节点 id（继续执行）或 null（表示该分支结束）；通过节点 params 中的输入/输出引用表达依赖关系，不需要显式绘制 edges。\n"
        "5) 当结构完成时调用 finalize_workflow。\n\n"
        "特别注意：只有 action 节点需要 out_params_schema，condition 节点没有该属性；out_params_schema 的格式应为 {\"参数名\": \"类型\"}，仅需列出业务 action 输出参数的名称与类型，不要添加额外描述或示例。\n\n"
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
        "4. 循环节点的内部数据只能通过 loop.exports 暴露给外部，下游引用循环结果时必须使用 result_of.<loop_id>.items（或 result_of.<loop_id>.exports.items）/ result_of.<loop_id>.aggregates.*，禁止直接引用 body 子图的节点。\n"
        "5. loop.exports 应定义在 params.exports 下，请勿写在 body_subgraph 内。\n"
        "6. 允许嵌套循环，但需要通过 parent_node_id 或 sub_graph_nodes 明确将子循环纳入父循环的 body_subgraph；"
        "   外部节点引用循环内部数据时仍需通过 loop.exports，而不是直接指向子图节点。\n\n"
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
    dependency_retry = 0
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
        log_llm_message(OPENAI_MODEL, msg, operation="structure_planning")
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

            log_tool_call(
                source="plan_workflow_structure",
                tool_name=func_name,
                tool_call_id=tool_call_id,
                args=args or raw_args,
            )

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

            elif func_name == "add_action_node":
                action_id = args.get("action_id")
                parent_node_id = args.get("parent_node_id")

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
                elif parent_node_id is not None and not isinstance(parent_node_id, str):
                    tool_result = {
                        "status": "error",
                        "message": "parent_node_id 需要是字符串或 null。",
                    }
                else:
                    original_params = args.get("params") or {}
                    cleaned_params, removed_fields = _filter_supported_params(
                        node_type="action",
                        params=original_params,
                        action_schemas=action_schemas,
                        action_id=action_id,
                    )
                    builder.add_node(
                        node_id=args["id"],
                        node_type="action",
                        action_id=action_id,
                        display_name=args.get("display_name"),
                        out_params_schema=args.get("out_params_schema"),
                        params=cleaned_params,
                        parent_node_id=parent_node_id if isinstance(parent_node_id, str) else None,
                    )
                    removed_node_fields = _sanitize_builder_node_fields(builder, args["id"])
                    if removed_fields or removed_node_fields:
                        tool_result = {
                            "status": "error",
                            "message": "action 节点仅支持 id/type/action_id/display_name/params/out_params_schema 字段，params 仅支持 arg_schema 字段，已移除不支持的字段。",
                            "removed_param_fields": removed_fields,
                            "removed_node_fields": removed_node_fields,
                            "node_id": args["id"],
                        }
                    else:
                        tool_result = {"status": "ok", "type": "node_added", "node_id": args["id"]}

            elif func_name == "add_loop_node":
                parent_node_id = args.get("parent_node_id")
                loop_kind = args.get("loop_kind")
                source = args.get("source")
                item_alias = args.get("item_alias")
                sub_graph_nodes, sub_graph_error = _normalize_sub_graph_nodes(
                    args.get("sub_graph_nodes"), builder=builder
                )

                missing_fields = [
                    name
                    for name, value in (
                        ("loop_kind", loop_kind),
                        ("source", source),
                        ("item_alias", item_alias),
                    )
                    if value is None
                ]
                invalid_fields = []
                if loop_kind is not None and loop_kind not in {"for_each", "while"}:
                    invalid_fields.append("loop_kind")
                if source is not None and not isinstance(source, (str, Mapping)):
                    invalid_fields.append("source")
                if item_alias is not None and not isinstance(item_alias, str):
                    invalid_fields.append("item_alias")

                if parent_node_id is not None and not isinstance(parent_node_id, str):
                    tool_result = {
                        "status": "error",
                        "message": "parent_node_id 需要是字符串或 null。",
                    }
                elif missing_fields or invalid_fields:
                    tool_result = {
                        "status": "error",
                        "message": "loop 节点需要提供合法的 loop_kind/source/item_alias 参数。",
                        "missing_fields": missing_fields,
                        "invalid_fields": invalid_fields,
                    }
                elif sub_graph_error:
                    tool_result = {"status": "error", **sub_graph_error}
                else:
                    params = args.get("params") or {}
                    params.update({
                        "loop_kind": loop_kind,
                        "source": source,
                        "item_alias": item_alias,
                    })
                    cleaned_params, removed_fields = _filter_supported_params(
                        node_type="loop",
                        params=params,
                        action_schemas=action_schemas,
                    )
                    builder.add_node(
                        node_id=args["id"],
                        node_type="loop",
                        action_id=None,
                        display_name=args.get("display_name"),
                        out_params_schema=None,
                        params=cleaned_params,
                        parent_node_id=parent_node_id if isinstance(parent_node_id, str) else None,
                    )
                    _attach_sub_graph_nodes(builder, args["id"], sub_graph_nodes)
                    removed_node_fields = _sanitize_builder_node_fields(builder, args["id"])
                    if removed_fields or removed_node_fields:
                        tool_result = {
                            "status": "error",
                            "message": "loop 节点的 params 仅支持 loop_kind/source/condition/item_alias/body_subgraph/exports，且 exports 只能包含 items/aggregates。",
                            "removed_fields": removed_fields,
                            "removed_node_fields": removed_node_fields,
                            "node_id": args["id"],
                        }
                    else:
                        tool_result = {"status": "ok", "type": "node_added", "node_id": args["id"]}

            elif func_name == "add_condition_node":
                true_to_node = args.get("true_to_node")
                false_to_node = args.get("false_to_node")
                parent_node_id = args.get("parent_node_id")
                condition_kind = args.get("kind")
                params = args.get("params")

                missing_fields = [
                    name
                    for name in ("kind", "true_to_node", "false_to_node")
                    if name not in args
                ]
                non_str_fields = [
                    name
                    for name, value in (
                        ("kind", condition_kind),
                        ("true_to_node", true_to_node),
                        ("false_to_node", false_to_node),
                    )
                    if value is not None and not isinstance(value, str)
                ]

                params_error: Optional[Dict[str, Any]] = None
                normalized_params: Dict[str, Any] = {}
                if isinstance(params, Mapping):
                    normalized_params = dict(params)
                    params_kind = params.get("kind")
                    if params_kind is not None and params_kind != condition_kind:
                        params_error = {
                            "status": "error",
                            "message": "params.kind 与 kind 参数不一致，请确保两者相同。",
                            "params_kind": params_kind,
                            "kind": condition_kind,
                        }
                elif params is not None:
                    params_error = {
                        "status": "error",
                        "message": "condition 节点的 params 需要是对象。",
                    }

                if missing_fields or non_str_fields:
                    tool_result = {
                        "status": "error",
                        "message": (
                            "condition 节点需要提供 kind 以及 true_to_node/false_to_node 字段，分支跳转可为节点 id（继续执行）"
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

                if not isinstance(condition_kind, str) or condition_kind not in CONDITION_ALLOWED_KINDS:
                    tool_result = {
                        "status": "error",
                        "message": "condition 节点需要提供合法的 kind。",
                        "invalid_fields": ["kind"],
                        "allowed_kinds": sorted(CONDITION_ALLOWED_KINDS),
                    }
                elif params_error:
                    tool_result = params_error
                elif parent_node_id is not None and not isinstance(parent_node_id, str):
                    tool_result = {
                        "status": "error",
                        "message": "parent_node_id 需要是字符串或 null。",
                    }
                else:
                    normalized_params["kind"] = condition_kind
                    cleaned_params, removed_fields = _filter_supported_params(
                        node_type="condition",
                        params=normalized_params,
                        action_schemas=action_schemas,
                    )
                    builder.add_node(
                        node_id=args["id"],
                        node_type="condition",
                        action_id=None,
                        display_name=args.get("display_name"),
                        out_params_schema=None,
                        params=cleaned_params,
                        true_to_node=true_to_node if isinstance(true_to_node, str) else None,
                        false_to_node=false_to_node if isinstance(false_to_node, str) else None,
                        parent_node_id=parent_node_id if isinstance(parent_node_id, str) else None,
                    )
                    removed_node_fields = _sanitize_builder_node_fields(builder, args["id"])
                    if removed_fields or removed_node_fields:
                        tool_result = {
                            "status": "error",
                            "message": "condition 节点的 params 仅支持 kind/source/field/value/threshold/min/max/bands。",
                            "removed_fields": removed_fields,
                            "removed_node_fields": removed_node_fields,
                            "node_id": args["id"],
                        }
                    else:
                        tool_result = {"status": "ok", "type": "node_added", "node_id": args["id"]}

            elif func_name == "add_switch_node":
                parent_node_id = args.get("parent_node_id")
                cases = args.get("cases")
                params = args.get("params") or {}
                default_to_node = args.get("default_to_node")

                if not isinstance(cases, list):
                    tool_result = {
                        "status": "error",
                        "message": "switch 节点需要提供 cases 数组。",
                    }
                elif parent_node_id is not None and not isinstance(parent_node_id, str):
                    tool_result = {
                        "status": "error",
                        "message": "parent_node_id 需要是字符串或 null。",
                    }
                else:
                    normalized_cases: list[Dict[str, Any]] = []
                    invalid_case_indices: list[int] = []
                    for idx, case in enumerate(cases):
                        if not isinstance(case, Mapping):
                            invalid_case_indices.append(idx)
                            continue
                        to_node = case.get("to_node") if "to_node" in case else None
                        if to_node is not None and not isinstance(to_node, str):
                            invalid_case_indices.append(idx)
                            continue
                        normalized_cases.append(dict(case))

                    if invalid_case_indices:
                        tool_result = {
                            "status": "error",
                            "message": "cases 中的 to_node 需要是字符串或 null。",
                            "invalid_case_indices": invalid_case_indices,
                        }
                    elif default_to_node is not None and not isinstance(default_to_node, str):
                        tool_result = {
                            "status": "error",
                            "message": "default_to_node 需要是字符串或 null。",
                        }
                    elif params is not None and not isinstance(params, Mapping):
                        tool_result = {
                            "status": "error",
                            "message": "switch 节点的 params 需要是对象。",
                        }
                    else:
                        cleaned_params, removed_fields = _filter_supported_params(
                            node_type="switch",
                            params=params,
                            action_schemas=action_schemas,
                        )
                        builder.add_node(
                            node_id=args["id"],
                            node_type="switch",
                            action_id=None,
                            display_name=args.get("display_name"),
                            params=cleaned_params,
                            cases=normalized_cases,
                            default_to_node=default_to_node if isinstance(default_to_node, str) else None,
                            parent_node_id=parent_node_id if isinstance(parent_node_id, str) else None,
                        )
                        removed_node_fields = _sanitize_builder_node_fields(builder, args["id"])
                        if removed_fields or removed_node_fields:
                            tool_result = {
                                "status": "error",
                                "message": "switch 节点的 params 仅支持 source/field 字段，其他字段已移除。",
                                "removed_fields": removed_fields,
                                "removed_node_fields": removed_node_fields,
                                "node_id": args["id"],
                            }
                        else:
                            tool_result = {"status": "ok", "type": "node_added", "node_id": args["id"]}

            elif func_name in {
                "update_action_node",
                "update_condition_node",
                "update_loop_node",
                "update_switch_node",
            }:
                node_id = args.get("id")
                parent_node_id = args.get("parent_node_id")
                sub_graph_nodes, sub_graph_error = _normalize_sub_graph_nodes(
                    args.get("sub_graph_nodes"), builder=builder
                )
                expected_type = {
                    "update_action_node": "action",
                    "update_condition_node": "condition",
                    "update_loop_node": "loop",
                    "update_switch_node": "switch",
                }[func_name]

                if not isinstance(node_id, str):
                    tool_result = {"status": "error", "message": f"{func_name} 需要提供字符串类型的 id。"}
                elif node_id not in builder.nodes:
                    tool_result = {"status": "error", "message": f"节点 {node_id} 尚未创建，无法更新。"}
                elif builder.nodes.get(node_id, {}).get("type") != expected_type:
                    tool_result = {
                        "status": "error",
                        "message": f"节点 {node_id} 类型不是 {expected_type}，无法使用 {func_name}。",
                    }
                elif parent_node_id is not None and not isinstance(parent_node_id, str):
                    tool_result = {
                        "status": "error",
                        "message": "parent_node_id 需要是字符串或 null。",
                    }
                elif sub_graph_error:
                    tool_result = {"status": "error", **sub_graph_error}
                else:
                    removed_param_fields: List[str] = []
                    removed_node_fields: List[str] = []

                    if func_name == "update_action_node":
                        new_action_id = args.get("action_id") if "action_id" in args else None
                        new_params = args.get("params") if "params" in args else None
                        if "params" in args and not isinstance(new_params, Mapping):
                            tool_result = {
                                "status": "error",
                                "message": "action 节点的 params 需要是对象。",
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": json.dumps(tool_result, ensure_ascii=False),
                                }
                            )
                            continue
                        if "action_id" in args:
                            if not last_action_candidates:
                                tool_result = {
                                    "status": "error",
                                    "message": "更新 action_id 前请先调用 search_business_actions 以获取候选。",
                                }
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": json.dumps(tool_result, ensure_ascii=False),
                                    }
                                )
                                continue
                            if new_action_id not in last_action_candidates:
                                tool_result = {
                                    "status": "error",
                                    "message": "action_id 必须是最近一次 search_business_actions 返回的 candidates.id 之一。",
                                    "allowed_action_ids": last_action_candidates,
                                }
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": json.dumps(tool_result, ensure_ascii=False),
                                    }
                                )
                                continue
                        updates: Dict[str, Any] = {}
                        if "display_name" in args:
                            updates["display_name"] = args.get("display_name")
                        if "out_params_schema" in args:
                            updates["out_params_schema"] = args.get("out_params_schema")
                        if "action_id" in args:
                            updates["action_id"] = new_action_id
                        if "parent_node_id" in args:
                            updates["parent_node_id"] = parent_node_id
                        if "params" in args:
                            cleaned_params, removed_param_fields = _filter_supported_params(
                                node_type="action",
                                params=new_params or {},
                                action_schemas=action_schemas,
                                action_id=new_action_id
                                if isinstance(new_action_id, str)
                                else builder.nodes.get(node_id, {}).get("action_id"),
                            )
                            updates["params"] = cleaned_params
                        builder.update_node(node_id, **updates)
                        removed_param_fields.extend(
                            _sanitize_builder_node_params(builder, node_id, action_schemas)
                        )
                        removed_node_fields = _sanitize_builder_node_fields(builder, node_id)
                        if removed_param_fields or removed_node_fields:
                            tool_result = {
                                "status": "error",
                                "message": "action 节点仅支持 id/type/action_id/display_name/params/out_params_schema 字段，params 仅支持 arg_schema 字段，已移除不支持的字段。",
                                "removed_param_fields": removed_param_fields,
                                "removed_node_fields": removed_node_fields,
                                "node_id": node_id,
                            }
                        else:
                            tool_result = {"status": "ok", "type": "node_updated", "node_id": node_id}
                    elif func_name == "update_condition_node":
                        new_params = args.get("params") if "params" in args else None
                        if "params" in args and not isinstance(new_params, Mapping):
                            tool_result = {
                                "status": "error",
                                "message": "condition 节点的 params 需要是对象。",
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": json.dumps(tool_result, ensure_ascii=False),
                                }
                            )
                            continue
                        if "true_to_node" in args and args.get("true_to_node") is not None and not isinstance(args.get("true_to_node"), str):
                            tool_result = {
                                "status": "error",
                                "message": "condition 的 true_to_node 只能是节点 id 或 null。",
                                "invalid_fields": ["true_to_node"],
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": json.dumps(tool_result, ensure_ascii=False),
                                }
                            )
                            continue
                        if "false_to_node" in args and args.get("false_to_node") is not None and not isinstance(args.get("false_to_node"), str):
                            tool_result = {
                                "status": "error",
                                "message": "condition 的 false_to_node 只能是节点 id 或 null。",
                                "invalid_fields": ["false_to_node"],
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": json.dumps(tool_result, ensure_ascii=False),
                                }
                            )
                            continue

                        updates: Dict[str, Any] = {}
                        if "display_name" in args:
                            updates["display_name"] = args.get("display_name")
                        if "true_to_node" in args:
                            updates["true_to_node"] = args.get("true_to_node")
                        if "false_to_node" in args:
                            updates["false_to_node"] = args.get("false_to_node")
                        if "parent_node_id" in args:
                            updates["parent_node_id"] = parent_node_id
                        if "params" in args:
                            normalized_params = dict(new_params or {})
                            existing_kind = builder.nodes.get(node_id, {}).get("params", {}).get("kind")
                            params_kind = normalized_params.get("kind", existing_kind)
                            if params_kind and params_kind not in CONDITION_ALLOWED_KINDS:
                                tool_result = {
                                    "status": "error",
                                    "message": "condition 节点需要提供合法的 kind。",
                                    "invalid_fields": ["kind"],
                                    "allowed_kinds": sorted(CONDITION_ALLOWED_KINDS),
                                }
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": json.dumps(tool_result, ensure_ascii=False),
                                    }
                                )
                                continue
                            cleaned_params, removed_param_fields = _filter_supported_params(
                                node_type="condition",
                                params={**normalized_params, "kind": params_kind} if params_kind else normalized_params,
                                action_schemas=action_schemas,
                            )
                            updates["params"] = cleaned_params
                        builder.update_node(node_id, **updates)
                        removed_param_fields.extend(
                            _sanitize_builder_node_params(builder, node_id, action_schemas)
                        )
                        removed_node_fields = _sanitize_builder_node_fields(builder, node_id)
                        if removed_param_fields or removed_node_fields:
                            tool_result = {
                                "status": "error",
                                "message": "condition 节点仅支持 id/type/display_name/params/true_to_node/false_to_node 字段，params 仅支持 kind/source/field/value/threshold/min/max/bands，已移除不支持的字段。",
                                "removed_fields": removed_param_fields,
                                "removed_node_fields": removed_node_fields,
                                "node_id": node_id,
                            }
                        else:
                            tool_result = {"status": "ok", "type": "node_updated", "node_id": node_id}
                    elif func_name == "update_switch_node":
                        new_params = args.get("params") if "params" in args else None
                        if "params" in args and not isinstance(new_params, Mapping):
                            tool_result = {
                                "status": "error",
                                "message": "switch 节点的 params 需要是对象。",
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": json.dumps(tool_result, ensure_ascii=False),
                                }
                            )
                            continue
                        invalid_case_indices: list[int] = []
                        normalized_cases: list[Dict[str, Any]] = []
                        if "cases" in args:
                            cases = args.get("cases")
                            if not isinstance(cases, list):
                                tool_result = {
                                    "status": "error",
                                    "message": "switch 的 cases 需要是数组。",
                                }
                                messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": tool_call_id,
                                        "content": json.dumps(tool_result, ensure_ascii=False),
                                    }
                                )
                                continue
                            for idx, case in enumerate(cases):
                                if not isinstance(case, Mapping):
                                    invalid_case_indices.append(idx)
                                    continue
                                to_node = case.get("to_node") if "to_node" in case else None
                                if to_node is not None and not isinstance(to_node, str):
                                    invalid_case_indices.append(idx)
                                    continue
                                normalized_cases.append(dict(case))

                        if "default_to_node" in args and args.get("default_to_node") is not None and not isinstance(args.get("default_to_node"), str):
                            tool_result = {
                                "status": "error",
                                "message": "switch 的 default_to_node 只能是节点 id 或 null。",
                                "invalid_fields": ["default_to_node"],
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": json.dumps(tool_result, ensure_ascii=False),
                                }
                            )
                            continue
                        if invalid_case_indices:
                            tool_result = {
                                "status": "error",
                                "message": "cases 中的 to_node 需要是字符串或 null。",
                                "invalid_case_indices": invalid_case_indices,
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": json.dumps(tool_result, ensure_ascii=False),
                                }
                            )
                            continue

                        updates: Dict[str, Any] = {}
                        if "display_name" in args:
                            updates["display_name"] = args.get("display_name")
                        if "parent_node_id" in args:
                            updates["parent_node_id"] = parent_node_id
                        if "default_to_node" in args:
                            updates["default_to_node"] = args.get("default_to_node")
                        if "cases" in args:
                            updates["cases"] = normalized_cases
                        if "params" in args:
                            cleaned_params, removed_param_fields = _filter_supported_params(
                                node_type="switch",
                                params=new_params or {},
                                action_schemas=action_schemas,
                            )
                            updates["params"] = cleaned_params
                        builder.update_node(node_id, **updates)
                        removed_param_fields.extend(
                            _sanitize_builder_node_params(builder, node_id, action_schemas)
                        )
                        removed_node_fields = _sanitize_builder_node_fields(builder, node_id)
                        if removed_param_fields or removed_node_fields:
                            tool_result = {
                                "status": "error",
                                "message": "switch 节点仅支持 id/type/display_name/params/cases/default_to_node 字段，params 仅支持 source/field，已移除不支持的字段。",
                                "removed_fields": removed_param_fields,
                                "removed_node_fields": removed_node_fields,
                                "node_id": node_id,
                            }
                        else:
                            tool_result = {"status": "ok", "type": "node_updated", "node_id": node_id}
                    else:  # update_loop_node
                        new_params = args.get("params") if "params" in args else None
                        if "params" in args and not isinstance(new_params, Mapping):
                            tool_result = {
                                "status": "error",
                                "message": "loop 节点的 params 需要是对象。",
                            }
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tool_call_id,
                                    "content": json.dumps(tool_result, ensure_ascii=False),
                                }
                            )
                            continue
                        updates: Dict[str, Any] = {}
                        if "display_name" in args:
                            updates["display_name"] = args.get("display_name")
                        if "parent_node_id" in args:
                            updates["parent_node_id"] = parent_node_id
                        if "params" in args:
                            cleaned_params, removed_param_fields = _filter_supported_params(
                                node_type="loop",
                                params=new_params or {},
                                action_schemas=action_schemas,
                            )
                            updates["params"] = cleaned_params
                        builder.update_node(node_id, **updates)
                        if func_name == "update_loop_node":
                            _attach_sub_graph_nodes(builder, node_id, sub_graph_nodes)
                        removed_param_fields.extend(
                            _sanitize_builder_node_params(builder, node_id, action_schemas)
                        )
                        removed_node_fields = _sanitize_builder_node_fields(builder, node_id)
                        if removed_param_fields or removed_node_fields:
                            tool_result = {
                                "status": "error",
                                "message": "loop 节点仅支持 id/type/display_name/params 字段，params 仅支持 loop_kind/source/condition/item_alias/body_subgraph/exports，且 exports 只能包含 items/aggregates，已移除不支持的字段。",
                                "removed_fields": removed_param_fields,
                                "removed_node_fields": removed_node_fields,
                                "node_id": node_id,
                            }
                        else:
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
                nodes_without_upstream = _find_nodes_without_upstream(skeleton)
                needs_dependency_review = bool(nodes_without_upstream)
                tool_result = {
                    "status": "ok" if is_covered else "needs_more_coverage",
                    "type": "finalized",
                    "notes": args.get("notes"),
                    "coverage": coverage,
                    "nodes_without_upstream": nodes_without_upstream,
                }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(tool_result, ensure_ascii=False),
                    }
                )

                if needs_dependency_review:
                    dependency_retry += 1
                    log_info(
                        "[Planner] 存在无上游依赖的节点，将提示 LLM 检查是否遗漏引用。",
                        f"nodes={nodes_without_upstream}",
                    )
                    dependency_feedback = _build_dependency_feedback_message(
                        workflow=skeleton, nodes_without_upstream=nodes_without_upstream
                    )
                    messages.append({"role": "system", "content": dependency_feedback})

                if is_covered:
                    if needs_dependency_review and dependency_retry <= max_dependency_refine_rounds:
                        log_info("🔧 覆盖度通过，但需要进一步检查无上游依赖的节点，继续规划。")
                    else:
                        finalized = True
                        log_success("[Planner] 覆盖度检查通过，结束结构规划。")
                        if needs_dependency_review and dependency_retry > max_dependency_refine_rounds:
                            log_warn(
                                "已超过无上游依赖节点检查次数，继续后续流程。"
                            )
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

            elif func_name == "dump_model":
                workflow_snapshot = _attach_inferred_edges(builder.to_workflow())
                latest_skeleton = workflow_snapshot
                tool_result = {
                    "status": "ok",
                    "type": "dump_model",
                    "summary": {
                        "node_count": len(workflow_snapshot.get("nodes") or []),
                        "edge_count": len(workflow_snapshot.get("edges") or []),
                    },
                    "workflow": workflow_snapshot,
                }

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

    return latest_skeleton


__all__ = ["plan_workflow_structure_with_llm"]
