"""Structure planning logic for the planner."""

import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    log_debug,
    log_event,
    log_info,
    log_json,
    log_section,
    log_success,
    log_warn,
)
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.approval import detect_missing_approval_nodes
from velvetflow.planner.connectivity import ensure_edges_connectivity
from velvetflow.planner.coverage import (
    check_requirement_coverage_with_llm,
    refine_workflow_structure_with_llm,
)
from velvetflow.planner.llm_edges import synthesize_edges_with_llm
from velvetflow.planner.tools import PLANNER_TOOLS
from velvetflow.planner.workflow_builder import WorkflowBuilder
from velvetflow.search import HybridActionSearchService


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
        "3) 使用 add_edge 连接节点形成有向图（DAG），包含必要的条件/分支/循环/并行等。\n"
        "4) 当结构完成时调用 finalize_workflow。\n\n"
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
        "4. 循环节点的内部数据只能通过 loop.exports 暴露给外部，下游引用循环结果时必须使用 result_of.<loop_id>.items 或 result_of.<loop_id>.aggregates.*，禁止直接引用 body 子图的节点。\n\n"
        "【覆盖度要求】\n"
        "你必须确保工作流结构能够完全覆盖用户自然语言需求中的每个子任务，而不是只覆盖前半部分：\n"
        "例如，如果需求包含：触发 + 查询 + 筛选 + 总结 + 通知，你不能只实现触发 + 查询，\n"
        "必须在结构里显式包含筛选、总结、通知等对应节点和数据流。\n"
        "当你确信所有子需求都有对应的节点和边时，再调用 finalize_workflow。"
    )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": nl_requirement},
    ]

    finalized = False

    # ---------- 结构规划（多轮 tool-calling） ----------
    for round_idx in range(max_rounds):
        log_section(f"结构规划 Round {round_idx + 1}")
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=PLANNER_TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )
        msg = resp.choices[0].message
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": msg.tool_calls,
        })

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
                        )
                        tool_result = {"status": "ok", "type": "node_added", "node_id": args["id"]}
                else:
                    builder.add_node(
                        node_id=args["id"],
                        node_type=node_type,
                        action_id=action_id,
                        display_name=args.get("display_name"),
                        params=args.get("params") or {},
                    )
                    tool_result = {"status": "ok", "type": "node_added", "node_id": args["id"]}

            elif func_name == "add_edge":
                builder.add_edge(
                    from_node=args["from_node"],
                    to_node=args["to_node"],
                    condition=args.get("condition"),
                )
                tool_result = {"status": "ok", "type": "edge_added"}

            elif func_name == "finalize_workflow":
                finalized = True
                tool_result = {"status": "ok", "type": "finalized", "notes": args.get("notes")}

            else:
                tool_result = {"status": "error", "message": f"未知工具 {func_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

        if finalized:
            log_success("[Planner] 收到 finalize_workflow，结束结构规划。")
            break

    # ---------- 接线 + 连通性补全 ----------
    skeleton = builder.to_workflow()
    nodes = skeleton.get("nodes", [])
    edges = skeleton.get("edges", [])

    if not edges:
        log_warn("[Planner] 第一阶段没有生成任何 edges，调用 LLM 进行自动接线...")
        auto_edges = synthesize_edges_with_llm(nodes=nodes, nl_requirement=nl_requirement)
        if auto_edges:
            log_info(f"[Planner] LLM 自动生成了 {len(auto_edges)} 条 edges。")
            skeleton["edges"] = auto_edges
        else:
            log_warn("[Planner] LLM 自动接线失败，使用保底线性串联方式生成 edges。")
            start_nodes = [n for n in nodes if n.get("type") == "start"]
            end_nodes = [n for n in nodes if n.get("type") == "end"]
            middle_nodes = [n for n in nodes if n.get("type") not in ("start", "end")]

            ordered = start_nodes + middle_nodes + end_nodes
            auto_edges = []
            for i in range(len(ordered) - 1):
                auto_edges.append({
                    "from": ordered[i]["id"],
                    "to": ordered[i + 1]["id"],
                    "condition": None,
                })
            skeleton["edges"] = auto_edges

    skeleton["edges"] = ensure_edges_connectivity(nodes, skeleton["edges"])
    skeleton = ensure_registered_actions(
        skeleton, action_registry=action_registry, search_service=search_service
    )

    # ---------- 覆盖度校验 + 结构改进 ----------
    for refine_round in range(max_coverage_refine_rounds + 1):
        log_section(f"覆盖度校验轮次 {refine_round}")
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
        log_event("coverage_check", {"round": refine_round, "coverage": coverage})
        log_json("覆盖度检查结果", coverage)

        if coverage.get("is_covered", False):
            log_success("当前结构已经被判定为“完全覆盖”用户需求。")
            break

        missing_points = coverage.get("missing_points", []) or []
        if not missing_points:
            log_warn("覆盖度检查认为不完整，但 missing_points 为空，不再尝试结构改进。")
            break

        if refine_round == max_coverage_refine_rounds:
            log_warn("已达到最大结构改进轮次，仍认为不完全覆盖，保留当前结构继续后续阶段。")
            break

        log_info("🔧 检测到未覆盖的需求点，将调用 LLM 对工作流结构进行增量改进：")
        for mp in missing_points:
            log_debug(f" - {mp}")

        refined = refine_workflow_structure_with_llm(
            nl_requirement=nl_requirement,
            current_workflow=skeleton,
            missing_points=missing_points,
            model=OPENAI_MODEL,
        )

        refined_nodes = refined.get("nodes", [])
        refined_edges = refined.get("edges", [])
        refined["edges"] = ensure_edges_connectivity(refined_nodes, refined_edges)
        refined = ensure_registered_actions(
            refined, action_registry=action_registry, search_service=search_service
        )
        skeleton = refined

    return skeleton


__all__ = ["plan_workflow_structure_with_llm"]
