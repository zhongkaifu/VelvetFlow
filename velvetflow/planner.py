"""Planner utilities and workflow orchestration logic."""
import copy
import json
import os
from dataclasses import asdict
from collections import deque
from typing import Any, Dict, List, Mapping, Optional, Union

from openai import OpenAI

from velvetflow.action_registry import get_action_by_id
from velvetflow.config import OPENAI_MODEL
from velvetflow.models import PydanticValidationError, ValidationError, Workflow
from velvetflow.search import HybridActionSearchService

# ===================== 5. Planner 工具定义 =====================

PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_business_actions",
            "description": (
                "在海量业务动作库中按自然语言查询可用动作。返回 candidates 列表，"
                "后续 add_node(type='action') 时 action_id 必须取自最近一次 candidates.id。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_workflow_meta",
            "description": "设置工作流的基本信息（名称、描述）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["workflow_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_node",
            "description": (
                "在工作流中新增一个节点。\n"
                "- type='action'：请先调用 search_business_actions 选出合适的 action_id。\n"
                "  action_id 必须是最近一次 search_business_actions 返回的 candidates.id 之一。\n"
                "- type='condition'：请在 params 中使用结构化条件，例如：\n"
                "  {\"kind\": \"any_greater_than\", "
                "\"source\": \"result_of.some_node.items\", "
                "\"field\": \"value\", \"threshold\": 10 }\n"
                "或 {\"kind\": \"equals\", "
                "\"source\": \"result_of.some_node.count\", \"value\": 0 }"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "节点唯一 ID"},
                    "type": {
                        "type": "string",
                        "enum": ["start", "end", "action", "condition", "loop", "parallel"],
                    },
                    "action_id": {
                        "type": "string",
                        "description": "type='action' 时指定 action_id。",
                        "nullable": True,
                    },
                    "display_name": {"type": "string"},
                    "params": {
                        "type": "object",
                        "description": "节点参数，可为空，但稍后会在第二阶段补全。",
                        "additionalProperties": True,
                    },
                },
                "required": ["id", "type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_edge",
            "description": "新增一条有向边（from -> to），可选 condition（true/false 等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_node": {"type": "string"},
                    "to_node": {"type": "string"},
                    "condition": {"type": "string", "nullable": True},
                },
                "required": ["from_node", "to_node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_workflow",
            "description": "当你认为结构已经覆盖需求时调用，结束规划阶段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "ready": {"type": "boolean", "default": True},
                    "notes": {"type": "string"},
                },
                "required": ["ready"],
            },
        },
    },
]


# ===================== 6. WorkflowBuilder =====================

class WorkflowBuilder:
    def __init__(self):
        self.workflow_name: str = "unnamed_workflow"
        self.description: str = ""
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Dict[str, Any]] = []

    def set_meta(self, name: str, description: Optional[str]):
        if name:
            self.workflow_name = name
        if description:
            self.description = description or ""

    def add_node(self, node_id: str, node_type: str,
                 action_id: Optional[str],
                 display_name: Optional[str],
                 params: Optional[Dict[str, Any]]):
        if node_id in self.nodes:
            print(f"[Builder] 节点 {node_id} 已存在，将覆盖。")
        self.nodes[node_id] = {
            "id": node_id,
            "type": node_type,
            "action_id": action_id,
            "display_name": display_name,
            "params": params or {},
        }

    def add_edge(self, from_node: str, to_node: str, condition: Optional[str]):
        self.edges.append({"from": from_node, "to": to_node, "condition": condition})

    def to_workflow(self) -> Dict[str, Any]:
        return {
            "workflow_name": self.workflow_name,
            "description": self.description,
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
        }


# ===================== 7. 保证 edges 覆盖所有节点 =====================

def ensure_edges_connectivity(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    目标：
    - 保留已有 edge（Planner/LLM 已经设计好的结构）
    - 保证所有节点至少在一条 edge 中出现（除非只有一个节点）
    - 保证从某个 start 节点能到所有节点（补充必要的边）
    """

    if not nodes:
        return []

    node_ids = [n["id"] for n in nodes]
    id_set = set(node_ids)

    # 1) 过滤非法边
    cleaned_edges: List[Dict[str, Any]] = []
    for e in edges:
        frm = e.get("from")
        to = e.get("to")
        if frm in id_set and to in id_set:
            cleaned_edges.append({"from": frm, "to": to, "condition": e.get("condition")})
    edges = cleaned_edges

    # 2) 找 start 节点
    start_nodes = [n["id"] for n in nodes if n.get("type") == "start"]
    if not start_nodes:
        to_ids = {e["to"] for e in edges}
        start_nodes = [nid for nid in node_ids if nid not in to_ids]

    if not start_nodes:
        start_nodes = [node_ids[0]]

    # 3) BFS 找 reachable
    adj: Dict[str, List[str]] = {}
    for e in edges:
        adj.setdefault(e["from"], []).append(e["to"])

    reachable: set = set()
    dq = deque(start_nodes)
    while dq:
        x = dq.popleft()
        if x in reachable:
            continue
        reachable.add(x)
        for y in adj.get(x, []):
            if y not in reachable:
                dq.append(y)

    # 4) 把不可达节点挂上去
    unreachable = [nid for nid in node_ids if nid not in reachable]
    if not unreachable:
        return edges  # 已经全连通

    from_ids = {e["from"] for e in edges}
    to_ids = {e["to"] for e in edges}
    tail_candidates = [nid for nid in node_ids if nid in from_ids and nid not in to_ids]
    if tail_candidates:
        current_tail = tail_candidates[0]
    else:
        current_tail = list(reachable)[-1] if reachable else start_nodes[0]

    for u in unreachable:
        if current_tail == u:
            continue
        edges.append({"from": current_tail, "to": u, "condition": None})
        current_tail = u

    return edges


# ===================== 8. Action 校验与纠偏 =====================

def ensure_registered_actions(
    workflow: Union[Workflow, Dict[str, Any]],
    action_registry: List[Dict[str, Any]],
    search_service: Optional[HybridActionSearchService] = None,
) -> Union[Workflow, Dict[str, Any]]:
    """
    确保所有 action 节点引用的 action_id 都存在于注册表。

    - 如果 action_id 合法，保持不变；
    - 如果 action_id 缺失或未注册，尝试用节点 display_name 进行搜索替换；
    - 如果依然无法匹配，则清空 action_id，避免携带非法 ID 进入后续阶段。
    """

    actions_by_id = _index_actions_by_id(action_registry)
    original_type = Workflow if isinstance(workflow, Workflow) else dict
    workflow_dict = (
        workflow.model_dump(by_alias=True)
        if isinstance(workflow, Workflow)
        else copy.deepcopy(workflow)
    )

    nodes = workflow_dict.get("nodes", []) if isinstance(workflow_dict, dict) else []

    for node in nodes:
        if node.get("type") != "action":
            continue

        aid = node.get("action_id")
        if aid and aid in actions_by_id:
            continue

        nid = node.get("id", "<unknown>")
        display_name = node.get("display_name") or ""

        replacement: Optional[str] = None
        if search_service and display_name:
            candidates = search_service.search(query=display_name, top_k=1)
            if candidates:
                replacement = candidates[0].get("action_id")

        if replacement:
            print(
                f"[ActionGuard] 节点 '{nid}' 的 action_id='{aid}' 未注册，"
                f"已根据 display_name='{display_name}' 替换为 '{replacement}'。"
            )
            node["action_id"] = replacement
        else:
            if aid:
                print(
                    f"[ActionGuard] 节点 '{nid}' 的 action_id='{aid}' 未注册且无法自动替换，"
                    "已清空该字段以便后续流程重新补齐。"
                )
            node["action_id"] = None

    if original_type is Workflow:
        return Workflow.model_validate(workflow_dict)
    return workflow_dict


# ===================== 9. edges 为空时用 LLM 接线 =====================

def synthesize_edges_with_llm(
    nodes: List[Dict[str, Any]],
    nl_requirement: str,
    model: str = OPENAI_MODEL,
) -> List[Dict[str, Any]]:
    """
    当第一阶段没有生成任何 edges 时，让 LLM 根据节点列表和需求补一份 edges。
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    node_brief = [
        {
            "id": n["id"],
            "type": n.get("type"),
            "action_id": n.get("action_id"),
            "display_name": n.get("display_name"),
        }
        for n in nodes
    ]

    system_prompt = (
        "你是一个工作流接线助手。\n"
        "现在有一组已经确定的节点 nodes（每个节点有 id/type/action_id/display_name），\n"
        "但是 edges 为空。\n"
        "你的任务是：\n"
        "1. 根据用户的自然语言需求和这些节点的含义，推理它们的执行顺序和分支结构。\n"
        "2. 生成一组 edges，每条 edge 形如：\n"
        "   {\"from\": \"节点ID\", \"to\": \"节点ID\", \"condition\": \"true/false 或 null\"}\n"
        "3. 整体必须是一个有向无环图（DAG），通常从 type='start' 节点开始，到 type='end' 节点结束。\n"
        "4. 如果存在条件节点(type='condition')，请用 edge.condition 表示 true/false 分支，\n"
        "   无条件顺序执行时 condition 用 null。\n"
        "5. 返回的 JSON 必须是：{\"edges\": [ ... ]}，不要包含其它字段，也不要加代码块标记。"
    )

    user_payload = {
        "nl_requirement": nl_requirement,
        "nodes": node_brief,
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.2,
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
        obj = json.loads(text)
        edges = obj.get("edges", [])
        if not isinstance(edges, list):
            raise ValueError("edges 不是 list")
        for e in edges:
            if "from" not in e or "to" not in e:
                raise ValueError("edge 缺少 from/to")
        return edges
    except Exception as e:
        print("[synthesize_edges_with_llm] 无法解析/使用 LLM 返回的 edges，错误：", e)
        print("原始内容：", content)
        return []


# ===================== 10. 需求覆盖校验 + 结构改进 =====================

def check_requirement_coverage_with_llm(
    nl_requirement: str,
    workflow: Dict[str, Any],
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    """
    让 LLM 审核：当前 workflow 是否完全覆盖 nl_requirement。
    返回结构示例：
    {
      "is_covered": true/false,
      "missing_points": ["...", "..."],
      "analysis": "..."
    }
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system_prompt = (
        "你是一个严谨的工作流需求覆盖度审查员。\n"
        "给定：\n"
        "1) 用户的自然语言需求 nl_requirement\n"
        "2) 当前的 workflow（workflow_name/description/nodes/edges）\n\n"
        "你的任务：\n"
        "1. 先把 nl_requirement 中的关键子需求拆分成若干个“原子能力”，例如：\n"
        "   - 某个触发方式（定时 / 事件 / 手动等）\n"
        "   - 若干个数据读取 / 查询步骤\n"
        "   - 若干个过滤 / 条件判断步骤\n"
        "   - 若干个聚合 / 统计 / 总结步骤\n"
        "   - 若干个对外动作（通知、写入数据库、调用外部系统等）\n"
        "   这里的例子仅用于说明“拆分粒度”，不要把具体业务词带入其它任务。\n"
        "2. 再逐项检查当前 workflow 是否对这些原子能力都有完整的支持：\n"
        "   - 是否有对应的节点；\n"
        "   - 节点之间的连接顺序是否合理；\n"
        "   - 是否存在明显缺失（例如：需求中提到“通知”，但 workflow 中完全没有任何通知/写入相关节点）。\n"
        "3. 如果完全覆盖，则 is_covered=true，missing_points 列表为空。\n"
        "4. 如果有任何一条需求没有被覆盖或只被部分覆盖，则 is_covered=false，\n"
        "   并在 missing_points 中用简短中文列出缺失点（例如：“缺少对特定条件的过滤”、“缺少结果汇总后发送给用户的步骤”等）。\n\n"
        "输出格式（非常重要）：\n"
        "返回一个 JSON 对象，形如：\n"
        "{\n"
        "  \"is_covered\": true/false,\n"
        "  \"missing_points\": [\"...\", \"...\"],\n"
        "  \"analysis\": \"详细分析\"\n"
        "}\n"
        "不要添加额外字段，不要输出代码块标记。"
    )

    payload = {
        "nl_requirement": nl_requirement,
        "workflow": workflow,
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
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
        result = json.loads(text)
    except json.JSONDecodeError:
        print("[check_requirement_coverage_with_llm] 无法解析 JSON，原始内容：")
        print(content)
        result = {
            "is_covered": False,
            "missing_points": ["LLM 覆盖度检查解析失败"],
            "analysis": content,
        }

    if "is_covered" not in result:
        result["is_covered"] = False
    if "missing_points" not in result or not isinstance(result["missing_points"], list):
        result["missing_points"] = []
    if "analysis" not in result:
        result["analysis"] = ""

    return result


def refine_workflow_structure_with_llm(
    nl_requirement: str,
    current_workflow: Dict[str, Any],
    missing_points: List[str],
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    """
    当发现 workflow 未完全覆盖需求时，请 LLM 在现有结构基础上进行改进，
    补充缺失的节点/分支/条件等。
    """
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    system_prompt = (
        "你是一个工作流架构师，负责在现有 workflow 基础上做“增量改进”，\n"
        "以便完全满足用户的自然语言需求。\n\n"
        "已知：\n"
        "1) nl_requirement 是用户的完整自然语言需求；\n"
        "2) current_workflow 是当前 workflow（workflow_name/description/nodes/edges），\n"
        "   它已经实现了部分需求，但还不完整；\n"
        "3) missing_points 是已经识别出的缺失点列表，例如：\n"
        "   - \"缺少对某个特定条件的过滤步骤\"\n"
        "   - \"缺少对结果进行汇总/统计/总结的步骤\"\n"
        "   - \"缺少将结果发送给指定用户或写入外部系统的步骤\"\n"
        "   这些示例只是缺失类型的演示，不代表具体业务。\n\n"
        "你的任务：\n"
        "1. 在 current_workflow 的基础上添加或调整节点/edges，以补齐 missing_points 指出的能力；\n"
        "2. 尽量复用已有节点和数据流（result_of.<node_id>），避免推倒重来；\n"
        "3. 只在必要时新增节点（例如：专门用于过滤/聚合/通知的新节点）；\n"
        "4. 确保整体依然是一个有向无环图（DAG），通常从 start 到 end；\n"
        "5. 不要删除已经正确实现需求的部分，除非必须重构；\n"
        "6. 不需要补全 params 的所有细节（第二阶段会做），但应显式添加对应的节点和 edges。\n\n"
        "输出格式：\n"
        "返回一个完整的 JSON 对象，形如：\n"
        "{\n"
        "  \"workflow_name\": \"...\",\n"
        "  \"description\": \"...\",\n"
        "  \"nodes\": [ ... ],\n"
        "  \"edges\": [ ... ]\n"
        "}\n"
        "不要包含其他顶层字段，不要输出代码块标记。"
    )

    payload = {
        "nl_requirement": nl_requirement,
        "current_workflow": current_workflow,
        "missing_points": missing_points,
    }

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        temperature=0.2,
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
        refined = json.loads(text)
    except json.JSONDecodeError:
        print("[refine_workflow_structure_with_llm] 无法解析 JSON，原始内容：")
        print(content)
        return current_workflow

    if not isinstance(refined, dict) or not isinstance(refined.get("nodes"), list) or not isinstance(refined.get("edges"), list):
        print("[refine_workflow_structure_with_llm] LLM 返回的结构不完整，回退到 current_workflow。")
        return current_workflow

    return refined


# ===================== 11. 第一阶段：结构规划 LLM =====================

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
        print(f"\n===== 结构规划 Round {round_idx + 1} =====")
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
            print("[Planner] 本轮没有 tool_calls，提前结束。")
            break

        for tc in msg.tool_calls:
            func_name = tc.function.name
            raw_args = tc.function.arguments
            tool_call_id = tc.id
            try:
                args = json.loads(raw_args) if raw_args else {}
            except json.JSONDecodeError:
                print(f"[Error] 解析工具参数失败: {raw_args}")
                args = {}

            print(f"[Planner] 调用工具: {func_name}({args})")

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
            print("[Planner] 收到 finalize_workflow，结束结构规划。")
            break

    # ---------- 接线 + 连通性补全 ----------
    skeleton = builder.to_workflow()
    nodes = skeleton.get("nodes", [])
    edges = skeleton.get("edges", [])

    if not edges:
        print("\n[Planner] 第一阶段没有生成任何 edges，调用 LLM 进行自动接线...")
        auto_edges = synthesize_edges_with_llm(nodes=nodes, nl_requirement=nl_requirement)
        if auto_edges:
            print(f"[Planner] LLM 自动生成了 {len(auto_edges)} 条 edges。")
            skeleton["edges"] = auto_edges
        else:
            print("[Planner] LLM 自动接线失败，使用保底线性串联方式生成 edges。")
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
        print(f"\n==== 覆盖度校验轮次 {refine_round} ====\n")
        coverage = check_requirement_coverage_with_llm(
            nl_requirement=nl_requirement,
            workflow=skeleton,
            model=OPENAI_MODEL,
        )
        print("覆盖度检查结果：", json.dumps(coverage, ensure_ascii=False, indent=2))

        if coverage.get("is_covered", False):
            print("✅ 当前结构已经被判定为“完全覆盖”用户需求。")
            break

        missing_points = coverage.get("missing_points", []) or []
        if not missing_points:
            print("⚠️ 覆盖度检查认为不完整，但 missing_points 为空，不再尝试结构改进。")
            break

        if refine_round == max_coverage_refine_rounds:
            print("⚠️ 已达到最大结构改进轮次，仍认为不完全覆盖，保留当前结构继续后续阶段。")
            break

        print("🔧 检测到未覆盖的需求点，将调用 LLM 对工作流结构进行增量改进：")
        for mp in missing_points:
            print(" -", mp)

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


# ===================== 12. 第二阶段辅助：节点上下游关系 =====================

def build_node_relations(workflow_skeleton: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    nodes = workflow_skeleton.get("nodes", [])
    edges = workflow_skeleton.get("edges", [])
    node_ids = {n["id"] for n in nodes}
    relations: Dict[str, Dict[str, List[str]]] = {
        nid: {"upstream": [], "downstream": []} for nid in node_ids
    }
    for e in edges:
        frm = e.get("from")
        to = e.get("to")
        if frm in node_ids and to in node_ids:
            relations[frm]["downstream"].append(to)
            relations[to]["upstream"].append(frm)
    return relations


# ===================== 13. 第二阶段：参数补全 LLM =====================

def fill_params_with_llm(
    workflow_skeleton: Dict[str, Any],
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

    node_relations = build_node_relations(workflow_skeleton)

    system_prompt = (
        "你是一个工作流参数补全助手。\n"
        "已有一个工作流 skeleton：节点(id/type/action_id/display_name)和 edges 已确定，"
        "但很多节点的 params 为空或不完整。\n"
        "另有 action_schemas（action_id -> arg_schema/output_schema）和 node_relations（每个节点的上下游关系）。\n\n"
        "【重要说明：示例仅为模式，不代表具体业务】\n"
        "下面所有示例（包括列表字段名、格式模板中的文字等）仅用于说明 DSL 的使用方式，\n"
        "实际任务中必须根据当前 action 的 output_schema 和业务语义来选取字段名、节点名和字符串内容，\n"
        "不要在与当前任务无关的场景中硬套“员工/体温/新闻/Nvidia”等具体词汇。\n\n"
        "【任务】\n"
        "1. 对 type='action' 且有 action_id 的节点，根据对应 arg_schema 填充 params，\n"
        "   覆盖所有 required 字段，并给出合理的占位值（可以是示例值，但要语义合理）。\n"
        "2. 当某个字段的值需要来自上游节点输出时，请使用“数据绑定 DSL”：\n"
        "   2.1 最简单：从上游直接取值或计数，例如：\n"
        "       - 直接引用：\n"
        "         {\"__from__\": \"result_of.some_node.items\", \"__agg__\": \"identity\"}\n"
        "       - 按条件计数：\n"
        "         {\n"
        "           \"__from__\": \"result_of.some_node.items\",\n"
        "           \"__agg__\": \"count_if\",\n"
        "           \"field\": \"value\",  // 这里的 value 只是示意字段名\n"
        "           \"op\": \">\",\n"
        "           \"value\": 10\n"
        "         }\n"
        "   2.2 对于“先过滤再格式化成消息文本”的情况，推荐使用 **pipeline 聚合 DSL**：\n"
        "       示例（只说明结构，不代表具体业务含义）：\n"
        "       {\n"
        "         \"__from__\": \"result_of.list_node.items\",\n"
        "         \"__agg__\": \"pipeline\",\n"
        "         \"steps\": [\n"
        "           {\"op\": \"filter\", \"field\": \"score\", \"cmp\": \">\", \"value\": 0.8},\n"
        "           {\"op\": \"map\", \"field\": \"id\"},\n"
        "           {\"op\": \"format_join\", \"format\": \"ID={value} 异常\", \"sep\": \"\\n\"}\n"
        "         ]\n"
        "       }\n"
        "       - filter 步骤：保留满足条件的元素（cmp 支持 >,>=,<,<=,==）。\n"
        "       - map 步骤：从每个元素中取某个字段组成新的列表。\n"
        "       - format_join 步骤：对列表中每个元素用 format 中的 {value} 替换，然后用 sep 拼接成一个字符串。\n"
        "   2.3 为兼容已有写法，也允许使用简化的 filter_map（执行器会自动翻译成 pipeline）：\n"
        "       {\n"
        "         \"__from__\": \"result_of.list_node.items\",\n"
        "         \"__agg__\": \"filter_map\",\n"
        "         \"filter_field\": \"score\",\n"
        "         \"filter_op\": \">\",\n"
        "         \"filter_value\": 0.8,\n"
        "         \"map_field\": \"id\",\n"
        "         \"format\": \"ID={value} 异常\"\n"
        "       }\n\n"
        "3. 对 type='condition' 的节点，根据 display_name、上下游关系和整体语义，\n"
        "   使用结构化条件 params，例如（只是模式示例）：\n"
        "   {\"kind\": \"any_greater_than\", "
        "\"source\": \"result_of.some_node.items\", "
        "\"field\": \"score\", \"threshold\": 0.8 }\n"
        "   或 {\"kind\": \"equals\", "
        "\"source\": \"result_of.other_node.count\", \"value\": 0 }。\n"
        "   这里的 some_node / items / score / count 都是示范性的占位名，\n"
        "   实际需要根据该节点选择的 action 的 output_schema 来决定。\n\n"
        "4. start/end 节点允许 params 为空 {}。\n"
        "5. 返回的 JSON 结构必须与输入 workflow_skeleton 相同，只是节点的 params 更完整。\n"
        "6. 只返回 JSON 对象本身，不要加代码块标记。"
    )

    user_payload = {
        "workflow_skeleton": workflow_skeleton,
        "node_relations": node_relations,
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
        completed_workflow = json.loads(text)
    except json.JSONDecodeError:
        print("[fill_params_with_llm] 无法解析模型返回 JSON，原始内容：")
        print(content)
        raise

    return completed_workflow


# ===================== 14. 静态校验工具函数 =====================

def _index_actions_by_id(action_registry: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {a["action_id"]: a for a in action_registry}


def _index_nodes_by_id(workflow: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in workflow.get("nodes", [])}


def _check_output_path_against_schema(
    source_path: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """
    对诸如 "result_of.fetch_temperatures.data" 或 "result_of.node_id.foo.bar" 做静态校验：
    - result_of.<node_id> 必须存在
    - 该 node 必须有 action_id
    - 对应 action 的 output_schema 必须包含第一层字段（data / foo）

    返回:
      - None: 校验通过
      - str: 具体错误信息
    """
    if not isinstance(source_path, str):
        return f"source/__from__ 应该是字符串，但收到类型: {type(source_path)}"

    parts = source_path.split(".")
    if len(parts) < 2 or parts[0] != "result_of":
        return None

    node_id = parts[1]
    rest_path = parts[2:]

    if node_id not in nodes_by_id:
        return f"路径 '{source_path}' 引用的节点 '{node_id}' 不存在。"

    node = nodes_by_id[node_id]
    action_id = node.get("action_id")
    if not action_id:
        return f"路径 '{source_path}' 引用的节点 '{node_id}' 没有 action_id，无法从 output_schema 校验。"

    action_def = actions_by_id.get(action_id)
    if not action_def:
        return f"路径 '{source_path}' 引用的节点 '{node_id}' 的 action_id='{action_id}' 不在 Action Registry 中。"

    output_schema = action_def.get("output_schema")
    if not isinstance(output_schema, dict):
        return f"action_id='{action_id}' 没有定义 output_schema，无法校验路径 '{source_path}'。"

    if not rest_path:
        return None

    err = _schema_path_error(output_schema, rest_path)
    if err:
        return f"路径 '{source_path}' 无效：{err}"

    return None


def _schema_path_error(schema: Mapping[str, Any], fields: List[str]) -> Optional[str]:
    """Check whether a dotted field path exists in a JSON schema."""

    if not isinstance(schema, Mapping):
        return "output_schema 不是对象，无法校验字段路径。"

    current: Mapping[str, Any] = schema
    idx = 0
    while idx < len(fields):
        name = fields[idx]
        typ = current.get("type")

        if typ == "array":
            current = current.get("items") or {}
            continue

        if typ == "object":
            props = current.get("properties") or {}
            if name not in props:
                return f"字段 '{name}' 不存在，已知字段有: {list(props.keys())}"
            current = props[name]
            idx += 1
            continue

        return f"字段路径 '{'.'.join(fields)}' 与 schema 类型 '{typ}' 不匹配（期望 object/array）。"

    return None


def _get_array_item_schema_from_output(
    source_path: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    给一个 source/__from__ 路径（例如 "result_of.fetch_temperatures.data"），
    如果它指向的是某个 action 的 output_schema 中的数组字段（例如 data: array[...]），
    就返回这个数组的 item schema（即 items），否则返回 None。
    """
    if not isinstance(source_path, str):
        return None

    parts = source_path.split(".")
    if len(parts) < 3 or parts[0] != "result_of":
        return None

    src_node_id = parts[1]
    first_field = parts[2]

    node = nodes_by_id.get(src_node_id)
    if not node:
        return None

    action_id = node.get("action_id")
    if not action_id:
        return None

    action_def = actions_by_id.get(action_id)
    if not action_def:
        return None

    output_schema = action_def.get("output_schema")
    if not isinstance(output_schema, dict):
        return None

    props = output_schema.get("properties") or {}
    field_schema = props.get(first_field)
    if not isinstance(field_schema, dict):
        return None

    if field_schema.get("type") == "array" and isinstance(field_schema.get("items"), dict):
        return field_schema["items"]

    return None


def _check_array_item_field(
    source_path: str,
    field_name: str,
    nodes_by_id: Dict[str, Dict[str, Any]],
    actions_by_id: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    """
    检查 "result_of.xxx.data" 这种路径指向的数组元素 schema 里，
    是否存在 field_name 这个字段。
    - 返回 None 表示没问题（或无法判断）
    - 返回 str 表示确定存在问题，这个 str 就是错误信息
    """
    if not field_name:
        return None

    item_schema = _get_array_item_schema_from_output(source_path, nodes_by_id, actions_by_id)
    if not item_schema:
        return None

    props = item_schema.get("properties") or {}
    if field_name not in props:
        return (
            f"source 路径 '{source_path}' 指向的数组元素 schema 中不存在字段 '{field_name}'，"
            f"已知的字段有: {list(props.keys())}"
        )
    return None


# ===================== 15. 后端校验 =====================

def validate_completed_workflow(
    workflow: Dict[str, Any],
    action_registry: List[Dict[str, Any]],
) -> List[ValidationError]:
    errors: List[ValidationError] = []

    nodes = workflow.get("nodes", [])
    edges = workflow.get("edges", [])

    nodes_by_id = _index_nodes_by_id(workflow)
    node_ids = set(nodes_by_id.keys())
    actions_by_id = _index_actions_by_id(action_registry)

    # ---------- edges 校验 ----------
    for e in edges:
        frm = e.get("from")
        to = e.get("to")
        if frm not in node_ids:
            errors.append(
                ValidationError(
                    code="INVALID_EDGE",
                    node_id=frm,
                    field="from",
                    message=f"Edge from '{frm}' -> '{to}' 中，from 节点不存在。",
                )
            )
        if to not in node_ids:
            errors.append(
                ValidationError(
                    code="INVALID_EDGE",
                    node_id=to,
                    field="to",
                    message=f"Edge from '{frm}' -> '{to}' 中，to 节点不存在。",
                )
            )

    # ---------- 图连通性校验 ----------
    start_nodes = [n["id"] for n in nodes if n.get("type") == "start"]
    reachable: set = set()
    if nodes and start_nodes:
        adj: Dict[str, List[str]] = {}
        for e in edges:
            frm = e.get("from")
            to = e.get("to")
            if frm in node_ids and to in node_ids:
                adj.setdefault(frm, []).append(to)

        dq = deque(start_nodes)
        while dq:
            nid = dq.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for nxt in adj.get(nid, []):
                if nxt not in reachable:
                    dq.append(nxt)

        for nid in node_ids - reachable:
            errors.append(
                ValidationError(
                    code="DISCONNECTED_GRAPH",
                    node_id=nid,
                    field=None,
                    message=f"节点 '{nid}' 无法从 start 节点到达。",
                )
            )

    # ---------- 节点校验 ----------
    for n in nodes:
        nid = n["id"]
        ntype = n.get("type")
        action_id = n.get("action_id")
        params = n.get("params", {})

        # 1) action 节点
        if ntype == "action" and action_id:
            action_def = actions_by_id.get(action_id)
            if not action_def:
                errors.append(
                    ValidationError(
                        code="UNKNOWN_ACTION_ID",
                        node_id=nid,
                        field="action_id",
                        message=f"节点 '{nid}' 的 action_id '{action_id}' 不在 Action Registry 中。",
                    )
                )
            else:
                schema = action_def.get("arg_schema") or {}
                required_fields = (schema.get("required") or []) if isinstance(schema, dict) else []

                if not isinstance(params, dict) or len(params) == 0:
                    if required_fields:
                        for field in required_fields:
                            errors.append(
                                ValidationError(
                                    code="MISSING_REQUIRED_PARAM",
                                    node_id=nid,
                                    field=field,
                                    message=(
                                        f"action 节点 '{nid}' 的 params 为空，但 action '{action_id}' 有必填字段 '{field}'。"
                                    ),
                                )
                            )
                else:
                    for field in required_fields:
                        if field not in params:
                            errors.append(
                                ValidationError(
                                    code="MISSING_REQUIRED_PARAM",
                                    node_id=nid,
                                    field=field,
                                    message=(
                                        f"action 节点 '{nid}' 的 params 缺少必填字段 '{field}' (action_id='{action_id}')"
                                    ),
                                )
                            )

            # 绑定 DSL 静态校验
            def _walk_params_for_from(obj: Any, path_prefix: str = ""):
                if isinstance(obj, dict):
                    if "__from__" in obj:
                        source_path = obj["__from__"]
                        err = _check_output_path_against_schema(
                            source_path=source_path,
                            nodes_by_id=nodes_by_id,
                            actions_by_id=actions_by_id,
                        )
                        if err:
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field=path_prefix or "params",
                                    message=(
                                        f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）无效：{err}"
                                    ),
                                )
                            )

                        agg = obj.get("__agg__")

                        if agg == "count_if":
                            fld = obj.get("field")
                            if isinstance(fld, str):
                                item_err = _check_array_item_field(
                                    source_path, fld, nodes_by_id, actions_by_id
                                )
                                if item_err:
                                    errors.append(
                                        ValidationError(
                                            code="SCHEMA_MISMATCH",
                                            node_id=nid,
                                            field=f"{path_prefix or 'params'}.field",
                                            message=(
                                                f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）中 count_if.field='{fld}' 无效：{item_err}"
                                            ),
                                        )
                                    )

                        if agg == "filter_map":
                            for fld_key in ("filter_field", "map_field"):
                                fld = obj.get(fld_key)
                                if isinstance(fld, str):
                                    item_err = _check_array_item_field(
                                        source_path, fld, nodes_by_id, actions_by_id
                                    )
                                    if item_err:
                                        errors.append(
                                            ValidationError(
                                                code="SCHEMA_MISMATCH",
                                                node_id=nid,
                                                field=f"{path_prefix or 'params'}.{fld_key}",
                                                message=(
                                                    f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）中 {agg}.{fld_key}='{fld}' 无效：{item_err}"
                                                ),
                                            )
                                        )

                        if agg == "pipeline":
                            steps = obj.get("steps") or []
                            for idx, step in enumerate(steps):
                                if not isinstance(step, dict):
                                    continue
                                fld = step.get("field")
                                if isinstance(fld, str):
                                    item_err = _check_array_item_field(
                                        source_path, fld, nodes_by_id, actions_by_id
                                    )
                                    if item_err:
                                        errors.append(
                                            ValidationError(
                                                code="SCHEMA_MISMATCH",
                                                node_id=nid,
                                                field=f"{path_prefix or 'params'}.pipeline.steps[{idx}].field",
                                                message=(
                                                    f"action 节点 '{nid}' 的参数绑定（{path_prefix or '<root>'}）中 pipeline.steps[{idx}].field='{fld}' 无效：{item_err}"
                                                ),
                                            )
                                        )

                    for k, v in obj.items():
                        new_prefix = f"{path_prefix}.{k}" if path_prefix else k
                        _walk_params_for_from(v, new_prefix)
                elif isinstance(obj, list):
                    for idx, v in enumerate(obj):
                        new_prefix = f"{path_prefix}[{idx}]"
                        _walk_params_for_from(v, new_prefix)

            _walk_params_for_from(params)

        # 2) condition 节点
        if ntype == "condition":
            if not isinstance(params, dict) or len(params) == 0:
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=nid,
                        field="params",
                        message=f"condition 节点 '{nid}' 的 params 为空，至少需要 kind/source 等字段。",
                    )
                )
            else:
                kind = params.get("kind")
                if not kind:
                    errors.append(
                        ValidationError(
                            code="MISSING_REQUIRED_PARAM",
                            node_id=nid,
                            field="kind",
                            message=f"condition 节点 '{nid}' 缺少 kind 字段。",
                        )
                    )
                else:
                    if kind == "any_greater_than":
                        for field in ["source", "field", "threshold"]:
                            if field not in params:
                                errors.append(
                                    ValidationError(
                                        code="MISSING_REQUIRED_PARAM",
                                        node_id=nid,
                                        field=field,
                                        message=(
                                            f"condition 节点 '{nid}' (kind=any_greater_than) 缺少字段 '{field}'。"
                                        ),
                                    )
                                )
                        src = params.get("source")
                        fld = params.get("field")
                        if isinstance(src, str) and isinstance(fld, str):
                            item_err = _check_array_item_field(src, fld, nodes_by_id, actions_by_id)
                            if item_err:
                                errors.append(
                                    ValidationError(
                                        code="SCHEMA_MISMATCH",
                                        node_id=nid,
                                        field="field",
                                        message=f"condition 节点 '{nid}' 的 field='{fld}' 无效：{item_err}",
                                    )
                                )

                    elif kind == "equals":
                        for field in ["source", "value"]:
                            if field not in params:
                                errors.append(
                                    ValidationError(
                                        code="MISSING_REQUIRED_PARAM",
                                        node_id=nid,
                                        field=field,
                                        message=(
                                            f"condition 节点 '{nid}' (kind=equals) 缺少字段 '{field}'。"
                                        ),
                                    )
                                )

                source = params.get("source")
                if isinstance(source, str):
                    if source.startswith("result_of."):
                        try:
                            rest = source[len("result_of."):]
                            node_part = rest.split(".", 1)[0]
                            if node_part not in node_ids:
                                errors.append(
                                    ValidationError(
                                        code="INVALID_EDGE",
                                        node_id=nid,
                                        field="source",
                                        message=(
                                            f"condition 节点 '{nid}' 的 source='{source}' 引用了不存在的节点 ID '{node_part}'。"
                                        ),
                                    )
                                )
                        except Exception:
                            errors.append(
                                ValidationError(
                                    code="SCHEMA_MISMATCH",
                                    node_id=nid,
                                    field="source",
                                    message=(
                                        f"condition 节点 '{nid}' 的 source='{source}' 格式异常。"
                                    ),
                                )
                            )

                    schema_err = _check_output_path_against_schema(
                        source_path=source,
                        nodes_by_id=nodes_by_id,
                        actions_by_id=actions_by_id,
                    )
                    if schema_err:
                        errors.append(
                            ValidationError(
                                code="SCHEMA_MISMATCH",
                                node_id=nid,
                                field="source",
                                message=(
                                    f"condition 节点 '{nid}' 的 source='{source}' 与上游 output_schema 不匹配：{schema_err}"
                                ),
                            )
                        )

    return errors


# ===================== 16. 自修复 LLM =====================

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
        "总体目标：在“尽量不改变工作流整体结构”的前提下，修复这些错误，使 workflow 通过静态校验。\n\n"
        "具体要求（很重要，请严格遵守）：\n"
        "1. 结构保持稳定：\n"
        "   - 不要增加或删除节点；\n"
        "   - 不要随意增加或删除 edges；\n"
        "   - 只能在必要时局部调整 edge.condition（true/false/null），一般情况下保持 edges 原样。\n\n"
        "2. action 节点修复优先级：\n"
        "   - 首先根据 action_schemas[action_id].arg_schema 补齐 params 里缺失的必填字段，或修正错误类型；\n"
        "   - 如果 action_id 本身是合法的（存在于 action_schemas 中），优先“修 params”，不要改 action_id；\n"
        "   - 只有当 validation_errors 明确指出 action_id 不存在时，才考虑把 action_id 改成一个更合理的候选，"
        "     并同步更新该节点的 params 使之符合新的 arg_schema。\n\n"
        "3. 关于 source/__from__ 与 output_schema 的错误：\n"
        "   - 当 validation_errors 提示“路径与上游 output_schema 不匹配”时，"
        "     优先修改路径本身（source 或 __from__ 的字符串），而不要改 action_id 或删除节点；\n"
        "   - 修改路径时的策略：\n"
        "       a) node_id 部分应指向一个存在且有 action_id 的节点；\n"
        "       b) 字段部分应与该 action 的 output_schema.properties 中的字段对齐；\n"
        "       c) 若只是字段拼写错误，尽量只改字段名；\n"
        "       d) 若引用了错误的节点，则优先改为真正产生该数据的上游 action 节点。\n\n"
        "4. 数组元素字段相关错误：\n"
        "   - 当错误信息中包含“数组元素 schema 中不存在字段”之类描述时，\n"
        "     请优先修正这些字段名本身，而不是修改 action_id 或删除节点。\n"
        "   - 字段名应从该数组元素 schema 的 properties 中选择最合理的候选。\n\n"
        "5. condition 节点修复：\n"
        "   - 继续使用结构化 params，例如 any_greater_than / equals；\n"
        "   - 补齐 kind/source/field/threshold/value 等必需字段；\n"
        "   - source 必须引用一个真实存在的节点，且路径前缀 result_of.<node_id> 与该节点的 output_schema 一致。\n\n"
        "6. 参数绑定 DSL 修复（__from__ 及其聚合逻辑）：\n"
        "   - 对于 {\"__from__\": \"result_of.xxx.data\", \"__agg__\": \"...\", ...}：\n"
        "       - 检查 __from__ 路径是否合法、与 output_schema 对齐；\n"
        "       - 检查 count_if/filter_map/pipeline 中的 field/filter_field/map_field 是否存在于数组元素 schema 中；\n"
        "   - 当错误涉及这些字段时，优先只改字段名（根据元素 schema 的 properties），保持聚合逻辑不变。\n\n"
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
        print("[repair_workflow_with_llm] 无法解析模型返回 JSON，原始内容：")
        print(content)
        raise

    return repaired_workflow



# ===================== 17. 总控：两阶段 + 自修复 =====================

def plan_workflow_with_two_pass(
    nl_requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    max_rounds: int = 10,
    max_repair_rounds: int = 3,
) -> Workflow:
    skeleton_raw = plan_workflow_structure_with_llm(
        nl_requirement=nl_requirement,
        search_service=search_service,
        action_registry=action_registry,
        max_rounds=max_rounds,
        max_coverage_refine_rounds=2,
    )
    print("\n==== 第一阶段结果：Workflow Skeleton ====\n")
    print(json.dumps(skeleton_raw, indent=2, ensure_ascii=False))

    skeleton = Workflow.model_validate(skeleton_raw)
    last_good_workflow: Workflow = skeleton

    completed_workflow_raw = fill_params_with_llm(
        workflow_skeleton=skeleton.model_dump(by_alias=True),
        action_registry=action_registry,
        model=OPENAI_MODEL,
    )

    try:
        completed_workflow = Workflow.model_validate(completed_workflow_raw)
        completed_workflow = ensure_registered_actions(
            completed_workflow,
            action_registry=action_registry,
            search_service=search_service,
        )
        if isinstance(completed_workflow, Workflow):
            current_workflow = completed_workflow
        else:
            current_workflow = Workflow.model_validate(completed_workflow)
        last_good_workflow = current_workflow
    except PydanticValidationError as e:
        print(
            "\n[plan_workflow_with_two_pass] 警告：fill_params_with_llm 返回的结构无法通过校验，", e
        )
        current_workflow = last_good_workflow

    for repair_round in range(max_repair_rounds + 1):
        print(f"\n==== 校验 + 自修复轮次 {repair_round} ====\n")
        print("当前 workflow：")
        print(json.dumps(current_workflow.model_dump(by_alias=True), indent=2, ensure_ascii=False))

        errors = validate_completed_workflow(
            current_workflow.model_dump(by_alias=True),
            action_registry=action_registry,
        )

        if not errors:
            print("\n==== 校验通过，无需进一步修复 ====\n")
            last_good_workflow = current_workflow
            return current_workflow

        print("\n==== 校验未通过，错误列表 ====")
        for e in errors:
            print(
                " -",
                f"[code={e.code}] node={e.node_id} field={e.field} message={e.message}",
            )

        if repair_round == max_repair_rounds:
            print("\n==== 已到最大修复轮次，仍有错误，返回最后一个合法结构版本 ====\n")
            return last_good_workflow

        print(f"\n==== 调用 LLM 进行第 {repair_round + 1} 次修复 ====\n")
        repaired_raw = repair_workflow_with_llm(
            broken_workflow=current_workflow.model_dump(by_alias=True),
            validation_errors=errors,
            action_registry=action_registry,
            model=OPENAI_MODEL,
        )

        try:
            repaired_workflow = Workflow.model_validate(repaired_raw)
            repaired_workflow = ensure_registered_actions(
                repaired_workflow,
                action_registry=action_registry,
                search_service=search_service,
            )
            if isinstance(repaired_workflow, Workflow):
                current_workflow = repaired_workflow
            else:
                current_workflow = Workflow.model_validate(repaired_workflow)
            last_good_workflow = current_workflow
        except PydanticValidationError:
            print(
                "[plan_workflow_with_two_pass] 警告：repair_workflow_with_llm 返回的结构不包含合法的 nodes/edges，本轮修复结果被忽略。"
            )

    return last_good_workflow


