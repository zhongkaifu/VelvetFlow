# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""
build_workflow.py

原有的单文件 demo 已拆分为可复用的模块：
- velvetflow.action_registry：业务动作注册表
- velvetflow.search：混合检索相关组件
- velvetflow.planner：LLM 驱动的工作流规划与修复
- velvetflow.executor：工作流执行器

运行前：
  pip install --upgrade openai numpy
  export OPENAI_API_KEY="你的APIKey"
"""

import json
import os
import traceback
from typing import List, Optional

from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import log_error
from velvetflow.metrics import RunManager
from velvetflow.planner import plan_workflow_with_two_pass
from velvetflow.visualization import render_workflow_dag
from velvetflow.search import (
    HybridActionSearchService,
    build_default_search_service,
    get_openai_client,
)

DEFAULT_WORKFLOW_JSON = "workflow_output.json"


def create_default_search_service() -> HybridActionSearchService:
    return build_default_search_service()


def prompt_requirement() -> str:
    user_nl = input("请输入你的流程需求（直接回车使用默认示例）：\n> ").strip()
    if not user_nl:
        user_nl = (
            "每天早上 5 点，从某信息源中获取当日的若干条记录，"
            "如果存在满足特定关键字条件的记录，请对这些记录进行总结，并发送通知给我。"
        )
        print("\n使用默认示例：", user_nl)
    return user_nl


def plan_workflow(user_nl: str, search_service: Optional[HybridActionSearchService] = None):
    hybrid_searcher = search_service or create_default_search_service()
    return plan_workflow_with_two_pass(
        nl_requirement=user_nl,
        search_service=hybrid_searcher,
        action_registry=BUSINESS_ACTIONS,
        max_rounds=50,
    )


def _is_empty_workflow(workflow) -> bool:
    nodes = getattr(workflow, "nodes", None)
    return not nodes


def _suggest_requirement_additions(user_requirement: str, *, model: str = OPENAI_MODEL) -> List[str]:
    """Use an LLM to propose clarifications that make the workflow plannable."""

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是业务流程规划助手，会帮用户把含糊的需求拆解得更具体。"
                        "请基于用户输入的需求、想法和目的给出 3 条补充思路，帮助用户明确：数据来源/触发时机、"
                        "关键条件或过滤规则、具体输出/通知形式、成功判定标准。"
                        "请用简短中文短句返回 JSON 数组，不要添加其他说明。"
                    ),
                },
                {
                    "role": "user",
                    "content": user_requirement,
                },
            ],
        )

        if not resp.choices:
            raise RuntimeError("未获得补充建议")

        content = (resp.choices[0].message.content or "").strip()
        return json.loads(content)
    except Exception as exc:  # pragma: no cover - 网络/模型依赖环境
        log_error(f"[suggestion] 无法从 LLM 获取补充建议：{exc}")
        return []


def _workflow_missing_business_tools(workflow, action_registry: List[dict]) -> bool:
    nodes = getattr(workflow, "nodes", None) or []
    if not nodes:
        return True
    actions_by_id = {action.get("action_id") for action in action_registry if action.get("action_id")}
    for node in nodes:
        if getattr(node, "type", None) != "action":
            continue
        action_id = getattr(node, "action_id", None)
        if action_id and action_id in actions_by_id:
            return False
    return True


def _suggest_tool_gap_guidance(
    user_requirement: str, *, model: str = OPENAI_MODEL
) -> tuple[str, List[str]]:
    """Use an LLM to inform the user that no suitable tools were found and suggest adjustments."""

    fallback_message = "当前动作库中暂未找到适合该需求的业务工具，可能需要调整需求描述或提供可用的系统信息。"
    fallback_suggestions = [
        "补充你期望使用的业务系统/数据源名称或接口类型。",
        "说明可接受的替代流程（例如改为通知/记录/导出）。",
        "明确触发时机、输入字段与输出目标，便于匹配到可用动作。",
    ]

    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是业务流程规划助手。当前动作库里没有发现能够支撑该需求的业务工具。"
                        "请输出 JSON 对象，包含 message 和 suggestions 两个字段："
                        "message 用 1-2 句中文明确告知没有匹配的业务工具，并从用户输入的需求、想法和目的出发提示可以如何调整；"
                        "suggestions 提供 3 条修改建议（简短中文短句），必须基于用户的需求意图。"
                        "仅返回 JSON，不要添加其他说明。"
                    ),
                },
                {"role": "user", "content": user_requirement},
            ],
        )

        if not resp.choices:
            raise RuntimeError("未获得业务工具缺失建议")

        content = (resp.choices[0].message.content or "").strip()
        payload = json.loads(content)
        message = payload.get("message") if isinstance(payload, dict) else None
        suggestions = payload.get("suggestions") if isinstance(payload, dict) else None

        normalized_message = message.strip() if isinstance(message, str) and message.strip() else fallback_message
        normalized_suggestions = (
            [item for item in suggestions if isinstance(item, str) and item.strip()]
            if isinstance(suggestions, list)
            else fallback_suggestions
        )
        if not normalized_suggestions:
            normalized_suggestions = fallback_suggestions
        return normalized_message, normalized_suggestions
    except Exception as exc:  # pragma: no cover - 网络/模型依赖环境
        log_error(f"[tool-gap] 无法从 LLM 获取业务工具缺失建议：{exc}")
        return fallback_message, fallback_suggestions


def prompt_additional_requirement(suggestions: List[str]) -> str:
    while True:
        if suggestions:
            print("\n为便于规划，请补充更多细节，可参考：")
            for idx, idea in enumerate(suggestions, start=1):
                print(f"  {idx}. {idea}")

        extra = input("\n请补充更具体的需求或限制条件：\n> ").strip()
        if extra:
            return extra
        print("请提供非空的补充信息，帮助完善工作流规划。")


def persist_workflow(workflow, json_path: str, dag_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(workflow.model_dump(by_alias=True), f, indent=2, ensure_ascii=False)
    print(f"\n已将工作流以 JSON 格式保存至：{json_path}")

    saved_path = render_workflow_dag(workflow, output_path=dag_path)
    print(f"已将最终工作流 DAG 保存为 JPEG：{saved_path}")


def main():
    print("=== 通用业务 + Hybrid Action Registry + 覆盖度校验 + 自修复多轮 function-calling Demo ===")
    if not os.environ.get("OPENAI_API_KEY"):
        print("请先设置环境变量 OPENAI_API_KEY 再运行。")
        return

    base_requirement = prompt_requirement()
    combined_requirement = base_requirement
    supplements: list[str] = []

    with RunManager(workflow_name="planning") as run_manager:
        while True:
            try:
                workflow = plan_workflow(combined_requirement)
                run_manager.workflow_name = workflow.workflow_name or run_manager.workflow_name
                run_manager.metrics.extra["planned_nodes"] = len(workflow.nodes)
            except Exception as e:
                print("\n[main] 工作流生成/补参/修复失败：", repr(e))
                print("[main] 异常类型：", type(e).__name__)
                print("[main] 需求输入：", combined_requirement)
                print("[main] Call stack:\n", traceback.format_exc())
                return

            needs_tool_guidance = _workflow_missing_business_tools(workflow, BUSINESS_ACTIONS)
            if not _is_empty_workflow(workflow) and not needs_tool_guidance:
                break

            if needs_tool_guidance:
                message, suggestions = _suggest_tool_gap_guidance(combined_requirement)
                if message:
                    print(f"\n{message}")
            else:
                suggestions = _suggest_requirement_additions(combined_requirement)

            extra_requirement = prompt_additional_requirement(suggestions)
            supplements.append(extra_requirement)
            supplement_lines = [f"补充信息 {idx + 1}：{item}" for idx, item in enumerate(supplements)]
            combined_requirement = f"{base_requirement}\n\n" + "\n".join(supplement_lines)

    print("\n==== 最终用于保存的 Workflow DSL ====\n")
    print(json.dumps(workflow.model_dump(by_alias=True), indent=2, ensure_ascii=False))

    json_path = os.path.join(os.getcwd(), DEFAULT_WORKFLOW_JSON)
    dag_path = os.path.join(os.getcwd(), "workflow_dag.jpg")

    try:
        persist_workflow(workflow, json_path=json_path, dag_path=dag_path)
        print("\n现在可以使用 execute_workflow.py 从保存的 JSON 执行该流程。")
    except Exception as e:
        print("\n[warning] 工作流持久化失败：", repr(e))
        print("[warning] 异常类型：", type(e).__name__)
        print("[warning] 消息：", str(e))
        print("[warning] Call stack:\n", traceback.format_exc())


if __name__ == "__main__":
    main()
