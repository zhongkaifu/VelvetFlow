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
from typing import Optional

from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.metrics import RunManager
from velvetflow.planner import plan_workflow_with_two_pass
from velvetflow.visualization import render_workflow_dag
from velvetflow.search import (
    HybridActionSearchService,
    build_default_search_service,
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
        max_rounds=100,
        max_repair_rounds=30,
    )


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

    user_nl = prompt_requirement()

    with RunManager(workflow_name="planning") as run_manager:
        try:
            workflow = plan_workflow(user_nl)
            run_manager.workflow_name = workflow.workflow_name or run_manager.workflow_name
            run_manager.metrics.extra["planned_nodes"] = len(workflow.nodes)
        except Exception as e:
            print("\n[main] 工作流生成/补参/修复失败：", repr(e))
            print("[main] 异常类型：", type(e).__name__)
            print("[main] 需求输入：", user_nl)
            print("[main] Call stack:\n", traceback.format_exc())
            return

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
