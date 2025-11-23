"""
workflow_fc_hybrid_actions_demo.py

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

from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.executor import DynamicActionExecutor, load_simulation_data
from velvetflow.planner import plan_workflow_with_two_pass
from velvetflow.search import (
    GLOBAL_VOCAB,
    FakeElasticsearch,
    HybridActionSearchService,
    VectorClient,
    embed_text_local,
)


def build_default_search_service() -> HybridActionSearchService:
    fake_es = FakeElasticsearch(BUSINESS_ACTIONS)
    dim = len(GLOBAL_VOCAB)
    vec_client = VectorClient(dim=dim)

    for action in BUSINESS_ACTIONS:
        text = (
            (action.get("name", "") or "")
            + " "
            + (action.get("description", "") or "")
            + " "
            + (action.get("domain", "") or "")
            + " "
            + " ".join(action.get("tags", []) or [])
        )
        emb = embed_text_local(text)
        vec_client.upsert(action["action_id"], emb)

    return HybridActionSearchService(
        es=fake_es,
        vector_client=vec_client,
        embed_fn=embed_text_local,
        alpha=0.6,
    )


def main():
    print("=== 通用业务 + Hybrid Action Registry + 覆盖度校验 + 自修复多轮 function-calling Demo ===")
    if not os.environ.get("OPENAI_API_KEY"):
        print("请先设置环境变量 OPENAI_API_KEY 再运行。")
        return

    user_nl = input("请输入你的流程需求（直接回车使用默认示例）：\n> ").strip()
    if not user_nl:
        user_nl = (
            "每天早上 5 点，从某信息源中获取当日的若干条记录，"
            "如果存在满足特定关键字条件的记录，请对这些记录进行总结，并发送通知给我。"
        )
        print("\n使用默认示例：", user_nl)

    hybrid_searcher = build_default_search_service()

    try:
        workflow = plan_workflow_with_two_pass(
            nl_requirement=user_nl,
            search_service=hybrid_searcher,
            action_registry=BUSINESS_ACTIONS,
            max_rounds=10,
            max_repair_rounds=3,
        )
    except Exception as e:
        print("\n[main] 工作流生成/补参/修复失败：", e)
        return

    print("\n==== 最终用于执行的 Workflow DSL ====\n")
    print(json.dumps(workflow, indent=2, ensure_ascii=False))

    simulation_path = os.path.join(os.path.dirname(__file__), "velvetflow", "simulation_data.json")
    simulation_data = load_simulation_data(simulation_path)

    executor = DynamicActionExecutor(workflow, simulations=simulation_data)
    executor.run()


if __name__ == "__main__":
    main()
