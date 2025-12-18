# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Command-line entrypoint for the unified workflow planning Agent.

This tool drives the unified Agent/Runner pipeline that can build the
workflow structure, fill parameters, validate, repair, and update the
workflow as needed. The resulting workflow DSL will be persisted to disk.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from velvetflow.action_registry import BUSINESS_ACTIONS
from velvetflow.config import OPENAI_MODEL
from velvetflow.planner.unified_agent import run_workflow_planning_agent
from velvetflow.search import build_default_search_service
from velvetflow.visualization import render_workflow_dag

DEFAULT_OUTPUT_JSON = "workflow_unified_output.json"
DEFAULT_OUTPUT_DAG = "workflow_unified_dag.jpg"


def _prompt_requirement(default_text: str) -> str:
    user_nl = input("请输入你的流程需求（直接回车使用默认示例）：\n> ").strip()
    if not user_nl:
        user_nl = default_text
        print("\n使用默认示例：", user_nl)
    return user_nl


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the unified workflow planning Agent.")
    parser.add_argument(
        "--requirement",
        type=str,
        help="自然语言需求描述，不提供时会提示输入（回车使用示例）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_JSON,
        help=f"输出 workflow JSON 文件路径（默认: {DEFAULT_OUTPUT_JSON})",
    )
    parser.add_argument(
        "--dag",
        type=str,
        default=DEFAULT_OUTPUT_DAG,
        help=f"输出 workflow DAG 图片路径（默认: {DEFAULT_OUTPUT_DAG})",
    )
    args = parser.parse_args(argv)

    if not os.environ.get("OPENAI_API_KEY"):
        print("请先设置环境变量 OPENAI_API_KEY 再运行。")
        return 1

    default_requirement = (
        "每天早上 5 点，从某信息源中获取当日的若干条记录，"
        "如果存在满足特定关键字条件的记录，请对这些记录进行总结，并发送通知给我。"
    )
    requirement = args.requirement or _prompt_requirement(default_requirement)

    search_service = build_default_search_service()

    try:
        workflow = run_workflow_planning_agent(
            nl_requirement=requirement,
            action_registry=BUSINESS_ACTIONS,
            search_service=search_service,
            model=OPENAI_MODEL,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        print("\n[unified_agent] 工作流规划失败：", repr(exc))
        return 1

    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(workflow.model_dump(by_alias=True), f, indent=2, ensure_ascii=False)
        print(f"\n已将工作流以 JSON 格式保存至：{args.output}")

        dag_path = render_workflow_dag(workflow, output_path=args.dag)
        print(f"已将最终工作流 DAG 保存为 JPEG：{dag_path}")
    except Exception as exc:  # pragma: no cover - CLI surface
        print("\n[warning] 工作流持久化失败：", repr(exc))
        return 1

    print("\n现在可以使用 execute_workflow.py 从保存的 JSON 执行该流程。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
