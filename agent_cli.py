# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""命令行入口：使用 OpenAI Agent SDK 运行 VelvetFlow workflow 工具。

示例：
    export OPENAI_API_KEY="<your_api_key>"
    python agent_cli.py --prompt "为 HR 的健康检查场景规划一个流程" \
      --action-registry tools/business_actions

本脚本会创建一个带有构建/校验/修复/更新工具的 Agent，将命令行提供的 prompt
原样发送给 Agent，并把返回内容打印到标准输出。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.workflow_agent import (
    AgentSdkNotInstalled,
    WorkflowAgentConfig,
    coerce_agent_output,
    create_workflow_agent,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run VelvetFlow workflow tools via OpenAI Agent SDK")
    parser.add_argument("--prompt", required=True, help="发送给 Agent 的自然语言指令")
    parser.add_argument(
        "--action-registry",
        type=Path,
        help="自定义动作库路径（目录或 JSON），默认使用内置 business_actions",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Agent 使用的模型名称（默认: {OPENAI_MODEL})",
    )
    parser.add_argument(
        "--max-plan-rounds",
        type=int,
        default=50,
        help="规划阶段最多尝试的回合数",
    )
    parser.add_argument(
        "--max-repair-rounds",
        type=int,
        default=3,
        help="校验失败时 LLM 自修复的最大轮次",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="当提供此参数时，直接使用规划流水线生成 workflow 并保存到指定文件",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = WorkflowAgentConfig(
        model=args.model or OPENAI_MODEL,
        action_registry_path=args.action_registry,
        max_plan_rounds=args.max_plan_rounds,
        max_repair_rounds=args.max_repair_rounds,
    )

    try:
        agent = create_workflow_agent(client=OpenAI(), config=config)
    except AgentSdkNotInstalled as exc:
        print(exc, file=sys.stderr)
        print("请安装支持 Agent 的 openai SDK (>=1.60) 或官方 agent 扩展包。", file=sys.stderr)
        return 2

    # 如果指定 --output，仍然通过 Agent 的 build_workflow 工具来生成并保存结果
    if args.output:
        try:
            response = agent.run(
                "请直接调用 build_workflow 工具生成 workflow DSL，并仅返回工具输出的 JSON。"
                f"\n需求: {args.prompt}"
            )
            normalized = coerce_agent_output(response)
        except Exception as exc:  # noqa: BLE001 - CLI surface
            print(f"通过 Agent 规划失败: {exc}", file=sys.stderr)
            return 2

        try:
            args.output.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"已保存 workflow 到 {args.output.resolve()}")
        except Exception as exc:  # noqa: BLE001 - IO surface
            print(f"写入文件失败: {exc}", file=sys.stderr)
            return 2
        return 0

    response = agent.run(args.prompt)
    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
