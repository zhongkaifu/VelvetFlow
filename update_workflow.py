"""Standalone tool for updating a workflow JSON based on a new requirement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from velvetflow.models import Workflow
from velvetflow.config import OPENAI_MODEL
from velvetflow.planner.update import update_workflow_with_llm

from validate_workflow import _format_errors, _load_action_registry, validate_workflow_data


def _resolve_requirement(args: argparse.Namespace) -> str:
    requirement = args.requirement or ""
    if args.requirement_file:
        try:
            requirement = Path(args.requirement_file).read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            print(f"找不到需求文件: {args.requirement_file}", file=sys.stderr)
            sys.exit(2)
    if not requirement:
        print("请通过 --requirement 或 --requirement-file 提供自然语言需求。", file=sys.stderr)
        sys.exit(2)
    return requirement


def _save_workflow(path: Path, workflow_dict: Any) -> None:
    normalized = Workflow.model_validate(workflow_dict).model_dump(by_alias=True)
    path.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Update a workflow JSON based on a natural language requirement",
    )
    parser.add_argument("workflow", type=Path, help="Path to the workflow JSON to update")
    parser.add_argument(
        "--requirement",
        type=str,
        help="自然语言需求描述（可使用 --requirement-file 替代）",
    )
    parser.add_argument(
        "--requirement-file",
        type=Path,
        help="包含需求描述的文本文件路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("workflow_updated.json"),
        help="输出更新后 workflow 的文件路径（默认: workflow_updated.json）",
    )
    parser.add_argument(
        "--action-registry",
        type=Path,
        default=Path(__file__).resolve().parent / "velvetflow" / "business_actions.json",
        help="动作库 JSON 路径（默认使用内置动作库）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="用于更新的 OpenAI 模型名称（默认使用 velvetflow.config.OPENAI_MODEL）",
    )
    args = parser.parse_args(argv)

    requirement = _resolve_requirement(args)

    try:
        workflow_raw = json.loads(args.workflow.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"找不到 workflow 文件: {args.workflow}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"workflow 文件不是合法 JSON: {exc}", file=sys.stderr)
        return 2

    try:
        action_registry = _load_action_registry(args.action_registry)
    except Exception as exc:  # noqa: BLE001 - CLI convenience
        print(f"加载 Action Registry 失败: {exc}", file=sys.stderr)
        return 2

    try:
        updated = update_workflow_with_llm(
            workflow_raw,
            requirement=requirement,
            action_registry=action_registry,
            model=args.model or OPENAI_MODEL,
        )
    except Exception as exc:  # noqa: BLE001 - surface model/IO issues
        print(f"更新 workflow 失败: {exc}", file=sys.stderr)
        return 2

    errors = validate_workflow_data(updated, action_registry)
    repair_round = 0
    max_rounds = 3
    while errors and repair_round < max_rounds:
        repair_round += 1
        print(
            f"更新后的 workflow 未通过校验（第 {repair_round} 次尝试修复），问题列表：",
            file=sys.stderr,
        )
        print(_format_errors(errors), file=sys.stderr)

        try:
            updated = update_workflow_with_llm(
                updated,
                requirement=requirement,
                action_registry=action_registry,
                model=args.model or OPENAI_MODEL,
                validation_errors=errors,
            )
        except Exception as exc:  # noqa: BLE001 - surface model/IO issues
            print(f"LLM 修复失败: {exc}", file=sys.stderr)
            return 2

        errors = validate_workflow_data(updated, action_registry)

    if errors:
        print("多轮修复后仍未通过校验，问题列表：", file=sys.stderr)
        print(_format_errors(errors), file=sys.stderr)
        return 1

    _save_workflow(args.output, updated)
    print(f"更新完成并已保存到 {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
