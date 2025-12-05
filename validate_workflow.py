# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Standalone Workflow DSL validation tool.

This module exposes a small CLI that validates a workflow JSON file against
pydantic schema rules and the planner's static validation rules. It reuses the
existing DSL validation logic and produces readable error messages when issues
are found.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, List, Mapping

from velvetflow.models import PydanticValidationError, ValidationError, Workflow
from velvetflow.reference_utils import normalize_template_placeholders
from velvetflow.workflow_parser import WorkflowParseResult, parse_workflow_source
from velvetflow.verification import precheck_loop_body_graphs, validate_completed_workflow
from velvetflow.verification import generate_repair_suggestions
from velvetflow.verification.semantic_analysis import analyze_workflow_semantics


def _convert_pydantic_errors(
    workflow_raw: Any, error: PydanticValidationError
) -> List[ValidationError]:
    """Map Pydantic validation errors to generic ``ValidationError`` objects."""

    nodes = []
    if isinstance(workflow_raw, dict):
        nodes = workflow_raw.get("nodes") or []

    def _node_id_from_index(index: int):
        if 0 <= index < len(nodes):
            node = nodes[index]
            if isinstance(node, dict):
                return node.get("id")
            if hasattr(node, "id"):
                return getattr(node, "id")
        return None

    validation_errors: List[ValidationError] = []
    for err in error.errors():
        loc = err.get("loc", ()) or ()
        msg = err.get("msg", "")

        node_id = None
        field = None

        if loc:
            if loc[0] == "nodes" and len(loc) >= 2 and isinstance(loc[1], int):
                node_id = _node_id_from_index(loc[1])
                if len(loc) >= 3:
                    field = str(loc[2])
            elif loc[0] == "edges" and len(loc) >= 2 and isinstance(loc[1], int):
                if len(loc) >= 3 and isinstance(loc[-1], str):
                    field = str(loc[-1])
                else:
                    field = "edges"
            else:
                field = ".".join(str(part) for part in loc)

        validation_errors.append(
            ValidationError(
                code="INVALID_SCHEMA",
                node_id=node_id,
                field=field,
                message=msg,
            )
        )

    return validation_errors


def _load_action_registry(path: Path) -> List[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Action registry 文件不存在: {path}")

    registry_raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registry_raw, list):
        raise ValueError("Action registry 应该是 JSON 数组。")
    return registry_raw


def _format_errors(errors: Iterable[ValidationError]) -> str:
    lines = []
    for idx, err in enumerate(errors, start=1):
        location_bits = []
        if err.node_id:
            location_bits.append(f"node={err.node_id}")
        if err.field:
            location_bits.append(f"field={err.field}")
        location = f" ({', '.join(location_bits)})" if location_bits else ""
        lines.append(f"{idx}. [{err.code}]{location} {err.message}")
    return "\n".join(lines)


def _convert_parser_issues(parse_result: WorkflowParseResult) -> List[ValidationError]:
    parser_errors: List[ValidationError] = []

    for syntax_err in parse_result.syntax_errors:
        message = f"语法错误（行 {syntax_err.line}, 列 {syntax_err.column}）：{syntax_err.message}"
        if syntax_err.expected:
            message += f"；期望: {', '.join(syntax_err.expected)}"
        parser_errors.append(
            ValidationError(
                code="SYNTAX_ERROR",
                node_id=None,
                field=None,
                message=message,
            )
        )

    for issue in parse_result.grammar_issues:
        message = issue.message
        if issue.expected:
            message += f"；期望: {', '.join(issue.expected)}"
        if issue.recovery:
            message += f"（建议 {issue.recovery.description}）"
        parser_errors.append(
            ValidationError(
                code="GRAMMAR_VIOLATION",
                node_id=issue.node_id,
                field=issue.path,
                message=message,
            )
        )

    return parser_errors


def validate_workflow_data(
    workflow_raw: Any, action_registry: List[dict[str, Any]], *, parser_result: WorkflowParseResult | None = None
) -> List[ValidationError]:
    """Validate workflow DSL structure and static rules.

    The function first runs grammar-aware parsing, then Pydantic schema
    validation, then applies the planner's static rules (action references,
    graph connectivity, bindings, etc.). All found errors are returned for
    reporting by the caller.
    """

    errors: List[ValidationError] = []

    parse_result = parser_result or parse_workflow_source(workflow_raw)
    errors.extend(_convert_parser_issues(parse_result))
    if errors:
        return errors

    workflow_parsed = parse_result.ast if parse_result.ast is not None else workflow_raw
    workflow_parsed = normalize_template_placeholders(workflow_parsed)

    semantic_errors = analyze_workflow_semantics(workflow_parsed, action_registry)
    errors.extend(semantic_errors)

    precheck_errors = precheck_loop_body_graphs(workflow_parsed)
    errors.extend(precheck_errors)

    try:
        workflow_model = Workflow.model_validate(workflow_parsed)
    except PydanticValidationError as exc:  # pragma: no cover - exercised via unit test
        errors.extend(_convert_pydantic_errors(workflow_parsed, exc))
        return errors
    except Exception as exc:  # pragma: no cover
        errors.append(
            ValidationError(
                code="INVALID_SCHEMA",
                node_id=None,
                field=None,
                message=str(exc),
            )
        )
        return errors

    workflow_dict = workflow_model.model_dump(by_alias=True)
    errors.extend(validate_completed_workflow(workflow_dict, action_registry))
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a workflow DSL JSON file.")
    parser.add_argument("workflow", type=Path, help="Path to workflow JSON file")
    parser.add_argument(
        "--action-registry",
        type=Path,
        default=Path(__file__).resolve().parent
        / "tools"
        / "business_actions.json",
        help="Path to action registry JSON file (defaults to built-in registry).",
    )
    parser.add_argument(
        "--print-normalized",
        action="store_true",
        help="Print normalized workflow JSON after successful validation.",
    )
    parser.add_argument(
        "--suggest-fixes",
        action="store_true",
        help="在校验失败时输出基于 AST 模板/约束求解的修复建议。",
    )
    args = parser.parse_args(argv)

    try:
        workflow_text = args.workflow.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"找不到 workflow 文件: {args.workflow}", file=sys.stderr)
        return 2

    try:
        action_registry = _load_action_registry(args.action_registry)
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"加载 Action Registry 失败: {exc}", file=sys.stderr)
        return 2

    parser_result = parse_workflow_source(workflow_text)
    errors = validate_workflow_data(parser_result.ast or {}, action_registry, parser_result=parser_result)
    if errors:
        print("校验未通过，发现以下问题：", file=sys.stderr)
        print(_format_errors(errors), file=sys.stderr)
        if args.suggest_fixes:
            patched, suggestions = generate_repair_suggestions(
                parser_result.ast or {}, action_registry, errors=errors
            )
            if suggestions:
                print("\n自动修复建议：", file=sys.stderr)
                for idx, suggestion in enumerate(suggestions, start=1):
                    print(
                        f"{idx}. [{suggestion.strategy}] {suggestion.description}"
                        f"（路径: {suggestion.path}, 置信度: {suggestion.confidence:.2f}）",
                        file=sys.stderr,
                    )
                    print(f"    建议补丁: {suggestion.patch}", file=sys.stderr)
            else:
                print("未生成自动修复建议。", file=sys.stderr)
        return 1

    print("workflow DSL 校验通过。")
    if args.print_normalized:
        normalized = Workflow.model_validate(parser_result.ast or {}).model_dump(by_alias=True)
        print(json.dumps(normalized, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
