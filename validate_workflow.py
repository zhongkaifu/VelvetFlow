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
from velvetflow.verification import precheck_loop_body_graphs, validate_completed_workflow


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


def validate_workflow_data(
    workflow_raw: Any, action_registry: List[dict[str, Any]]
) -> List[ValidationError]:
    """Validate workflow DSL structure and static rules.

    The function first runs Pydantic schema validation, then applies the
    planner's static rules (action references, graph connectivity, bindings,
    etc.). All found errors are returned for reporting by the caller.
    """

    errors: List[ValidationError] = []

    precheck_errors = precheck_loop_body_graphs(workflow_raw)
    if precheck_errors:
        return precheck_errors

    try:
        workflow_model = Workflow.model_validate(workflow_raw)
    except PydanticValidationError as exc:  # pragma: no cover - exercised via unit test
        return _convert_pydantic_errors(workflow_raw, exc)
    except Exception as exc:  # pragma: no cover
        return [
            ValidationError(
                code="INVALID_SCHEMA",
                node_id=None,
                field=None,
                message=str(exc),
            )
        ]

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
        / "velvetflow"
        / "business_actions.json",
        help="Path to action registry JSON file (defaults to built-in registry).",
    )
    parser.add_argument(
        "--print-normalized",
        action="store_true",
        help="Print normalized workflow JSON after successful validation.",
    )
    args = parser.parse_args(argv)

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
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"加载 Action Registry 失败: {exc}", file=sys.stderr)
        return 2

    errors = validate_workflow_data(workflow_raw, action_registry)
    if errors:
        print("校验未通过，发现以下问题：", file=sys.stderr)
        print(_format_errors(errors), file=sys.stderr)
        return 1

    print("workflow DSL 校验通过。")
    if args.print_normalized:
        normalized = Workflow.model_validate(workflow_raw).model_dump(by_alias=True)
        print(json.dumps(normalized, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
