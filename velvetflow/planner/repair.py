# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""LLM-driven repair helpers used when validation fails."""

import asyncio
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
import itertools
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    log_debug,
    log_error,
    log_info,
    log_success,
    log_warn,
)
from velvetflow.models import Node, PydanticValidationError, ValidationError, Workflow
from velvetflow.planner.agent_runtime import Agent, Runner, function_tool
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.repair_tools import apply_repair_tool
from velvetflow.verification import (
    generate_repair_suggestions,
    precheck_loop_body_graphs,
    validate_completed_workflow,
)


def _convert_pydantic_errors(
    workflow_raw: Any, error: PydanticValidationError
) -> List[ValidationError]:
    """Map Pydantic validation errors to generic ValidationError objects."""

    nodes = []
    if isinstance(workflow_raw, dict):
        nodes = workflow_raw.get("nodes") or []

    def _node_id_from_index(index: int) -> Optional[str]:
        if 0 <= index < len(nodes):
            node = nodes[index]
            if isinstance(node, dict):
                return node.get("id")
            if isinstance(node, Node):
                return node.id
        return None

    validation_errors: List[ValidationError] = []
    for err in error.errors():
        loc = err.get("loc", ()) or ()
        msg = err.get("msg", "")

        node_id: Optional[str] = None
        field: Optional[str] = None

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


def _looks_like_patch(text: str) -> bool:
    """Return True if a string resembles a unified diff patch."""

    stripped = text.lstrip()
    return stripped.startswith("diff --git") or stripped.startswith("--- ") or stripped.startswith("*** ")


def _rewrite_patch_paths(patch_text: str, target_path: Path) -> str:
    """Rewrite patch file headers to point to the temporary workflow path."""

    target_name = target_path.name
    old_name = f"a/{target_name}"
    new_name = f"b/{target_name}"
    rewritten_lines = []
    for line in patch_text.splitlines():
        if line.startswith("diff --git"):
            rewritten_lines.append(f"diff --git {old_name} {new_name}")
        elif line.startswith("--- "):
            rewritten_lines.append(f"--- {old_name}")
        elif line.startswith("+++ "):
            rewritten_lines.append(f"+++ {new_name}")
        else:
            rewritten_lines.append(line)

    if patch_text.endswith("\n"):
        return "\n".join(rewritten_lines) + "\n"
    return "\n".join(rewritten_lines)


def _apply_patch_output(workflow: Mapping[str, Any], patch_text: str) -> Optional[Dict[str, Any]]:
    """Apply a unified diff patch to the serialized workflow using git."""

    workflow_json = json.dumps(workflow, ensure_ascii=False, indent=2, sort_keys=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        workflow_path = Path(tmpdir) / "workflow.json"
        workflow_path.write_text(workflow_json, encoding="utf-8")

        rewritten_patch = _rewrite_patch_paths(patch_text, workflow_path)
        result = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", "--unsafe-paths", "-"],
            input=rewritten_patch,
            text=True,
            capture_output=True,
            cwd=workflow_path.parent,
        )

        if result.returncode != 0:
            log_error("[repair_workflow_with_llm] 无法应用补丁，git apply 失败。")
            if result.stdout or result.stderr:
                log_debug(
                    "[repair_workflow_with_llm] git apply 输出:\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )

            manual = _apply_unified_diff_text(workflow_json, patch_text)
            if manual is None:
                return None

            try:
                return json.loads(manual)
            except json.JSONDecodeError:
                log_error("[repair_workflow_with_llm] 补丁应用后不是合法 JSON，忽略该补丁。")
                return None

        patched_content = workflow_path.read_text(encoding="utf-8")

    try:
        return json.loads(patched_content)
    except json.JSONDecodeError:
        log_error("[repair_workflow_with_llm] 补丁应用后不是合法 JSON，忽略该补丁。")
    return None


_HUNK_HEADER_RE = re.compile(r"@@ -(?P<old_start>\d+)(?:,(?P<old_len>\d+))? +\+(?P<new_start>\d+)(?:,(?P<new_len>\d+))? @@")


def _apply_unified_diff_text(original: str, patch_text: str) -> Optional[str]:
    """Fallback parser to apply simple unified diff patches in pure Python."""

    original_lines = original.splitlines(keepends=True)
    patched_lines: List[str] = []
    idx = 0

    lines_iter = iter(patch_text.splitlines())
    for line in lines_iter:
        if not line.startswith("@@ "):
            continue

        match = _HUNK_HEADER_RE.match(line)
        if not match:
            return None

        old_start = int(match.group("old_start")) - 1

        if old_start > len(original_lines):
            return None

        patched_lines.extend(original_lines[idx:old_start])
        idx = old_start

        for hunk_line in lines_iter:
            if hunk_line.startswith("@@ "):
                lines_iter = itertools.chain([hunk_line], lines_iter)
                break
            if hunk_line.startswith("diff --git") or hunk_line.startswith("--- ") or hunk_line.startswith("+++ "):
                continue
            if hunk_line.startswith(" "):
                content = hunk_line[1:]
                if idx >= len(original_lines) or original_lines[idx].rstrip("\n") != content:
                    return None
                patched_lines.append(original_lines[idx])
                idx += 1
            elif hunk_line.startswith("-"):
                content = hunk_line[1:]
                if idx >= len(original_lines) or original_lines[idx].rstrip("\n") != content:
                    return None
                idx += 1
            elif hunk_line.startswith("+"):
                patched_lines.append(hunk_line[1:] + "\n")
            elif hunk_line == "":
                patched_lines.append("\n")
            else:
                return None
        else:
            pass

    patched_lines.extend(original_lines[idx:])

    return "".join(patched_lines)


def _make_failure_validation_error(message: str) -> ValidationError:
    return ValidationError(
        code="INVALID_SCHEMA", node_id=None, field=None, message=message
    )


def _summarize_validation_errors_for_llm(
    errors: Sequence[ValidationError],
    *,
    workflow: Mapping[str, Any] | None = None,
    action_registry: Sequence[Mapping[str, Any]] | None = None,
    previous_attempts: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Convert validation errors to an LLM-friendly, human-readable summary."""

    if not errors:
        return "未提供可用的错误信息。"

    lines: List[str] = [
        "请逐条修复下列校验错误（必须全部清零才能继续，系统只有有限的自动修复轮次）："
    ]
    loop_hints: List[str] = []
    schema_hints: List[str] = []
    repair_prompts: List[str] = []
    exploration_prompts: List[str] = []

    action_map: Dict[str, Mapping[str, Any]] = {}
    node_to_action: Dict[str, str] = {}
    previous_attempts = previous_attempts or {}

    def _make_error_key(e: ValidationError) -> str:
        return f"{e.code}:{e.node_id or 'global'}:{e.field or 'global'}"
    if action_registry:
        action_map = {
            str(a.get("action_id")): a
            for a in action_registry
            if isinstance(a, Mapping) and a.get("action_id")
        }

    if workflow and isinstance(workflow, Mapping):
        for n in workflow.get("nodes", []) or []:
            if not isinstance(n, Mapping):
                continue
            node_id = n.get("id")
            action_id = n.get("action_id") if n.get("type") == "action" else None
            if isinstance(node_id, str) and isinstance(action_id, str):
                node_to_action[node_id] = action_id
    for idx, err in enumerate(errors, start=1):
        locations = []
        if err.node_id:
            locations.append(f"节点 {err.node_id}")
        if err.field:
            locations.append(f"字段 {err.field}")

        location = f"（{', '.join(locations)}）" if locations else ""
        lines.append(f"{idx}. [{err.code}]{location}：{err.message}")

        # 提供针对 loop 节点的额外修复提示，帮助模型快速定位核心问题
        if err.field and (
            "loop_kind" in err.field
            or err.field.startswith("exports")
            or err.field in {"source", "item_alias"}
        ):
            loop_hints.append(
                "- loop 节点需同时包含 params.loop_kind（for_each/while）、params.source、"
                "params.item_alias；exports 需使用 {key: Jinja表达式} 结构暴露循环体字段，表达式必须引用 body_subgraph 节点字段"
            )

        if err.node_id and err.field:
            action_id = node_to_action.get(err.node_id)
            action_def = action_map.get(action_id) if action_id else None
            properties = (
                action_def.get("arg_schema", {}).get("properties")
                if isinstance(action_def, Mapping)
                else None
            )
            if isinstance(properties, Mapping) and err.field in properties:
                field_schema = properties.get(err.field)
                if isinstance(field_schema, Mapping):
                    expected_type = field_schema.get("type") or field_schema.get("anyOf")
                    schema_hints.append(
                        f"- 动作 {action_id} 的字段 {err.field} 期望类型/结构：{expected_type}，请按 schema 修正。"
                    )

        prompt = _ERROR_TYPE_PROMPTS.get(err.code)
        if prompt:
            repair_prompts.append(f"- [{err.code}] {prompt}")
        else:
            # 当没有固定的修复模板时，提供可探索的方向，帮助 LLM 主动推理修复思路
            directions: List[str] = [
                "检查节点/字段的引用是否存在或命名一致，必要时重建引用路径。",
                "对照可用的 action schema 或上游输出，确认类型与结构是否兼容。",
                "如果依赖缺失，可使用工具为该节点补充上游或拆分节点降低耦合。",
            ]
            scope = "、".join(
                part
                for part in [
                    f"节点 {err.node_id}" if err.node_id else "",
                    f"字段 {err.field}" if err.field else "",
                ]
                if part
            )
            scope_hint = f"（定位 {scope}）" if scope else ""
            exploration_prompts.append(
                f"- [{err.code}] 暂无固定修复模板{scope_hint}，建议沿以下方向探索：{' '.join(directions)}"
            )

        if (
            err.code == "SCHEMA_MISMATCH"
            and err.message
            and re.search(r"loop[^']*\.item\.\w+", err.message)
        ):
            loop_ref = err.node_id or "loop 节点"
            match = re.search(r"([\w-]+)\.item\.\w+", err.message)
            if match:
                loop_ref = match.group(1)
            repair_prompts.append(
                "- [SCHEMA_MISMATCH] loop item 引用修复：如果在 loop 体内使用，请改为"
                " `loop.item.<字段>` 并确保 params.source 的元素 schema 含该字段；"
                "如果在 loop 外部引用，请改为 `result_of.<loop_id>.exports.<key>`。".replace(
                    "<loop_id>", str(loop_ref)
                )
            )

        history = previous_attempts.get(_make_error_key(err))
        if history:
            lines.append("    历史修复尝试：")
            for h in history:
                lines.append(f"    - {h}")

    if schema_hints:
        lines.append("")
        lines.append("Schema 提示（按提示修改可减少重复校验）：")
        lines.extend(sorted(set(schema_hints)))

    if loop_hints:
        lines.append("")
        lines.append("Loop 修复提示（补齐必填字段/字段名需与 source schema 对齐）：")
        lines.extend(sorted(set(loop_hints)))

    guidance_prompts = sorted(set(repair_prompts + exploration_prompts))
    if guidance_prompts:
        lines.append("")
        lines.append("错误特定提示/修复指导（请结合上下文尝试以下思路）：")
        lines.extend(guidance_prompts)

    return "\n".join(lines)


_ERROR_TYPE_PROMPTS: Dict[str, str] = {
    "MISSING_REQUIRED_PARAM": "Fill in params.<field>; prefer binding upstream outputs that match arg_schema or use schema defaults. Example: if an action requires params.prompt but it is missing, bind result.output from an upstream copywriting node.",
    "UNKNOWN_ACTION_ID": "Replace with an action_id that exists in the Action Registry, keeping the node's intent closest and fixing params accordingly. Example: change an unknown action_id to text.generate in the registry and keep parameters such as prompt/temperature.",
    "UNKNOWN_PARAM": "Remove or rename params fields not declared in arg_schema so they align with the schema. Example: if the action schema only accepts prompt/temperature, delete an unexpected max_token field.",
    "DISCONNECTED_GRAPH": "Connect nodes with no in-degree/out-degree so every node sits on a reachable path. Example: add an edge from a prior generation node to an isolated summary node and route its result to downstream aggregation nodes.",
    "INVALID_EDGE": "Fix edge.from/edge.to to reference existing nodes and keep condition fields valid. Example: if edge.to points to missing node-3, change it to an existing reviewer node.",
    "SCHEMA_MISMATCH": "Adjust bound fields or aggregation so types are compatible with upstream output_schema/arg_schema. Example: if upstream output is a list but the current node expects a string, join the list or pick a single element.",
    "INVALID_SCHEMA": "Fill or correct field types per the DSL structure (nodes/edges/params) to avoid missing JSON pieces. Example: if the workflow lacks an edges field, add an empty array and ensure nodes is a list.",
    "INVALID_LOOP_BODY": "Fill loop.body_subgraph.nodes (at least one action) and ensure exports use the {key: Jinja expression} shape referencing body_subgraph node fields.",
    "STATIC_RULES_SUMMARY": "Follow the static rules summary step by step, prioritizing connectivity and schema mismatches. Example: fix disconnected edges first per the summary, then fill missing required params.",
    "EMPTY_PARAM_VALUE": "Populate empty parameters with non-empty values or bindings; avoid empty strings/objects. Example: change an empty params.prompt to a concrete prompt or bind it to an upstream result.",
    "EMPTY_PARAMS": "params cannot be an empty object—provide values or references for required fields and use tools to fill uncertain parts. Example: if a node type is action but params is {}, fill required fields like prompt/template according to arg_schema.",
    "SELF_REFERENCE": "Remove a node's reference to its own result_of.<node>; use upstream outputs instead, or split the node and apply tools. Example: if node-a.params.input references result_of.node-a, change it to result_of.previous-node.",
    "SYNTAX_ERROR": "Fix DSL syntax (brackets, commas, quotes, etc.) so the parser can build the AST. Example: add missing quotes or commas to make the JSON parseable.",
    "GRAMMAR_VIOLATION": "Rearrange nodes/fields according to grammar constraints, respecting expected tokens/productions. Example: move a node definition mistakenly placed in edges back to the nodes list.",
}


def apply_rule_based_repairs(
    workflow_raw: Mapping[str, Any],
    action_registry: Sequence[Mapping[str, Any]],
    validation_errors: Sequence[ValidationError],
) -> Tuple[Dict[str, Any], List[ValidationError]]:
    """Run deterministic repairs before falling back to LLM fixes.

    The repair logic relies on parser-backed AST cloning plus schema-driven templates
    and constraint solving to patch common issues. Remaining validation errors are
    returned so callers can decide whether an LLM hand-off is still required.
    """

    patched_workflow, _suggestions = generate_repair_suggestions(
        workflow_raw, action_registry, errors=validation_errors
    )
    if any(err.code == "UNDEFINED_REFERENCE" for err in validation_errors):
        patched_workflow, drop_summary = apply_repair_tool(
            "drop_invalid_references", patched_workflow, remove_edges=True
        )
        log_info(
            "[AutoRepair] UNDEFINED_REFERENCE 已自动清理，待提交给 LLM 分析。",
            f"summary={drop_summary}",
        )
    try:
        remaining_errors: List[ValidationError] = []
        remaining_errors.extend(precheck_loop_body_graphs(patched_workflow))
        remaining_errors.extend(
            validate_completed_workflow(patched_workflow, list(action_registry))
        )
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] 规则修复后重跑校验失败：{exc}")
        remaining_errors = list(validation_errors)

    return patched_workflow, remaining_errors


def _safe_repair_invalid_loop_body(workflow_raw: Mapping[str, Any]) -> Workflow:
    """Best-effort patch for loop body graphs with missing nodes.

    当 loop.body_subgraph 缺失节点时，自动补一个占位 action，避免后续校验直接失败。
    """

    workflow_dict: Dict[str, Any] = dict(workflow_raw)
    nodes = workflow_dict.get("nodes") if isinstance(workflow_dict.get("nodes"), list) else []

    for node in nodes:
        if not isinstance(node, Mapping) or node.get("type") != "loop":
            continue

        params = node.get("params") if isinstance(node.get("params"), dict) else {}
        node["params"] = params

        body = params.get("body_subgraph") if isinstance(params, Mapping) else None
        if not isinstance(body, dict):
            body = {}

        body_nodes = body.get("nodes") if isinstance(body.get("nodes"), list) else []

        if not body_nodes:
            action_id = f"{node.get('id')}_body_action"
            body_nodes = [
                {"id": action_id, "type": "action"},
            ]

        body["nodes"] = body_nodes
        params["body_subgraph"] = body
        node["params"] = params

    workflow_dict["nodes"] = nodes
    return Workflow.model_validate(workflow_dict)


def _repair_with_llm_and_fallback(
    *,
    broken_workflow: Dict[str, Any],
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
    search_service,
    reason: str,
    previous_attempts: Mapping[str, Sequence[str]] | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Workflow:
    log_info(f"[AutoRepair] {reason}，将错误上下文提交给 LLM 尝试修复。")

    patched_workflow, remaining_errors = apply_rule_based_repairs(
        broken_workflow, action_registry, validation_errors
    )
    if not remaining_errors:
        log_success("[AutoRepair] 规则/模板修复已清零错误，无需进入 LLM。")
        return Workflow.model_validate(patched_workflow)

    if len(remaining_errors) < len(validation_errors):
        log_info(
            f"[AutoRepair] 规则修复降低错误数量：{len(validation_errors)} -> {len(remaining_errors)}"
        )

    broken_workflow = patched_workflow
    validation_errors = list(remaining_errors)

    error_summary = _summarize_validation_errors_for_llm(
        validation_errors,
        workflow=broken_workflow,
        action_registry=action_registry,
        previous_attempts=previous_attempts,
    )
    try:
        repaired_raw = repair_workflow_with_llm(
            broken_workflow=broken_workflow,
            validation_errors=validation_errors,
            action_registry=action_registry,
            error_summary=error_summary,
            previous_failed_attempts=previous_attempts,
            model=OPENAI_MODEL,
            progress_callback=progress_callback,
        )
        repaired = Workflow.model_validate(repaired_raw)
        repaired = ensure_registered_actions(
            repaired,
            action_registry=action_registry,
            search_service=search_service,
        )
        if not isinstance(repaired, Workflow):
            repaired = Workflow.model_validate(repaired)
        log_success("[AutoRepair] LLM 修复成功，继续构建 workflow。")
        return repaired
    except Exception as err:  # noqa: BLE001
        log_warn(f"[AutoRepair] LLM 修复失败：{err}")
        try:
            fallback = _safe_repair_invalid_loop_body(broken_workflow)
            fallback = ensure_registered_actions(
                fallback,
                action_registry=action_registry,
                search_service=search_service,
            )
            if not isinstance(fallback, Workflow):
                fallback = Workflow.model_validate(fallback)
            log_warn("[AutoRepair] 使用未修复的结构作为回退，继续后续流程。")
            return fallback
        except Exception as inner_err:  # noqa: BLE001
            log_error(
                "[AutoRepair] 回退到原始结构失败，保留失败的结构以便继续修复："
                f"{inner_err}"
            )
            # Preserve the last broken structure so the outer planner loop can keep
            # attempting repairs instead of returning a synthetic empty workflow.
            if isinstance(broken_workflow, Workflow):
                return broken_workflow

            if isinstance(broken_workflow, Mapping):
                return Workflow(
                    workflow_name=str(
                        broken_workflow.get("workflow_name", "unfixed_workflow")
                    ),
                    description=str(broken_workflow.get("description", "")),
                    nodes=broken_workflow.get("nodes") or [],
                )

            # If we cannot introspect the broken payload, fall back to a minimal
            # structure with the original data attached for future repairs.
            return Workflow(workflow_name="unfixed_workflow", nodes=[broken_workflow])


def repair_workflow_with_llm(
    broken_workflow: Dict[str, Any],
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
    error_summary: str | None = None,
    previous_failed_attempts: Mapping[str, Sequence[str]] | None = None,
    model: str = OPENAI_MODEL,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Dict[str, Any]:
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("请先设置 OPENAI_API_KEY 环境变量再进行自动修复。")

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

    system_prompt = """
You are a workflow repair assistant.
[Workflow DSL syntax and semantics (must follow)]
- workflow = {workflow_name, description, nodes: []}. Only return valid JSON (edges will be auto-inferred by the system from node bindings; do not generate them).
- Node base structure: {id, type, display_name, params, action_id?, out_params_schema?, loop/subgraph/branches?}.
  type only allows action/condition/loop/parallel. No need for start/end/exit nodes.
  action nodes must include action_id (from the action registry) and params; only action nodes may have out_params_schema.
  condition nodes must include params.expression that returns a boolean (valid Jinja expression) and true_to_node/false_to_node (string or null).
  loop nodes include loop_kind/iter/source/body_subgraph/exports. The value of exports must reference fields of nodes inside body_subgraph (e.g., {{ result_of.node.field }}). Outside the loop, only exports.<key> can be referenced. body_subgraph only contains a nodes array.
  loop.body_subgraph does not need and must not explicitly declare edges, entry, or exit nodes—remove them if present.
  - params must directly use Jinja expressions to reference upstream results (e.g., {{ result_of.<node_id>.<field_path> }} or {{ loop.item.xxx }}); do not output object-style bindings.
    <node_id> must exist and the fields must align with upstream output_schema or loop.exports.
  You have a workflow JSON and a set of structured validation_errors.
  validation_errors is a JSON array whose elements contain code/node_id/field/message.
  These errors come from:
  - Missing action parameters or mismatches with arg_schema
  - Incomplete condition definitions
  - source paths referencing nonexistent nodes
  - source paths not matching the upstream action's output_schema
  - Fields missing from the array element schema referenced by source

  - previous_failed_attempts records past repair attempts for the same error. Avoid repeating failed methods and try different fix paths.

  - workflow structure not conforming to DSL schema (e.g., invalid node type)

  Overall goal: repair these errors so the workflow passes static validation while “changing the overall workflow structure as little as possible.”

  Specific requirements (very important, follow strictly):
 1. Keep the structure stable:
    - Do not add or delete nodes;
    - edges will be auto-derived by the system based on node references. Do not manually add/remove or adjust conditions.

 2. Action node repair priority:
    - First, fill missing required fields in params or fix incorrect types based on action_schemas[action_id].arg_schema;
    - If the action_id itself is valid (exists in action_schemas), prioritize “fixing params” instead of changing action_id;
    - Only when validation_errors explicitly indicate the action_id does not exist should you consider changing action_id to a more reasonable candidate and simultaneously update that node’s params to match the new arg_schema.

 3. Condition node repair:
    - Ensure params.expression exists and is a valid Jinja boolean expression;
    - expression should directly express the logic. To reference upstream results use {{ result_of.<node>.<field_path> }} or {{ loop.item.xxx }};
    - Do not generate kind/field/op/value fields; all conditions must be within expression.

 4. loop nodes: ensure loop_kind (for_each/while) and source are valid, and expose body results via exports (expressions must reference body_subgraph node fields). Downstream may only reference result_of.<loop_id>.exports.<key>.
 5. parallel nodes: branches must be a non-empty array.

6. Parameter binding repairs:
   - Only output Jinja expressions or literals; do not fall back to object-style bindings;
   - Put aggregation/filtering/concatenation logic directly in Jinja expressions or filters and ensure field paths align with upstream schemas;
   - Common error: downstream references missing the exports segment—add result_of.<loop_id>.exports.<key>.

 7. Minimize the scope of changes:
    - When multiple fixes exist, prefer the minimal change closest to the original intent (e.g., rename a single field instead of rewriting the entire params).
    - When validation_error_summary provides schema hints or path information, prioritize adjusting field types/structures per the hint to avoid repeating mistakes.

 8. Repair rounds are limited and must close the loop:
    - The system revalidates within limited iterations; if you ignore any validation_error, the process may proceed to the next round or terminate.
    - Address every item in validation_errors until there are zero issues before outputting results to avoid leaving problems behind.

 9. Output requirements:
    - Keep the top-level structure workflow_name/description/nodes unchanged (only adjust node internals; edges are inferred by the system);
    - Node id/type must remain unchanged;
    - You must use the toolchain to apply modifications: adjust the workflow through repair tools and directly use the latest result—avoid natural language or Markdown code blocks.
 10. Available tools: when you need structured modifications, prioritize the provided tools (deterministic, no LLM dependency): fix_loop_body_references/fill_action_required_params/update_node_field/normalize_binding_paths/replace_reference_paths/drop_invalid_references.
"""

    working_workflow: Dict[str, Any] = broken_workflow
    def _log_tool_call(tool_name: str, payload: Mapping[str, Any] | None = None) -> None:
        if payload:
            log_info(
                f"[WorkflowRepairer] tool={tool_name}",
                json.dumps(payload, ensure_ascii=False),
            )
        else:
            log_info(f"[WorkflowRepairer] tool={tool_name}")

    def _emit_canvas_update(label: str, workflow_obj: Mapping[str, Any]) -> None:
        if not progress_callback:
            return
        try:
            progress_callback(label, workflow_obj)
        except Exception:
            log_debug(f"[WorkflowRepairer] progress_callback {label} 调用失败，已忽略。")

    def _apply_named_repair(tool_name: str, args: Mapping[str, Any]) -> Dict[str, Any]:
        nonlocal working_workflow
        patched_workflow, summary = apply_repair_tool(
            tool_name=tool_name,
            args=dict(args),
            workflow=working_workflow,
            validation_errors=validation_errors,
            action_registry=action_registry,
        )
        if summary.get("applied"):
            working_workflow = patched_workflow
        return {"workflow": working_workflow, "summary": summary}

    @function_tool(strict_mode=False)
    def fix_loop_body_references(node_id: str) -> Mapping[str, Any]:
        """修复 loop 节点 body_subgraph 内部引用错误。

        适用场景：循环内节点引用 loop.exports 或上游路径不合规时。

        Args:
            node_id: 需要修复的 loop 节点 ID。

        Returns:
            包含修复摘要与当前 workflow 的结果字典。
        """
        _log_tool_call("fix_loop_body_references", {"node_id": node_id})
        result = _apply_named_repair("fix_loop_body_references", {"node_id": node_id})
        _emit_canvas_update("fix_loop_body_references", result["workflow"])
        return result

    @function_tool(strict_mode=False)
    def fill_action_required_params(node_id: str) -> Mapping[str, Any]:
        """补全 action 节点缺失的必填参数。

        适用场景：节点参数校验缺少必填项时自动修补。

        Args:
            node_id: 需要补全的 action 节点 ID。

        Returns:
            包含修复摘要与当前 workflow 的结果字典。
        """
        _log_tool_call("fill_action_required_params", {"node_id": node_id})
        result = _apply_named_repair("fill_action_required_params", {"node_id": node_id})
        _emit_canvas_update("fill_action_required_params", result["workflow"])
        return result

    @function_tool(strict_mode=False)
    def update_node_field(node_id: str, field_path: str, value: Any) -> Mapping[str, Any]:
        """更新节点指定字段路径的值。

        适用场景：修正字段类型、表达式或分支指向等具体字段。

        Args:
            node_id: 需要更新的节点 ID。
            field_path: 字段路径（点号分隔，例如 params.expression）。
            value: 新值。

        Returns:
            包含修复摘要与当前 workflow 的结果字典。
        """
        _log_tool_call(
            "update_node_field",
            {"node_id": node_id, "field_path": field_path},
        )
        result = _apply_named_repair(
            "update_node_field", {"node_id": node_id, "field_path": field_path, "value": value}
        )
        _emit_canvas_update("update_node_field", result["workflow"])
        return result

    @function_tool(strict_mode=False)
    def normalize_binding_paths() -> Mapping[str, Any]:
        """规范化所有节点的参数绑定路径。

        适用场景：统一 Jinja 引用路径格式，修复旧 DSL 或路径缺失。

        Returns:
            包含修复摘要与当前 workflow 的结果字典。
        """
        _log_tool_call("normalize_binding_paths")
        result = _apply_named_repair("normalize_binding_paths", {})
        _emit_canvas_update("normalize_binding_paths", result["workflow"])
        return result

    @function_tool(strict_mode=False)
    def replace_reference_paths(old: str, new: str, include_edges: bool = True) -> Mapping[str, Any]:
        """批量替换 workflow 中的引用路径。

        适用场景：字段路径变更，需要全量替换引用时。

        Args:
            old: 需要替换的旧路径。
            new: 替换后的新路径。
            include_edges: 是否同时更新 edges 引用。

        Returns:
            包含修复摘要与当前 workflow 的结果字典。
        """
        _log_tool_call(
            "replace_reference_paths",
            {"old": old, "new": new, "include_edges": include_edges},
        )
        result = _apply_named_repair(
            "replace_reference_paths", {"old": old, "new": new, "include_edges": include_edges}
        )
        _emit_canvas_update("replace_reference_paths", result["workflow"])
        return result

    @function_tool(strict_mode=False)
    def drop_invalid_references(remove_edges: bool = True) -> Mapping[str, Any]:
        """移除无法解析的引用或无效依赖。

        适用场景：引用路径不存在或节点被删除导致的残留引用。

        Args:
            remove_edges: 是否同时删除失效 edges。

        Returns:
            包含修复摘要与当前 workflow 的结果字典。
        """
        _log_tool_call("drop_invalid_references", {"remove_edges": remove_edges})
        result = _apply_named_repair("drop_invalid_references", {"remove_edges": remove_edges})
        _emit_canvas_update("drop_invalid_references", result["workflow"])
        return result

    agent = Agent(
        name="WorkflowRepairer",
        instructions=system_prompt,
        tools=[
            fix_loop_body_references,
            fill_action_required_params,
            update_node_field,
            normalize_binding_paths,
            replace_reference_paths,
            drop_invalid_references,
        ],
        model=model,
    )

    run_input = json.dumps(
        {
            "workflow": broken_workflow,
            "validation_error_summary": error_summary,
            "validation_errors": [asdict(e) for e in validation_errors],
            "previous_failed_attempts": previous_failed_attempts,
            "action_schemas": action_schemas,
        },
        ensure_ascii=False,
    )

    try:
        Runner.run_sync(agent, run_input, max_turns=12)
    except TypeError:
        coro = Runner.run(agent, run_input)  # type: ignore[call-arg]
        result = coro if not asyncio.iscoroutine(coro) else asyncio.run(coro)
        _ = result

    return working_workflow


__all__ = [
    "_convert_pydantic_errors",
    "_make_failure_validation_error",
    "_repair_with_llm_and_fallback",
    "apply_rule_based_repairs",
    "repair_workflow_with_llm",
]
