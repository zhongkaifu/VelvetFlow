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
            log_error(
                "[repair_workflow_with_llm] Failed to apply patch: git apply returned a non-zero status."
            )
            if result.stdout or result.stderr:
                log_debug(
                    "[repair_workflow_with_llm] git apply output:\n"
                    f"stdout: {result.stdout}\nstderr: {result.stderr}"
                )

            manual = _apply_unified_diff_text(workflow_json, patch_text)
            if manual is None:
                return None

            try:
                return json.loads(manual)
            except json.JSONDecodeError:
                log_error(
                    "[repair_workflow_with_llm] Patched content is not valid JSON; ignoring this patch."
                )
                return None

        patched_content = workflow_path.read_text(encoding="utf-8")

    try:
        return json.loads(patched_content)
    except json.JSONDecodeError:
        log_error(
            "[repair_workflow_with_llm] Patched content is not valid JSON; ignoring this patch."
        )
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
        return "No actionable validation errors were provided."

    lines: List[str] = [
        "Resolve each validation error below (all must be cleared to continue; the system only allows a limited number of automated repair rounds):"
    ]
    loop_hints: List[str] = []
    schema_hints: List[str] = []
    repair_prompts: List[str] = []
    exploration_prompts: List[str] = []

    action_map: Dict[str, Mapping[str, Any]] = {}
    node_to_action: Dict[str, str] = {}
    missing_param_context: Dict[str, Dict[str, Any]] = {}
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
                action_def = action_map.get(action_id, {})
                arg_schema = action_def.get("arg_schema") if isinstance(action_def, Mapping) else None
                required_fields = []
                if isinstance(arg_schema, Mapping):
                    req = arg_schema.get("required")
                    if isinstance(req, list):
                        required_fields = req
                params = n.get("params") if isinstance(n.get("params"), Mapping) else {}
                missing = [
                    f
                    for f in required_fields
                    if f not in params
                    or params.get(f) is None
                    or (isinstance(params.get(f), str) and params.get(f).strip() == "")
                    or (isinstance(params.get(f), (list, dict)) and len(params.get(f)) == 0)
                ]
                if missing:
                    missing_param_context[node_id] = {
                        "action_id": action_id,
                        "missing": missing,
                        "required": required_fields,
                        "provided": sorted(params.keys()),
                    }
    for idx, err in enumerate(errors, start=1):
        locations = []
        if err.node_id:
            locations.append(f"Node {err.node_id}")
        if err.field:
            locations.append(f"Field {err.field}")

        location = f" ({', '.join(locations)})" if locations else ""
        lines.append(f"{idx}. [{err.code}]{location}: {err.message}")

        # Provide additional repair hints for loop nodes to help the model quickly locate core issues
        if err.field and (
            "loop_kind" in err.field
            or err.field.startswith("exports")
            or err.field in {"source", "item_alias"}
        ):
            loop_hints.append(
                "- Loop nodes must include params.loop_kind (for_each/while), params.source, and params.item_alias."
                " exports must use the {key: Jinja expression} structure to expose loop body fields, and expressions must reference body_subgraph node fields."
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
                        f"- Action {action_id} expects field {err.field} to match type/structure: {expected_type}; please align it with the schema."
                    )

        if err.code == "MISSING_REQUIRED_PARAM" and err.node_id:
            ctx = missing_param_context.get(err.node_id)
            if ctx:
                required_list = ", ".join(ctx.get("required", []))
                missing_list = ", ".join(ctx.get("missing", [])) or err.field
                provided_list = ", ".join(ctx.get("provided", []))
                schema_hints.append(
                    "- Action {aid} is missing required params [{missing}] (required: [{required}], provided: [{provided}]); use fill_action_required_params or update bindings to satisfy arg_schema.".format(
                        aid=ctx.get("action_id", ""),
                        missing=missing_list,
                        required=required_list,
                        provided=provided_list,
                    )
                )

        prompt = _ERROR_TYPE_PROMPTS.get(err.code)
        if prompt:
            repair_prompts.append(f"- [{err.code}] {prompt}")
        else:
            # When no fixed repair template exists, offer exploratory directions to help the LLM reason through potential fixes
            directions: List[str] = [
                "Check whether node/field references exist and are consistently named; rebuild reference paths if necessary.",
                "Compare against available action schemas or upstream outputs to ensure type and structure compatibility.",
                "If dependencies are missing, use tools to add upstream providers for this node or split nodes to reduce coupling.",
            ]
            scope = "、".join(
                part
                for part in [
                    f"Node {err.node_id}" if err.node_id else "",
                    f"Field {err.field}" if err.field else "",
                ]
                if part
            )
            scope_hint = f" (Scope: {scope})" if scope else ""
            exploration_prompts.append(
                f"- [{err.code}] No fixed repair template{scope_hint}; explore the following directions: {' '.join(directions)}"
            )

        if (
            err.code == "SCHEMA_MISMATCH"
            and err.message
            and re.search(r"loop[^']*\.item\.\w+", err.message)
        ):
            loop_ref = err.node_id or "loop node"
            match = re.search(r"([\w-]+)\.item\.\w+", err.message)
            if match:
                loop_ref = match.group(1)
            repair_prompts.append(
                "- [SCHEMA_MISMATCH] loop item 引用修复 (Loop item reference fix): if used inside the loop body, change to"
                " `loop.item.<field>` and ensure the params.source element schema contains that field;"
                " if used outside the loop, change to `result_of.<loop_id>.exports.<key>`.".replace(
                    "<loop_id>", str(loop_ref)
                )
            )

        history = previous_attempts.get(_make_error_key(err))
    if history:
        lines.append("    历史修复尝试 / Previous repair attempts:")
        for h in history:
            lines.append(f"    - {h}")

    if schema_hints:
        lines.append("")
        lines.append("Schema hints (applying these should reduce repeated validation failures):")
        lines.extend(sorted(set(schema_hints)))

    if loop_hints:
        lines.append("")
        lines.append("Loop repair hints (fill required fields and align names with the source schema):")
        lines.extend(sorted(set(loop_hints)))

    guidance_prompts = sorted(set(repair_prompts + exploration_prompts))
    if guidance_prompts:
        lines.append("")
        lines.append("错误特定提示 / Error-specific tips (use these directions with the surrounding context):")
        lines.extend(guidance_prompts)

    return "\n".join(lines)


_ERROR_TYPE_PROMPTS: Dict[str, str] = {
    "MISSING_REQUIRED_PARAM": "Fill params.<field>; prefer binding upstream outputs that satisfy arg_schema or use schema defaults. Use the fill_action_required_params tool with node_id to auto-fill placeholders before applying custom bindings. Example: if an action requires params.prompt but it is missing, bind it to result.output from an upstream copywriting node.",
    "UNKNOWN_ACTION_ID": "Replace with an action_id that exists in the Action Registry while keeping semantics similar and adjusting params accordingly. Example: change an unknown action_id to text.generate in the registry and retain prompt/temperature parameters.",
    "UNKNOWN_PARAM": "Remove or rename params fields not declared in the arg_schema so they align with the schema. Example: if the action schema only accepts prompt/temperature, delete an unexpected max_token field.",
    "DISCONNECTED_GRAPH": "Connect nodes with no inputs/outputs so every node sits on a reachable path. Example: add an edge from an upstream generation node to an isolated summary node and route its result to downstream aggregation.",
    "INVALID_EDGE": "修复边 / 连接无入度: Fix edge from/to values to reference existing nodes and keep condition fields valid. Example: if edge.to points to nonexistent node-3, change it to a real reviewer node.",
    "SCHEMA_MISMATCH": "Adjust bound fields or aggregation so types are compatible with upstream output_schema/arg_schema. Example: when upstream output is a list but the current node expects a string, join the list or pick a single element.",
    "INVALID_SCHEMA": "Complete or correct field types per the DSL structure (nodes/edges/params) to avoid missing JSON structures. Example: if the workflow lacks an edges field, add an empty array and ensure nodes is a list.",
    "INVALID_LOOP_BODY": "Fill loop.body_subgraph.nodes (at least one action) and ensure exports use the {key: Jinja expression} structure referencing body_subgraph node fields.",
    "STATIC_RULES_SUMMARY": "Follow the static rule summary step by step, prioritizing connectivity and schema mismatches. Example: fix disconnected edges first, then fill missing required params.",
    "EMPTY_PARAM_VALUE": "Populate empty parameters with non-empty values or bindings; avoid empty strings/objects. Example: change an empty params.prompt to a concrete prompt or bind upstream output.",
    "EMPTY_PARAMS": "params cannot be an empty object; provide values or references for required fields and use tools to fill uncertain entries. Example: if a node is an action but params is {}, populate prompt/template and other required fields per arg_schema.",
    "SELF_REFERENCE": "Remove a node's self reference to result_of.<node> and use upstream output or split nodes, applying changes via tools. Example: if node-a.params.input references result_of.node-a, change it to result_of.previous-node.",
    "SYNTAX_ERROR": "Fix DSL syntax (brackets, commas, quotes, etc.) so the parser can build an AST. Example: add missing quotes or commas to ensure the JSON can be parsed.",
    "GRAMMAR_VIOLATION": "Rearrange nodes/fields per grammar constraints, following the expected tokens/productions. Example: move a node definition mistakenly placed in edges back to the nodes list.",
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
            "[AutoRepair] UNDEFINED_REFERENCE was auto-cleaned; handing off to the LLM for analysis.",
            f"summary={drop_summary}",
        )
    try:
        remaining_errors: List[ValidationError] = []
        remaining_errors.extend(precheck_loop_body_graphs(patched_workflow))
        remaining_errors.extend(
            validate_completed_workflow(patched_workflow, list(action_registry))
        )
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] Validation rerun after rule-based fixes failed: {exc}")
        remaining_errors = list(validation_errors)

    return patched_workflow, remaining_errors


def _safe_repair_invalid_loop_body(workflow_raw: Mapping[str, Any]) -> Workflow:
    """Best-effort patch for loop body graphs with missing nodes.

    When loop.body_subgraph lacks nodes, automatically insert a placeholder action
    to avoid immediate validation failure in later steps.
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
    log_info(f"[AutoRepair] {reason}; submitting error context to the LLM for repair.")

    patched_workflow, remaining_errors = apply_rule_based_repairs(
        broken_workflow, action_registry, validation_errors
    )
    if not remaining_errors:
        log_success("[AutoRepair] Rule/template repairs cleared all errors; no LLM round needed.")
        return Workflow.model_validate(patched_workflow)

    if len(remaining_errors) < len(validation_errors):
        log_info(
            f"[AutoRepair] Rule-based fixes reduced error count: {len(validation_errors)} -> {len(remaining_errors)}"
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
        log_success("[AutoRepair] LLM repair succeeded; continuing workflow construction.")
        return repaired
    except Exception as err:  # noqa: BLE001
        log_warn(f"[AutoRepair] LLM repair failed: {err}")
        try:
            fallback = _safe_repair_invalid_loop_body(broken_workflow)
            fallback = ensure_registered_actions(
                fallback,
                action_registry=action_registry,
                search_service=search_service,
            )
            if not isinstance(fallback, Workflow):
                fallback = Workflow.model_validate(fallback)
            log_warn(
                "[AutoRepair] Using the unresolved structure as a fallback; continuing the workflow process."
            )
            return fallback
        except Exception as inner_err:  # noqa: BLE001
            log_error(
                f"[AutoRepair] Failed to fall back to the original structure; keeping the failed structure for continued repair: {inner_err}"
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
        raise RuntimeError(
            "Set the OPENAI_API_KEY environment variable before running auto-repair."
        )

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
- workflow = {workflow_name, description, nodes: []}; always return valid JSON (edges will be inferred from node bindings and do not need to be generated).
- Node structure: {id, type, display_name, params, action_id?, out_params_schema?, loop/subgraph/branches?}.
  type only allows action/condition/loop/parallel. No start/end/exit nodes are needed.
  Action nodes must provide action_id (from the action registry) and params; only action nodes may set out_params_schema.
  Condition nodes must include params.expression (a valid Jinja boolean expression) and true_to_node/false_to_node (string or null).
  Loop nodes contain loop_kind/iter/source/body_subgraph/exports; export values must reference fields from body_subgraph nodes (e.g., {{ result_of.node.field }}). Outside the loop you may only reference exports.<key>; body_subgraph only contains a nodes array.
  loop.body_subgraph must not declare edges, entry, or exit nodes—remove them if present.
  - params must directly use Jinja expressions to reference upstream results (e.g., {{ result_of.<node_id>.<field_path> }} or {{ loop.item.xxx }}); do not output object-style bindings.
    <node_id> must exist and its fields must align with upstream output_schema or loop.exports.
  You are given a workflow JSON and a list of structured validation errors validation_errors.
  validation_errors is a JSON array whose elements include code/node_id/field/message.
  These errors come from:
  - missing or schema-mismatched action parameters
  - incomplete condition settings
  - source paths referencing nonexistent nodes
  - source paths that do not match an upstream action's output_schema
  - source paths pointing to array element schemas that lack a referenced field

  - previous_failed_attempts records past fixes for the same error; avoid repeating failed approaches and try different repair paths.

  - workflow structure violating the DSL schema (e.g., invalid node type)

  Overall goal: fix these errors while keeping the workflow structure as stable as possible so it passes static validation.

  Detailed requirements (follow strictly):
 1. Keep the structure stable:
    - Do not add or delete nodes;
    - edges are inferred from node references; do not manually add/remove or adjust conditions.

 2. Action node repair priority:
    - First fill missing required fields in params based on action_schemas[action_id].arg_schema or correct invalid types;
    - If action_id itself is valid (exists in action_schemas), prefer "fix params" over changing action_id;
    - Only when validation_errors explicitly say the action_id is unknown should you consider switching to a better candidate and align that node's params with the new arg_schema.

 3. Condition node repair:
    - Ensure params.expression exists and is a valid Jinja boolean expression;
    - expression should encode the logic directly; when referencing upstream results use {{ result_of.<node>.<field_path> }} or {{ loop.item.xxx }};
    - Do not generate kind/field/op/value fields—fold all conditions into expression.

 4. Loop nodes: ensure loop_kind (for_each/while) and source are valid, and use exports to expose body results (expressions must reference body_subgraph node fields); downstream nodes may only reference result_of.<loop_id>.exports.<key>.
 5. Parallel nodes: branches must be a non-empty array.

6. Parameter binding repairs:
   - Output only Jinja expressions or literals; do not revert to object-style bindings;
   - Aggregation/filtering/concatenation logic should be written directly in Jinja expressions or filters, with field paths aligned to upstream schemas;
   - Common mistake: downstream references missing the exports segment—complete them as result_of.<loop_id>.exports.<key>.

 7. Minimize the change scope:
    - When multiple fixes are possible, choose the smallest change that preserves intent (e.g., rename a field instead of rewriting params).
    - When validation_error_summary provides schema hints or path details, follow them first to avoid repeated mistakes.

 8. Repair rounds are limited and must converge:
    - The system re-validates within limited rounds; if you ignore any validation_error the process proceeds to the next round or stops.
    - Address every item in validation_errors until the count is zero before outputting results.

 9. Output requirements:
    - Keep top-level workflow_name/description/nodes unchanged (only adjust node internals; edges are inferred);
    - Do not change node id/type;
    - Use the toolchain to apply changes: modify the workflow via repair tools and use the latest result directly; avoid natural language or Markdown code blocks.
 10. Available tools: when structural edits are needed, prefer the provided deterministic tools (no LLM dependency): fix_loop_body_references/fill_action_required_params/update_node_field/normalize_binding_paths/replace_reference_paths/drop_invalid_references.
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
            log_debug(
                f"[WorkflowRepairer] progress_callback {label} invocation failed and was ignored."
            )

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
        """Fix invalid references inside a loop node's body_subgraph.

        Use when: nodes inside the loop body reference loop.exports or upstream paths incorrectly.

        Args:
            node_id: ID of the loop node to repair.

        Returns:
            Result dictionary containing the repair summary and the current workflow.
        """
        _log_tool_call("fix_loop_body_references", {"node_id": node_id})
        result = _apply_named_repair("fix_loop_body_references", {"node_id": node_id})
        _emit_canvas_update("fix_loop_body_references", result["workflow"])
        return result

    @function_tool(strict_mode=False)
    def fill_action_required_params(node_id: str) -> Mapping[str, Any]:
        """Fill missing required parameters for an action node.

        Use when: validation reports missing required parameters on an action node.

        Args:
            node_id: ID of the action node that needs completion.

        Returns:
            Result dictionary containing the repair summary and the current workflow.
        """
        _log_tool_call("fill_action_required_params", {"node_id": node_id})
        result = _apply_named_repair("fill_action_required_params", {"node_id": node_id})
        _emit_canvas_update("fill_action_required_params", result["workflow"])
        return result

    @function_tool(strict_mode=False)
    def update_node_field(node_id: str, field_path: str, value: Any) -> Mapping[str, Any]:
        """Update the value at a specific field path on a node.

        Use when: adjusting field types, expressions, or branch targets for a given field.

        Args:
            node_id: ID of the node to update.
            field_path: Dot-delimited field path (for example, params.expression).
            value: New value to apply.

        Returns:
            Result dictionary containing the repair summary and the current workflow.
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
        """Normalize parameter binding paths across all nodes.

        Use when: unifying Jinja reference formats or fixing missing/legacy binding paths.

        Returns:
            Result dictionary containing the repair summary and the current workflow.
        """
        _log_tool_call("normalize_binding_paths")
        result = _apply_named_repair("normalize_binding_paths", {})
        _emit_canvas_update("normalize_binding_paths", result["workflow"])
        return result

    @function_tool(strict_mode=False)
    def replace_reference_paths(old: str, new: str, include_edges: bool = True) -> Mapping[str, Any]:
        """Bulk replace reference paths within the workflow.

        Use when: a field path changes and all references must be updated.

        Args:
            old: Path to replace.
            new: Replacement path.
            include_edges: Whether to update references on edges as well.

        Returns:
            Result dictionary containing the repair summary and the current workflow.
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
        """Remove unresolved references or invalid dependencies.

        Use when: reference paths point to missing nodes or leftover dependencies after deletions.

        Args:
            remove_edges: Whether to delete invalid edges as well.

        Returns:
            Result dictionary containing the repair summary and the current workflow.
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
