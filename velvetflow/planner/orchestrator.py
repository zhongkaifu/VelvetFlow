# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Top-level orchestrator for the two-pass planner.

This module orchestrates structure planning, parameter completion, and error repair.
It runs a two-pass reasoning flow with an LLM to draft the workflow structure and fill
parameters, then supplies validation context back to the LLM for self-repair when
needed. The main entry ``plan_workflow_with_two_pass`` handles:

* Skeleton stage: invoke the LLM with the natural-language requirement to generate
  initial nodes, ensuring Action Registry coverage.
* Parameter stage: fill each action's parameters under the agreed-upon prompt context
  and validate the structure against the ``Workflow`` Pydantic model.
* Repair stage: when validation fails, return the ``ValidationError`` list and last
  valid structure to the LLM, asking for a fully serialized workflow dictionary; if
  multiple repair rounds still fail, return the most recent valid version.

All callers receive a ``Workflow`` object and do not need to handle the raw JSON
returned by the LLM.
"""

import copy
import json
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    TraceContext,
    child_span,
    log_debug,
    log_error,
    log_event,
    log_info,
    log_json,
    log_llm_usage,
    log_section,
    log_success,
    log_warn,
    use_trace_context,
)
from velvetflow.models import PydanticValidationError, ValidationError, Workflow
from velvetflow.loop_dsl import index_loop_body_nodes, iter_workflow_and_loop_body_nodes
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.repair import (
    _convert_pydantic_errors,
    _make_failure_validation_error,
    _repair_with_llm_and_fallback,
)
from velvetflow.planner.requirement_analysis import analyze_user_requirement
from velvetflow.planner.repair_tools import (
    align_loop_body_alias_references,
    fill_loop_exports_defaults,
    fix_missing_loop_exports_items,
    normalize_binding_paths,
    _apply_local_repairs_for_unknown_params,
)
from velvetflow.planner.structure import plan_workflow_structure_with_llm
from velvetflow.verification import (
    precheck_loop_body_graphs,
    run_lightweight_static_rules,
    validate_completed_workflow,
)
from velvetflow.verification.jinja_validation import (
    normalize_condition_params_to_jinja,
    normalize_params_to_jinja,
)
from velvetflow.verification.validation import (
    _get_array_item_schema_from_output,
    _get_field_schema_from_item,
    _get_output_schema_at_path,
    _is_numeric_schema_type,
)
from velvetflow.search import HybridActionSearchService


def _index_actions_by_id(
    action_registry: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Build a quick lookup table for action metadata keyed by `action_id`."""

    return {a["action_id"]: a for a in action_registry}


def _parse_expected_format_schema(expected_format: Any) -> Optional[Dict[str, Any]]:
    """Parse expected_format into a JSON schema mapping if possible."""

    if isinstance(expected_format, Mapping):
        return dict(expected_format)

    if isinstance(expected_format, str):
        stripped = expected_format.strip()

        # Handle fenced code blocks such as ```json ... ```.
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = stripped.strip("`")
            if "\n" in stripped:
                stripped = stripped.split("\n", 1)[1]

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None

        if isinstance(parsed, Mapping):
            return dict(parsed)

    return None


def _is_missing_required_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    if isinstance(value, (list, dict)):
        return len(value) == 0
    return False


def _collect_missing_required_param_errors(
    workflow: Workflow, action_registry: List[Dict[str, Any]]
) -> List[ValidationError]:
    """Return errors for action nodes whose required params are absent or empty."""

    actions_by_id = _index_actions_by_id(action_registry)
    errors: List[ValidationError] = []

    for node in iter_workflow_and_loop_body_nodes(workflow.model_dump(by_alias=True)):
        if node.get("type") != "action":
            continue

        action_id = node.get("action_id") if isinstance(node.get("action_id"), str) else None
        action_def = actions_by_id.get(action_id) if action_id else None
        if not action_def:
            continue

        arg_schema = action_def.get("arg_schema") if isinstance(action_def, Mapping) else None
        required_fields = arg_schema.get("required") if isinstance(arg_schema, Mapping) else []
        if not isinstance(required_fields, list):
            continue

        params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
        provided_keys = ", ".join(sorted(params.keys())) if params else ""

        for field in required_fields:
            value = params.get(field)
            if field not in params or _is_missing_required_value(value):
                errors.append(
                    ValidationError(
                        code="MISSING_REQUIRED_PARAM",
                        node_id=node.get("id"),
                        field=field,
                        message=(
                            "Action node '{nid}' with tool '{aid}' is missing required "
                            "param '{field}' or it is empty; required fields: {required}; "
                            "provided: {provided}"
                        ).format(
                            nid=node.get("id"),
                            aid=action_id,
                            field=field,
                            required=", ".join(required_fields),
                            provided=provided_keys,
                        ),
                    )
                )

    return errors


def _schema_primary_type(schema: Optional[Mapping[str, Any]]) -> Any:
    if not isinstance(schema, Mapping):
        return None

    typ = schema.get("type")
    if isinstance(typ, list):
        return typ[0] if typ else None
    return typ


def _schema_types(schema: Optional[Mapping[str, Any]]) -> set[str]:
    if not isinstance(schema, Mapping):
        return set()

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return {str(t) for t in schema_type}
    if isinstance(schema_type, str):
        return {schema_type}
    return set()


def _is_empty_fallback_workflow(workflow: Workflow) -> bool:
    """Return True if we only have the synthetic empty fallback workflow."""

    return workflow.workflow_name == "fallback_workflow" and not workflow.nodes


def _schemas_compatible(expected: Optional[Mapping[str, Any]], actual: Optional[Mapping[str, Any]]) -> bool:
    """Lightweight JSON Schema compatibility check for planner-time validation."""

    if not expected:
        return True
    if not actual:
        return False

    expected_types = _schema_types(expected)
    actual_types = _schema_types(actual)
    if not expected_types or not actual_types:
        return True

    overlapping = expected_types & actual_types
    if not overlapping:
        return False

    if "array" in overlapping:
        expected_items = expected.get("items") if isinstance(expected.get("items"), Mapping) else None
        actual_items = actual.get("items") if isinstance(actual.get("items"), Mapping) else None
        if expected_items or actual_items:
            return _schemas_compatible(expected_items, actual_items)

    return True


def _find_unregistered_action_nodes(
    workflow_dict: Dict[str, Any],
    actions_by_id: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Find workflow nodes whose `action_id` is missing or not registered."""

    invalid: List[Dict[str, str]] = []
    for node in iter_workflow_and_loop_body_nodes(workflow_dict):
        if node.get("type") != "action":
            continue
        action_id = node.get("action_id")
        if not action_id or action_id not in actions_by_id:
            invalid.append({
                "id": node.get("id", "<unknown>"),
                "action_id": action_id or "",
            })
    return invalid


def _attach_out_params_schema(
    workflow: Workflow, actions_by_id: Mapping[str, Mapping[str, Any]]
) -> Workflow:
    wf_dict = workflow.model_dump(by_alias=True)
    changed = False

    for node in iter_workflow_and_loop_body_nodes(wf_dict):
        if not isinstance(node, Mapping) or node.get("type") != "action":
            continue

        action_id = node.get("action_id")
        schema = actions_by_id.get(action_id, {}).get("output_schema") if action_id else None
        if not isinstance(schema, Mapping):
            continue

        params = node.get("params") if isinstance(node.get("params"), Mapping) else {}
        expected_format = params.get("expected_format") if isinstance(params, Mapping) else None

        schema_to_set = schema

        if action_id == "common.ask_ai.v1":
            expected_schema = _parse_expected_format_schema(expected_format)
            if expected_schema:
                schema_to_set = copy.deepcopy(schema)
                properties = schema_to_set.setdefault("properties", {})
                properties["results"] = expected_schema
                required = schema_to_set.setdefault("required", [])
                if "results" not in required:
                    required.append("results")

        if node.get("out_params_schema") != schema_to_set:
            node["out_params_schema"] = schema_to_set
            changed = True

    return workflow if not changed else Workflow.model_validate(wf_dict)


def _evaluate_schema_alignment(
    node_params: Mapping[str, Any], candidate_schema: Mapping[str, Any]
) -> tuple[float, List[str], List[str], List[str]]:
    """Score how well the node params align with the candidate arg_schema."""

    if not isinstance(node_params, Mapping):
        return 0.0, [], [], ["Node params are not a mapping; cannot compare."]

    if not isinstance(candidate_schema, Mapping):
        return 0.0, [], sorted(node_params.keys()), [
            "Candidate action is missing arg_schema; skipping alignment score.",
        ]

    properties = candidate_schema.get("properties")
    if not isinstance(properties, Mapping):
        return 0.0, [], sorted(node_params.keys()), [
            "arg_schema.properties missing; skipping alignment score.",
        ]

    provided_keys = set(node_params.keys())
    schema_keys = set(properties.keys())
    matched = sorted(provided_keys & schema_keys)
    unmatched = sorted(provided_keys - schema_keys)

    if not provided_keys:
        return 0.6, matched, unmatched, [
            "Node provided no parameter keys; assigning medium default confidence.",
        ]

    if not schema_keys:
        return 0.0, matched, unmatched, [
            "Candidate arg_schema has no properties; cannot compare parameter keys.",
        ]

    coverage = len(matched) / max(len(provided_keys), len(schema_keys))
    notes: List[str] = []
    if unmatched:
        notes.append(
            "Parameter keys are missing from candidate arg_schema: " + ",".join(unmatched)
        )

    return coverage, matched, unmatched, notes


def _auto_replace_unregistered_actions(
    workflow_dict: Dict[str, Any],
    invalid_nodes: List[Dict[str, str]],
    search_service: HybridActionSearchService,
    actions_by_id: Mapping[str, Dict[str, Any]],
) -> Optional[Workflow]:
    """Attempt a one-shot auto replacement for unknown action_id to avoid LLM hops."""

    changed = False
    MIN_SEARCH_CONFIDENCE = 0.6
    MIN_SCHEMA_SCORE = 0.4

    for item in invalid_nodes:
        node_id = item.get("id")
        if not node_id:
            continue

        node = next(
            (n for n in iter_workflow_and_loop_body_nodes(workflow_dict) if n.get("id") == node_id),
            None,
        )
        if not isinstance(node, Mapping):
            continue

        display_name = node.get("display_name") or ""
        original_action_id = node.get("action_id") or ""

        query = display_name or original_action_id
        if not query:
            continue

        candidates = search_service.search(query=query, top_k=1) if search_service else []
        candidate = candidates[0] if candidates else None
        candidate_id = candidate.get("action_id") if candidate else None
        search_confidence = candidate.get("score") if candidate else None
        if not candidate_id or candidate_id not in actions_by_id:
            continue

        if search_confidence is None or search_confidence < MIN_SEARCH_CONFIDENCE:
            log_warn(
                f"[ActionGuard] Candidate action '{candidate_id}' for node '{node_id}' has confidence"
                f" {search_confidence} below threshold {MIN_SEARCH_CONFIDENCE}; skipping auto replacement."
            )
            continue

        schema_score, matched_params, unmatched_params, schema_notes = _evaluate_schema_alignment(
            node.get("params", {}), candidate.get("arg_schema", {})
        )
        if schema_score < MIN_SCHEMA_SCORE:
            log_warn(
                f"[ActionGuard] Candidate action '{candidate_id}' for node '{node_id}' has param alignment"
                f" {schema_score:.2f} below threshold {MIN_SCHEMA_SCORE}; skipping auto replacement."
            )
            continue

        node["action_id"] = candidate_id
        changed = True
        log_info(
            f"[ActionGuard] Node '{node_id}' had unregistered action_id='{original_action_id}',",
            f"Auto-replaced with '{candidate_id}' (search={search_confidence:.2f}, schema={schema_score:.2f})."
        )

    if not changed:
        return None

    try:
        return Workflow.model_validate(workflow_dict)
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[ActionGuard] Auto replacement for action_id failed: {exc}")
        return None


def _schema_default_value(field_schema: Mapping[str, Any]) -> Any:
    """Return a predictable placeholder value for a JSON schema field."""

    if not isinstance(field_schema, Mapping):
        return None

    if "default" in field_schema:
        return field_schema.get("default")

    schema_type = field_schema.get("type")
    if isinstance(schema_type, list):
        schema_type = schema_type[0] if schema_type else None

    if schema_type == "string":
        return ""
    if schema_type == "integer":
        return 0
    if schema_type == "number":
        return 0
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        return []
    if schema_type == "object":
        return {}

    enum_values = field_schema.get("enum")
    if isinstance(enum_values, list) and enum_values:
        return enum_values[0]

    return None


def _coerce_value_to_schema_type(value: Any, field_schema: Mapping[str, Any]) -> Optional[Any]:
    """Coerce a value into the basic JSON schema type when possible."""

    if not isinstance(field_schema, Mapping):
        return None

    schema_type = field_schema.get("type")
    if isinstance(schema_type, list):
        schema_type = schema_type[0] if schema_type else None

    if schema_type == "string" and not isinstance(value, str):
        return str(value)

    if schema_type == "integer":
        if isinstance(value, bool):
            return int(value)
        if not isinstance(value, int):
            try:
                return int(value)
            except Exception:  # noqa: BLE001
                return None

    if schema_type == "number":
        if isinstance(value, bool):
            return float(value)
        if not isinstance(value, (int, float)):
            try:
                return float(value)
            except Exception:  # noqa: BLE001
                return None

    if schema_type == "boolean" and not isinstance(value, bool):
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "false"}:
                return lowered == "true"
        return bool(value)

    if schema_type == "array" and not isinstance(value, list):
        return [value]

    if schema_type == "object" and not isinstance(value, Mapping):
        return {"value": value}

    return None


def _align_reference_field_types(
    workflow: Workflow, *, action_registry: List[Dict[str, Any]]
) -> tuple[Workflow, List[ValidationError]]:
    """Legacy placeholder: reference bindings are no longer auto-aligned."""

    return workflow, []


def _align_binding_and_param_types(
    workflow: Workflow, *, action_registry: List[Dict[str, Any]]
) -> tuple[Workflow, List[ValidationError]]:
    """Legacy placeholder: binding/param type alignment for __from__/__agg__ is removed."""

    return workflow, []




def _apply_local_repairs_for_missing_params(
    current_workflow: Workflow,
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
) -> Optional[Workflow]:
    """Handle predictable missing/typing issues before invoking the LLM.

    When parameter completion leaves required fields missing or minor type issues, use arg_schema
    placeholders or type coercion to reduce LLM repair cycles.
    """

    actions_by_id = _index_actions_by_id(action_registry)
    workflow_dict = current_workflow.model_dump(by_alias=True)
    nodes: List[Dict[str, Any]] = [
        n for n in workflow_dict.get("nodes", []) if isinstance(n, Mapping)
    ]
    nodes_by_id: Dict[str, Dict[str, Any]] = {n.get("id"): n for n in nodes}

    changed = False

    for err in validation_errors:
        if err.code != "MISSING_REQUIRED_PARAM" or not err.node_id:
            continue

        node = nodes_by_id.get(err.node_id)
        if not node or node.get("type") != "action":
            continue

        action_id = node.get("action_id")
        action_def = actions_by_id.get(action_id)
        if not action_def:
            continue

        arg_schema = action_def.get("arg_schema") or {}
        properties = arg_schema.get("properties") if isinstance(arg_schema, Mapping) else None
        if not isinstance(properties, Mapping):
            continue

        field_schema = properties.get(err.field) if err.field else None
        if not field_schema:
            continue

        params = node.get("params")
        if not isinstance(params, dict):
            params = {}
            node["params"] = params
            changed = True

        if err.field and err.field not in params:
            params[err.field] = _schema_default_value(field_schema)
            changed = True
            continue

        if err.field and err.field in params:
            coerced = _coerce_value_to_schema_type(params[err.field], field_schema)
            if coerced is not None and coerced != params[err.field]:
                params[err.field] = coerced
                changed = True

    if not changed:
        return None

    try:
        return Workflow.model_validate(workflow_dict)
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] Local fix failed: {exc}")
        return None


def _apply_local_repairs_for_schema_mismatch(
    current_workflow: Workflow,
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
) -> Optional[Workflow]:
    """Coerce obvious type mismatches in action params to avoid unnecessary LLM calls."""

    actions_by_id = _index_actions_by_id(action_registry)
    workflow_dict = current_workflow.model_dump(by_alias=True)
    nodes: List[Dict[str, Any]] = [
        n for n in workflow_dict.get("nodes", []) if isinstance(n, Mapping)
    ]
    nodes_by_id: Dict[str, Dict[str, Any]] = {n.get("id"): n for n in nodes}

    changed = False

    for err in validation_errors:
        if err.code != "SCHEMA_MISMATCH" or not err.node_id or not err.field:
            continue

        node = nodes_by_id.get(err.node_id)
        if not node or node.get("type") != "action":
            continue

        action_id = node.get("action_id")
        action_def = actions_by_id.get(action_id)
        if not action_def:
            continue

        properties = (
            action_def.get("arg_schema", {}).get("properties")
            if isinstance(action_def, Mapping)
            else None
        )
        if not isinstance(properties, Mapping):
            continue

        field_schema = properties.get(err.field)
        if not isinstance(field_schema, Mapping):
            continue

        params = node.get("params")
        if not isinstance(params, dict) or err.field not in params:
            continue

        coerced = _coerce_value_to_schema_type(params[err.field], field_schema)
        if coerced is not None and coerced != params[err.field]:
            params[err.field] = coerced
            changed = True

    if not changed:
        return None

    try:
        return Workflow.model_validate(workflow_dict)
    except Exception as exc:  # noqa: BLE001
        log_warn(f"[AutoRepair] Schema type coercion failed: {exc}")
        return None


def _ensure_actions_registered_or_repair(
    workflow: Workflow,
    action_registry: List[Dict[str, Any]],
    search_service: HybridActionSearchService,
    reason: str,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Workflow:
    """Ensure all action nodes reference a registered action, otherwise trigger repair.

    If the workflow has action nodes missing or referencing unregistered `action_id`, this function
    builds corresponding `ValidationError` entries and passes the error context, Action Registry, and call
    reason to the LLM for auto repair. The LLM must return a complete workflow JSON,
    which Pydantic will then convert into a `Workflow` instance.
    """

    guarded = ensure_registered_actions(
        workflow,
        action_registry=action_registry,
        search_service=search_service,
    )
    guarded = guarded if isinstance(guarded, Workflow) else Workflow.model_validate(guarded)

    actions_by_id = _index_actions_by_id(action_registry)
    guarded = _attach_out_params_schema(guarded, actions_by_id)
    invalid_nodes = _find_unregistered_action_nodes(
        guarded.model_dump(by_alias=True),
        actions_by_id=actions_by_id,
    )
    if not invalid_nodes:
        return guarded

    auto_repaired = _auto_replace_unregistered_actions(
        guarded.model_dump(by_alias=True),
        invalid_nodes=invalid_nodes,
        search_service=search_service,
        actions_by_id=actions_by_id,
    )
    if auto_repaired is not None:
        log_info(
            "[ActionGuard] Auto replacement based on display_name/action_id similarity complete; running single validation."
        )
        return _attach_out_params_schema(auto_repaired, actions_by_id)

    validation_errors = [
        ValidationError(
            code="UNKNOWN_ACTION_ID",
            node_id=item["id"],
            field="action_id",
            message=(
                f"Node '{item['id']}' has action_id '{item['action_id']}' that is not in the Action Registry,"
                "please replace it with a registered action."
                if item.get("action_id")
                else f"Node '{item['id']}' is missing action_id; please supply a registered action."
            ),
        )
        for item in invalid_nodes
    ]

    repaired = _repair_with_llm_and_fallback(
        broken_workflow=guarded.model_dump(by_alias=True),
        validation_errors=validation_errors,
        action_registry=action_registry,
        search_service=search_service,
        reason=reason,
        progress_callback=progress_callback,
    )
    final_workflow = repaired if isinstance(repaired, Workflow) else Workflow.model_validate(repaired)
    return _attach_out_params_schema(final_workflow, actions_by_id)


def _validate_and_repair_workflow(
    current_workflow: Workflow,
    *,
    action_registry: List[Dict[str, Any]],
    search_service: HybridActionSearchService,
    max_repair_rounds: int,
    last_good_workflow: Workflow | None = None,
    trace_event_prefix: str = "planner",
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Workflow:
    failed_repair_history: Dict[str, List[str]] = {}
    pending_attempts: Dict[str, Dict[str, Any]] = {}

    def _error_key(err: ValidationError) -> str:
        return f"{err.code}:{err.node_id or 'global'}:{err.field or 'global'}"

    last_good_workflow = last_good_workflow or current_workflow

    def _emit_progress(label: str, workflow_obj: Workflow | Mapping[str, Any]) -> None:
        if not progress_callback:
            return
        try:
            payload = workflow_obj.model_dump(by_alias=True) if isinstance(workflow_obj, Workflow) else workflow_obj
            progress_callback(label, payload)
        except Exception:
            log_debug(f"[validate_and_repair] progress_callback {label} failed and was ignored.")

    for repair_round in range(max_repair_rounds + 1):
        log_section(f"Validation + auto-repair round {repair_round}")
        log_json("Current workflow", current_workflow.model_dump(by_alias=True))

        jinja_params_workflow, jinja_params_summary, jinja_params_errors = normalize_params_to_jinja(
            current_workflow.model_dump(by_alias=True)
        )
        if jinja_params_summary.get("applied") or jinja_params_summary.get("forbidden_paths"):
            log_info(
                "[JinjaGuard] Normalized workflow params and marked non-Jinja DSL entries.",
                f"summary={jinja_params_summary}",
            )
            try:
                current_workflow = Workflow.model_validate(jinja_params_workflow)
                last_good_workflow = current_workflow
            except PydanticValidationError as exc:
                log_warn(
                    f"[AutoRepair] Validation failed after Jinja normalization; passing to LLM. error={exc}"
                )
                validation_errors = _convert_pydantic_errors(jinja_params_workflow, exc) or [
                    _make_failure_validation_error(str(exc))
                ]
                current_workflow = _repair_with_llm_and_fallback(
                    broken_workflow=jinja_params_workflow if isinstance(jinja_params_workflow, dict) else {},
                    validation_errors=validation_errors,
                    action_registry=action_registry,
                    search_service=search_service,
                    reason="Validation failed after Jinja normalization",
                    progress_callback=progress_callback,
                )
                current_workflow = _ensure_actions_registered_or_repair(
                    current_workflow,
                    action_registry=action_registry,
                    search_service=search_service,
                    reason="Validate action_id after Jinja normalization repair",
                    progress_callback=progress_callback,
                )
                last_good_workflow = current_workflow
                _emit_progress(f"{trace_event_prefix}_repair_round_{repair_round}", current_workflow)
                continue

        jinja_checked_workflow, jinja_summary, jinja_errors = normalize_condition_params_to_jinja(
            current_workflow.model_dump(by_alias=True)
        )
        if jinja_summary.get("applied"):
            log_info(
                "[JinjaGuard] Normalized references in condition.params to Jinja strings.",
                f"replacements={jinja_summary.get('replacements')}",
            )
            current_workflow = Workflow.model_validate(jinja_checked_workflow)
            last_good_workflow = current_workflow

        current_workflow, binding_type_errors = _align_binding_and_param_types(
            current_workflow, action_registry=action_registry
        )

        current_workflow, reference_errors = _align_reference_field_types(
            current_workflow, action_registry=action_registry
        )

        loop_exports_workflow, loop_export_summary = fill_loop_exports_defaults(
            current_workflow.model_dump(by_alias=True),
            action_registry=action_registry,
        )
        if loop_export_summary.get("applied"):
            log_info(
                "[AutoRepair] loop.exports were missing or incomplete; auto-filled:",
                f"summary={loop_export_summary}",
            )
            current_workflow = Workflow.model_validate(loop_exports_workflow)
            last_good_workflow = current_workflow
            _emit_progress(f"{trace_event_prefix}_repair_round_{repair_round}", current_workflow)

        aligned_alias_workflow, alias_summary = align_loop_body_alias_references(
            current_workflow.model_dump(by_alias=True),
            action_registry=action_registry,
        )
        if alias_summary.get("applied"):
            log_info(
                "[AutoRepair] loop.body_subgraph referenced a stale alias; aligned to current item_alias.",
                f"summary={alias_summary}",
            )
            current_workflow = Workflow.model_validate(aligned_alias_workflow)
            last_good_workflow = current_workflow

        normalized_workflow, normalize_summary = normalize_binding_paths(
            current_workflow.model_dump(by_alias=True)
        )
        if normalize_summary.get("applied"):
            log_info(
                "[AutoRepair] Normalized binding paths to reduce LLM repairs.",
                f"summary={normalize_summary}",
            )
            current_workflow = Workflow.model_validate(normalized_workflow)
            last_good_workflow = current_workflow

        static_errors = run_lightweight_static_rules(
            current_workflow.model_dump(by_alias=True),
            action_registry=action_registry,
        )

        # Some validation passes (e.g., optional type coercion) may be skipped; ensure the
        # error accumulator is initialized even when those phases are absent to avoid
        # NameError crashes.
        coercion_errors: List[ValidationError] = []

        errors: List[ValidationError] = []
        errors.extend(jinja_params_errors)
        errors.extend(jinja_errors)
        errors.extend(coercion_errors)
        errors.extend(binding_type_errors)
        errors.extend(reference_errors)

        if static_errors:
            errors.extend(static_errors)
        else:
            errors.extend(
                validate_completed_workflow(
                    current_workflow.model_dump(by_alias=True),
                    action_registry=action_registry,
                )
            )

        existing_error_keys = {(e.code, e.node_id, e.field) for e in errors}
        for err in _collect_missing_required_param_errors(current_workflow, action_registry):
            if (err.code, err.node_id, err.field) not in existing_error_keys:
                errors.append(err)
                existing_error_keys.add((err.code, err.node_id, err.field))

        if _is_empty_fallback_workflow(current_workflow):
            log_warn("Detected empty fallback workflow; triggering replanning automatically")
            errors.append(
                _make_failure_validation_error(
                    "workflow rollback produced an empty fallback_workflow; replanning required"
                )
            )

        fixed_workflow, fix_summary = fix_missing_loop_exports_items(
            current_workflow.model_dump(by_alias=True), errors
        )
        if fix_summary.get("applied"):
            log_info(
                "[AutoRepair] Detected loop.exports reference missing the exports section and auto-completed it.",
                f"replacements={fix_summary.get('replacements')}",
            )
            current_workflow = Workflow.model_validate(fixed_workflow)
            last_good_workflow = current_workflow
            _emit_progress(f"{trace_event_prefix}_repair_round_{repair_round}", current_workflow)
            continue

        current_errors_by_key = {_error_key(e): e for e in errors}
        if pending_attempts:
            for key, attempt in list(pending_attempts.items()):
                if key in current_errors_by_key:
                    note = (
                        f"Attempt round {attempt.get('round')} with {attempt.get('method')} did not fix errors:"
                        f"{current_errors_by_key[key].message}"
                    )
                    failed_repair_history.setdefault(key, []).append(note)
                pending_attempts.pop(key, None)

        if not errors:
            log_success("Validation passed; no further repair needed")
            last_good_workflow = current_workflow
            _emit_progress(f"{trace_event_prefix}_repair_round_{repair_round}", current_workflow)
            log_event(
                f"{trace_event_prefix}_completed",
                {
                    "status": "success",
                    "repair_rounds": repair_round,
                    "has_errors": False,
                },
            )
            return current_workflow

        code_counts: Dict[str, int] = {}
        for e in errors:
            code_counts[e.code] = code_counts.get(e.code, 0) + 1

        log_event(
            "validation_errors_distribution",
            {
                "round": repair_round,
                "counts": code_counts,
                "total": len(errors),
            },
        )
        log_warn("Validation failed; error list:")
        for e in errors:
            log_error(
                f"[code={e.code}] node={e.node_id} field={e.field} message={e.message}"
            )

        locally_repaired = _apply_local_repairs_for_unknown_params(
            current_workflow=current_workflow,
            validation_errors=errors,
            action_registry=action_registry,
        )
        if locally_repaired is None:
            locally_repaired = _apply_local_repairs_for_missing_params(
                current_workflow=current_workflow,
                validation_errors=errors,
                action_registry=action_registry,
            )
        if locally_repaired is None:
            locally_repaired = _apply_local_repairs_for_schema_mismatch(
                current_workflow=current_workflow,
                validation_errors=errors,
                action_registry=action_registry,
            )
        if locally_repaired is not None:
            log_info("[AutoRepair] Predictable missing/extra/type mismatch fields detected; fixed locally and revalidated.")
            current_workflow = locally_repaired
            last_good_workflow = current_workflow

            errors = validate_completed_workflow(
                current_workflow.model_dump(by_alias=True),
                action_registry=action_registry,
            )

            if not errors:
                log_success("Validation passed after local fixes; no LLM repair needed.")
                _emit_progress(f"{trace_event_prefix}_repair_round_{repair_round}", current_workflow)
                log_event(
                    f"{trace_event_prefix}_completed",
                    {
                        "status": "success_after_local_repair",
                        "repair_rounds": repair_round,
                        "has_errors": False,
                    },
                )
                return current_workflow

                log_warn("Errors remain after local fixes; continuing with LLM repair:")
                for e in errors:
                    log_error(
                        f"[code={e.code}] node={e.node_id} field={e.field} message={e.message}"
                    )

        if repair_round == max_repair_rounds:
            log_warn("Reached max repair rounds with remaining errors; returning last valid structure")
            log_event(
                f"{trace_event_prefix}_completed",
                {
                    "status": "reached_max_rounds",
                    "repair_rounds": repair_round,
                    "last_good_returned": True,
                },
            )
            _emit_progress(f"{trace_event_prefix}_repair_round_{repair_round}", last_good_workflow)
            return last_good_workflow

        current_errors_by_key = {_error_key(e): e for e in errors}
        previous_attempts = {
            key: failed_repair_history.get(key, [])
            for key in current_errors_by_key
            if failed_repair_history.get(key)
        }

        log_info(f"Calling LLM for repair round {repair_round + 1}")
        current_workflow = _repair_with_llm_and_fallback(
            broken_workflow=current_workflow.model_dump(by_alias=True),
            validation_errors=errors,
            action_registry=action_registry,
            search_service=search_service,
            reason=f"Repair round {repair_round + 1}",
            previous_attempts=previous_attempts,
            progress_callback=progress_callback,
        )
        for key in current_errors_by_key:
            pending_attempts[key] = {
                "round": repair_round + 1,
                "method": "LLM repair",
            }
        current_workflow = _ensure_actions_registered_or_repair(
            current_workflow,
            action_registry=action_registry,
            search_service=search_service,
            reason=f"Validate action_id after repair round {repair_round + 1}",
            progress_callback=progress_callback,
        )
        last_good_workflow = current_workflow
        _emit_progress(f"{trace_event_prefix}_repair_round_{repair_round}", current_workflow)

    log_event(
        f"{trace_event_prefix}_completed",
        {
            "status": "exhausted_rounds",
            "repair_rounds": max_repair_rounds,
            "last_good_returned": True,
        },
    )
    _emit_progress(f"{trace_event_prefix}_repair_round_{max_repair_rounds}", last_good_workflow)
    return last_good_workflow


def plan_workflow_with_two_pass(
    nl_requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    *,
    max_rounds: int = 10,
    max_repair_rounds: int = 3,
    trace_context: TraceContext | None = None,
    trace_id: str | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Workflow:
    """Plan a workflow in two passes with structured validation and LLM repair.

    Args
    ----
    nl_requirement:
        Natural-language requirement for the LLM; must clarify business goal and key inputs/outputs.
    search_service:
        Retrieval service to help the LLM select business actions matching the requirement.
    action_registry:
        List of registered business actions; each element must include a unique `action_id`.
    max_rounds:
        Allowed LLM iterations in the structure planning stage, controlling coverage/connectivity fixes.
    max_repair_rounds:
        Maximum rounds for validation + self-repair after parameter completion; beyond this, return the
        last version that passed validation.

    Returns
    ----
    Workflow
        Final Workflow object that passes Pydantic validation with all action_id registered.

    LLM response format
    -------------------
    * Structure planning and parameter completion require the LLM to return JSON parseable by
      ``Workflow``, at minimum containing the ``nodes`` array (edges are inferred from node bindings).
    * Repair rounds include ``validation_errors``; the LLM must output the corrected full JSON without
      extra explanatory text.

    Edge cases
    --------
    * If structure planning or parameter fill throws or fails validation, reuse the last valid
      Workflow and invoke LLM self-repair.
    * When ``max_repair_rounds`` is reached with remaining errors, return the most recent validated
      version so callers always receive a valid ``Workflow`` instance.
    """

    parsed_requirement = analyze_user_requirement(
        nl_requirement,
        max_rounds=max_rounds,
    )

    # Allow one restart with a fresh trace context if parameter completion fails,
    # so downstream callers still get a valid workflow instead of an exception.
    restart_attempted = False

    while True:
        context = (
            trace_context
            if not restart_attempted and trace_context is not None
            else TraceContext.create(
                trace_id=trace_id if not restart_attempted else None, span_name="orchestrator"
            )
        )
        with use_trace_context(context):
            log_event(
                "plan_start",
                {
                    "nl_requirement": nl_requirement,
                    "max_rounds": max_rounds,
                    "max_repair_rounds": max_repair_rounds,
                },
                context=context,
            )

            with child_span("structure_planning"):
                skeleton_raw = plan_workflow_structure_with_llm(
                    parsed_requirement=parsed_requirement,
                    search_service=search_service,
                    action_registry=action_registry,
                    max_rounds=max_rounds,
                    progress_callback=progress_callback,
                )
            log_section("Phase one result: Workflow Skeleton")
            log_json("Workflow Skeleton", skeleton_raw)
            if progress_callback:
                try:
                    progress_callback("structure_completed", skeleton_raw)
                except Exception:
                    log_debug("[plan_workflow_with_two_pass] progress_callback failed during structure stage and was ignored.")

            precheck_errors = precheck_loop_body_graphs(skeleton_raw)
            if precheck_errors:
                log_warn(
                    "[plan_workflow_with_two_pass] Structure stage found loop body referencing missing nodes; delegating to LLM repair."
                )
                skeleton = _repair_with_llm_and_fallback(
                    broken_workflow=skeleton_raw if isinstance(skeleton_raw, dict) else {},
                    validation_errors=precheck_errors,
                    action_registry=action_registry,
                    search_service=search_service,
                    reason="Structure planning validation failed",
                    progress_callback=progress_callback,
                )
            else:
                try:
                    skeleton = Workflow.model_validate(skeleton_raw)
                except PydanticValidationError as e:
                    log_warn(
                        "[plan_workflow_with_two_pass] Structure planning validation failed; sending errors to LLM repair before continuing."
                    )
                    validation_errors = _convert_pydantic_errors(skeleton_raw, e)
                    if not validation_errors:
                        validation_errors = [_make_failure_validation_error(str(e))]
                    skeleton = _repair_with_llm_and_fallback(
                        broken_workflow=skeleton_raw if isinstance(skeleton_raw, dict) else {},
                        validation_errors=validation_errors,
                        action_registry=action_registry,
                        search_service=search_service,
                        reason="Structure planning validation failed",
                        progress_callback=progress_callback,
                    )
                except Exception as e:  # noqa: BLE001
                    log_warn(
                        "[plan_workflow_with_two_pass] Structure stage encountered a fatal error; delegating to LLM repair before continuing."
                    )
                    skeleton = _repair_with_llm_and_fallback(
                        broken_workflow=skeleton_raw if isinstance(skeleton_raw, dict) else {},
                        validation_errors=[_make_failure_validation_error(str(e))],
                        action_registry=action_registry,
                        search_service=search_service,
                        reason="Structure planning validation failed",
                        progress_callback=progress_callback,
                    )
            if not isinstance(skeleton, Workflow):
                payload = (
                    skeleton
                    if isinstance(skeleton, Mapping)
                    else skeleton.model_dump(by_alias=True)
                    if hasattr(skeleton, "model_dump")
                    else {}
                )
                skeleton = Workflow.model_validate(payload)
            elif any(isinstance(n, dict) for n in getattr(skeleton, "nodes", [])):
                skeleton = Workflow.model_validate(
                    {
                        "workflow_name": getattr(skeleton, "workflow_name", "unnamed_workflow"),
                        "description": getattr(skeleton, "description", ""),
                        "nodes": getattr(skeleton, "nodes", []),
                    }
                )

            log_event("plan_structure_done", {"workflow": skeleton.model_dump()})
            skeleton = _ensure_actions_registered_or_repair(
                skeleton,
                action_registry=action_registry,
                search_service=search_service,
                reason="Fix unregistered action_id after structure planning",
                progress_callback=progress_callback,
            )
            last_good_workflow: Workflow = skeleton
            current_workflow = skeleton
            last_good_workflow = current_workflow

            planned_workflow = _validate_and_repair_workflow(
                current_workflow,
                action_registry=action_registry,
                search_service=search_service,
                max_repair_rounds=max_repair_rounds,
                last_good_workflow=last_good_workflow,
                trace_event_prefix="planner",
                progress_callback=progress_callback,
            )

        if not _is_empty_fallback_workflow(planned_workflow):
            return planned_workflow

        if restart_attempted:
            return planned_workflow

        log_warn(
            "[plan_workflow_with_two_pass] Detected empty fallback workflow; current context cannot continue repair,"
            "replanning with fresh context."
        )
        restart_attempted = True
        trace_context = None
        trace_id = None


def update_workflow_with_two_pass(
    existing_workflow: Mapping[str, Any],
    requirement: str,
    search_service: HybridActionSearchService,
    action_registry: List[Dict[str, Any]],
    *,
    max_rounds: int = 100,
    max_repair_rounds: int = 3,
    trace_context: TraceContext | None = None,
    trace_id: str | None = None,
    progress_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
) -> Workflow:
    """Update an existing workflow using the same validation/repair pipeline as planning.

    The ``existing_workflow`` is treated as a trusted DSL template whenever possible. We
    attempt to preserve untouched nodes/edges while satisfying the new ``requirement``
    through incremental changes plus validation-driven self-repair.
    """

    # Reuse upstream trace context if provided so logs stay correlated; otherwise
    # create a dedicated span for this update run.
    context = trace_context or TraceContext.create(trace_id=trace_id, span_name="update_orchestrator")
    with use_trace_context(context):
        log_event(
            "update_start",
            {
                "max_rounds": max_rounds,
                "max_repair_rounds": max_repair_rounds,
            },
            context=context,
        )

        try:
            base_workflow = Workflow.model_validate(existing_workflow)
        except PydanticValidationError as e:
            log_warn("[update_workflow_with_two_pass] Input workflow failed validation; entering repair loop.")
            # Input DSL might come from hand edits; repair it first so the planner
            # can safely continue from a validated baseline.
            validation_errors = _convert_pydantic_errors(existing_workflow, e) or [
                _make_failure_validation_error(str(e))
            ]
            base_workflow = _repair_with_llm_and_fallback(
                broken_workflow=existing_workflow if isinstance(existing_workflow, dict) else {},
                validation_errors=validation_errors,
                action_registry=action_registry,
                search_service=search_service,
                reason="Input validation before update failed",
                progress_callback=progress_callback,
            )

        base_workflow = _ensure_actions_registered_or_repair(
            base_workflow,
            action_registry=action_registry,
            search_service=search_service,
            reason="Validate action_id before update",
            progress_callback=progress_callback,
        )
        last_good_workflow = base_workflow

        parsed_requirement = analyze_user_requirement(
            requirement,
            existing_workflow=base_workflow.model_dump(by_alias=True),
            max_rounds=max_rounds,
        )

        with child_span("structure_planning"):
            updated_raw = plan_workflow_structure_with_llm(
                parsed_requirement=parsed_requirement,
                search_service=search_service,
                action_registry=action_registry,
                max_rounds=max_rounds,
                progress_callback=progress_callback,
                existing_workflow=base_workflow.model_dump(by_alias=True),
            )

        # Sanity-check loop bodies early so repair prompts include missing IDs
        # before the more expensive validation steps run.
        precheck_errors = precheck_loop_body_graphs(updated_raw)
        if precheck_errors:
            log_warn(
                "[update_workflow_with_two_pass] Structure stage found loop body referencing missing nodes; delegating to LLM repair."
            )
            updated_workflow = _repair_with_llm_and_fallback(
                broken_workflow=updated_raw if isinstance(updated_raw, dict) else {},
                validation_errors=precheck_errors,
                action_registry=action_registry,
                search_service=search_service,
                reason="Structure planning validation failed",
                progress_callback=progress_callback,
            )
        else:
            try:
                updated_workflow = Workflow.model_validate(updated_raw)
            except PydanticValidationError as e:
                log_warn(
                    "[update_workflow_with_two_pass] Structure planning validation failed; sending errors to LLM repair before continuing."
                )
                # Convert schema errors into a friendly shape before retrying via LLM.
                validation_errors = _convert_pydantic_errors(updated_raw, e) or [
                    _make_failure_validation_error(str(e))
                ]
                updated_workflow = _repair_with_llm_and_fallback(
                    broken_workflow=updated_raw if isinstance(updated_raw, dict) else {},
                    validation_errors=validation_errors,
                    action_registry=action_registry,
                    search_service=search_service,
                    reason="Structure planning validation failed",
                    progress_callback=progress_callback,
            )

        if not isinstance(updated_workflow, Workflow):
            payload = (
                updated_workflow
                if isinstance(updated_workflow, Mapping)
                else updated_workflow.model_dump(by_alias=True)
                if hasattr(updated_workflow, "model_dump")
                else {}
            )
            updated_workflow = Workflow.model_validate(payload)
        elif any(isinstance(n, dict) for n in getattr(updated_workflow, "nodes", [])):
            updated_workflow = Workflow.model_validate(
                {
                    "workflow_name": getattr(updated_workflow, "workflow_name", "unnamed_workflow"),
                    "description": getattr(updated_workflow, "description", ""),
                    "nodes": getattr(updated_workflow, "nodes", []),
                }
            )

        updated_workflow = _ensure_actions_registered_or_repair(
            updated_workflow,
            action_registry=action_registry,
            search_service=search_service,
            reason="Validate action_id after update",
            progress_callback=progress_callback,
        )
        last_good_workflow = updated_workflow

        if progress_callback:
            try:
                progress_callback("update_completed", updated_workflow.model_dump(by_alias=True))
            except Exception:
                log_debug("[update_workflow_with_two_pass] progress_callback failed during update stage and was ignored.")

        # Reuse the same validation/repair pipeline to ensure params and bindings
        # are executable after structural updates.
        return _validate_and_repair_workflow(
            updated_workflow,
            action_registry=action_registry,
            search_service=search_service,
            max_repair_rounds=max_repair_rounds,
            last_good_workflow=last_good_workflow,
            trace_event_prefix="update",
            progress_callback=progress_callback,
        )


__all__ = ["plan_workflow_with_two_pass", "update_workflow_with_two_pass"]
