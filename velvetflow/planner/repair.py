# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""LLM-driven repair helpers used when validation fails."""

import json
import os
import subprocess
import tempfile
from dataclasses import asdict
import itertools
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    child_span,
    log_debug,
    log_error,
    log_info,
    log_llm_message,
    log_llm_usage,
    log_success,
    log_tool_call,
    log_warn,
)
from velvetflow.models import Node, PydanticValidationError, ValidationError, Workflow
from velvetflow.planner.action_guard import ensure_registered_actions
from velvetflow.planner.repair_tools import REPAIR_TOOLS, apply_repair_tool
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
            or err.field.startswith("exports.items")
            or err.field in {"source", "item_alias"}
        ):
            loop_hints.append(
                "- loop 节点需同时包含 params.loop_kind（for_each/while）、params.source、"
                "params.item_alias；exports.items.fields 只能包含 body_subgraph 或 source 元素 schema 中存在的字段"
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
    "MISSING_REQUIRED_PARAM": "补齐 params.<field>，优先绑定上游符合 arg_schema 的输出或使用 schema 默认值。示例：如果 action 需要 params.prompt 且缺失，可绑定上游文案生成节点的 result.output。",
    "UNKNOWN_ACTION_ID": "替换为 Action Registry 中存在的 action_id，保持节点含义最接近并同步校正 params。示例：将未知的 action_id 改为 registry 中的 text.generate，并保留 prompt/temperature 等参数。",
    "UNKNOWN_PARAM": "移除或重命名 params 中未在 arg_schema 声明的字段，使其与 schema 对齐。示例：action schema 仅接受 prompt/temperature，应删除意外出现的 max_token 字段。",
    "DISCONNECTED_GRAPH": "连接无入度/出度节点，确保每个节点都位于可达路径上。示例：为孤立的 summary 节点添加来自前置生成节点的边，并把结果导向下游汇总节点。",
    "INVALID_EDGE": "修复边的 from/to 以引用存在的节点，并保持条件字段合法。示例：若 edge.to 指向不存在的 node-3，可改为实际存在的 reviewer 节点。",
    "SCHEMA_MISMATCH": "调整绑定字段或聚合方式，使类型与上游 output_schema/arg_schema 兼容。示例：如果上游输出是 list 而当前节点期望 string，可改为 join 列表或选择单个元素。",
    "INVALID_SCHEMA": "按 DSL 结构补齐/修正字段类型（nodes/edges/params），避免 JSON 结构缺失。示例：在 workflow 缺少 edges 字段时补空数组，并确保 nodes 为列表。",
    "INVALID_LOOP_BODY": "补齐 loop.body_subgraph.nodes（至少一个 action），并确保 exports.items 字段匹配 body 输出。示例：为 body_subgraph 增加默认 action 节点，exports.items.fields 仅保留 body 输出中存在的字段。",
    "STATIC_RULES_SUMMARY": "遵循静态规则摘要中的提示逐项修复，优先处理连通性和 schema 不匹配。示例：按摘要要求先修复未连接的边，再补齐缺失的必填参数。",
    "EMPTY_PARAM_VALUE": "为空的参数应填入非空值或绑定，避免空字符串/空对象。示例：将空的 params.prompt 改为具体提示词或绑定上游结果。",
    "EMPTY_PARAMS": "params 不能是空对象，为必填字段提供值或引用，并使用工具完成不确定的填充。示例：若节点类型为 action 但 params 为 {}，根据 arg_schema 填入 prompt/template 等必填字段。",
    "SELF_REFERENCE": "移除节点对自身 result_of.<node> 的引用，改用上游输出或拆分节点并通过工具应用修改。示例：若 node-a.params.input 引用 result_of.node-a，可改为 result_of.previous-node。",
    "SYNTAX_ERROR": "修正 DSL 语法（括号、逗号、引号等），使解析器能生成 AST。示例：补全缺失的引号或逗号，确保 JSON 能被解析。",
    "GRAMMAR_VIOLATION": "遵循文法约束调整节点/字段位置，按期望的 token/产生式重新组织。示例：将误放在 edges 中的节点定义移回 nodes 列表。",
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
                f"[AutoRepair] 回退到原始结构失败，将返回空的 fallback workflow：{inner_err}"
            )
            return Workflow(
                workflow_name="fallback_workflow", nodes=[]
            )


def repair_workflow_with_llm(
    broken_workflow: Dict[str, Any],
    validation_errors: List[ValidationError],
    action_registry: List[Dict[str, Any]],
    error_summary: str | None = None,
    previous_failed_attempts: Mapping[str, Sequence[str]] | None = None,
    model: str = OPENAI_MODEL,
) -> Dict[str, Any]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
你是一个工作流修复助手。
【Workflow DSL 语法与语义（务必遵守）】
- workflow = {workflow_name, description, nodes: []}，只能返回合法 JSON（edges 会由系统基于节点绑定自动推导，不需要生成）。
- node 基本结构：{id, type, display_name, params, action_id?, out_params_schema?, loop/subgraph/branches?}。
  type 仅允许 action/condition/loop/parallel。无需 start/end/exit 节点。
  action 节点必须填写 action_id（来自动作库）与 params；只有 action 节点允许 out_params_schema。
  condition 节点需包含返回布尔值的 params.expression（合法 Jinja 表达式），以及 true_to_node/false_to_node（字符串或 null）。
  loop 节点包含 loop_kind/iter/source/body_subgraph/exports，循环外部只能引用 exports.items 或 exports.aggregates，body_subgraph 仅包含 nodes 数组。
  loop.body_subgraph 内不需要也不允许显式声明 edges、entry 或 exit 节点，如发现请直接删除。
  - params 必须直接使用 Jinja 表达式引用上游结果（如 {{ result_of.<node_id>.<field_path> }} 或 {{ loop.item.xxx }}），不再允许 __from__/__agg__ DSL。
    <node_id> 必须存在且字段需与上游 output_schema 或 loop.exports 对齐。
  当前有一个 workflow JSON 和一组结构化校验错误 validation_errors。
  validation_errors 是 JSON 数组，元素包含 code/node_id/field/message。
  这些错误来自：
  - action 参数缺失或不符合 arg_schema
  - condition 条件不完整
  - source 路径引用了不存在的节点
  - source 路径与上游 action 的 output_schema 不匹配
  - source 指向的数组元素 schema 中不存在某个字段

  - previous_failed_attempts 会记录同一错误的历史修复尝试，请避免重复失败的方法，尝试不同的修复路径。

  - workflow 结构不符合 DSL schema（例如节点 type 非法）

  总体目标：在“尽量不改变工作流整体结构”的前提下，修复这些错误，使 workflow 通过静态校验。

  具体要求（很重要，请严格遵守）：
 1. 结构保持稳定：
    - 不要增加或删除节点；
    - edges 会由系统根据节点引用自动推导，不需要手动增删或调整 condition。

 2. action 节点修复优先级：
    - 首先根据 action_schemas[action_id].arg_schema 补齐 params 里缺失的必填字段，或修正错误类型；
    - 如果 action_id 本身是合法的（存在于 action_schemas 中），优先“修 params”，不要改 action_id；
    - 只有当 validation_errors 明确指出 action_id 不存在时，才考虑把 action_id 改成一个更合理的候选，
      并同步更新该节点的 params 使之符合新的 arg_schema。

 3. condition 节点修复：
    - 确保 params.expression 存在且是合法的 Jinja 布尔表达式；
    - expression 应直接编写判断逻辑，引用上游结果请使用 {{ result_of.<node>.<field_path> }} 或 {{ loop.item.xxx }}；
    - 不要生成 kind/field/op/value 等字段，所有条件均需融入 expression。

 4. loop 节点：确保 loop_kind（for_each/while）和 source 字段合法，并使用 exports 暴露 body 结果，下游只能引用 result_of.<loop_id>.items / aggregates。
 5. parallel 节点：branches 必须是非空数组。

 6. 参数绑定修复：
    - 仅输出 Jinja 表达式或字面量，禁止回退到 __from__/__agg__ 对象；
    - 聚合/过滤/拼接逻辑请直接写在 Jinja 表达式或过滤器里，并确保字段路径与上游 schema 对齐；
    - 常见错误：loop.exports.items.fields 只有包装字段，但条件需要访问内部子字段，请将路径写成 <字段>.<子字段>。

 7. 修改范围尽量最小化：
    - 当有多种修复方式时，优先选择改动最小、语义最接近原意的方案（如只改一个字段名，而不是重写整个 params）。
    - 当 validation_error_summary 提供 Schema 提示或路径信息时，优先按提示矫正字段类型/结构，避免多轮重复犯错。

 8. 修复回合有限且必须闭环：
    - 系统会在有限轮次内重新校验；如果你忽略任何一条 validation_error，流程将直接进入下一轮甚至终止。
    - 请逐条对照 validation_errors，把所有问题修到为 0 再输出结果，避免留存隐患。

 9. 输出要求：
    - 保持顶层结构：workflow_name/description/nodes 不变（仅节点内部内容可调整，edges 由系统推导）；

7. 修改范围尽量最小化：
   - 当有多种修复方式时，优先选择改动最小、语义最接近原意的方案（如只改一个字段名，而不是重写整个 params）。
   - 当 validation_error_summary 提供 Schema 提示或路径信息时，优先按提示矫正字段类型/结构，避免多轮重复犯错。

8. 修复回合有限且必须闭环：
   - 系统会在有限轮次内重新校验；如果你忽略任何一条 validation_error，流程将直接进入下一轮甚至终止。
   - 请逐条对照 validation_errors，把所有问题修到为 0 再输出结果，避免留存隐患。

9. 输出要求：
   - 保持顶层结构：workflow_name/description/nodes 不变（仅节点内部内容可调整，edges 由系统推导）；
   - 节点的 id/type 不变；
   - 返回修复后的 workflow JSON，只返回 JSON 对象本身，不要包含代码块标记。
   - 如果修改量较大，也可以返回针对 workflow.json 的 unified diff 补丁；系统会尝试用 git apply 将补丁合并回 workflow 并继续校验。
10. 可用工具：当你需要结构化修改时，优先调用提供的工具（无 LLM 依赖、结果确定），用来修复 loop body 引用、补齐必填参数或写入指定字段。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                {
                    "workflow": broken_workflow,
                    "validation_error_summary": error_summary,
                    "validation_errors": [asdict(e) for e in validation_errors],
                    "previous_failed_attempts": previous_failed_attempts,
                    "action_schemas": action_schemas,
                },
                ensure_ascii=False,
            ),
        },
    ]

    working_workflow: Dict[str, Any] = broken_workflow

    while True:
        with child_span("repair_llm"):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=REPAIR_TOOLS,
                tool_choice="auto",
                temperature=0.1,
            )
        log_llm_usage(model, getattr(resp, "usage", None), operation="repair_workflow")
        if not resp.choices:
            raise RuntimeError("repair_workflow_with_llm 未返回任何候选消息")

        msg = resp.choices[0].message
        log_llm_message(
            model,
            msg,
            operation="repair_workflow",
        )
        if msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls})
            for tc in msg.tool_calls:
                func_name = tc.function.name
                raw_args = tc.function.arguments
                try:
                    args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    log_error(f"[repair_workflow_with_llm] 无法解析工具参数: {raw_args}")
                    args = {}

                log_tool_call(
                    source="repair_workflow_with_llm",
                    tool_name=func_name,
                    tool_call_id=tc.id,
                    args=args or raw_args,
                )
                patched_workflow, summary = apply_repair_tool(
                    tool_name=func_name,
                    args=args,
                    workflow=working_workflow,
                    validation_errors=validation_errors,
                    action_registry=action_registry,
                )
                log_tool_call(
                    source="repair_workflow_with_llm",
                    tool_name=f"{func_name} 完成",
                    tool_call_id=tc.id,
                    args=summary,
                )
                if summary.get("applied"):
                    working_workflow = patched_workflow

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(
                            {
                                "workflow": working_workflow,
                                "summary": summary,
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
            continue

        content = msg.content or ""
        text = content.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if "\n" in text:
                first_line, rest = text.split("\n", 1)
                if first_line.strip().lower().startswith("json"):
                    text = rest

        if not text:
            log_warn("[repair_workflow_with_llm] LLM 返回空内容，使用最近一次工具输出。")
            return working_workflow

        if _looks_like_patch(text):
            patched_workflow = _apply_patch_output(working_workflow, text)
            if patched_workflow is not None:
                log_info("[repair_workflow_with_llm] 接收到补丁输出，已通过 git apply 合并。")
                return patched_workflow

            log_warn("[repair_workflow_with_llm] 补丁输出应用失败，尝试直接解析 JSON。")

        try:
            repaired_workflow = json.loads(text)
        except json.JSONDecodeError:
            log_error("[repair_workflow_with_llm] 无法解析模型返回 JSON")
            log_debug(content)
            raise

        return repaired_workflow


__all__ = [
    "_convert_pydantic_errors",
    "_make_failure_validation_error",
    "_repair_with_llm_and_fallback",
    "apply_rule_based_repairs",
    "repair_workflow_with_llm",
]
