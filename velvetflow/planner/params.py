# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""LLM-powered parameter completion utilities."""

import copy
import json
import os
from collections import deque
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from openai import OpenAI

from velvetflow.config import OPENAI_MODEL
from velvetflow.logging_utils import (
    child_span,
    log_debug,
    log_error,
    log_event,
    log_llm_message,
    log_llm_usage,
    log_tool_call,
)
from velvetflow.models import ALLOWED_PARAM_AGGREGATORS, Node, Workflow
from velvetflow.planner.params_tools import build_param_completion_tool
from velvetflow.planner.relations import get_referenced_nodes
from velvetflow.reference_utils import normalize_reference_path
from velvetflow.verification.validation import (
    _check_array_item_field,
    _check_output_path_against_schema,
)


def _find_start_nodes_for_params(workflow: Workflow) -> List[str]:
    starts = [n.id for n in workflow.nodes if n.type == "start"]
    if starts:
        return starts

    to_ids = {e.to_node for e in workflow.edges}
    candidates = [n.id for n in workflow.nodes if n.id not in to_ids]
    if candidates:
        return candidates

    return [workflow.nodes[0].id] if workflow.nodes else []


def _traverse_order(workflow: Workflow) -> List[str]:
    """按 start -> downstream 的顺序遍历，保证上游节点先被处理。"""

    adj: Dict[str, List[str]] = {}
    for e in workflow.edges:
        adj.setdefault(e.from_node, []).append(e.to_node)

    visited: set[str] = set()
    order: List[str] = []
    dq: deque[str] = deque(_find_start_nodes_for_params(workflow))

    while dq:
        nid = dq.popleft()
        if nid in visited:
            continue
        visited.add(nid)
        order.append(nid)
        for nxt in adj.get(nid, []):
            if nxt not in visited:
                dq.append(nxt)

    for node in workflow.nodes:
        if node.id not in visited:
            order.append(node.id)

    return order


def _looks_like_entity_key(field_name: str) -> bool:
    lower = field_name.lower()
    return any(token in lower for token in ["id", "code", "key", "uuid"])


def _extract_binding_sources(
    obj: Any, *, path_prefix: str = "params"
) -> List[Tuple[str, str]]:
    """Return all bindings within a params object as (path, __from__) pairs."""

    bindings: List[Tuple[str, str]] = []

    if isinstance(obj, dict):
        if isinstance(obj.get("__from__"), str):
            bindings.append(
                (path_prefix, normalize_reference_path(obj["__from__"]))
            )
        for k, v in obj.items():
            child_prefix = f"{path_prefix}.{k}"
            bindings.extend(_extract_binding_sources(v, path_prefix=child_prefix))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            bindings.extend(
                _extract_binding_sources(item, path_prefix=f"{path_prefix}[{idx}]")
            )

    return bindings


def _build_binding_memory(
    filled_params: Mapping[str, Mapping[str, Any]], processed_node_ids: Sequence[str]
) -> Dict[str, str]:
    """Track key entity bindings from already补全的节点，供跨节点一致性检查使用。"""

    memory: Dict[str, str] = {}

    for nid in processed_node_ids:
        params = filled_params.get(nid)
        if not isinstance(params, Mapping):
            continue
        for path, source in _extract_binding_sources(params):
            field = path.split(".")[-1]
            if _looks_like_entity_key(field) and field not in memory:
                memory[field] = source

    return memory


def _summarize_output_fields_from_schema(
    schema: Mapping[str, Any] | None,
) -> List[str]:
    if not isinstance(schema, Mapping):
        return []
    props = schema.get("properties")
    if isinstance(props, Mapping):
        return [k for k in props.keys() if isinstance(k, str)]
    return []


def _summarize_node_outputs(
    node: Node, action_schemas: Mapping[str, Dict[str, Any]]
) -> List[str]:
    if node.type == "loop":
        exports = node.params.get("exports") if isinstance(node.params, Mapping) else {}
        fields: List[str] = []
        if isinstance(exports, Mapping):
            items = exports.get("items")
            if isinstance(items, Mapping):
                item_fields = items.get("fields")
                if isinstance(item_fields, list):
                    fields.extend([f for f in item_fields if isinstance(f, str)])
            aggregates = exports.get("aggregates")
            if isinstance(aggregates, Mapping):
                fields.extend([f"aggregates.{k}" for k in aggregates.keys() if isinstance(k, str)])
        return fields

    if not node.action_id:
        return []
    schema = getattr(node, "out_params_schema", None) or action_schemas.get(
        node.action_id, {}
    ).get("output_schema")
    return _summarize_output_fields_from_schema(schema)


def _build_global_context(
    *,
    workflow: Workflow,
    action_schemas: Mapping[str, Dict[str, Any]],
    filled_params: Mapping[str, Mapping[str, Any]],
    processed_node_ids: Sequence[str],
    binding_memory: Mapping[str, str],
) -> Dict[str, Any]:
    upstream_map: Dict[str, List[str]] = {}
    for e in workflow.edges:
        upstream_map.setdefault(e.to_node, []).append(e.from_node)

    node_summaries: List[Dict[str, Any]] = []
    for n in workflow.nodes:
        schema = action_schemas.get(n.action_id, {}) if n.action_id else {}
        node_summaries.append(
            {
                "id": n.id,
                "type": n.type,
                "action_id": n.action_id,
                "display_name": n.display_name,
                "domain": schema.get("domain"),
                "out_params_schema": getattr(n, "out_params_schema", None)
                or schema.get("output_schema"),
                "output_fields": _summarize_node_outputs(n, action_schemas),
                "arg_required_fields": (
                    schema.get("arg_schema", {}).get("required") if isinstance(schema.get("arg_schema"), Mapping) else None
                ),
                "upstream": upstream_map.get(n.id, []),
                "params_snapshot": filled_params.get(n.id) if n.id in processed_node_ids else None,
            }
        )

    return {
        "workflow": {
            "name": workflow.workflow_name,
            "description": workflow.description,
        },
        "node_summaries": node_summaries,
        "entity_binding_hints": [
            {"field": field, "source": src} for field, src in binding_memory.items()
        ],
    }


def _collect_binding_issues(
    params: Dict[str, Any],
    upstream_nodes: List[Node],
    action_schemas: Dict[str, Dict[str, Any]],
    binding_memory: Mapping[str, str] | None = None,
) -> List[str]:
    """Validate __from__ / field references against upstream schemas.

    The function only checks bindings that point to ``result_of.<node_id>``
    sources. When a binding or its nested pipeline/count_if field refers to a
    missing node or field, the issue is returned as a human-readable string so
    that the caller can surface it early and trigger automated repair tools.
    """

    nodes_by_id: Dict[str, Dict[str, Any]] = {
        n.id: n.model_dump(by_alias=True) for n in upstream_nodes
    }
    actions_by_id = {aid: schema for aid, schema in action_schemas.items()}

    issues: List[str] = []

    def _walk(obj: Any, path_prefix: str = "params") -> None:
        if isinstance(obj, dict):
            if "__from__" in obj:
                src = normalize_reference_path(obj.get("__from__"))
                schema_err = _check_output_path_against_schema(
                    src, nodes_by_id, actions_by_id
                )
                if schema_err:
                    issues.append(f"{path_prefix}: {schema_err}")

                field_name = path_prefix.split(".")[-1]
                remembered = (binding_memory or {}).get(field_name)
                if remembered and normalize_reference_path(remembered) != src:
                    issues.append(
                        f"{path_prefix}: 字段 {field_name} 需要与之前的绑定保持一致（之前来源 {remembered}，当前 {src}）。"
                    )

                agg = obj.get("__agg__")
                if agg is not None and agg not in ALLOWED_PARAM_AGGREGATORS:
                    issues.append(
                        f"{path_prefix}: __agg__ 取值非法（{agg}），可选值：{', '.join(ALLOWED_PARAM_AGGREGATORS)}。"
                    )

                field_checks: List[tuple[str, str]] = []
                if agg == "count_if":
                    fld = obj.get("field")
                    if isinstance(fld, str):
                        field_checks.append(("count_if.field", fld))
                if agg == "pipeline":
                    steps = obj.get("steps")
                    if isinstance(steps, list):
                        for idx, step in enumerate(steps):
                            if not isinstance(step, dict):
                                continue
                            if step.get("op") == "filter" and isinstance(
                                step.get("field"), str
                            ):
                                field_checks.append(
                                    (
                                        f"pipeline.steps[{idx}].field",
                                        step["field"],
                                    )
                                )

                for field_label, fld in field_checks:
                    item_err = _check_array_item_field(
                        src, fld, nodes_by_id, actions_by_id
                    )
                    if item_err:
                        issues.append(
                            f"{path_prefix}.{field_label}: {item_err}"
                        )

            for k, v in obj.items():
                new_prefix = f"{path_prefix}.{k}" if path_prefix else k
                _walk(v, new_prefix)

        elif isinstance(obj, list):
            for idx, v in enumerate(obj):
                _walk(v, f"{path_prefix}[{idx}]")

    _walk(params)

    return issues


def _validate_node_params(
    *,
    node: Node,
    params: Dict[str, Any],
    upstream_nodes: List[Node],
    action_schemas: Dict[str, Dict[str, Any]],
    binding_memory: Mapping[str, str] | None = None,
) -> List[str]:
    """Validate a single node's params after every tool submission.

    The checks cover:
    - required/unknown fields against the node's arg_schema when available
    - binding legality for result_of references via ``_collect_binding_issues``
    """

    errors: List[str] = []

    if not isinstance(params, dict):
        return ["params 必须是对象"]

    schema = action_schemas.get(node.action_id or "", {}).get("arg_schema", {})
    required_fields = []
    properties = None
    allow_additional = True

    if isinstance(schema, dict):
        required_fields = schema.get("required") or []
        properties = (
            schema.get("properties") if isinstance(schema.get("properties"), dict) else None
        )
        allow_additional = bool(schema.get("additionalProperties", True))

    for field in required_fields:
        if field not in params:
            errors.append(f"缺少必填字段 {field}")

    if isinstance(properties, dict) and not allow_additional:
        for field in params:
            if field not in properties:
                errors.append(f"参数 {field} 未在 arg_schema 中定义")

    errors.extend(
        _collect_binding_issues(
            params,
            upstream_nodes=upstream_nodes,
            action_schemas=action_schemas,
            binding_memory=binding_memory,
        )
    )

    return errors


def _build_validation_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "validate_node_params",
            "description": "校验当前节点的 params 是否满足 arg_schema 与绑定规则，必须在每次提交后立即调用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "要校验的节点 id"},
                    "params": {
                        "type": "object",
                        "description": "待校验的参数对象，不传则使用最近一次提交的 params。",
                    },
                },
                "required": ["id"],
                "additionalProperties": False,
            },
        },
    }


def fill_params_with_llm(
    workflow_skeleton: Dict[str, Any],
    action_registry: List[Dict[str, Any]],
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

    system_prompt = (
        "你是一个工作流参数补全助手。一次只处理一个节点，请根据给定的 arg_schema 补齐当前节点的 params。\n"
        "请务必调用 submit_node_params 工具提交结果，禁止返回自然语言。每次提交后必须调用 validate_node_params 工具进行校验，收到校验错误要重新分析并再次提交。\n"
        "当某个字段需要引用其他节点的输出时，必须直接写成 Jinja 表达式或模板字符串，例如 {{ result_of.<node_id>.<field> }}，"
        "node_id 只能来自 allowed_node_ids，field 必须存在于该节点的 output_schema。禁止再使用旧的 __from__/__agg__ 绑定 DSL。\n"
        "你的所有 params 值都会直接交给 Jinja2 引擎解析，任何非 Jinja 语法（包括残留的绑定对象或伪代码）都会被视为错误并触发自动修复，请务必输出可被 Jinja 渲染的字符串或字面量。\n"
        "循环场景下可使用 loop.item/loop.index 以及 loop.exports.* 暴露的字段，依然通过 Jinja 表达式引用，禁止直接引用 loop body 的节点。\n"
        "引用 loop.exports 结构时必须指向 exports 内部的具体字段或子结构，不能只写 result_of.<loop_id>.exports。\n"
        "若 loop.exports.items.fields 仅包含用来包裹完整输出的字段（如 data/record 等），需要通过 <字段>.<子字段> 的形式访问内部属性，不能直接写子字段名。\n"
        "所有节点的 params 不得为空；如无 obvious 默认值，需要分析上下游语义后调用工具补全。\n"
        "聚合/筛选/格式化也请用 Jinja 过滤器或表达式完成（例如 {{ result_of.a.items | selectattr('score', '>', 80) | list | length }} 或 {{ result_of.a.items | map(attribute='name') | join(', ') }}），不要再输出带 __agg__ 的对象。"
    )

    workflow = Workflow.model_validate(workflow_skeleton)
    nodes_by_id: Dict[str, Node] = {n.id: n for n in workflow.nodes}
    filled_params: Dict[str, Dict[str, Any]] = {
        n.id: copy.deepcopy(n.params) for n in workflow.nodes
    }

    processed_node_ids: List[str] = []

    for node_id in _traverse_order(workflow):
        node = nodes_by_id[node_id]
        upstream_nodes = get_referenced_nodes(workflow, node_id)
        allowed_node_ids = [n.id for n in upstream_nodes]

        binding_memory = _build_binding_memory(
            filled_params, processed_node_ids
        )

        global_context = _build_global_context(
            workflow=workflow,
            action_schemas=action_schemas,
            filled_params=filled_params,
            processed_node_ids=processed_node_ids,
            binding_memory=binding_memory,
        )

        upstream_context = []
        for n in upstream_nodes:
            action_schema = action_schemas.get(n.action_id, {}) if n.action_id else {}
            upstream_context.append(
                {
                    "id": n.id,
                    "type": n.type,
                    "action_id": n.action_id,
                    "output_schema": action_schema.get("output_schema"),
                    "params": filled_params.get(n.id, n.params),
                }
            )

        target_action_schema = action_schemas.get(node.action_id, {}) if node.action_id else {}
        user_payload = {
            "target_node": {
                "id": node.id,
                "type": node.type,
                "action_id": node.action_id,
                "display_name": node.display_name,
                "existing_params": filled_params[node.id],
            },
            "arg_schema": target_action_schema.get("arg_schema"),
            "allowed_node_ids": allowed_node_ids,
            "allowed_upstream_nodes": upstream_context,
            "global_context": global_context,
        }

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        validated_params: Dict[str, Any] | None = None
        last_submitted_params: Dict[str, Any] | None = None

        for round_idx in range(4):
            with child_span(f"fill_params_{node.id}_round_{round_idx}"):
                resp = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=[
                        build_param_completion_tool(
                            node,
                            action_schemas=action_schemas,
                            allowed_node_ids=allowed_node_ids,
                        ),
                        _build_validation_tool(),
                    ],
                    tool_choice="auto",
                    temperature=0.1,
                )
            log_llm_usage(model, getattr(resp, "usage", None), operation="fill_params")
            if not resp.choices:
                raise RuntimeError(f"fill_params_with_llm({node.id}) 未返回任何候选消息")

            message = resp.choices[0].message
            log_llm_message(
                model,
                message,
                operation="fill_params",
                node_id=node.id,
                action_id=node.action_id,
            )
            tool_calls = getattr(message, "tool_calls", None) or []
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": tool_calls,
                }
            )

            validation_performed = False
            validation_errors: List[str] = []

            for tc in tool_calls:
                tool_call_id = getattr(tc, "id", None)
                func_name = tc.function.name
                raw_args = tc.function.arguments

                log_event(
                    "params_tool_call",
                    {
                        "node_id": node.id,
                        "tool_name": func_name,
                        "tool_call_id": tool_call_id,
                        "raw_arguments": raw_args,
                    },
                    node_id=node.id,
                    action_id=node.action_id,
                )

                try:
                    args = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    log_error("[fill_params_with_llm] 无法解析 tool_call 参数 JSON")
                    log_debug(raw_args)
                    args = {}

                log_tool_call(
                    source=f"fill_params:{node.id}",
                    tool_name=func_name,
                    tool_call_id=tool_call_id,
                    args=args or raw_args,
                )

                if func_name == "submit_node_params":
                    if args.get("id") != node.id:
                        continue
                    tool_params = args.get("params", {})
                    if isinstance(tool_params, dict):
                        last_submitted_params = tool_params
                        log_event(
                            "params_tool_result",
                            {
                                "node_id": node.id,
                                "tool_name": func_name,
                                "params": tool_params,
                            },
                            node_id=node.id,
                            action_id=node.action_id,
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call_id,
                                "content": json.dumps(
                                    {
                                        "status": "received",
                                        "message": "已收到提交，将立即进行校验。",
                                    },
                                    ensure_ascii=False,
                                ),
                            }
                        )

                elif func_name == "validate_node_params":
                    if args.get("id") != node.id:
                        continue
                    validation_performed = True
                    params_to_check = (
                        args.get("params") if isinstance(args.get("params"), dict) else last_submitted_params
                    )
                    if params_to_check is None:
                        validation_errors = ["缺少待校验的 params（请先提交或在参数中提供 params）"]
                    else:
                        validation_errors = _validate_node_params(
                            node=node,
                            params=params_to_check,
                            upstream_nodes=upstream_nodes,
                            action_schemas=action_schemas,
                            binding_memory=binding_memory,
                        )

                    tool_payload = {
                        "status": "error" if validation_errors else "ok",
                        "errors": validation_errors or None,
                    }
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(tool_payload, ensure_ascii=False),
                        }
                    )
                    if not validation_errors and params_to_check is not None:
                        validated_params = params_to_check
                        filled_params[node.id] = params_to_check

                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": json.dumps(
                                {
                                    "status": "error",
                                    "message": f"未知工具 {func_name}",
                                },
                                ensure_ascii=False,
                            ),
                        }
                    )

            if validated_params is not None:
                break

            if last_submitted_params is not None and not validation_performed:
                validation_errors = _validate_node_params(
                    node=node,
                    params=last_submitted_params,
                    upstream_nodes=upstream_nodes,
                    action_schemas=action_schemas,
                    binding_memory=binding_memory,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "status": "error" if validation_errors else "ok",
                                "errors": validation_errors or None,
                                "message": "自动校验提交的 params。请根据校验结果修正参数并再次调用 submit_node_params 和 validate_node_params。",
                            },
                            ensure_ascii=False,
                        ),
                    }
                )
                if not validation_errors:
                    validated_params = last_submitted_params
                    filled_params[node.id] = last_submitted_params
                    break

        if validated_params is None:
            raise ValueError(f"节点 {node.id} 的参数补全未通过校验，请检查工具调用返回。")

        processed_node_ids.append(node.id)

    completed_nodes = []
    for node in workflow.nodes:
        data = node.model_dump(by_alias=True)
        data["params"] = filled_params.get(node.id, node.params)
        completed_nodes.append(data)

    completed_workflow = {
        "workflow_name": workflow.workflow_name,
        "description": workflow.description,
        "nodes": completed_nodes,
        "edges": [e.model_dump(by_alias=True) for e in workflow.edges],
    }

    return completed_workflow


__all__ = ["fill_params_with_llm"]
