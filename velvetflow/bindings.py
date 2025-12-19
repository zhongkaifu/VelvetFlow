# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Parameter binding resolution for workflow execution."""

import ast
import copy
import json
import re
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from velvetflow.aggregation import (
    JinjaExprValidationError,
    _is_instance_of_type,
    eval_jinja_expression,
    validate_jinja_expression,
)
from velvetflow.action_registry import get_action_by_id
from velvetflow.loop_dsl import build_loop_output_schema
from velvetflow.logging_utils import log_warn
from velvetflow.models import ALLOWED_PARAM_AGGREGATORS, Node, Workflow
from velvetflow.reference_utils import (
    canonicalize_template_placeholders,
    normalize_reference_path,
    parse_field_path,
)
from velvetflow.jinja_utils import render_jinja_template


class BindingContext:
    def __init__(
        self,
        workflow: Workflow,
        results: Dict[str, Any],
        *,
        extra_nodes: Optional[Mapping[str, Node]] = None,
        loop_ctx: Optional[Dict[str, Any]] = None,
        loop_id: Optional[str] = None,
    ):
        self.workflow = workflow
        self.results = results  # node_id -> result object
        self.loop_ctx = loop_ctx or {}
        self.loop_id = loop_id
        self._nodes = {n.id: n for n in workflow.nodes}
        if extra_nodes:
            self._nodes.update(extra_nodes)

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of the binding context."""

        return {
            "results": copy.deepcopy(self.results),
            "loop_ctx": copy.deepcopy(self.loop_ctx),
            "loop_id": self.loop_id,
        }

    @classmethod
    def from_snapshot(
        cls,
        workflow: Workflow,
        snapshot: Mapping[str, Any],
        *,
        extra_nodes: Optional[Mapping[str, Node]] = None,
    ) -> "BindingContext":
        """Restore a binding context from a serialized snapshot."""

        return cls(
            workflow,
            copy.deepcopy(snapshot.get("results", {})),
            extra_nodes=extra_nodes,
            loop_ctx=copy.deepcopy(snapshot.get("loop_ctx", {})),
            loop_id=snapshot.get("loop_id"),
        )

    @staticmethod
    def save_snapshot_to_file(snapshot: Mapping[str, Any], file_path: str | Path) -> None:
        """Persist a binding context snapshot to disk as JSON."""

        with open(file_path, "w", encoding="utf-8") as fp:
            json.dump(snapshot, fp, ensure_ascii=False, indent=2)

    @classmethod
    def load_snapshot_from_file(
        cls,
        workflow: Workflow,
        file_path: str | Path,
        *,
        extra_nodes: Optional[Mapping[str, Node]] = None,
    ) -> "BindingContext":
        """Load a binding context snapshot from disk and restore the context."""

        with open(file_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        return cls.from_snapshot(workflow, payload, extra_nodes=extra_nodes)

    def _format_field_path(self, parts: List[Any]) -> str:
        path_str = ""
        for part in parts:
            if isinstance(part, int):
                path_str += f"[{part}]"
            else:
                path_str += ("." if path_str else "") + str(part)
        return path_str

    def _qualify_context_path(self, path: Any) -> Any:
        """Normalize context references and prepend ``result_of.`` when omitted.

        Some authoring tools emit bare node references (e.g.,
        ``search_nvidia_news.results[*].snippet``) inside template placeholders.
        To keep compatibility with the rest of the resolver, this helper detects
        such cases and rewrites them to the canonical ``result_of.<node_id>``
        form. Non-string inputs or paths already pointing to loop context or
        ``result_of`` are returned unchanged.
        """

        normalized = normalize_reference_path(path)
        if not isinstance(normalized, str):
            return normalized

        try:
            parts = parse_field_path(normalized)
        except Exception:
            return normalized

        if not parts:
            return normalized

        head = parts[0]
        if head in {self.loop_id, "loop"} or head in self.loop_ctx:
            return normalized

        if head == "result_of" or head not in self._nodes:
            return normalized

        return f"result_of.{normalized}"

    def build_jinja_context(self) -> Dict[str, Any]:
        """Return a runtime context for rendering Jinja templates.

        This includes loop state, workflow execution state (if provided by the
        caller), and the accumulated node results so params can freely refer to
        ``workflow.*``, ``loop.*`` and ``result_of.*`` inside templates.
        """

        workflow_ctx: Any = None
        # Prefer a dedicated ``context`` field if the workflow carries one;
        # otherwise expose the serialized workflow object for callers that
        # inject execution_date or other metadata via the workflow payload.
        if hasattr(self.workflow, "context"):
            workflow_ctx = getattr(self.workflow, "context")
        else:  # pragma: no cover - fallback for custom workflow payloads
            try:
                workflow_ctx = self.workflow.model_dump(by_alias=True)
            except Exception:
                workflow_ctx = None

        ctx: Dict[str, Any] = {
            "result_of": self.results,
            "loop": self.loop_ctx,
            "loop_ctx": self.loop_ctx,
            "loop_id": self.loop_id,
            **self.loop_ctx,
            "env": dict(os.environ),
            "system": {"date": datetime.now().date().isoformat()},
        }

        if workflow_ctx:
            ctx["workflow"] = workflow_ctx

        return ctx

    def _infer_loop_reference_path(self, src_path: str) -> str:
        """Try to recover missing loop exports segments in result paths."""

        try:
            parts = parse_field_path(src_path)
        except Exception:
            return src_path

        if len(parts) < 2 or parts[0] != "result_of":
            return src_path

        node_id = parts[1]
        node = self._nodes.get(node_id)
        if not node or node.type != "loop":
            return src_path

        rest_path = parts[2:]
        params = node.params or {}
        exports = params.get("exports") if isinstance(params, Mapping) else None
        if not isinstance(exports, Mapping):
            body_graph = params.get("body_subgraph") if isinstance(params, Mapping) else None
            if isinstance(body_graph, Mapping):
                exports = body_graph.get("exports") if isinstance(body_graph.get("exports"), Mapping) else None

        if not isinstance(exports, Mapping):
            return src_path

        normalized_rest = rest_path[1:] if rest_path and rest_path[0] == "exports" else rest_path
        if not normalized_rest:
            return src_path

        export_keys = {key for key in exports.keys() if isinstance(key, str)}
        if normalized_rest[0] not in export_keys:
            return src_path

        corrected_parts = ["result_of", node_id, "exports", *normalized_rest]
        return self._format_field_path(corrected_parts)

    def resolve_binding(self, binding: dict) -> Any:
        """解析 __from__, __agg__, pipeline 等绑定 DSL。"""

        def _render(val: Any, fmt: Optional[str], field: Optional[str] = None) -> str:
            template_str = fmt or "{{ value }}"
            normalized = canonicalize_template_placeholders(template_str)
            if "{{" not in normalized and "}}" not in normalized and "{" in normalized:
                normalized = re.sub(r"\{([^{}]+)\}", r"{{ \1 }}", normalized)

            context: Dict[str, Any] = {"value": val, "item": val, "field": field}
            if isinstance(val, Mapping):
                context.update(val)

            try:
                return render_jinja_template(normalized, context)
            except Exception as exc:  # pragma: no cover - fallback path
                log_warn(
                    f"[binding] 模板 {normalized!r} 渲染失败: {exc}，回退到 str.format"
                )
                if isinstance(val, Mapping):
                    selected = val.get(field) if field else val
                    data: Mapping[str, Any] = _SafeDict(val)
                    if fmt:
                        try:
                            return fmt.format_map(data)
                        except Exception:
                            pass
                    return "" if selected is None else str(selected)

                selected = val
                if fmt:
                    try:
                        return fmt.format_map(_SafeDict({}))
                    except Exception:
                        pass
                return "" if selected is None else str(selected)

        if not isinstance(binding, dict) or "__from__" not in binding:
            return binding

        src_path = normalize_reference_path(binding["__from__"])
        resolved_path = src_path
        try:
            self._validate_result_reference(resolved_path)
        except ValueError:
            inferred_path = self._infer_loop_reference_path(resolved_path)
            if inferred_path == resolved_path:
                raise
            self._validate_result_reference(inferred_path)
            log_warn(
                f"[binding] 自动补全 loop 引用 '{src_path}' -> '{inferred_path}'"
            )
            resolved_path = inferred_path

        data = self._get_from_context(resolved_path)
        agg_spec = binding.get("__agg__", "identity")
        agg_op = agg_spec.get("op") if isinstance(agg_spec, Mapping) else agg_spec
        if agg_op not in ALLOWED_PARAM_AGGREGATORS:
            raise ValueError(
                f"__agg__ 取值非法（{agg_op}），允许值：{', '.join(ALLOWED_PARAM_AGGREGATORS)}"
            )

        input_type = agg_spec.get("input_type") if isinstance(agg_spec, Mapping) else None
        output_type = agg_spec.get("output_type") if isinstance(agg_spec, Mapping) else None
        on_empty = agg_spec.get("on_empty") if isinstance(agg_spec, Mapping) else None
        condition_expr = agg_spec.get("condition") if isinstance(agg_spec, Mapping) else None
        if condition_expr is not None:
            try:
                validate_jinja_expression(condition_expr, path="__agg__.condition")
            except JinjaExprValidationError as exc:
                raise ValueError(f"__agg__.condition 非法: {exc}") from exc

        def _apply_on_empty(result: Any) -> Any:
            is_empty = result is None
            if not is_empty:
                try:
                    is_empty = len(result) == 0  # type: ignore[arg-type]
                except Exception:
                    is_empty = False

            if not is_empty:
                return result

            if isinstance(on_empty, Mapping) and "default" in on_empty:
                return on_empty.get("default")
            if on_empty == "error" or (
                isinstance(on_empty, Mapping) and on_empty.get("mode") == "error"
            ):
                raise ValueError("聚合结果为空且 on_empty=error")
            return result

        def _finalize(result: Any) -> Any:
            result = _apply_on_empty(result)
            if not _is_instance_of_type(result, output_type):
                raise ValueError(
                    f"__agg__={agg_op} 输出类型不符，期望 {output_type}, 收到 {type(result).__name__}"
                )
            return result

        if not _is_instance_of_type(data, input_type):
            raise ValueError(
                f"__agg__={agg_op} 期望输入类型 {input_type or '任意'}，收到 {type(data).__name__}"
            )

        if agg_op == "identity":
            return _finalize(data)

        if agg_op == "count":
            if isinstance(data, list):
                return _finalize(len(data))
            if isinstance(data, Mapping):
                return _finalize(len(data))
            return _finalize(0 if data is None else 1)

        if agg_op == "count_if":
            if not isinstance(data, list):
                return 0
            field = binding.get("field") or binding.get("filter_field")
            op = binding.get("op") or binding.get("filter_op") or "=="
            target = binding.get("value") or binding.get("filter_value")
            count = 0

            def _match(item: Any) -> bool:
                if condition_expr is not None:
                    try:
                        return bool(
                            eval_jinja_expression(
                                condition_expr,
                                {"item": item, "value": item, "field": field},
                            )
                        )
                    except Exception:
                        return False
                v = item.get(field) if isinstance(item, Mapping) else item
                try:
                    if op == ">" and v is not None and v > target:
                        return True
                    if op == ">=" and v is not None and v >= target:
                        return True
                    if op == "<" and v is not None and v < target:
                        return True
                    if op == "<=" and v is not None and v <= target:
                        return True
                    if op == "==" and v == target:
                        return True
                    if op == "!=" and v != target:
                        return True
                except TypeError:
                    return False
                return False

            for item in data:
                if _match(item):
                    count += 1
            return _finalize(count)

        if agg_op == "join":
            sep = binding.get("separator")
            if sep is None:
                sep = binding.get("sep", "")
            if not isinstance(sep, str):
                sep = str(sep)

            if isinstance(data, list):
                return _finalize(sep.join("" if v is None else str(v) for v in data))

            return _finalize("" if data is None else str(data))

        if agg_op == "format_join":
            field = binding.get("field")
            fmt = binding.get("format") or (f"{{{field}}}" if field else None)
            sep = binding.get("sep", "\n")

            if not isinstance(data, list):
                return _render(data, fmt, field)

            rendered: List[str] = []
            for item in data:
                rendered.append(_render(item, fmt, field))
            return _finalize(sep.join(rendered))

        if agg_op == "filter_map":
            filter_field = binding.get("filter_field")
            filter_op = binding.get("filter_op", "==")
            filter_value = binding.get("filter_value")
            map_field = binding.get("map_field")
            fmt = binding.get("format") or (f"{{{map_field}}}" if map_field else None)
            sep = binding.get("sep", "\n")

            if not isinstance(data, list):
                return _finalize(_render(data, fmt))

            lines: List[str] = []
            for item in data:
                if not isinstance(item, Mapping):
                    continue
                v = item.get(filter_field)
                passed = False
                if condition_expr is not None:
                    try:
                        passed = bool(
                            eval_jinja_expression(
                                condition_expr,
                                {"item": item, "value": v, "field": filter_field},
                            )
                        )
                    except Exception:
                        passed = False
                elif filter_op == ">" and v is not None and v > filter_value:
                    passed = True
                elif filter_op == ">=" and v is not None and v >= filter_value:
                    passed = True
                elif filter_op == "<" and v is not None and v < filter_value:
                    passed = True
                elif filter_op == "<=" and v is not None and v <= filter_value:
                    passed = True
                elif filter_op == "==" and v == filter_value:
                    passed = True
                elif filter_op == "!=" and v != filter_value:
                    passed = True
                elif filter_op in (None, "", "always"):
                    passed = True
                if passed:
                    lines.append(_render(item, fmt, map_field))

            return _finalize(sep.join(lines))

        if agg_op == "pipeline":
            steps = binding.get("steps") or []
            current = data

            for step in steps:
                if not isinstance(step, dict):
                    continue
                op = step.get("op")

                if op == "filter":
                    if not isinstance(current, list):
                        continue
                    field = step.get("field") or step.get("filter_field")
                    cmp_op = (
                        step.get("cmp")
                        or step.get("filter_op")
                        or step.get("operator")
                        or "=="
                    )
                    cmp_val = step.get("value") or step.get("filter_value")
                    cond_expr = step.get("condition")
                    if cond_expr is not None:
                        try:
                            validate_jinja_expression(cond_expr, path="pipeline.steps[].condition")
                        except JinjaExprValidationError:
                            cond_expr = None

                    filtered = []
                    for item in current:
                        if not isinstance(item, dict):
                            continue
                        v = item.get(field)
                        passed = False
                        if cond_expr is not None:
                            try:
                                passed = bool(
                                    eval_jinja_expression(
                                        cond_expr,
                                        {"item": item, "value": v, "field": field},
                                    )
                                )
                            except Exception:
                                passed = False
                        elif cmp_op == ">" and v is not None and v > cmp_val:
                            passed = True
                        elif cmp_op == ">=" and v is not None and v >= cmp_val:
                            passed = True
                        elif cmp_op == "<" and v is not None and v < cmp_val:
                            passed = True
                        elif cmp_op == "<=" and v is not None and v <= cmp_val:
                            passed = True
                        elif cmp_op == "==" and v == cmp_val:
                            passed = True
                        elif cmp_op == "!=" and v != cmp_val:
                            passed = True
                        elif cmp_op in (None, "", "always"):
                            passed = True
                        if passed:
                            filtered.append(item)
                    current = filtered

                elif op == "format_join":
                    field = step.get("field")
                    fmt = step.get("format") or (f"{{{field}}}" if field else None)
                    sep = step.get("sep", "\n")

                    if not isinstance(current, list):
                        return _finalize(_render(current, fmt, field))
                    lines: List[str] = []
                    for v in current:
                        lines.append(_render(v, fmt, field))
                    return _finalize(sep.join(lines))

            return _finalize(current)

        return _finalize(data)

    def _schema_has_path(self, schema: Mapping[str, Any], fields: List[Any]) -> bool:
        if not isinstance(schema, Mapping):
            return False

        current: Mapping[str, Any] = schema
        idx = 0
        while idx < len(fields):
            name = fields[idx]
            typ = current.get("type")

            if isinstance(name, int):
                if typ != "array":
                    return False
                current = current.get("items") or {}
                idx += 1
                continue

            if name == "*":
                if typ != "array":
                    return False
                current = current.get("items") or {}
                idx += 1
                continue

            if typ == "array":
                if name in {"length", "count"}:
                    current = {"type": "integer"}
                    idx += 1
                    continue
                current = current.get("items") or {}
                # array 本身不消费字段名，继续在 items 上检查同一个字段
                continue

            if typ == "object" or typ is None:
                if name == "count":
                    current = {"type": "integer"}
                    idx += 1
                    continue
                props = current.get("properties") or {}
                if name not in props:
                    return False
                current = props[name]
                idx += 1
                continue

            if idx == len(fields) - 1:
                return True

            return False

        return True

    def _validate_result_reference(self, src_path: str) -> None:
        """Ensure __from__ references point to valid node outputs."""

        src_path = normalize_reference_path(src_path)
        if not isinstance(src_path, str) or not src_path.startswith("result_of."):
            return

        try:
            parts = parse_field_path(src_path)
        except Exception:
            raise ValueError(f"__from__ 路径 '{src_path}' 不是合法的 result_of 引用")

        if len(parts) < 2 or parts[0] != "result_of":
            raise ValueError(f"__from__ 路径 '{src_path}' 不是合法的 result_of 引用")

        node_id = parts[1]
        node = self._nodes.get(node_id)
        if not node:
            raise ValueError(f"__from__ 引用的节点 '{node_id}' 不存在")

        # loop 节点有虚拟 output_schema（exports）
        if node.type == "loop":
            loop_schema = build_loop_output_schema(node.params or {})
            if not loop_schema:
                raise ValueError(f"__from__ 引用的 loop 节点 '{node_id}' 未定义 exports")
            field_path = parts[2:]
            if field_path and field_path[0] == "exports":
                field_path = field_path[1:]
            if (
                field_path
                and field_path[-1] == "count"
                and self._schema_has_path(loop_schema, field_path[:-1])
            ):
                return

            if not self._schema_has_path(loop_schema, field_path):
                raise ValueError(
                    f"__from__ 路径 '{src_path}' 引用了 loop '{node_id}' 输出中不存在的字段"
                )
            return

        # 控制节点（condition/start/end 等）也允许被引用，缺少 action_id 时跳过 schema 校验
        if node.type != "action" or not node.action_id:
            return

        action_id = node.action_id
        action = get_action_by_id(action_id) if action_id else None
        if not action:
            raise ValueError(f"__from__ 引用的节点 '{node_id}' 缺少合法的 action_id")

        output_schema = node.out_params_schema or action.get("output_schema")
        arg_schema = action.get("arg_schema")
        field_path = parts[2:]
        if not field_path:
            return

        if field_path[0] == "params":
            arg_fields = field_path[1:]
            # 如果引用的是上游的输入参数，使用 arg_schema 进行校验（若缺失则跳过校验）
            if arg_fields and arg_schema and not self._schema_has_path(arg_schema, arg_fields):
                raise ValueError(
                    f"__from__ 路径 '{src_path}' 引用了 action '{action_id}' 输入中不存在的字段"
                )
            return

        if (
            field_path
            and field_path[-1] == "count"
            and self._schema_has_path(output_schema, field_path[:-1])
        ):
            return

        if not self._schema_has_path(output_schema, field_path):
            raise ValueError(
                f"__from__ 路径 '{src_path}' 引用了 action '{action_id}' 输出中不存在的字段"
            )

    def get_value(self, path: str) -> Any:
        """Public helper for resolving arbitrary context paths.

        condition 节点等控制流会直接把 source 作为字符串传入，这里复用
        _get_from_context 的逻辑，让调用方不需要了解内部实现细节。
        """

        return self._get_from_context(path)

    def _get_from_context(self, path: str):
        path = self._qualify_context_path(path)
        if not path:
            raise KeyError("空路径")

        try:
            parts = parse_field_path(path)
        except Exception:
            raise KeyError("空路径")

        resolved_parts: List[str] = []

        def _fmt_path(extra: Optional[str] = None) -> str:
            path_parts = resolved_parts.copy()
            if extra is not None:
                path_parts.append(extra)
            return ".".join(path_parts)

        def _append_token(token: Any) -> None:
            if isinstance(token, int) and resolved_parts:
                resolved_parts[-1] = f"{resolved_parts[-1]}[{token}]"
            else:
                resolved_parts.append(str(token))

        if self.loop_id and parts[0] == self.loop_id:
            cur = self.loop_ctx
            _append_token(parts[0])
            rest = parts[1:]
        elif parts[0] == "loop":
            cur: Any = self.loop_ctx
            _append_token("loop")
            for p in parts[1:]:
                if isinstance(p, int):
                    raise KeyError("loop_ctx 顶层不支持列表索引访问")

                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                    _append_token(p)
                    continue
                raise KeyError(f"{_fmt_path(str(p))}: 在 loop_ctx 中找不到字段")
            return cur

        elif parts[0] in self.loop_ctx:
            cur = self.loop_ctx[parts[0]]
            _append_token(parts[0])
            rest = parts[1:]
        elif parts[0] == "result_of" and len(parts) >= 2:
            first_key = parts[1]
            if first_key not in self.results:
                raise KeyError(f"result_of.{first_key}: 上游节点未执行或没有结果")
            cur = self.results[first_key]
            resolved_parts.extend(["result_of", str(first_key)])
            rest = parts[2:]
            node = self._nodes.get(first_key)
            if node and node.type == "loop" and rest and rest[0] == "exports":
                rest = rest[1:]
        else:
            context = {f"result_of.{nid}": value for nid, value in self.results.items()}
            if parts[0] not in context:
                raise KeyError(f"{parts[0]}: 上下文中不存在")
            cur = context[parts[0]]
            _append_token(parts[0])
            rest = parts[1:]

        for p in rest:
            if isinstance(p, str):
                func_match = re.match(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)$", p)
                if func_match:
                    func_name = func_match.group("name")
                    args_literal = func_match.group("args").strip()
                    try:
                        args = () if args_literal == "" else ast.literal_eval(f"({args_literal},)")
                    except Exception as exc:
                        raise ValueError(f"{_fmt_path(p)}: 函数参数解析失败: {exc}")

                    if func_name == "join":
                        sep = args[0] if args else ""
                        if not isinstance(sep, str):
                            sep = str(sep)
                        if isinstance(cur, list):
                            cur = sep.join("" if v is None else str(v) for v in cur)
                            _append_token(p)
                            continue
                        raise TypeError(
                            f"{_fmt_path()}: 值类型为 {type(cur).__name__}，不支持 join 函数"
                        )

                    raise ValueError(f"{_fmt_path(p)}: 不支持的内建函数 '{func_name}'")

                if p == "length":
                    try:
                        cur = len(cur)
                    except Exception:
                        raise TypeError(
                            f"{_fmt_path()}: 值类型为 {type(cur).__name__}，不支持 length 访问"
                        )
                    _append_token(p)
                    continue

                if p == "count":
                    try:
                        cur = len(cur)
                    except Exception:
                        raise TypeError(
                            f"{_fmt_path()}: 值类型为 {type(cur).__name__}，不支持 count 访问"
                        )
                    _append_token(p)
                    continue

            if p == "*":
                if not isinstance(cur, list):
                    raise TypeError(
                        f"{_fmt_path()}: 值类型为 {type(cur).__name__}，不支持通配访问"
                    )
                _append_token("*")
                continue

            if isinstance(p, int):
                if isinstance(cur, list):
                    if p < 0 or p >= len(cur):
                        raise KeyError(f"{_fmt_path()}[{p}]: 列表索引越界")
                    cur = cur[p]
                    _append_token(p)
                    continue
                raise TypeError(f"{_fmt_path()}: 当前值类型为 {type(cur).__name__}，不支持索引 {p}")

            if isinstance(cur, dict):
                if p not in cur:
                    raise KeyError(
                        f"{_fmt_path(p)}: 字段不存在，当前可用键: {sorted(cur.keys())}"
                    )
                cur = cur[p]
                _append_token(p)
                continue

            if isinstance(cur, list):
                if not cur:
                    raise KeyError(f"{_fmt_path()}: 列表为空，无法获取字段 '{p}'")

                extracted = []
                missing_count = 0
                invalid_types = set()
                for item in cur:
                    if isinstance(item, dict):
                        if p in item:
                            extracted.append(item[p])
                        else:
                            missing_count += 1
                    else:
                        invalid_types.add(type(item).__name__)

                if extracted:
                    cur = extracted
                    _append_token(p)
                    continue

                details = []
                if missing_count:
                    details.append(f"{missing_count} 个字典缺少字段 '{p}'")
                if invalid_types:
                    details.append(f"存在非字典元素类型: {sorted(invalid_types)}")
                raise KeyError(
                    f"{_fmt_path(p)}: 列表元素不包含目标字段，" + ", ".join(details)
                )

            raise TypeError(
                f"{_fmt_path()}: 值类型为 {type(cur).__name__}，不支持字段 '{p}' 访问"
            )

        return cur


class _SafeDict(dict):
    """A dict that leaves unknown placeholders intact during formatting."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def eval_node_params(node: Node, ctx: BindingContext) -> Dict[str, Any]:
    """遍历 node.params，如果是绑定 DSL，就用 ctx.resolve_binding 解析。"""

    template_pattern = re.compile(
        r"\{\{\s*([^{}]+?)\s*\}\}|\$\{\{\s*([^{}]+?)\s*\}\}|\$\{\s*([^{}]+?)\s*\}"
    )

    decoder = json.JSONDecoder()

    def _render_json_bindings(text: str) -> str:
        out: List[str] = []
        sentinel = object()
        idx = 0
        while idx < len(text):
            if text[idx] != "{":
                out.append(text[idx])
                idx += 1
                continue

            try:
                obj, end = decoder.raw_decode(text, idx)
            except json.JSONDecodeError:
                out.append(text[idx])
                idx += 1
                continue

            replacement: Any = sentinel
            if isinstance(obj, dict) and "__from__" in obj:
                try:
                    replacement = ctx.resolve_binding(obj)
                except Exception as e:  # pragma: no cover - log-only path
                    log_warn(
                        f"[param-resolver] 字符串内嵌绑定 {obj} 解析失败: {e}，使用原值"
                    )

            if replacement is sentinel:
                out.append(text[idx:end])
            else:
                out.append("" if replacement is None else str(replacement))
            idx = end

        return "".join(out)

    def _render_each_templates(text: str) -> str:
        each_pattern = re.compile(
            r"\{\{\s*#each\s+([^{}]+?)\s*\}\}(.*?)\{\{\s*/each\s*\}\}",
            re.S,
        )
        missing = object()

        def _from_item(item: Any, path: str) -> Any:
            trimmed = path
            if trimmed.startswith("this."):
                trimmed = trimmed[len("this.") :]
            elif trimmed == "this":
                trimmed = ""
            elif trimmed.startswith("."):
                trimmed = trimmed[1:]

            if not trimmed:
                return item

            try:
                tokens = parse_field_path(trimmed)
            except Exception:
                return missing

            cur = item
            for token in tokens:
                if isinstance(cur, list):
                    if not isinstance(token, int) or token >= len(cur) or token < 0:
                        return missing
                    cur = cur[token]
                    continue

                if isinstance(cur, dict):
                    if token not in cur:
                        return missing
                    cur = cur[token]
                    continue

                return missing

            return cur

        def _replace(match: re.Match[str]) -> str:
            raw_path = match.group(1) or ""
            body = match.group(2) or ""
            normalized_path = normalize_reference_path(raw_path)
            qualified_path = ctx._qualify_context_path(normalized_path)
            try:
                iterable = ctx.get_value(qualified_path)
            except Exception as exc:  # pragma: no cover - log-only path
                log_warn(
                    f"[param-resolver] #each 路径 {raw_path} 解析失败: {exc}，保留原样"
                )
                return match.group(0)

            if not isinstance(iterable, list):
                log_warn(
                    f"[param-resolver] #each 路径 {raw_path} 对应值类型 {type(iterable).__name__}，保留原样"
                )
                return match.group(0)

            rendered_blocks: List[str] = []
            for item in iterable:
                def _render_this_placeholders(text: str) -> str:
                    def _replace_this(m: re.Match[str]) -> str:
                        raw = m.group(1) or m.group(2) or m.group(3) or ""
                        normalized = normalize_reference_path(raw)
                        if normalized.startswith("this") or normalized.startswith("."):
                            value = _from_item(item, normalized)
                            if value is missing:
                                return m.group(0)
                            return "" if value is None else str(value)
                        return m.group(0)

                    return template_pattern.sub(_replace_this, text)

                rendered_body = _render_each_templates(body)
                rendered_blocks.append(_render_this_placeholders(rendered_body))

            return "".join(rendered_blocks)

        return each_pattern.sub(_replace, text)

    def _resolve(value: Any, path: str = "") -> Any:
        if isinstance(value, dict):
            if "__from__" in value:
                try:
                    return ctx.resolve_binding(value)
                except Exception as e:
                    log_warn(f"[param-resolver] 解析参数 {path or '<root>'} 失败: {e}，使用原值")
                    return value

            return {k: _resolve(v, f"{path}.{k}" if path else k) for k, v in value.items()}

        if isinstance(value, list):
            return [_resolve(v, f"{path}[{idx}]") for idx, v in enumerate(value)]

        if isinstance(value, str):
            normalized_templates = canonicalize_template_placeholders(value)
            stripped_template = normalized_templates.strip()
            if stripped_template.startswith("{{") and stripped_template.endswith("}}"):
                inner = stripped_template[2:-2].strip()
                try:
                    return eval_jinja_expression(inner, ctx.build_jinja_context())
                except Exception:
                    pass
            rendered_inline = _render_json_bindings(normalized_templates)
            rendered_with_each = _render_each_templates(rendered_inline)
            jinja_context: Dict[str, Any] = {
                "result_of": ctx.results,
                "loop": ctx.loop_ctx,
                "loop_ctx": ctx.loop_ctx,
                "loop_id": ctx.loop_id,
                **ctx.loop_ctx,
            }

            def _jinja_get(path: str) -> Any:
                normalized = normalize_reference_path(path)
                qualified = ctx._qualify_context_path(normalized)
                return ctx.get_value(qualified)

            jinja_context["get"] = _jinja_get
            try:
                rendered_with_each = render_jinja_template(rendered_with_each, jinja_context)
            except Exception:
                pass
            str_val = rendered_with_each.strip()
            # 允许 text 等字段以字符串形式携带绑定表达式（常见于外部序列化后传入）
            if str_val.startswith("{") and str_val.endswith("}") and "__from__" in str_val:
                try:
                    parsed = json.loads(str_val)
                except Exception:
                    parsed = None
                if parsed is None:
                    try:
                        parsed = ast.literal_eval(str_val)
                    except Exception:
                        parsed = None
                if isinstance(parsed, dict) and "__from__" in parsed:
                    try:
                        return ctx.resolve_binding(parsed)
                    except Exception as e:
                        log_warn(f"[param-resolver] 字符串绑定 {value} 解析失败: {e}，使用原值")

            normalized_v = normalize_reference_path(rendered_with_each)
            qualified_v = ctx._qualify_context_path(normalized_v)
            head = qualified_v.split(".", 1)[0]
            # 仅当整个字符串就是绑定路径时才解析；包含插值占位符的混合字符串将保留原样
            if (
                head in ctx.loop_ctx
                or head == ctx.loop_id
                or qualified_v.startswith("loop.")
                or qualified_v.startswith("result_of.")
            ):
                try:
                    return ctx.get_value(qualified_v)
                except Exception as e:
                    log_warn(
                        f"[param-resolver] 路径字符串 {value} 解析失败: {e}，使用原值"
                    )

            resolved_with_templates = rendered_with_each
            replaced = False

            def _replace(match: re.Match[str]) -> str:
                nonlocal replaced
                raw_path = match.group(1) or match.group(2) or match.group(3)
                normalized_path = normalize_reference_path(raw_path)
                qualified_path = ctx._qualify_context_path(normalized_path)
                head = qualified_path.split(".", 1)[0]
                if (
                    head not in ctx.loop_ctx
                    and head != ctx.loop_id
                    and not qualified_path.startswith("loop.")
                    and not qualified_path.startswith("result_of.")
                ):
                    return match.group(0)
                try:
                    value = ctx.get_value(qualified_path)
                except Exception as e:
                    log_warn(
                        f"[param-resolver] 模板占位符 {match.group(0)} 解析失败: {e}，保留原样"
                    )
                    return match.group(0)

                replaced = True
                return "" if value is None else str(value)

            resolved_with_templates = template_pattern.sub(
                _replace, resolved_with_templates
            )
            if replaced or resolved_with_templates != value:
                return resolved_with_templates

            return value

        return value

    def _render_runtime_templates(val: Any) -> Any:
        if isinstance(val, str):
            try:
                return render_jinja_template(val, ctx.build_jinja_context())
            except Exception as exc:  # pragma: no cover - runtime log only
                log_warn(f"[param-resolver] Jinja 渲染失败，保留原值: {exc}")
                return val

        if isinstance(val, list):
            return [_render_runtime_templates(v) for v in val]

        if isinstance(val, dict):
            return {k: _render_runtime_templates(v) for k, v in val.items()}

        return val

    resolved = {k: _resolve(v, k) for k, v in (node.params or {}).items()}
    return _render_runtime_templates(resolved)
