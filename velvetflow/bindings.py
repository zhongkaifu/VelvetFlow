"""Parameter binding resolution for workflow execution."""

import json
import re
from typing import Any, Dict, List, Mapping, Optional

from velvetflow.action_registry import get_action_by_id
from velvetflow.loop_dsl import build_loop_output_schema
from velvetflow.logging_utils import log_warn
from velvetflow.models import ALLOWED_PARAM_AGGREGATORS, Node, Workflow
from velvetflow.reference_utils import (
    canonicalize_template_placeholders,
    normalize_reference_path,
    parse_field_path,
)


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

    def _infer_loop_reference_path(self, src_path: str) -> str:
        """Try to recover missing loop exports/items/aggregates segments.

        Some LLM-generated bindings may omit ``exports`` or ``items`` when
        referencing loop outputs. This helper attempts to insert the missing
        segments based on the loop's ``exports`` definition. If multiple
        completions are possible, a ``ValueError`` will be raised to surface the
        ambiguity instead of guessing.
        """

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
        if normalized_rest and normalized_rest[0] in {"items", "aggregates"}:
            return src_path

        items_fields: List[List[Any]] = []
        items_spec = exports.get("items")
        if isinstance(items_spec, Mapping):
            for field in items_spec.get("fields", []):
                if not isinstance(field, str):
                    continue
                try:
                    items_fields.append(parse_field_path(field))
                except Exception:
                    items_fields.append([field])

        aggregate_names = {
            agg.get("name")
            for agg in exports.get("aggregates", [])
            if isinstance(agg, Mapping) and isinstance(agg.get("name"), str)
        }

        candidates: List[List[Any]] = []
        for field_tokens in items_fields:
            if normalized_rest[: len(field_tokens)] == field_tokens:
                candidates.append(["items", *normalized_rest])
                candidates.append(["exports", "items", *normalized_rest])

        if normalized_rest and normalized_rest[0] in aggregate_names:
            candidates.append(["aggregates", *normalized_rest])
            candidates.append(["exports", "aggregates", *normalized_rest])

        if not candidates:
            return src_path

        loop_schema = build_loop_output_schema(params)
        validated_candidates: List[List[Any]] = []
        for cand in candidates:
            effective = cand[1:] if cand and cand[0] == "exports" else cand
            if loop_schema and self._schema_has_path(loop_schema, effective):
                validated_candidates.append(cand)

        if not validated_candidates:
            return src_path

        deduped: List[List[Any]] = []
        seen_effective: set = set()
        for cand in validated_candidates:
            effective = tuple(cand[1:] if cand and cand[0] == "exports" else cand)
            if effective in seen_effective:
                continue
            seen_effective.add(effective)
            deduped.append(cand)

        if len(deduped) > 1:
            readable = [
                self._format_field_path(["result_of", node_id, *cand]) for cand in deduped
            ]
            raise ValueError(
                f"__from__ 引用 '{src_path}' 存在歧义，可能的有效路径: {', '.join(readable)}"
            )

        corrected_parts = ["result_of", node_id, *deduped[0]]
        return self._format_field_path(corrected_parts)

    def resolve_binding(self, binding: dict) -> Any:
        """解析 __from__, __agg__, pipeline 等绑定 DSL。"""

        def _render(val: Any, fmt: Optional[str], field: Optional[str] = None) -> str:
            if isinstance(val, Mapping):
                selected = val.get(field) if field else val
                data: Mapping[str, Any] = _SafeDict(val)
                if fmt:
                    try:
                        return fmt.format_map(data)
                    except Exception:
                        pass
                return "" if selected is None else str(selected)

            # 非字典元素：直接字符串化或用默认字段选择
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
        agg = binding.get("__agg__", "identity")
        if agg not in ALLOWED_PARAM_AGGREGATORS:
            raise ValueError(
                f"__agg__ 取值非法（{agg}），允许值：{', '.join(ALLOWED_PARAM_AGGREGATORS)}"
            )

        if agg == "identity":
            return data

        if agg == "count":
            if isinstance(data, list):
                return len(data)
            if isinstance(data, Mapping):
                return len(data)
            return 0 if data is None else 1

        if agg == "count_if":
            if not isinstance(data, list):
                return 0
            field = binding.get("field") or binding.get("filter_field")
            op = binding.get("op") or binding.get("filter_op") or "=="
            target = binding.get("value") or binding.get("filter_value")
            count = 0
            for item in data:
                # Support both list-of-dicts and list-of-scalars
                if isinstance(item, dict):
                    v = item.get(field)
                else:
                    v = item

                try:
                    if op == ">" and v is not None and v > target:
                        count += 1
                    elif op == ">=" and v is not None and v >= target:
                        count += 1
                    elif op == "<" and v is not None and v < target:
                        count += 1
                    elif op == "<=" and v is not None and v <= target:
                        count += 1
                    elif op == "==" and v == target:
                        count += 1
                    elif op == "!=" and v != target:
                        count += 1
                except TypeError:
                    # If comparison fails (e.g., string vs number), skip the item
                    continue
            return count

        if agg == "join":
            sep = binding.get("separator")
            if sep is None:
                sep = binding.get("sep", "")
            if not isinstance(sep, str):
                sep = str(sep)

            if isinstance(data, list):
                return sep.join("" if v is None else str(v) for v in data)

            return "" if data is None else str(data)

        if agg == "format_join":
            field = binding.get("field")
            fmt = binding.get("format") or (f"{{{field}}}" if field else None)
            sep = binding.get("sep", "\n")

            if not isinstance(data, list):
                return _render(data, fmt, field)

            rendered: List[str] = []
            for item in data:
                rendered.append(_render(item, fmt, field))
            return sep.join(rendered)

        if agg == "filter_map":
            filter_field = binding.get("filter_field")
            filter_op = binding.get("filter_op", "==")
            filter_value = binding.get("filter_value")
            map_field = binding.get("map_field")
            fmt = binding.get("format") or (f"{{{map_field}}}" if map_field else None)
            sep = binding.get("sep", "\n")

            if not isinstance(data, list):
                return _render(data, fmt)

            lines: List[str] = []
            for item in data:
                if not isinstance(item, Mapping):
                    continue
                v = item.get(filter_field)
                passed = False
                if filter_op == ">" and v is not None and v > filter_value:
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

            return sep.join(lines)

        if agg == "pipeline":
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
                    filtered = []
                    for item in current:
                        if not isinstance(item, dict):
                            continue
                        v = item.get(field)
                        passed = False
                        if cmp_op == ">" and v is not None and v > cmp_val:
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
                        return _render(current, fmt, field)
                    lines: List[str] = []
                    for v in current:
                        lines.append(_render(v, fmt, field))
                    return sep.join(lines)

            return current

        return binding

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
                current = current.get("items") or {}
                # array 本身不消费字段名，继续在 items 上检查同一个字段
                continue

            if typ == "object" or typ is None:
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
            rendered_inline = _render_json_bindings(normalized_templates)
            str_val = rendered_inline.strip()
            # 允许 text 等字段以字符串形式携带绑定表达式（常见于外部序列化后传入）
            if str_val.startswith("{") and str_val.endswith("}") and "__from__" in str_val:
                try:
                    parsed = json.loads(str_val)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict) and "__from__" in parsed:
                    try:
                        return ctx.resolve_binding(parsed)
                    except Exception as e:
                        log_warn(f"[param-resolver] 字符串绑定 {value} 解析失败: {e}，使用原值")

            normalized_v = normalize_reference_path(rendered_inline)
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

            resolved_with_templates = rendered_inline
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

    return {k: _resolve(v, k) for k, v in (node.params or {}).items()}

