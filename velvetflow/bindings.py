"""Parameter binding resolution for workflow execution."""

from typing import Any, Dict, List, Mapping, Optional

from velvetflow.action_registry import get_action_by_id
from velvetflow.loop_dsl import build_loop_output_schema
from velvetflow.logging_utils import log_warn
from velvetflow.models import Node, Workflow


class BindingContext:
    def __init__(
        self,
        workflow: Workflow,
        results: Dict[str, Any],
        *,
        extra_nodes: Optional[Mapping[str, Node]] = None,
        loop_ctx: Optional[Dict[str, Any]] = None,
    ):
        self.workflow = workflow
        self.results = results  # node_id -> result object
        self.loop_ctx = loop_ctx or {}
        self._nodes = {n.id: n for n in workflow.nodes}
        if extra_nodes:
            self._nodes.update(extra_nodes)

    def resolve_binding(self, binding: dict) -> Any:
        """解析 __from__, __agg__, pipeline 等绑定 DSL。"""

        def _render(val: Any, fmt: str, field: Optional[str] = None) -> str:
            if isinstance(val, Mapping):
                selected = val.get(field) if field else val
                data: Mapping[str, Any] = _SafeDict(val)
                data.setdefault("value", selected)
            else:
                selected = val
                data = _SafeDict({"value": selected})
            try:
                return fmt.format_map(data)
            except Exception:
                try:
                    return fmt.replace("{value}", str(selected))
                except Exception:
                    return str(selected)

        if not isinstance(binding, dict) or "__from__" not in binding:
            return binding

        src_path = binding["__from__"]
        self._validate_result_reference(src_path)
        data = self._get_from_context(src_path)
        agg = binding.get("__agg__", "identity")

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

        if agg == "format_join":
            fmt = binding.get("format", "{value}")
            sep = binding.get("sep", "\n")
            field = binding.get("field")

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
            fmt = binding.get("format", "{value}")
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
                    fmt = step.get("format", "{value}")
                    sep = step.get("sep", "\n")
                    field = step.get("field")

                    if not isinstance(current, list):
                        return _render(current, fmt, field)
                    lines: List[str] = []
                    for v in current:
                        lines.append(_render(v, fmt, field))
                    return sep.join(lines)

            return current

        return binding

    def _schema_has_path(self, schema: Mapping[str, Any], fields: List[str]) -> bool:
        if not isinstance(schema, Mapping):
            return False

        current: Mapping[str, Any] = schema
        idx = 0
        while idx < len(fields):
            name = fields[idx]
            typ = current.get("type")

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

        if not isinstance(src_path, str) or not src_path.startswith("result_of."):
            return

        parts = src_path.split(".")
        if len(parts) < 2:
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

        output_schema = action.get("output_schema")
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
        if not path:
            raise KeyError("空路径")

        parts = path.split(".")

        if parts[0] == "loop":
            cur: Any = self.loop_ctx
            for p in parts[1:]:
                if isinstance(cur, dict) and p in cur:
                    cur = cur[p]
                    continue
                raise KeyError(p)
            return cur

        if parts[0] in self.loop_ctx:
            cur = self.loop_ctx[parts[0]]
            rest = parts[1:]
        elif parts[0] == "result_of" and len(parts) >= 2:
            first_key = parts[1]
            if first_key not in self.results:
                raise KeyError(f"result_of.{first_key}")
            cur: Any = self.results[first_key]
            rest = parts[2:]
        else:
            context = {f"result_of.{nid}": value for nid, value in self.results.items()}
            if parts[0] not in context:
                raise KeyError(parts[0])
            cur = context[parts[0]]
            rest = parts[1:]

        for p in rest:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
                continue

            # When the current value is a list of dicts, allow selecting a
            # sub-field from every element. This prevents parameter resolving
            # from failing on paths like ``result_of.xxx.data.employee_id``
            # where ``data`` is a list of objects.
            if isinstance(cur, list):
                extracted = []
                for item in cur:
                    if isinstance(item, dict) and p in item:
                        extracted.append(item[p])
                if extracted:
                    cur = extracted
                    continue
                raise KeyError(p)

                raise KeyError(p)

        return cur


class _SafeDict(dict):
    """A dict that leaves unknown placeholders intact during formatting."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def eval_node_params(node: Node, ctx: BindingContext) -> Dict[str, Any]:
    """遍历 node.params，如果是绑定 DSL，就用 ctx.resolve_binding 解析。"""

    resolved = {}
    for k, v in (node.params or {}).items():
        if isinstance(v, dict) and "__from__" in v:
            try:
                resolved_value = ctx.resolve_binding(v)
            except Exception as e:
                log_warn(f"[param-resolver] 解析参数 {k} 失败: {e}，使用原值")
                resolved[k] = v
            else:
                resolved[k] = resolved_value
        elif isinstance(v, str):
            head = v.split(".", 1)[0]
            if head in ctx.loop_ctx or v.startswith("loop."):
                try:
                    resolved[k] = ctx.get_value(v)
                    continue
                except Exception as e:
                    log_warn(f"[param-resolver] 路径字符串 {v} 解析失败: {e}，使用原值")

            resolved[k] = v
        else:
            resolved[k] = v
    return resolved

