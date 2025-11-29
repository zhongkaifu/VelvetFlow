"""Control/data-flow and property analysis for workflows.

This module applies compiler-inspired static analysis to detect structural
problems (dead cycles, missing exits), data-flow smells (unused outputs,
unsatisfied dependencies), non-functional contract violations (timeouts,
retries/idempotency), and simple security/compliance risks (sensitive data
leaking to external domains without explicit opt-in).
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from velvetflow.dsl_spec import BindingLexer
from velvetflow.models import ValidationError, infer_edges_from_bindings
from velvetflow.reference_utils import normalize_reference_path


class WorkflowStaticAnalyzer:
    """Run control/data-flow, property, and security checks."""

    _SENSITIVE_TOKENS = {"secret", "token", "password", "credential", "key"}

    def __init__(self, action_registry: Sequence[Mapping[str, Any]]):
        self._actions_by_id = {str(a.get("action_id")): dict(a) for a in action_registry}
        self._lexer = BindingLexer()

    # ------------------------------------------------------------------
    def analyze(self, workflow: Mapping[str, Any]) -> List[ValidationError]:
        nodes = workflow.get("nodes") if isinstance(workflow, Mapping) else None
        edges = workflow.get("edges") if isinstance(workflow, Mapping) else None
        if not isinstance(nodes, list):
            return []

        explicit_edges = edges if isinstance(edges, list) else []
        inferred_edges = infer_edges_from_bindings(nodes)
        merged_edges = self._merge_edges(explicit_edges, inferred_edges, nodes)

        nodes_by_id = {n.get("id"): n for n in nodes if isinstance(n, Mapping)}

        issues: List[ValidationError] = []
        issues.extend(self._check_control_flow(nodes_by_id, merged_edges))
        issues.extend(self._check_data_flow(nodes_by_id, merged_edges))
        issues.extend(self._check_attributes(nodes_by_id))
        issues.extend(self._check_security(nodes_by_id))
        return issues

    # ------------------------------------------------------------------
    def _merge_edges(
        self,
        explicit: Sequence[Mapping[str, Any]],
        inferred: Sequence[Mapping[str, Any]],
        nodes: Sequence[Mapping[str, Any]],
    ) -> List[Tuple[str, str]]:
        valid_ids = {
            n.get("id")
            for n in nodes
            if isinstance(n, Mapping) and isinstance(n.get("id"), str)
        }
        merged: List[Tuple[str, str]] = []
        seen: set[Tuple[str, str]] = set()

        for edge in list(explicit) + list(inferred):
            src = edge.get("from")
            dst = edge.get("to")
            if not isinstance(src, str) or not isinstance(dst, str):
                continue
            if src not in valid_ids or dst not in valid_ids:
                continue
            pair = (src, dst)
            if pair in seen:
                continue
            seen.add(pair)
            merged.append(pair)
        return merged

    # Control-flow ------------------------------------------------------
    def _check_control_flow(
        self, nodes_by_id: Mapping[str, Mapping[str, Any]], edges: Sequence[Tuple[str, str]]
    ) -> List[ValidationError]:
        issues: List[ValidationError] = []

        adjacency: Dict[str, List[str]] = defaultdict(list)
        reverse: Dict[str, List[str]] = defaultdict(list)
        for src, dst in edges:
            adjacency[src].append(dst)
            reverse[dst].append(src)

        # Detect cycles that have no exit to an end node or outside the SCC.
        for component in self._strongly_connected_components(adjacency, nodes_by_id.keys()):
            if len(component) == 1:
                node_id = next(iter(component))
                if node_id not in adjacency.get(node_id, ()):
                    continue
            has_exit = any(
                (succ not in component) for nid in component for succ in adjacency.get(nid, ())
            )
            contains_end = any(
                (nodes_by_id.get(nid, {}).get("type") == "end") for nid in component
            )
            if not has_exit and not contains_end:
                issues.append(
                    ValidationError(
                        code="CONTROL_FLOW_VIOLATION",
                        node_id=next(iter(component)),
                        field=None,
                        message="检测到无出口的循环/强连通分量，可能导致执行卡死。",
                    )
                )

        return issues

    def _strongly_connected_components(
        self, adjacency: Mapping[str, Iterable[str]], node_ids: Iterable[str]
    ) -> List[List[str]]:
        index = 0
        indices: Dict[str, int] = {}
        lowlink: Dict[str, int] = {}
        stack: List[str] = []
        on_stack: set[str] = set()
        result: List[List[str]] = []

        def strongconnect(v: str) -> None:
            nonlocal index
            indices[v] = index
            lowlink[v] = index
            index += 1
            stack.append(v)
            on_stack.add(v)

            for w in adjacency.get(v, ()):  # pragma: no branch - small loop
                if w not in indices:
                    strongconnect(w)
                    lowlink[v] = min(lowlink[v], lowlink[w])
                elif w in on_stack:
                    lowlink[v] = min(lowlink[v], indices[w])

            if lowlink[v] == indices[v]:
                component: List[str] = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    component.append(w)
                    if w == v:
                        break
                result.append(component)

        for nid in node_ids:
            if nid not in indices:
                strongconnect(nid)
        return result

    # Data-flow ---------------------------------------------------------
    def _check_data_flow(
        self, nodes_by_id: Mapping[str, Mapping[str, Any]], edges: Sequence[Tuple[str, str]]
    ) -> List[ValidationError]:
        issues: List[ValidationError] = []

        adjacency: Dict[str, List[str]] = defaultdict(list)
        inbound: Dict[str, int] = defaultdict(int)
        for src, dst in edges:
            adjacency[src].append(dst)
            inbound[dst] += 1

        reachable = self._reachable_from_roots(nodes_by_id, adjacency, inbound)

        consumers: MutableMapping[str, int] = defaultdict(int)
        for src, dst in edges:
            if dst in nodes_by_id:
                consumers[src] += 1

        implicit_outputs = {"end", "loop"}
        for nid, node in nodes_by_id.items():
            if nid not in reachable:
                continue
            ntype = node.get("type")
            if consumers.get(nid, 0) == 0 and ntype not in implicit_outputs:
                issues.append(
                    ValidationError(
                        code="DATA_FLOW_VIOLATION",
                        node_id=nid,
                        field=None,
                        message="节点输出未被任何下游消费，可能是冗余或遗漏了依赖。",
                    )
                )

        for nid, node in nodes_by_id.items():
            params = node.get("params") if isinstance(node, Mapping) else None
            for src in self._iter_binding_sources(params):
                parts = src.split(".")
                if len(parts) >= 2 and parts[0] == "result_of":
                    dep_id = parts[1]
                    if dep_id not in nodes_by_id:
                        issues.append(
                            ValidationError(
                                code="DATA_FLOW_VIOLATION",
                                node_id=nid,
                                field=None,
                                message=f"依赖的节点 '{dep_id}' 不存在，数据流无法满足前置条件。",
                            )
                        )

        return issues

    def _reachable_from_roots(
        self,
        nodes_by_id: Mapping[str, Mapping[str, Any]],
        adjacency: Mapping[str, List[str]],
        inbound: Mapping[str, int],
    ) -> set[str]:
        roots = [nid for nid, n in nodes_by_id.items() if n.get("type") == "start"]
        if not roots:
            roots = [nid for nid in nodes_by_id if inbound.get(nid, 0) == 0]
        dq = deque(roots)
        reachable: set[str] = set()
        while dq:
            nid = dq.popleft()
            if nid in reachable:
                continue
            reachable.add(nid)
            for nxt in adjacency.get(nid, []):
                dq.append(nxt)
        return reachable

    def _iter_binding_sources(self, params: Any) -> Iterable[str]:
        if isinstance(params, Mapping):
            if "__from__" in params:
                src = params.get("__from__")
                if isinstance(src, str):
                    normalized = self._lexer.normalize(src)
                    if isinstance(normalized, str):
                        yield normalized
            for v in params.values():
                yield from self._iter_binding_sources(v)
        elif isinstance(params, list):
            for item in params:
                yield from self._iter_binding_sources(item)

    # Attribute constraints ---------------------------------------------
    def _check_attributes(self, nodes_by_id: Mapping[str, Mapping[str, Any]]) -> List[ValidationError]:
        issues: List[ValidationError] = []

        for nid, node in nodes_by_id.items():
            params = node.get("params") if isinstance(node, Mapping) else None
            action_id = node.get("action_id") if isinstance(node, Mapping) else None
            action_def = self._actions_by_id.get(action_id or "")
            idempotent = bool(action_def.get("pure") or action_def.get("idempotent")) if action_def else False

            if isinstance(params, Mapping):
                timeout = params.get("timeout")
                retries = params.get("retries") if "retries" in params else params.get("retry")

                if timeout is not None and isinstance(timeout, (int, float)):
                    if timeout <= 0:
                        issues.append(
                            ValidationError(
                                code="PROPERTY_VIOLATION",
                                node_id=nid,
                                field="timeout",
                                message="timeout 需要是大于 0 的数值。",
                            )
                        )

                if retries is not None:
                    if not isinstance(retries, int) or retries < 0:
                        issues.append(
                            ValidationError(
                                code="PROPERTY_VIOLATION",
                                node_id=nid,
                                field="retries",
                                message="retries 必须是非负整数。",
                            )
                        )
                    elif retries > 0 and not idempotent:
                        issues.append(
                            ValidationError(
                                code="PROPERTY_VIOLATION",
                                node_id=nid,
                                field="retries",
                                message="非幂等操作配置了重试但缺少幂等/补偿保证，可能重复副作用。",
                            )
                        )

        return issues

    # Security/compliance ----------------------------------------------
    def _check_security(self, nodes_by_id: Mapping[str, Mapping[str, Any]]) -> List[ValidationError]:
        issues: List[ValidationError] = []

        for nid, node in nodes_by_id.items():
            params = node.get("params") if isinstance(node, Mapping) else None
            action_id = node.get("action_id") if isinstance(node, Mapping) else None
            target_domain = self._actions_by_id.get(action_id or "", {}).get("domain")

            if not isinstance(params, Mapping):
                continue

            for path, binding in self._iter_bindings(params):
                src = binding.get("__from__")
                normalized_src = normalize_reference_path(src) if isinstance(src, str) else None
                if not isinstance(normalized_src, str):
                    continue
                if not self._is_sensitive_path(path, normalized_src):
                    continue

                source_domain = self._source_domain(normalized_src, nodes_by_id)
                if self._is_external_domain(target_domain) and target_domain != source_domain:
                    issues.append(
                        ValidationError(
                            code="SECURITY_VIOLATION",
                            node_id=nid,
                            field=path,
                            message="敏感数据正流向外部域/跨域调用，缺少显式的访问控制或脱敏声明。",
                        )
                    )

        return issues

    def _iter_bindings(self, params: Any, prefix: str | None = None) -> Iterable[Tuple[str, Mapping[str, Any]]]:
        if isinstance(params, Mapping):
            if "__from__" in params:
                yield prefix or "params", params
            for key, value in params.items():
                next_prefix = f"{prefix}.{key}" if prefix else str(key)
                yield from self._iter_bindings(value, next_prefix)
        elif isinstance(params, list):
            for idx, item in enumerate(params):
                next_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
                yield from self._iter_bindings(item, next_prefix)

    def _is_sensitive_path(self, target_path: str, source_path: str) -> bool:
        tokens = {token.lower() for token in (target_path + "." + source_path).replace("[", ".").replace("]", "").split(".")}
        return any(tok in self._SENSITIVE_TOKENS for tok in tokens)

    def _source_domain(self, source_path: str, nodes_by_id: Mapping[str, Mapping[str, Any]]) -> Optional[str]:
        parts = source_path.split(".")
        if len(parts) < 2 or parts[0] != "result_of":
            return None
        src_node = nodes_by_id.get(parts[1])
        action_id = src_node.get("action_id") if isinstance(src_node, Mapping) else None
        action_def = self._actions_by_id.get(action_id or "")
        return action_def.get("domain") if isinstance(action_def, Mapping) else None

    def _is_external_domain(self, domain: Optional[str]) -> bool:
        if not domain:
            return False
        return any(token in domain.lower() for token in ("http", "web", "external", "notify", "email"))


__all__ = ["WorkflowStaticAnalyzer"]
