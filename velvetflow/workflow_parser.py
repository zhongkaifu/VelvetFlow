from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence, Tuple


WORKFLOW_GRAMMAR = r"""
workflow           ::= '{' workflow_fields '}'
workflow_fields    ::= workflow_field (',' workflow_field)*
workflow_field     ::= '"workflow_name"' ':' string
                     | '"description"' ':' string
                     | '"nodes"' ':' '[' node_list ']'
                     | '"edges"' ':' '[' edge_list ']'
node_list          ::= /* empty */ | node (',' node)*
node               ::= '{' '"id"' ':' string ',' '"type"' ':' string node_tail '}'
node_tail          ::= (',' '"action_id"' ':' string)?
                        (',' '"params"' ':' object)?
                        (',' '"exports"' ':' object)?
                        (',' '"description"' ':' string)?
edge_list          ::= /* empty */ | edge (',' edge)*
edge               ::= '{' '"from"' ':' string ',' '"to"' ':' string edge_tail '}'
edge_tail          ::= (',' '"condition"' ':' object)?
string             ::= '"' ( character )* '"'
object             ::= '{' (string ':' value (',' string ':' value)*)? '}'
value              ::= string | number | object | array | 'true' | 'false' | 'null'
array              ::= '[' (value (',' value)*)? ']'
"""


@dataclass
class RecoveryEdit:
    """Suggested minimal-cost change for syntax or grammar issues."""

    kind: str
    path: str
    description: str
    replacement: Any | None = None


@dataclass
class WorkflowSyntaxError:
    """Represents a low-level syntax issue when parsing workflow text."""

    message: str
    line: int
    column: int
    expected: Sequence[str] = field(default_factory=tuple)


@dataclass
class WorkflowGrammarIssue:
    """Represents a grammar violation after successful tokenization."""

    path: str
    message: str
    node_id: str | None = None
    expected: Sequence[str] = field(default_factory=tuple)
    recovery: RecoveryEdit | None = None


@dataclass
class WorkflowParseResult:
    ast: Mapping[str, Any] | None
    syntax_errors: list[WorkflowSyntaxError]
    grammar_issues: list[WorkflowGrammarIssue]
    recovery_edits: list[RecoveryEdit]
    recovered: bool = False
    reused_issues: int = 0
    changed_paths: Sequence[str] = field(default_factory=tuple)


def _decode_json(source: str) -> Tuple[Mapping[str, Any] | None, list[WorkflowSyntaxError]]:
    try:
        return json.loads(source), []
    except json.JSONDecodeError as exc:
        return None, [
            WorkflowSyntaxError(
                message=exc.msg,
                line=exc.lineno,
                column=exc.colno,
                expected=(),
            )
        ]


def _recover_json(source: str, error: WorkflowSyntaxError) -> Tuple[Mapping[str, Any] | None, list[RecoveryEdit]]:
    """Attempt a lightweight phrase-level recovery for common JSON mistakes."""

    edits: list[RecoveryEdit] = []
    patched = source

    if "property name" in error.message or "Expecting property name" in error.message:
        # Trailing comma or missing key name
        patched = patched[: error.column - 1] + ' "_placeholder": null' + patched[error.column - 1 :]
        edits.append(
            RecoveryEdit(
                kind="insert",
                path="$",
                description="自动插入占位字段以绕过缺失的属性名。",
            )
        )
    elif "Expecting ',' delimiter" in error.message:
        patched = patched[: error.column - 1] + "," + patched[error.column - 1 :]
        edits.append(
            RecoveryEdit(
                kind="insert",
                path="$",
                description="在解析错误处插入逗号以完成列表/对象分隔。",
            )
        )
    elif "Unterminated string" in error.message or "Expecting ':' delimiter" in error.message:
        patched = patched + "}"
        edits.append(
            RecoveryEdit(
                kind="insert",
                path="$",
                description="自动闭合对象，便于继续校验。",
            )
        )

    recovered_ast, syntax_errors = _decode_json(patched)
    if syntax_errors:
        return None, edits

    return recovered_ast, edits


def _validate_required_keys(
    ast: Mapping[str, Any]
) -> Tuple[list[WorkflowGrammarIssue], list[RecoveryEdit]]:
    issues: list[WorkflowGrammarIssue] = []
    recoveries: list[RecoveryEdit] = []

    if not isinstance(ast, Mapping):
        issues.append(
            WorkflowGrammarIssue(
                path="$",
                message="workflow 顶层必须是对象。",
                expected=["object"],
                recovery=RecoveryEdit(
                    kind="replace",
                    path="$",
                    description="将顶层替换为包含 nodes/edges 的对象。",
                    replacement={},
                ),
            )
        )
        return issues, recoveries

    if "nodes" not in ast:
        edit = RecoveryEdit(
            kind="insert",
            path="nodes",
            description="插入空的 nodes 数组。",
            replacement=[],
        )
        recoveries.append(edit)
        issues.append(
            WorkflowGrammarIssue(
                path="nodes",
                message="缺少必需的 nodes 字段。",
                expected=["nodes: list"],
                recovery=edit,
            )
        )
    if "edges" not in ast:
        edit = RecoveryEdit(
            kind="insert",
            path="edges",
            description="插入空的 edges 数组。",
            replacement=[],
        )
        recoveries.append(edit)
        issues.append(
            WorkflowGrammarIssue(
                path="edges",
                message="缺少必需的 edges 字段。",
                expected=["edges: list"],
                recovery=edit,
            )
        )

    return issues, recoveries


def _validate_nodes(nodes: Any) -> Tuple[list[WorkflowGrammarIssue], list[RecoveryEdit]]:
    issues: list[WorkflowGrammarIssue] = []
    recoveries: list[RecoveryEdit] = []

    if not isinstance(nodes, list):
        edit = RecoveryEdit(
            kind="replace",
            path="nodes",
            description="nodes 必须是数组，已建议替换为空数组。",
            replacement=[],
        )
        issues.append(
            WorkflowGrammarIssue(
                path="nodes",
                message="nodes 字段必须是数组。",
                expected=["list"],
                recovery=edit,
            )
        )
        recoveries.append(edit)
        return issues, recoveries

    for idx, node in enumerate(nodes):
        path = f"nodes[{idx}]"
        if not isinstance(node, Mapping):
            edit = RecoveryEdit(
                kind="replace",
                path=path,
                description="节点必须是对象，已建议替换为空对象。",
                replacement={},
            )
            issues.append(
                WorkflowGrammarIssue(
                    path=path,
                    message="节点必须是对象。",
                    expected=["object"],
                    recovery=edit,
                )
            )
            recoveries.append(edit)
            continue

        node_id = node.get("id") if isinstance(node.get("id"), str) else None
        if node_id is None:
            edit = RecoveryEdit(
                kind="insert",
                path=f"{path}.id",
                description="补充节点 id 字段。",
                replacement=f"auto_{idx}",
            )
            issues.append(
                WorkflowGrammarIssue(
                    path=f"{path}.id",
                    message="节点缺少 id 或类型错误。",
                    node_id=node_id,
                    expected=["string"],
                    recovery=edit,
                )
            )
            recoveries.append(edit)

        if not isinstance(node.get("type"), str):
            edit = RecoveryEdit(
                kind="replace",
                path=f"{path}.type",
                description="将节点类型替换为默认 action。",
                replacement="action",
            )
            issues.append(
                WorkflowGrammarIssue(
                    path=f"{path}.type",
                    message="节点缺少 type 或类型错误。",
                    node_id=node_id,
                    expected=["string"],
                    recovery=edit,
                )
            )
            recoveries.append(edit)

    return issues, recoveries


def _validate_edges(edges: Any) -> Tuple[list[WorkflowGrammarIssue], list[RecoveryEdit]]:
    issues: list[WorkflowGrammarIssue] = []
    recoveries: list[RecoveryEdit] = []

    if not isinstance(edges, list):
        edit = RecoveryEdit(
            kind="replace",
            path="edges",
            description="edges 必须是数组，已建议替换为空数组。",
            replacement=[],
        )
        issues.append(
            WorkflowGrammarIssue(
                path="edges",
                message="edges 字段必须是数组。",
                expected=["list"],
                recovery=edit,
            )
        )
        recoveries.append(edit)
        return issues, recoveries

    for idx, edge in enumerate(edges):
        path = f"edges[{idx}]"
        if not isinstance(edge, Mapping):
            edit = RecoveryEdit(
                kind="replace",
                path=path,
                description="边必须是对象，已建议替换为空对象。",
                replacement={},
            )
            issues.append(
                WorkflowGrammarIssue(
                    path=path,
                    message="边必须是对象。",
                    expected=["object"],
                    recovery=edit,
                )
            )
            recoveries.append(edit)
            continue

        if not isinstance(edge.get("from"), str):
            edit = RecoveryEdit(
                kind="replace",
                path=f"{path}.from",
                description="from 字段缺失或类型错误，建议替换为空字符串。",
                replacement="",
            )
            issues.append(
                WorkflowGrammarIssue(
                    path=f"{path}.from",
                    message="边缺少 from 字段。",
                    node_id=str(edge.get("from")),
                    expected=["string"],
                    recovery=edit,
                )
            )
            recoveries.append(edit)

        if not isinstance(edge.get("to"), str):
            edit = RecoveryEdit(
                kind="replace",
                path=f"{path}.to",
                description="to 字段缺失或类型错误，建议替换为空字符串。",
                replacement="",
            )
            issues.append(
                WorkflowGrammarIssue(
                    path=f"{path}.to",
                    message="边缺少 to 字段。",
                    node_id=str(edge.get("to")),
                    expected=["string"],
                    recovery=edit,
                )
            )
            recoveries.append(edit)

    return issues, recoveries


def _walk_changes(old: Any, new: Any, prefix: str = "$") -> list[str]:
    if type(old) != type(new):
        return [prefix]

    changes: list[str] = []
    if isinstance(old, MutableMapping):
        all_keys = set(old) | set(new)  # type: ignore[arg-type]
        for key in all_keys:
            old_val = old.get(key)
            new_val = new.get(key)
            child_prefix = f"{prefix}.{key}" if prefix != "$" else key
            changes.extend(_walk_changes(old_val, new_val, child_prefix))
    elif isinstance(old, list):
        max_len = max(len(old), len(new))
        for idx in range(max_len):
            old_val = old[idx] if idx < len(old) else None
            new_val = new[idx] if idx < len(new) else None
            child_prefix = f"{prefix}[{idx}]"
            changes.extend(_walk_changes(old_val, new_val, child_prefix))
    else:
        if old != new:
            changes.append(prefix)
    return changes


def parse_workflow_source(
    source: str | Mapping[str, Any], *, attempt_recovery: bool = True
) -> WorkflowParseResult:
    if isinstance(source, Mapping):
        ast = source
        syntax_errors: list[WorkflowSyntaxError] = []
        recovery_edits: list[RecoveryEdit] = []
        recovered = False
    else:
        ast, syntax_errors = _decode_json(source)
        recovery_edits = []
        recovered = False
        if syntax_errors and attempt_recovery:
            recovered_ast, edits = _recover_json(source, syntax_errors[0])
            recovery_edits = edits
            if recovered_ast is not None:
                ast = recovered_ast
                recovered = True
            else:
                return WorkflowParseResult(
                    ast=None,
                    syntax_errors=syntax_errors,
                    grammar_issues=[],
                    recovery_edits=recovery_edits,
                    recovered=recovered,
                )

    return _validate_ast(ast, syntax_errors, recovery_edits, recovered)


def _validate_ast(
    ast: Mapping[str, Any] | None,
    syntax_errors: list[WorkflowSyntaxError],
    recovery_edits: list[RecoveryEdit],
    recovered: bool,
) -> WorkflowParseResult:
    if ast is None:
        return WorkflowParseResult(
            ast=None,
            syntax_errors=syntax_errors,
            grammar_issues=[],
            recovery_edits=recovery_edits,
            recovered=recovered,
        )

    grammar_issues, recoveries = _validate_required_keys(ast)
    nodes_issues: list[WorkflowGrammarIssue] = []
    edges_issues: list[WorkflowGrammarIssue] = []

    if isinstance(ast, Mapping):
        node_candidates = ast.get("nodes")
        nodes_issues, node_recoveries = _validate_nodes(node_candidates)
        edge_candidates = ast.get("edges")
        edges_issues, edge_recoveries = _validate_edges(edge_candidates)
        recoveries.extend(node_recoveries + edge_recoveries)

    grammar_issues.extend(nodes_issues)
    grammar_issues.extend(edges_issues)

    return WorkflowParseResult(
        ast=ast,
        syntax_errors=syntax_errors,
        grammar_issues=grammar_issues,
        recovery_edits=recoveries + recovery_edits,
        recovered=recovered,
    )


class IncrementalWorkflowParser:
    """Incremental workflow parser that revalidates only changed regions."""

    def __init__(self) -> None:
        self._last_ast: Mapping[str, Any] | None = None
        self._last_issues: list[WorkflowGrammarIssue] = []

    def parse(self, source: str | Mapping[str, Any]) -> WorkflowParseResult:
        result = parse_workflow_source(source)
        if result.ast is None or self._last_ast is None:
            self._last_ast = result.ast
            self._last_issues = result.grammar_issues
            result.reused_issues = 0
            result.changed_paths = tuple(issue.path for issue in result.grammar_issues)
            return result

        changed_paths = _walk_changes(self._last_ast, result.ast)
        cached = [
            issue for issue in self._last_issues if issue.path not in set(changed_paths)
        ]
        combined_issues = cached + result.grammar_issues

        result.reused_issues = len(cached)
        result.grammar_issues = combined_issues
        result.changed_paths = tuple(changed_paths)

        self._last_ast = result.ast
        self._last_issues = combined_issues
        return result


__all__ = [
    "WORKFLOW_GRAMMAR",
    "RecoveryEdit",
    "WorkflowSyntaxError",
    "WorkflowGrammarIssue",
    "WorkflowParseResult",
    "IncrementalWorkflowParser",
    "parse_workflow_source",
]
