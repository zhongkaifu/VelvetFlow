"""Workflow DSL grammar and tokenization helpers.

The module exposes EBNF snippets for IDE/CLI integrations and a small
lexer dedicated to the ``__from__`` binding path. It complements the
existing JSON/Pydantic validation by making the grammar explicit and
machine-consumable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

# Identifiers reused by workflow / binding grammar.
IDENTIFIER_RE = r"[A-Za-z_][A-Za-z0-9_-]*"

WORKFLOW_EBNF = """
workflow      ::= "{" workflow_body "}"
workflow_body ::= "workflow_name" ":" string ","
                  "description" ":" string ","
                  "nodes" ":" node_list
                  ["," "edges" ":" edge_list]

node_list     ::= "[" node {"," node} "]"
node          ::= "{" "id" ":" identifier "," "type" ":" node_type
                   ["," "action_id" ":" identifier]
                   ["," "display_name" ":" string]
                   ["," "params" ":" param_obj]
                   ["," "true_to_node" ":" identifier]
                   ["," "false_to_node" ":" identifier]
                   "}"
node_type     ::= "start" | "end" | "action" | "condition" | "loop"

edge_list     ::= "[" edge {"," edge} "]"
edge          ::= "{" "from" ":" identifier "," "to" ":" identifier
                   ["," "condition" ":" ("true"|"false")] "}"

param_obj     ::= "{" {param_field} "}"
param_field   ::= identifier ":" param_value
param_value   ::= string | number | bool | null
                 | binding_expr | param_obj | param_array
param_array   ::= "[" {param_value ","} param_value? "]"
"""

BINDING_EBNF = """
binding_expr ::= "{" "__from__" ":" binding_source
                  ["," "__agg__" ":" agg_op
                    ["," agg_args]
                  ] "}"

binding_source ::= "result_of" "." identifier {"." identifier}
                 | "params" "." identifier {"." identifier}

agg_op       ::= "identity" | "count" | "count_if"
               | "filter_map" | "format_join" | "pipeline"
agg_args     ::= "field" ":" identifier
               | "sep" ":" string
               | "steps" ":" param_array
"""


@dataclass
class GrammarSpec:
    """Container object exposing EBNF fragments and token rules."""

    workflow: str = WORKFLOW_EBNF.strip()
    binding: str = BINDING_EBNF.strip()
    identifier_pattern: str = IDENTIFIER_RE


@dataclass
class Token:
    type: str
    value: str


class BindingLexer:
    """Tokenize ``result_of`` binding paths using the declared grammar.

    The lexer is intentionally tiny: it only accepts identifiers and dot
    separators, rejecting templating syntax (e.g. ``{{...}}``) before the
    semantic analyzer runs.
    """

    _token_re = re.compile(
        rf"(?P<IDENT>{IDENTIFIER_RE})|(?P<DOT>\.)",
    )

    def normalize(self, text: str) -> str:
        """Strip template delimiters (e.g. ``${{ ... }}``) and whitespace."""

        cleaned = text.strip()
        if cleaned.startswith("${{") and cleaned.endswith("}}"):
            return cleaned[3:-2].strip()
        if cleaned.startswith("{{") and cleaned.endswith("}}"):
            return cleaned[2:-2].strip()
        return cleaned

    def tokenize(self, text: str) -> List[Token]:
        normalized = self.normalize(text)
        tokens: List[Token] = []
        for match in self._token_re.finditer(normalized):
            kind = "IDENT" if match.lastgroup == "IDENT" else "DOT"
            tokens.append(Token(kind, match.group()))
        if tokens:
            reconstructed = "".join(tok.value for tok in tokens)
            if reconstructed != normalized:
                raise ValueError(f"非法绑定路径：{text!r}")
        elif normalized:
            raise ValueError(f"非法绑定路径：{text!r}")
        return tokens

    def iter_identifiers(self, text: str) -> Iterable[str]:
        """Yield identifier tokens only (ignoring dots)."""

        for tok in self.tokenize(text):
            if tok.type == "IDENT":
                yield tok.value


__all__ = [
    "GrammarSpec",
    "BindingLexer",
    "WORKFLOW_EBNF",
    "BINDING_EBNF",
    "IDENTIFIER_RE",
]
