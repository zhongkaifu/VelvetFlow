import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validate_workflow import validate_workflow_data
from velvetflow.workflow_parser import IncrementalWorkflowParser, parse_workflow_source


def test_parser_reports_syntax_error_and_recovery():
    text = '{"workflow_name": "demo" "nodes": [], "edges": []}'
    result = parse_workflow_source(text)

    assert result.syntax_errors
    assert result.recovery_edits
    assert result.recovered is True
    assert result.ast is not None


def test_validate_workflow_data_reports_grammar_issues():
    parser_result = parse_workflow_source("{}")
    errors = validate_workflow_data(parser_result.ast or {}, [], parser_result=parser_result)

    assert errors
    assert errors[0].code == "GRAMMAR_VIOLATION"
    assert "nodes" in errors[0].message


def test_incremental_parser_reuses_cached_issues():
    parser = IncrementalWorkflowParser()
    base = {"workflow_name": "demo", "nodes": [{"id": "n1"}], "edges": []}

    first = parser.parse(base)
    assert any(issue.path == "nodes[0].type" for issue in first.grammar_issues)

    updated = {**base, "description": "added"}
    second = parser.parse(updated)

    assert second.reused_issues >= 1
    assert "description" in second.changed_paths
    assert any(issue.path == "nodes[0].type" for issue in second.grammar_issues)
