from __future__ import annotations

import json

import pytest

from velvetflow import logging_utils
from velvetflow.metrics import RunManager, RunMetrics


def test_record_node_increments_counts() -> None:
    metrics = RunMetrics()

    metrics.record_node(success=True)
    metrics.record_node(success=False)

    assert metrics.total_nodes == 2
    assert metrics.failed_nodes == 1


def test_record_llm_usage_coerces_values() -> None:
    metrics = RunMetrics()

    metrics.record_llm_usage(
        {
            "prompt_tokens": "3",
            "completion_tokens": None,
            "total_tokens": "5",
        }
    )

    assert metrics.llm_calls == 1
    assert metrics.prompt_tokens == 3
    assert metrics.completion_tokens == 0
    assert metrics.total_tokens == 5


@pytest.mark.parametrize("workflow_name", [None, "demo-workflow"])
def test_to_prometheus_includes_labels_and_metrics(workflow_name: str | None) -> None:
    metrics = RunMetrics(
        total_nodes=2,
        failed_nodes=1,
        llm_calls=3,
        prompt_tokens=4,
        completion_tokens=5,
        total_tokens=9,
    )

    output = metrics.to_prometheus(run_id="run-123", workflow_name=workflow_name)

    labels = 'run_id="run-123"'
    if workflow_name:
        labels = f'{labels},workflow="{workflow_name}"'

    assert f"velvetflow_total_nodes{{{labels}}} 2" in output
    assert f"velvetflow_failed_nodes{{{labels}}} 1" in output
    assert f"velvetflow_llm_calls{{{labels}}} 3" in output
    assert f"velvetflow_prompt_tokens{{{labels}}} 4" in output
    assert f"velvetflow_completion_tokens{{{labels}}} 5" in output
    assert f"velvetflow_total_tokens{{{labels}}} 9" in output


@pytest.mark.parametrize(
    "metrics_format,expected_marker",
    [
        ("prometheus", "velvetflow_total_nodes"),
        ("json", '"total_nodes"'),
    ],
)
def test_run_manager_writes_metrics_file(
    tmp_path, metrics_format: str, expected_marker: str
) -> None:
    assert logging_utils.current_run_id() is None

    with RunManager(log_dir=tmp_path, metrics_format=metrics_format) as manager:
        assert logging_utils.current_run_id() == manager.run_id
        manager.metrics.record_node(success=True)

    assert logging_utils.current_run_id() is None
    assert manager.metrics_path.exists()

    content = manager.metrics_path.read_text(encoding="utf-8")
    assert expected_marker in content
    if metrics_format == "json":
        payload = json.loads(content)
        assert payload["run_id"] == manager.run_id
        assert payload["total_nodes"] == 1
