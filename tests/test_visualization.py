# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from pathlib import Path

import pytest

pytest.importorskip("PIL")

from velvetflow.models import Workflow
from velvetflow import visualization


def test_condition_branches_are_rendered_without_explicit_edges(tmp_path: Path):
    workflow = Workflow.model_validate(
        {
            "workflow_name": "demo",
            "nodes": [
                {"id": "start", "type": "start"},
                {
                    "id": "check", "type": "condition", "true_to_node": "next", "false_to_node": None
                },
                {"id": "next", "type": "action", "action_id": "do_something"},
            ],
        }
    )

    edges = visualization._resolve_display_edges(workflow)
    incoming, outgoing = visualization._build_edge_maps(workflow.nodes, edges)

    assert outgoing["check"] == ["next"]
    assert incoming["next"] == ["check"]

    output_path = tmp_path / "workflow.jpg"
    saved = visualization.render_workflow_dag(workflow, output_path=str(output_path))

    assert Path(saved).exists()

