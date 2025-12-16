from typing import Any, Dict, List

import pytest

from velvetflow.models import ValidationError, Workflow
from velvetflow.planner import repair


def test_repair_preserves_workflow_when_fallback_fails(monkeypatch):
    broken_workflow: Dict[str, Any] = {
        "workflow_name": "failing_flow",
        "description": "",
        "nodes": [
            {
                "id": "n1",
                "type": "action",
                "action_id": "demo.action.v1",
                "display_name": "demo",
                "params": {},
                "depends_on": [],
            }
        ],
    }

    validation_errors: List[ValidationError] = [
        ValidationError(
            code="SCHEMA_MISMATCH",
            node_id="n1",
            field="node",
            message="broken",
        )
    ]

    monkeypatch.setattr(
        repair, "apply_rule_based_repairs", lambda wf, *_args: (wf, validation_errors)
    )
    monkeypatch.setattr(
        repair, "repair_workflow_with_llm", lambda *_, **__: (_ for _ in ()).throw(RuntimeError("llm fail"))
    )
    monkeypatch.setattr(
        repair, "_safe_repair_invalid_loop_body", lambda *_: (_ for _ in ()).throw(RuntimeError("fallback fail"))
    )

    result = repair._repair_with_llm_and_fallback(  # pylint: disable=protected-access
        broken_workflow=broken_workflow,
        validation_errors=validation_errors,
        action_registry=[],
        search_service=None,
        reason="test",
        previous_attempts=None,
    )

    assert isinstance(result, Workflow)
    assert result.workflow_name == "failing_flow"
    assert result.nodes == broken_workflow["nodes"]
