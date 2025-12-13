from velvetflow.models import ValidationError
from velvetflow.planner.repair import _format_errors_for_llm_payload


def test_format_errors_for_llm_payload_matches_spec():
    errors = [
        ValidationError(
            code="INVALID_SCHEMA",
            node_id="cond_1",
            field="false_to_node",
            message="false_to_node references a branch that is unreachable",
            expected="Node must lead to a valid downstream execution path",
            actual_value="step_8",
        )
    ]

    payload = _format_errors_for_llm_payload(errors)

    assert payload == {
        "errors": [
            {
                "type": "InvalidSchema",
                "message": "false_to_node references a branch that is unreachable",
                "location": "/nodes/cond_1/false_to_node",
                "expected": "Node must lead to a valid downstream execution path",
                "actual_value": "step_8",
            }
        ]
    }


def test_format_errors_for_llm_payload_handles_missing_location():
    errors = [
        ValidationError(
            code="DISCONNECTED_GRAPH",
            node_id=None,
            field=None,
            message="graph is disconnected",
        )
    ]

    payload = _format_errors_for_llm_payload(errors)

    assert payload["errors"][0]["location"] is None
    assert payload["errors"][0]["type"] == "DisconnectedGraph"
