from validate_workflow import validate_workflow_data

from velvetflow.planner.repair_tools import connect_dangling_nodes_to_end


def test_connect_dangling_nodes_to_end_adds_paths():
    workflow = {
        "workflow_name": "temperature_monitor",
        "nodes": [
            {"id": "start_node", "type": "start"},
            {"id": "get_temperatures", "type": "action", "action_id": "stub.action"},
            {"id": "loop_check_temperature", "type": "action", "action_id": "stub.action"},
            {"id": "condition_any_alerts", "type": "action", "action_id": "stub.action"},
            {"id": "generate_alert_report", "type": "action", "action_id": "stub.action"},
            {"id": "generate_normal_report", "type": "action", "action_id": "stub.action"},
            {"id": "end", "type": "end"},
        ],
        "edges": [],
    }

    action_registry = [
        {"action_id": "stub.action", "arg_schema": {}, "output_schema": {}},
    ]

    errors_before = validate_workflow_data(workflow, action_registry)
    assert any(err.code == "CONTROL_FLOW_VIOLATION" for err in errors_before)
    assert any(err.code == "DATA_FLOW_VIOLATION" for err in errors_before)

    patched, summary = connect_dangling_nodes_to_end(workflow)

    assert summary["applied"] is True
    assert summary["added_edges"] == 6
    assert summary["end_node_id"] == "end"

    errors_after = validate_workflow_data(patched, action_registry)
    assert errors_after == []
