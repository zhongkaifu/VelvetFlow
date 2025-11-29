from velvetflow.verification.flow_analysis import WorkflowStaticAnalyzer


def test_detects_cycles_without_exit():
    workflow = {
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "a", "type": "action", "action_id": "compute", "params": {}},
            {"id": "b", "type": "action", "action_id": "compute", "params": {}},
        ],
        "edges": [
            {"from": "start", "to": "a"},
            {"from": "a", "to": "b"},
            {"from": "b", "to": "a"},
        ],
    }

    analyzer = WorkflowStaticAnalyzer([{"action_id": "compute", "arg_schema": {}, "output_schema": {}}])
    issues = analyzer.analyze(workflow)

    assert any(err.code == "CONTROL_FLOW_VIOLATION" for err in issues)


def test_property_and_security_checks():
    actions = [
        {
            "action_id": "fetch_secret",
            "domain": "internal",
            "arg_schema": {"type": "object", "properties": {}},
            "output_schema": {"type": "object", "properties": {"token": {"type": "string"}}},
        },
        {
            "action_id": "ship_external",
            "domain": "external_http",
            "arg_schema": {
                "type": "object",
                "properties": {"secret": {"type": "string"}},
                "required": ["secret"],
            },
            "output_schema": {},
        },
        {
            "action_id": "side_effect",
            "arg_schema": {"type": "object", "properties": {"retries": {"type": "integer"}}},
            "output_schema": {},
            "idempotent": False,
        },
        {"action_id": "end_action", "arg_schema": {}, "output_schema": {}},
    ]

    workflow = {
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "fetch", "type": "action", "action_id": "fetch_secret", "params": {}},
            {
                "id": "send",
                "type": "action",
                "action_id": "ship_external",
                "params": {"secret": {"__from__": "result_of.fetch.token"}},
            },
            {
                "id": "side",
                "type": "action",
                "action_id": "side_effect",
                "params": {"retries": 2},
            },
            {"id": "end", "type": "end", "action_id": "end_action", "params": {}},
        ],
        "edges": [
            {"from": "start", "to": "fetch"},
            {"from": "fetch", "to": "send"},
            {"from": "send", "to": "side"},
            {"from": "side", "to": "end"},
        ],
    }

    analyzer = WorkflowStaticAnalyzer(actions)
    issues = analyzer.analyze(workflow)

    codes = {err.code for err in issues}
    assert "SECURITY_VIOLATION" in codes
    assert "PROPERTY_VIOLATION" in codes


def test_unused_output_is_allowed():
    actions = [{"action_id": "noop", "arg_schema": {}, "output_schema": {}}]
    workflow = {
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "lonely", "type": "action", "action_id": "noop", "params": {}},
            {"id": "end", "type": "end"},
        ],
        "edges": [
            {"from": "start", "to": "lonely"},
            {"from": "start", "to": "end"},
        ],
    }

    analyzer = WorkflowStaticAnalyzer(actions)
    issues = analyzer.analyze(workflow)

    assert not any(err.code == "DATA_FLOW_VIOLATION" for err in issues)
