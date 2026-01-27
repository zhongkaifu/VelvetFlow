# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from velvetflow.verification.repair_suggestions import generate_repair_suggestions


def test_disconnected_nodes_receive_edge_suggestion():
    workflow = {
        "nodes": [
            {"id": "lonely", "type": "action", "action_id": "emit", "params": {}},
        ],
        "edges": [],
    }
    actions = [
        {
            "action_id": "emit",
            "arg_schema": {},
            "output_schema": {"type": "object", "properties": {"out": {"type": "string"}}},
        }
    ]

    patched, suggestions = generate_repair_suggestions(workflow, actions)

    assert patched.get("edges", []) == []
    assert all("连接未入度节点" not in s.description for s in suggestions)
