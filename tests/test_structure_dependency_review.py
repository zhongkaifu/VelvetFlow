# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from velvetflow.planner.structure import _find_nodes_without_upstream


def test_nodes_without_upstream_excludes_special_nodes():
    workflow = {
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "orphan", "type": "action", "action_id": "search"},
            {"id": "end", "type": "end"},
            {"id": "exit", "type": "exit"},
        ],
        "edges": [],
    }

    dangling = _find_nodes_without_upstream(workflow)

    assert [node["id"] for node in dangling] == ["orphan"]


def test_nodes_without_upstream_respects_incoming_edges():
    workflow = {
        "nodes": [
            {"id": "start", "type": "start"},
            {"id": "query", "type": "action", "action_id": "search"},
            {
                "id": "notify",
                "type": "action",
                "action_id": "notify_user",
                "display_name": "Send notice",
            },
        ],
        "edges": [],
    }

    dangling = _find_nodes_without_upstream(workflow)

    assert [node["id"] for node in dangling] == ["query", "notify"]

