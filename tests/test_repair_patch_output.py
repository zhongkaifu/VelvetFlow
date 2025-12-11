import json

from velvetflow.planner import repair


def test_apply_patch_output_merges_changes():
    workflow = {
        "workflow_name": "demo",
        "description": "",
        "nodes": [],
    }

    patch = (
        'diff --git a/workflow.json b/workflow.json\n'
        '--- a/workflow.json\n'
        '+++ b/workflow.json\n'
        '@@ -1,5 +1,10 @@\n'
        ' {\n'
        '-  "description": "",\n'
        '-  "nodes": [],\n'
        '+  "description": "patched",\n'
        '+  "nodes": [\n'
        '+    {\n'
        '+      "id": "start",\n'
        '+      "type": "start"\n'
        '+    }\n'
        '+  ],\n'
        '   "workflow_name": "demo"\n'
        ' }\n'
    )

    patched = repair._apply_patch_output(workflow, patch)

    assert patched is not None
    assert patched["description"] == "patched"
    assert patched["nodes"] == [{"id": "start", "type": "start"}]


def test_looks_like_patch_detection():
    assert repair._looks_like_patch("diff --git a/one b/one")
    assert repair._looks_like_patch("--- a")
    assert not repair._looks_like_patch(json.dumps({"a": 1}))
