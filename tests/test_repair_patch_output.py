import json
import inspect
import textwrap

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


def test_inject_line_numbers_from_context():
    workflow = {
        "workflow_name": "demo",
        "description": "",
        "nodes": [],
    }

    patch = textwrap.dedent(
        """
        --- workflow.json
        +++ workflow.json
        @@
        -  "description": "",
        +  "description": "patched",
        """
    )

    rewritten = repair._inject_line_numbers_into_patch(workflow, patch)

    assert rewritten is not None
    assert "@@ -2,1 +2,1 @@" in rewritten


def test_inject_line_numbers_guesses_when_context_missing(monkeypatch):
    workflow = {
        "workflow_name": "demo",
        "description": "",
        "nodes": [],
    }

    captured_warnings = []
    monkeypatch.setattr(repair, "log_warn", lambda message: captured_warnings.append(message))

    patch = textwrap.dedent(
        """
        --- workflow.json
        +++ workflow.json
        @@
        -  "missing": true
        +  "missing": false
        """
    )

    rewritten = repair._inject_line_numbers_into_patch(workflow, patch)

    assert rewritten is not None
    assert "@@ -1,1 +1,1 @@" in rewritten
    assert any("补丁缺少行号且无法通过上下文定位" in msg for msg in captured_warnings)


def test_repair_prompt_includes_git_apply_schema():
    source = inspect.getsource(repair.repair_workflow_with_llm)

    assert "--- workflow.json" in source
    assert "@@ -<旧起始行>,<旧长度> +<新起始行>,<新长度> @@" in source
    assert "git apply" in source
