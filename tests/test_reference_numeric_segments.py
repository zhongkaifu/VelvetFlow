# Author: OpenAI Assistant
# License: BSD 3-Clause License

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from velvetflow.bindings import BindingContext
from velvetflow.models import Workflow


def _build_ctx(results):
    workflow = Workflow.model_validate(
        {
            "workflow_name": "binding-numeric", 
            "nodes": [{"id": "search", "type": "action", "params": {}}],
            "edges": [],
        }
    )
    return BindingContext(workflow, results)


def test_dotted_numeric_segment_is_treated_as_list_index():
    ctx = _build_ctx(results={"search": {"results": [{"title": "foo"}]}})

    assert (
        ctx.resolve_binding({"__from__": "result_of.search.results.0.title"})
        == "foo"
    )

