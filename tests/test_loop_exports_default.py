from velvetflow.bindings import BindingContext
from velvetflow.executor.loops import LoopExecutionMixin
from velvetflow.models import Workflow


class _DummyLoopExecutor(LoopExecutionMixin):
    def __init__(self, workflow: Workflow) -> None:
        self.workflow = workflow


def test_exports_collect_values_from_loop_body():
    workflow = Workflow.model_validate(
        {"workflow_name": "dummy", "nodes": [{"id": "noop", "type": "action"}]}
    )
    dummy = _DummyLoopExecutor(workflow)
    binding = BindingContext(
        workflow,
        {"collect_high_temp": {"employee_id": "e1", "temperature": 39}},
        loop_ctx={"index": 0, "item": {"employee_id": "e1"}},
        loop_id="loop_employees",
    )
    exports_output = {}

    dummy._apply_loop_exports(
        {"items": "{{ result_of.collect_high_temp.employee_id }}", "employee_ids": "{{ result_of.collect_high_temp.employee_id }}"},
        exports_output,
        binding,
    )

    assert exports_output["items"] == ["e1"]
    assert exports_output["employee_ids"] == ["e1"]
