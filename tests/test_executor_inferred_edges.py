from velvetflow.executor import DynamicActionExecutor
from velvetflow.models import Workflow


def test_executor_inferrs_edges_from_bindings_when_missing():
    """Ensure execution order respects result_of references even without explicit edges."""

    workflow = Workflow.model_validate(
        {
            "workflow_name": "inferred_edges_demo",
            "nodes": [
                {
                    "id": "combine",
                    "type": "action",
                    "action_id": "hr.notify_human.v1",
                    "params": {
                        "text": "value={{result_of.producer.output}}",
                    },
                },
                {
                    "id": "producer",
                    "type": "action",
                    "action_id": "ops.create_incident.v1",
                },
            ],
            # 留空 edges，依赖执行器自动推导
            "edges": [],
        }
    )

    executor = DynamicActionExecutor(
        workflow,
        simulations={
            "ops.create_incident.v1": {"result": {"output": "P"}},
            "hr.notify_human.v1": {"result": {"status": "combined"}},
        },
    )

    results = executor.run()

    # combine 节点应在 producer 之后执行，模板被替换为 producer 的输出
    assert results["combine"]["params"]["text"] == "value=P"

