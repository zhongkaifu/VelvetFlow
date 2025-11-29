import sys
from types import SimpleNamespace

# Provide a lightweight stub to satisfy optional OpenAI imports during tests.
sys.modules.setdefault("openai", SimpleNamespace(OpenAI=object))

from velvetflow.executor import DynamicActionExecutor
from velvetflow.models import Workflow


def test_executor_advances_from_action_to_loop():
    workflow = Workflow.model_validate(
        {
            "workflow_name": "temperature_monitor",
            "nodes": [
                {
                    "id": "get_temperatures",
                    "type": "action",
                    "action_id": "hr.get_today_temperatures.v1",
                    "params": {"date": "2024-06-07"},
                },
                {
                    "id": "loop_check_temperature",
                    "type": "loop",
                    "params": {
                        "loop_kind": "for_each",
                        "source": "result_of.get_temperatures.data",
                        "item_alias": "employee",
                        "body_subgraph": {
                            "nodes": [
                                {
                                    "id": "check_temperature_condition",
                                    "type": "condition",
                                    "params": {
                                        "kind": "any_greater_than",
                                        "source": "employee",
                                        "field": "temperature",
                                        "threshold": 38,
                                    },
                                    "true_to_node": "add_to_warning_list",
                                },
                                {
                                    "id": "add_to_warning_list",
                                    "type": "action",
                                    "action_id": "hr.update_employee_health_profile.v1",
                                    "params": {
                                        "employee_id": "employee.employee_id",
                                        "last_temperature": "employee.temperature",
                                        "status": "fever",
                                    },
                                },
                            ]
                        },
                        "exports": {
                            "items": {
                                "from_node": "add_to_warning_list",
                                "fields": ["employee_id"],
                            }
                        },
                    },
                },
            ],
            "edges": [{"from": "get_temperatures", "to": "loop_check_temperature"}],
        }
    )

    executor = DynamicActionExecutor(
        workflow, simulations="velvetflow/simulation_data.json"
    )
    results = executor.run()

    assert "loop_check_temperature" in results
    items = results["loop_check_temperature"]["exports"]["items"]
    assert {"employee_id": "E002"} in items
