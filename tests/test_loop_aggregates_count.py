from pathlib import Path

from velvetflow.executor import DynamicActionExecutor, load_simulation_data
from velvetflow.models import Workflow


SIM_DATA_PATH = Path(__file__).resolve().parents[1] / "simulation_data.json"
SIM_DATA = load_simulation_data(str(SIM_DATA_PATH))


def test_loop_aggregates_count_accumulates_from_executed_body_nodes():
    workflow_dict = {
        "workflow_name": "loop_count_aggregates",
        "nodes": [
            {
                "id": "loop_employees",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": [
                        {"employee_id": "E100", "temperature": 39.2},
                        {"employee_id": "E101", "temperature": 37.4},
                        {"employee_id": "E102", "temperature": 38.5},
                    ],
                    "item_alias": "employee",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "condition_high_temp",
                                "type": "condition",
                                "params": {
                                    "expression": "{{ employee.temperature > 38 }}",
                                },
                                "true_to_node": "add_to_warning_list",
                                "false_to_node": None,
                                "parent_node_id": "loop_employees",
                            },
                            {
                                "id": "add_to_warning_list",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "params": {
                                    "employee_id": {"__from__": "loop.employee.employee_id"},
                                    "last_temperature": {"__from__": "loop.employee.temperature"},
                                    "status": "预警",
                                },
                                "parent_node_id": "loop_employees",
                                "depends_on": ["condition_high_temp"],
                            },
                        ],
                        "entry": "condition_high_temp",
                        "exit": "add_to_warning_list",
                    },
                    "exports": {
                        "items": {
                            "from_node": "add_to_warning_list",
                            "fields": ["employee_id", "last_temperature", "status"],
                        },
                        "aggregates": [
                            {
                                "name": "count",
                                "kind": "count",
                                "from_node": "add_to_warning_list",
                                "expr": {"kind": "count"},
                            }
                        ],
                    },
                },
            }
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(workflow, simulations=SIM_DATA)

    results = executor.run()

    loop_output = results["loop_employees"]
    assert loop_output["aggregates"]["count"] == 2
    assert loop_output["items"] == [
        {"employee_id": "E100", "last_temperature": "39.2", "status": "预警"},
        {"employee_id": "E102", "last_temperature": "38.5", "status": "预警"},
    ]
