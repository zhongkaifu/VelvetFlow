from pathlib import Path

from velvetflow.executor import DynamicActionExecutor, load_simulation_data
from velvetflow.models import Workflow


SIM_DATA_PATH = Path(__file__).resolve().parents[1] / "simulation_data.json"
SIM_DATA = load_simulation_data(str(SIM_DATA_PATH))


def test_loop_exports_collects_values_from_body_nodes():
    workflow_dict = {
        "workflow_name": "loop_exports_collects",
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
                                "id": "add_to_warning_list",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "params": {
                                    "employee_id": "{{ loop.employee.employee_id }}",
                                    "last_temperature": "{{ loop.employee.temperature }}",
                                    "status": "预警",
                                },
                                "parent_node_id": "loop_employees",
                            },
                        ],
                        "exit": "add_to_warning_list",
                    },
                    "exports": {
                        "records": "{{ result_of.add_to_warning_list }}",
                        "employee_ids": "{{ result_of.add_to_warning_list.employee_id }}",
                    },
                },
            }
        ],
    }

    workflow = Workflow.model_validate(workflow_dict)
    executor = DynamicActionExecutor(workflow, simulations=SIM_DATA)

    results = executor.run()

    loop_output = results["loop_employees"]
    assert loop_output["exports"]["employee_ids"] == ["E100", "E101", "E102"]
    assert loop_output["exports"]["records"] == [
        {"employee_id": "E100", "last_temperature": "39.2", "status": "预警"},
        {"employee_id": "E101", "last_temperature": "37.4", "status": "预警"},
        {"employee_id": "E102", "last_temperature": "38.5", "status": "预警"},
    ]
