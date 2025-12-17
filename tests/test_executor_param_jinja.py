import pytest

from velvetflow.bindings import BindingContext, eval_node_params
from velvetflow.models import Workflow


def _build_workflow():
    wf_dict = {
        "workflow_name": "jinja_runtime",
        "nodes": [
            {
                "id": "update_profile",
                "type": "action",
                "action_id": "hr.update_employee_health_profile.v1",
                "display_name": "更新员工健康档案",
                "params": {
                    "employee_id": "{{loop.item.employee_id}}",
                    "status": "{% if loop.item.temperature > 38 %}High Temperature{% else %}Normal{% endif %}",
                },
            }
        ],
    }
    return Workflow.model_validate(wf_dict)


@pytest.mark.parametrize(
    "temperature,expected",
    [
        (39.1, "High Temperature"),
        (36.5, "Normal"),
    ],
)
def test_loop_body_jinja_templates_are_rendered(temperature, expected):
    workflow = _build_workflow()
    node = workflow.nodes[0]
    ctx = BindingContext(
        workflow,
        results={},
        loop_ctx={"item": {"employee_id": "emp-1", "temperature": temperature}},
        loop_id="loop_employees",
    )

    resolved = eval_node_params(node, ctx)

    assert resolved["status"] == expected
    assert resolved["employee_id"] == "emp-1"


def test_env_variables_are_rendered(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    workflow = Workflow.model_validate(
        {
            "workflow_name": "jinja_runtime_env",
            "nodes": [
                {
                    "id": "use_env",
                    "type": "action",
                    "action_id": "common.web_scraper.search_and_crawl.v1",
                    "display_name": "使用环境变量",
                    "params": {
                        "openai_api_key": "{{env.OPENAI_API_KEY}}",
                    },
                }
            ],
        }
    )

    ctx = BindingContext(workflow, results={})
    node = workflow.nodes[0]

    resolved = eval_node_params(node, ctx)

    assert resolved["openai_api_key"] == "sk-test-key"
