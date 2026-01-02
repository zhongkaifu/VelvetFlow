import pytest

from velvetflow.models import Workflow


def _workflow_with_scalar_loop_source():
    return {
        "workflow_name": "test",
        "description": "",
        "nodes": [
            {
                "id": "fetch",
                "type": "action",
                "action_id": "hr.get_today_temperatures.v1",
                "display_name": "fetch",
                "params": {"date": "{{system.date}}"},
                "depends_on": [],
                "out_params_schema": {
                    "type": "object",
                    "properties": {
                        "date": {"type": "string"},
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "employee_id": {"type": "string"},
                                    "temperature": {"type": "number"},
                                },
                                "required": ["employee_id", "temperature"],
                            },
                        },
                    },
                    "required": ["date", "data"],
                },
            },
            {
                "id": "loop_invalid",
                "type": "loop",
                "display_name": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "{{ result_of.fetch.date }}",
                    "item_alias": "row",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "noop",
                                "type": "action",
                                "action_id": "hr.update_employee_health_profile.v1",
                                "display_name": "noop",
                                "params": {"employee_id": "{{ row }}"},
                                "out_params_schema": {
                                    "type": "object",
                                    "properties": {"employee_id": {"type": "string"}},
                                    "required": ["employee_id"],
                                },
                                "parent_node_id": "loop_invalid",
                                "depends_on": [],
                            }
                        ]
                    },
                    "exports": {"items": "{{ result_of.noop.employee_id }}"},
                },
                "depends_on": ["fetch"],
            },
        ],
    }


def test_loop_source_must_be_array_reports_actual_type():
    workflow = _workflow_with_scalar_loop_source()

    with pytest.raises(Exception) as excinfo:
        Workflow.model_validate(workflow)

    message = str(excinfo.value)
    assert "source 应该引用数组/序列" in message
    assert "类型为 string" in message or "类型为 未知" in message


def _workflow_with_concatenated_loop_source():
    return {
        "workflow_name": "test_concat_loop",
        "description": "",
        "nodes": [
            {
                "id": "search_nvidia",
                "type": "action",
                "action_id": "news.search_nvidia.v1",
                "display_name": "search nvidia",
                "params": {"query": "Nvidia"},
                "depends_on": [],
                "out_params_schema": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {"type": "object", "properties": {"title": {"type": "string"}}},
                        }
                    },
                },
            },
            {
                "id": "search_google",
                "type": "action",
                "action_id": "news.search_google.v1",
                "display_name": "search google",
                "params": {"query": "Google"},
                "depends_on": [],
                "out_params_schema": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {"type": "object", "properties": {"title": {"type": "string"}}},
                        }
                    },
                },
            },
            {
                "id": "summaries",
                "type": "loop",
                "display_name": "summaries",
                "params": {
                    "loop_kind": "for_each",
                    "source": "{{ result_of.search_nvidia.results + result_of.search_google.results }}",
                    "item_alias": "article",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "summarize",
                                "type": "action",
                                "action_id": "news.summarize.v1",
                                "display_name": "summarize",
                                "params": {"text": "{{ article.title }}"},
                                "out_params_schema": {
                                    "type": "object",
                                    "properties": {"summary": {"type": "string"}},
                                    "required": ["summary"],
                                },
                                "parent_node_id": "summaries",
                                "depends_on": [],
                            }
                        ]
                    },
                    "exports": {"items": "{{ result_of.summarize.summary }}"},
                },
                "depends_on": ["search_nvidia", "search_google"],
            },
        ],
    }


def test_loop_source_allows_jinja_concatenation_expression():
    workflow = _workflow_with_concatenated_loop_source()

    parsed = Workflow.model_validate(workflow)

    assert parsed.workflow_name == "test_concat_loop"
