import pytest

from velvetflow.models import PydanticValidationError, Workflow


def _edge_tuples(edges):
    return {(edge.from_node, edge.to_node, edge.condition) for edge in edges}


def test_minimal_workflow_from_docs_infers_edges():
    workflow = {
        "workflow_name": "send_newsletter",
        "description": "向 CRM 客户发送新品资讯并记录审批",
        "nodes": [
            {
                "id": "search_users",
                "type": "action",
                "action_id": "crm.search_customers",
                "params": {"segment": "recent_buyers"},
            },
            {
                "id": "approve",
                "type": "condition",
                "params": {"expression": "{{ result_of.search_users.count >= 10 }}"},
                "true_to_node": "send_email",
                "false_to_node": "end",
            },
            {
                "id": "send_email",
                "type": "action",
                "action_id": "crm.send_newsletter",
                "params": {
                    "audience": "{{ result_of.search_users.items }}",
                    "template_id": "spring_launch",
                },
            },
            {"id": "end", "type": "end"},
        ],
    }

    parsed = Workflow.model_validate(workflow)
    assert parsed.workflow_name == "send_newsletter"
    assert len(parsed.nodes) == 4

    edges = _edge_tuples(parsed.edges)
    expected = {
        ("search_users", "approve", None),
        ("search_users", "send_email", None),
        ("approve", "send_email", "true"),
        ("approve", "end", "false"),
    }
    assert expected.issubset(edges)


def test_action_node_rejects_branch_fields():
    workflow = {
        "workflow_name": "invalid_action_branch",
        "nodes": [
            {
                "id": "send_email",
                "type": "action",
                "action_id": "crm.send_newsletter",
                "params": {"template_id": "spring_launch"},
                "true_to_node": "end",
            }
        ],
    }

    with pytest.raises(PydanticValidationError) as excinfo:
        Workflow.model_validate(workflow)

    assert "action 节点不支持 true/false 分支字段" in str(excinfo.value)


def test_condition_branch_must_be_string():
    workflow = {
        "workflow_name": "invalid_condition_branch",
        "nodes": [
            {
                "id": "check_budget",
                "type": "condition",
                "params": {"expression": "{{ result_of.estimate.budget > 10000 }}"},
                "true_to_node": 123,
                "false_to_node": "auto_purchase",
            }
        ],
    }

    with pytest.raises(PydanticValidationError) as excinfo:
        Workflow.model_validate(workflow)

    assert "true_to_node 必须是字符串" in str(excinfo.value)


def test_switch_cases_require_objects_and_targets():
    workflow = {
        "workflow_name": "invalid_switch_cases",
        "nodes": [
            {
                "id": "route_channel",
                "type": "switch",
                "params": {"source": "{{ result_of.detect_user.preferences }}"},
                "cases": ["email", {"match": "sms", "to_node": 456}],
            }
        ],
    }

    with pytest.raises(PydanticValidationError) as excinfo:
        Workflow.model_validate(workflow)

    message = str(excinfo.value)
    assert "case 必须是对象" in message
    assert "to_node 必须是字符串" in message


def test_loop_body_requires_nodes_and_action():
    workflow = {
        "workflow_name": "invalid_loop_body",
        "nodes": [
            {
                "id": "batch_payments",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "loop.items",
                    "item_alias": "invoice",
                    "body_subgraph": {"nodes": []},
                },
            }
        ],
    }

    with pytest.raises(PydanticValidationError) as excinfo:
        Workflow.model_validate(workflow)

    assert "body_subgraph.nodes 不能为空" in str(excinfo.value)

    workflow["nodes"][0]["params"]["body_subgraph"] = {
        "nodes": [{"id": "only_start", "type": "start"}]
    }

    with pytest.raises(PydanticValidationError) as excinfo:
        Workflow.model_validate(workflow)

    assert "loop 子图必须包含至少一个 action 节点" in str(excinfo.value)


def test_complex_dag_from_docs_infers_edges():
    workflow = {
        "workflow_name": "async_order_pipeline",
        "nodes": [
            {
                "id": "search_orders",
                "type": "action",
                "action_id": "crm.list_orders",
                "params": {"days": 7},
                "out_params_schema": {
                    "type": "object",
                    "properties": {"items": {"type": "array"}},
                },
            },
            {
                "id": "if_has_orders",
                "type": "condition",
                "params": {"expression": "{{ result_of.search_orders.count > 0 }}"},
                "true_to_node": "for_each_order",
                "false_to_node": "end",
            },
            {
                "id": "for_each_order",
                "type": "loop",
                "params": {
                    "loop_kind": "for_each",
                    "source": "{{ result_of.search_orders.items }}",
                    "item_alias": "order",
                    "body_subgraph": {
                        "nodes": [
                            {
                                "id": "fetch_detail",
                                "type": "action",
                                "action_id": "crm.get_order_detail",
                                "params": {"order_id": "{{ loop.order.id }}"},
                            }
                        ]
                    },
                    "exports": {"orders": "{{ result_of.fetch_detail.id }}"},
                },
            },
            {
                "id": "ops_summary",
                "type": "action",
                "depends_on": ["for_each_order"],
                "action_id": "ops.send_report",
                "params": {"orders": "{{ result_of.for_each_order.exports.orders }}"},
            },
            {
                "id": "finance_summary",
                "type": "action",
                "depends_on": ["for_each_order"],
                "action_id": "finance.sync_erp",
                "params": {"orders": "{{ result_of.for_each_order.exports.orders }}"},
            },
            {"id": "end", "type": "end"},
        ],
    }

    parsed = Workflow.model_validate(workflow)
    edges = _edge_tuples(parsed.edges)

    expected = {
        ("search_orders", "if_has_orders", None),
        ("if_has_orders", "for_each_order", "true"),
        ("if_has_orders", "end", "false"),
        ("for_each_order", "ops_summary", None),
        ("for_each_order", "finance_summary", None),
    }
    assert expected.issubset(edges)
