"""Planner tool definitions for LLM tool-calling."""

PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_business_actions",
            "description": (
                "在海量业务动作库中按自然语言查询可用动作。返回 candidates 列表，"
                "后续 add_node(type='action') 时 action_id 必须取自最近一次 candidates.id。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_workflow_meta",
            "description": "设置工作流的基本信息（名称、描述）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "workflow_name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["workflow_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_node",
            "description": (
                "在工作流中新增一个节点。\n"
                "- type='action'：请先调用 search_business_actions 选出合适的 action_id。\n"
                "  action_id 必须是最近一次 search_business_actions 返回的 candidates.id 之一。\n"
                "- type='condition'：请在 params 中使用结构化条件，例如：\n"
                "  {\"kind\": \"any_greater_than\", \"source\": \"result_of.some_node.items\", \"field\": \"value\", \"threshold\": 10 }\n"
                "或 {\"kind\": \"equals\", \"source\": \"result_of.some_node.count\", \"value\": 0 }，也可以使用 between/not_equals/multi_band 等枚举。\n"
                "  必须同时提供 true_to_node 和 false_to_node，可为节点 id（继续执行）或 null（该分支结束）。\n"
                "- type='loop'：params 需要提供 loop_kind(for_each/while)、source/condition、item_alias、body_subgraph(nodes) 等。\n"
                "- 节点依赖关系通过 params 中的 result_of 引用自动推导，edges 会被隐式记录为上下文，无需手工 add_edge。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "节点唯一 ID"},
                    "type": {
                        "type": "string",
                        "enum": ["start", "end", "action", "condition", "loop", "parallel"],
                    },
                    "action_id": {
                        "type": "string",
                        "description": "type='action' 时指定 action_id。",
                        "nullable": True,
                    },
                    "display_name": {"type": "string"},
                    "out_params_schema": {
                        "type": "object",
                        "description": "action 节点的输出参数 Schema，字段列出可引用的 result_of 输出。",
                        "additionalProperties": True,
                    },
                    "params": {
                        "type": "object",
                        "description": "节点参数，可为空，但稍后会在第二阶段补全。",
                        "additionalProperties": True,
                    },
                    "true_to_node": {
                        "type": ["string", "null"],
                        "description": "condition 节点为真时跳转的目标节点 id，或 null 表示该分支结束（type=condition 必填）",
                    },
                    "false_to_node": {
                        "type": ["string", "null"],
                        "description": "condition 节点为假时跳转的目标节点 id，或 null 表示该分支结束（type=condition 必填）",
                    },
                },
                "required": ["id", "type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_node",
                "description": (
                "更新已创建节点的字段。使用 updates 传入操作列表，每项包含 key/value 与 op（add/modify/remove），"
                "例如 [{\"op\": \"modify\", \"key\": \"display_name\", \"value\": \"新的名称\"}, {\"op\": \"remove\", \"key\": \"params\"}]。"
                "对于 condition 节点，true_to_node/false_to_node 的值可为节点 id 或 null（表示该分支结束）。"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "要更新的节点 id"},
                    "updates": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {
                                    "type": "string",
                                    "enum": ["add", "modify", "remove"],
                                    "description": "对目标 key 执行的操作，默认为 modify。",
                                    "default": "modify",
                                },
                                "key": {"type": "string"},
                                "value": {
                                    "description": "要写入的值，op=remove 时可省略。",
                                    "nullable": True,
                                },
                            },
                            "required": ["key"],
                        },
                        "description": "要更新的字段列表，按顺序覆盖或删除。",
                    },
                },
                "required": ["id", "updates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_node",
            "description": "删除已创建的节点，可用于移除重复或错误节点。",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "要删除的节点 id"},
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_workflow",
            "description": "当你认为结构已经覆盖需求时调用，结束规划阶段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "ready": {"type": "boolean", "default": True},
                    "notes": {"type": "string"},
                },
                "required": ["ready"],
            },
        },
    },
]

__all__ = ["PLANNER_TOOLS"]
