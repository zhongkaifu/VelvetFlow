"""Planner tool definitions for LLM tool-calling."""

PLANNER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_business_actions",
            "description": (
                "在海量业务动作库中按自然语言查询可用动作。返回 candidates 列表，"
                "后续 add_action_node 时 action_id 必须取自最近一次 candidates.id。"
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
            "name": "add_action_node",
            "description": (
                "在工作流中新增一个 action 节点。请先调用 search_business_actions 选出合适的 action_id，"
                "action_id 必须是最近一次 search_business_actions 返回的 candidates.id 之一。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "节点唯一 ID"},
                    "action_id": {
                        "type": "string",
                        "description": "action 节点对应的业务动作 ID。",
                    },
                    "display_name": {"type": "string"},
                    "out_params_schema": {
                        "type": "object",
                        "description": "action 节点的输出参数 Schema，格式为 {\"参数名\": \"类型\"}。",
                        "additionalProperties": {"type": "string"},
                    },
                    "params": {
                        "type": "object",
                        "description": "节点参数，可为空，但稍后会在第二阶段补全。",
                        "additionalProperties": True,
                    },
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "父节点 ID（例如循环/并行等子图的宿主节点），无父节点则为 null。",
                    },
                },
                "required": ["id", "action_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_condition_node",
            "description": (
                "在工作流中新增一个 condition 节点，请使用结构化条件参数，并显式提供 true/false 分支目标。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "节点唯一 ID"},
                    "display_name": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "enum": [
                            "list_not_empty",
                            "any_greater_than",
                            "equals",
                            "contains",
                            "not_equals",
                            "greater_than",
                            "less_than",
                            "between",
                            "all_less_than",
                            "is_empty",
                            "not_empty",
                            "is_not_empty",
                            "multi_band",
                            "compare",
                        ],
                        "description": "condition.kind（枚举值）。",
                    },
                    "params": {
                        "type": "object",
                        "description": "condition 的结构化参数，例如 {\"kind\": \"equals\", ...}。",
                        "additionalProperties": True,
                    },
                    "true_to_node": {
                        "type": ["string", "null"],
                        "description": "条件为真时跳转的目标节点 id，或 null 表示该分支结束。",
                    },
                    "false_to_node": {
                        "type": ["string", "null"],
                        "description": "条件为假时跳转的目标节点 id，或 null 表示该分支结束。",
                    },
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "父节点 ID（例如循环/并行等子图的宿主节点），无父节点则为 null。",
                    },
                },
                "required": ["id", "kind", "params", "true_to_node", "false_to_node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_action_node",
            "description": (
                "更新已创建的 action 节点字段。使用 updates 传入操作列表（同 update_node 旧接口），"
                "如需变更 action_id，请先调用 search_business_actions 以获取候选；可同时通过 parent_node_id 指定新的父节点。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "要更新的 action 节点 id"},
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
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "可选的父节点 ID，提供时会覆盖现有的 parent_node_id。",
                    },
                },
                "required": ["id", "updates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_condition_node",
            "description": (
                "更新已创建的 condition 节点字段。使用 updates 传入操作列表（同 update_node 旧接口），"
                "true_to_node/false_to_node 的值可为节点 id 或 null；可通过 parent_node_id 指定新的父节点。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "要更新的 condition 节点 id"},
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
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "可选的父节点 ID，提供时会覆盖现有的 parent_node_id。",
                    },
                },
                "required": ["id", "updates"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_loop_node",
            "description": "在工作流中新增一个 loop 节点，params.body_subgraph 可留空，稍后补全。",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "节点唯一 ID"},
                    "display_name": {"type": "string"},
                    "loop_kind": {
                        "type": "string",
                        "enum": ["for_each", "while"],
                        "description": "循环类型，仅支持 for_each/while。",
                    },
                    "source": {
                        "description": "for_each 迭代来源，支持绑定或 result_of 路径。",
                        "anyOf": [
                            {"type": "string"},
                            {"type": "object", "additionalProperties": True},
                        ],
                    },
                    "item_alias": {
                        "type": "string",
                        "description": "循环体内访问当前元素时使用的变量名。",
                    },
                    "params": {
                        "type": "object",
                        "description": "循环参数（items/exports/body_subgraph 等），可为空，稍后补全。",
                        "additionalProperties": True,
                    },
                    "sub_graph_nodes": {
                        "type": "array",
                        "description": "加入 loop.body_subgraph 的已创建节点 id 列表，禁止放入 loop 节点（不允许嵌套循环）。",
                        "items": {"type": "string"},
                    },
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "父节点 ID，不允许指向其他 loop（禁止嵌套循环），无父节点则为 null。",
                    },
                },
                "required": ["id", "loop_kind", "source", "item_alias"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_loop_node",
            "description": "更新已创建的 loop 节点字段，支持与 update_node 相同的 {op,key,value} 列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "要更新的 loop 节点 id"},
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
                    "sub_graph_nodes": {
                        "type": "array",
                        "description": "将已有节点纳入 loop.body_subgraph 的 id 列表，禁止包含 loop 节点（不允许嵌套循环）。",
                        "items": {"type": "string"},
                    },
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "可选的父节点 ID，禁止指向 loop 节点（不允许嵌套循环）。",
                    },
                },
                "required": ["id", "updates"],
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
    {
        "type": "function",
        "function": {
            "name": "dump_model",
            "description": "输出当前 workflow 的完整 DSL（含 nodes/edges），便于分析下一步规划。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

__all__ = ["PLANNER_TOOLS"]
