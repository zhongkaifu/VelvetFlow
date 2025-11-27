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
                "- type='loop'：params 需要提供 loop_kind(for_each/while)、source/condition、item_alias、body_subgraph(nodes/edges/entry/exit) 等。"
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
                    "params": {
                        "type": "object",
                        "description": "节点参数，可为空，但稍后会在第二阶段补全。",
                        "additionalProperties": True,
                    },
                },
                "required": ["id", "type"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_edge",
            "description": "新增一条有向边（from -> to），可选 condition（true/false 等）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_node": {"type": "string"},
                    "to_node": {"type": "string"},
                    "condition": {"type": "string", "nullable": True},
                },
                "required": ["from_node", "to_node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "design_loop_exports",
            "description": (
                "为已存在的 loop 节点设计 exports（items/aggregates）。\n"
                "- items.from_node 必须引用 body_subgraph 内的节点，fields 是希望暴露的字段名。\n"
                "- aggregates 是可选的汇总指标，支持 count/count_if/max/min/sum/avg，from_node 同样只能引用 loop body 节点。\n"
                "- 如果只需简单收集列表，可仅填写 items 字段。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "loop_id": {"type": "string", "description": "已添加的 loop 节点 ID"},
                    "exports": {
                        "type": "object",
                        "properties": {
                            "items": {
                                "type": "object",
                                "properties": {
                                    "from_node": {"type": "string"},
                                    "fields": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "mode": {
                                        "type": "string",
                                        "enum": ["collect", "append", "extend"],
                                    },
                                },
                                "required": ["from_node", "fields"],
                                "additionalProperties": True,
                            },
                            "aggregates": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                        "from_node": {"type": "string"},
                                        "kind": {
                                            "type": "string",
                                            "enum": ["count", "count_if", "max", "min", "sum", "avg"],
                                        },
                                        "expr": {
                                            "type": "object",
                                            "properties": {
                                                "field": {"type": "string"},
                                                "op": {"type": "string"},
                                                "value": {},
                                            },
                                            "additionalProperties": True,
                                        },
                                    },
                                    "required": ["name", "kind", "from_node"],
                                    "additionalProperties": True,
                                },
                            },
                        },
                        "required": ["items"],
                        "additionalProperties": True,
                    },
                },
                "required": ["loop_id", "exports"],
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
