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

PARAM_COMPLETION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_node_params",
            "description": "更新指定节点的 params（用于参数补全阶段）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "params": {"type": "object", "additionalProperties": True},
                },
                "required": ["node_id", "params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_workflow",
            "description": "当所有节点参数都已补全时调用，提交最终 workflow。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

WORKFLOW_VALIDATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "validate_workflow",
            "description": (
                "对当前 workflow 运行校验，包括 params.kind 合法性、参数绑定引用、必填字段等问题。"
                "返回包含 code/node_id/field/message 的错误列表，若为空代表校验通过。"
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]

WORKFLOW_EDIT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "update_node",
            "description": "修改节点的 action_id / display_name / params。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "action_id": {"type": "string", "nullable": True},
                    "display_name": {"type": "string", "nullable": True},
                    "params": {"type": "object", "additionalProperties": True},
                },
                "required": ["node_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_node_params",
            "description": "仅更新节点 params，通常用于修复缺失字段或类型问题。",
            "parameters": {
                "type": "object",
                "properties": {
                    "node_id": {"type": "string"},
                    "params": {"type": "object", "additionalProperties": True},
                },
                "required": ["node_id", "params"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_edge",
            "description": "更新 edge 的 condition，可指定 scope_node_id 在 loop.body_subgraph 内修改。",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_node": {"type": "string"},
                    "to_node": {"type": "string"},
                    "condition": {"type": "string", "nullable": True},
                    "scope_node_id": {"type": "string", "description": "可选：loop 节点 id，用于更新其 body_subgraph.edges"},
                },
                "required": ["from_node", "to_node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_edge",
            "description": "新增一条 edge，可选 scope_node_id 在 loop.body_subgraph 内添加。",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_node": {"type": "string"},
                    "to_node": {"type": "string"},
                    "condition": {"type": "string", "nullable": True},
                    "scope_node_id": {"type": "string", "description": "可选：loop 节点 id，用于在其 body_subgraph 内添加 edge"},
                },
                "required": ["from_node", "to_node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_edge",
            "description": "删除一条 edge，可选 scope_node_id 在 loop.body_subgraph 内删除。",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_node": {"type": "string"},
                    "to_node": {"type": "string"},
                    "scope_node_id": {"type": "string", "description": "可选：loop 节点 id，用于在其 body_subgraph 内删除 edge"},
                },
                "required": ["from_node", "to_node"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_workflow",
            "description": "当所有修复完成时调用，返回当前 workflow。",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

__all__ = [
    "PLANNER_TOOLS",
    "PARAM_COMPLETION_TOOLS",
    "WORKFLOW_VALIDATION_TOOLS",
    "WORKFLOW_EDIT_TOOLS",
]
