# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

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
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "该节点依赖的上游节点 id 列表。",
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
                        "description": "condition 的结构化参数，仅支持列出的字段。",
                        "properties": {
                            "kind": {"type": "string"},
                            "source": {
                                "description": "可使用 result_of 路径或绑定对象。",
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "object", "additionalProperties": True},
                                ],
                            },
                            "field": {"type": "string"},
                            "value": {},
                            "threshold": {"type": "number"},
                            "min": {"type": "number"},
                            "max": {"type": "number"},
                            "bands": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "min": {"type": "number"},
                                        "max": {"type": "number"},
                                        "label": {"type": "string"},
                                    },
                                    "required": ["min", "max"],
                                    "additionalProperties": False,
                                },
                            },
                        },
                        "additionalProperties": False,
                    },
                    "true_to_node": {
                        "type": ["string", "null"],
                        "description": "条件为真时跳转的目标节点 id，或 null 表示该分支结束。",
                    },
                    "false_to_node": {
                        "type": ["string", "null"],
                        "description": "条件为假时跳转的目标节点 id，或 null 表示该分支结束。",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "当前条件节点依赖的上游节点 id 列表。",
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
            "name": "add_switch_node",
            "description": "在工作流中新增一个 switch 节点，用于按匹配值多路分支跳转。",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "节点唯一 ID"},
                    "display_name": {"type": "string"},
                    "params": {
                        "type": "object",
                        "description": "switch 的输入来源定义，支持 source/field。",
                        "properties": {
                            "source": {
                                "description": "可使用 result_of 路径或绑定对象。",
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "object", "additionalProperties": True},
                                ],
                            },
                            "field": {"type": "string", "description": "可选的子字段访问路径。"},
                        },
                        "additionalProperties": False,
                    },
                    "cases": {
                        "type": "array",
                        "description": "分支数组，每个元素包含 match 与跳转 to_node。",
                        "items": {
                            "type": "object",
                            "properties": {
                                "match": {},
                                "value": {},
                                "to_node": {"type": ["string", "null"], "description": "匹配时跳转的节点 id 或 null。"},
                            },
                            "required": ["to_node"],
                            "additionalProperties": True,
                        },
                    },
                    "default_to_node": {
                        "type": ["string", "null"],
                        "description": "无匹配时的默认跳转节点 id，或 null 表示结束。",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "该 switch 节点依赖的上游节点 id 列表。",
                    },
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "父节点 ID（如循环子图宿主节点），无父节点则为 null。",
                    },
                },
                "required": ["id", "cases"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_action_node",
            "description": (
                "更新已创建的 action 节点字段，支持直接覆盖 display_name/params/out_params_schema/action_id，"
                "如需变更 action_id，请先调用 search_business_actions 以获取候选；可同时通过 parent_node_id 指定新的父节点。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "要更新的 action 节点 id"},
                    "display_name": {"type": "string"},
                    "params": {
                        "type": "object",
                        "description": "action 入参，仅支持 arg_schema 中定义的字段。",
                        "additionalProperties": True,
                    },
                    "out_params_schema": {"type": "object"},
                    "action_id": {"type": "string"},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "覆盖当前 action 节点的 depends_on 列表。",
                    },
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "可选的父节点 ID，提供时会覆盖现有的 parent_node_id。",
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_condition_node",
            "description": (
                "更新已创建的 condition 节点字段，可直接覆盖 display_name/params/true_to_node/false_to_node，"
                "true_to_node/false_to_node 的值可为节点 id 或 null；可通过 parent_node_id 指定新的父节点。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "要更新的 condition 节点 id"},
                    "display_name": {"type": "string"},
                    "params": {
                        "type": "object",
                        "description": (
                            "condition 参数，仅支持 kind/source/field/value/threshold/min/max/bands，"
                            "多余字段将被忽略并视为非法。"
                        ),
                        "properties": {
                            "kind": {"type": "string"},
                            "source": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "object", "additionalProperties": True},
                                ]
                            },
                            "field": {"type": ["string", "null"]},
                            "value": {},
                            "threshold": {},
                            "min": {},
                            "max": {},
                            "bands": {
                                "description": "分段阈值配置，允许数组或对象形式。",
                                "anyOf": [
                                    {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "min": {"type": "number"},
                                                "max": {"type": "number"},
                                                "label": {"type": "string"},
                                            },
                                            "required": ["min", "max"],
                                            "additionalProperties": False,
                                        },
                                    },
                                    {"type": "object", "additionalProperties": True},
                                ],
                            },
                        },
                        "additionalProperties": False,
                    },
                    "true_to_node": {
                        "type": ["string", "null"],
                        "description": "条件为真时跳转的目标节点 id，或 null 表示该分支结束。",
                    },
                    "false_to_node": {
                        "type": ["string", "null"],
                        "description": "条件为假时跳转的目标节点 id，或 null 表示该分支结束。",
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "覆盖 condition 节点的 depends_on 依赖列表。",
                    },
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "可选的父节点 ID，提供时会覆盖现有的 parent_node_id。",
                    },
                },
                "required": ["id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_switch_node",
            "description": "更新已创建的 switch 节点字段，可覆盖 display_name/params/cases/default_to_node。",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "要更新的 switch 节点 id"},
                    "display_name": {"type": "string"},
                    "params": {
                        "type": "object",
                        "description": "switch 的输入来源定义，支持 source/field。",
                        "properties": {
                            "source": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "object", "additionalProperties": True},
                                ]
                            },
                            "field": {"type": "string"},
                        },
                        "additionalProperties": False,
                    },
                    "cases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "match": {},
                                "value": {},
                                "to_node": {"type": ["string", "null"]},
                            },
                            "required": ["to_node"],
                            "additionalProperties": True,
                        },
                    },
                    "default_to_node": {"type": ["string", "null"]},
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "覆盖 switch 节点的 depends_on 依赖列表。",
                    },
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "可选的父节点 ID，提供时会覆盖现有的 parent_node_id。",
                    },
                },
                "required": ["id"],
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
                        "description": (
                            "循环参数，仅支持 loop_kind/source/condition/item_alias/body_subgraph/exports 字段，"
                            "其中 exports 只能包含 items/aggregates，其他字段会被删除并报错。"
                        ),
                        "properties": {
                            "loop_kind": {"type": "string", "enum": ["for_each", "while"]},
                            "source": {
                                "description": "for_each 迭代来源，支持绑定或 result_of 路径。",
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "object", "additionalProperties": True},
                                ],
                            },
                            "condition": {
                                "description": "while 循环退出条件，支持绑定或 result_of 路径。",
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "object", "additionalProperties": True},
                                ],
                            },
                            "item_alias": {"type": "string"},
                            "body_subgraph": {"type": "object"},
                            "exports": {
                                "type": "object",
                                "description": "循环输出定义，仅支持 items 与 aggregates。",
                                "properties": {
                                    "items": {
                                        "type": "object",
                                        "description": "每轮收集的字段定义，必须引用 body_subgraph 中的节点。",
                                        "properties": {
                                            "from_node": {
                                                "type": "string",
                                                "description": "循环体内要暴露输出的节点 id（body_subgraph.nodes 中的节点）。",
                                            },
                                            "fields": {
                                                "type": "array",
                                                "minItems": 1,
                                                "description": "要收集的字段列表，非空字符串数组。",
                                                "items": {"type": "string"},
                                            },
                                            "mode": {
                                                "type": "string",
                                                "enum": ["collect", "first", "last"],
                                                "description": "收集模式：collect/first/last。",
                                            },
                                        },
                                        "required": ["from_node", "fields"],
                                        "additionalProperties": False,
                                    },
                                    "aggregates": {
                                        "type": "array",
                                        "description": "可选的聚合定义数组。",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string", "description": "聚合结果名称。"},
                                                "from_node": {
                                                    "type": "string",
                                                    "description": "循环体内参与聚合的节点 id。",
                                                },
                                                "kind": {
                                                    "type": "string",
                                                    "enum": [
                                                        "count",
                                                        "count_if",
                                                        "max",
                                                        "min",
                                                        "sum",
                                                        "avg",
                                                    ],
                                                    "description": "聚合类型。",
                                                },
                                                "source": {
                                                    "description": "聚合来源路径。",
                                                    "anyOf": [
                                                        {"type": "string"},
                                                        {"type": "object", "additionalProperties": True},
                                                    ],
                                                },
                                                "expr": {
                                                    "type": "object",
                                                    "description": "聚合表达式，count_if 需要 field/op/value，其他需要 field。",
                                                },
                                            },
                                            "required": ["name", "from_node", "kind", "source", "expr"],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                        "additionalProperties": False,
                    },
                    "sub_graph_nodes": {
                        "type": "array",
                        "description": "加入 loop.body_subgraph 的已创建节点 id 列表，可包含子循环。",
                        "items": {"type": "string"},
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "该 loop 节点依赖的上游节点 id 列表。",
                    },
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "父节点 ID，可指向上层 loop 以形成嵌套，无父节点则为 null。",
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
            "description": "更新已创建的 loop 节点字段，可直接覆盖 display_name/params，并可调整子图归属。",
            "parameters": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "要更新的 loop 节点 id"},
                    "display_name": {"type": "string"},
                    "params": {
                        "type": "object",
                        "description": (
                            "循环参数，仅支持 loop_kind/source/condition/item_alias/body_subgraph/exports 字段，"
                            "其中 exports 只能包含 items/aggregates，其他字段会被删除并报错。"
                        ),
                        "properties": {
                            "loop_kind": {"type": "string", "enum": ["for_each", "while"]},
                            "source": {
                                "description": "for_each 迭代来源，支持绑定或 result_of 路径。",
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "object", "additionalProperties": True},
                                ],
                            },
                            "condition": {
                                "description": "while 循环退出条件，支持绑定或 result_of 路径。",
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "object", "additionalProperties": True},
                                ],
                            },
                            "item_alias": {"type": "string"},
                            "body_subgraph": {"type": "object"},
                            "exports": {
                                "type": "object",
                                "description": "循环输出定义，仅支持 items 与 aggregates。",
                                "properties": {
                                    "items": {
                                        "type": "object",
                                        "description": "每轮收集的字段定义，必须引用 body_subgraph 中的节点。",
                                        "properties": {
                                            "from_node": {
                                                "type": "string",
                                                "description": "循环体内要暴露输出的节点 id（body_subgraph.nodes 中的节点）。",
                                            },
                                            "fields": {
                                                "type": "array",
                                                "minItems": 1,
                                                "description": "要收集的字段列表，非空字符串数组。",
                                                "items": {"type": "string"},
                                            },
                                            "mode": {
                                                "type": "string",
                                                "enum": ["collect", "first", "last"],
                                                "description": "收集模式：collect/first/last。",
                                            },
                                        },
                                        "required": ["from_node", "fields"],
                                        "additionalProperties": False,
                                    },
                                    "aggregates": {
                                        "type": "array",
                                        "description": "可选的聚合定义数组。",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string", "description": "聚合结果名称。"},
                                                "from_node": {
                                                    "type": "string",
                                                    "description": "循环体内参与聚合的节点 id。",
                                                },
                                                "kind": {
                                                    "type": "string",
                                                    "enum": [
                                                        "count",
                                                        "count_if",
                                                        "max",
                                                        "min",
                                                        "sum",
                                                        "avg",
                                                    ],
                                                    "description": "聚合类型。",
                                                },
                                                "source": {
                                                    "description": "聚合来源路径。",
                                                    "anyOf": [
                                                        {"type": "string"},
                                                        {"type": "object", "additionalProperties": True},
                                                    ],
                                                },
                                                "expr": {
                                                    "type": "object",
                                                    "description": "聚合表达式，count_if 需要 field/op/value，其他需要 field。",
                                                },
                                            },
                                            "required": ["name", "from_node", "kind", "source", "expr"],
                                            "additionalProperties": False,
                                        },
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                        "additionalProperties": False,
                    },
                    "sub_graph_nodes": {
                        "type": "array",
                        "description": "将已有节点纳入 loop.body_subgraph 的 id 列表，可包含嵌套 loop。",
                        "items": {"type": "string"},
                    },
                    "depends_on": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "覆盖 loop 节点的 depends_on 依赖列表。",
                    },
                    "parent_node_id": {
                        "type": ["string", "null"],
                        "description": "可选的父节点 ID，可指向其他 loop 以形成嵌套。",
                    },
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
