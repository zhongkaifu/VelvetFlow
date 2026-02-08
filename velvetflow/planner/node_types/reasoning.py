# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Reasoning node definitions."""

REASONING_PARAM_FIELDS = {
    "system_prompt",
    "task_prompt",
    "context",
    "expected_output_format",
    "toolset",
}

REASONING_NODE_FIELDS = {
    "id",
    "type",
    "display_name",
    "params",
    "out_params_schema",
    "parent_node_id",
    "depends_on",
}
