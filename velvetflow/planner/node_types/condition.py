# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

"""Condition node definitions."""

CONDITION_PARAM_FIELDS = {"expression"}

CONDITION_NODE_FIELDS = {
    "id",
    "type",
    "display_name",
    "params",
    "true_to_node",
    "false_to_node",
    "parent_node_id",
    "depends_on",
}
