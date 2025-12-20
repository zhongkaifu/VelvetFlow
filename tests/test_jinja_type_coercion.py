from velvetflow.jinja_utils import eval_jinja_expr


def test_jinja_expression_coerces_numeric_strings_for_comparison():
    context = {"result": {"last_temperature": "39"}}

    assert eval_jinja_expr("result.last_temperature > 38", context) is True
    assert eval_jinja_expr("38 < result.last_temperature", context) is True
