import pytest

from velvetflow.jinja_utils import validate_jinja_expr


def test_validate_jinja_expr_allows_registered_filters_and_tests():
    # Should not raise when using built-in filters and tests
    validate_jinja_expr("items | map('length') | select('truthy') | list", path="expr")
    validate_jinja_expr("result.value | length > 0")


def test_validate_jinja_expr_rejects_empty_or_non_string():
    with pytest.raises(ValueError, match="expression 需要非空字符串表达式"):
        validate_jinja_expr(" ")
    with pytest.raises(ValueError, match="expression 需要非空字符串表达式"):
        validate_jinja_expr(None)  # type: ignore[arg-type]


def test_validate_jinja_expr_rejects_invalid_syntax():
    with pytest.raises(ValueError, match="不是合法的 Jinja 表达式"):
        validate_jinja_expr("foo(", path="invalid")


def test_validate_jinja_expr_rejects_unknown_filters_and_tests():
    with pytest.raises(ValueError, match="使用了未注册的过滤器: unknown_filter"):
        validate_jinja_expr("items | unknown_filter")
    with pytest.raises(ValueError, match="使用了未注册的过滤器: unknown"):
        validate_jinja_expr("items | map('unknown')")
    with pytest.raises(ValueError, match="使用了未注册的测试: unknown_test"):
        validate_jinja_expr("items | select('unknown_test')")
