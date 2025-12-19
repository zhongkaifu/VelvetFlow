# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import pytest

from tools.builtin import _normalize_web_url, _resolve_search_url


@pytest.mark.parametrize(
    "raw_url, expected",
    [
        ("example.com/page", "https://example.com/page"),
        ("http://example.com", "http://example.com"),
        ("https://secure.example.com", "https://secure.example.com"),
        ("//cdn.example.com/resource", "https://cdn.example.com/resource"),
        ("", ""),
    ],
)
def test_normalize_web_url(raw_url: str, expected: str) -> None:
    assert _normalize_web_url(raw_url) == expected


def test_resolve_search_url_unwraps_and_normalizes_redirect() -> None:
    raw_redirect = "/l/?kh=-1&uddg=https%3A%2F%2Fexample.com%2Fpath%3Fq%3D1"
    assert _resolve_search_url(raw_redirect) == "https://example.com/path?q=1"


def test_resolve_search_url_adds_scheme_when_missing() -> None:
    assert _resolve_search_url("example.com/path") == "https://example.com/path"
