# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

import sys
import types

import pytest


def _install_dummy_module(name: str, attrs: dict[str, object]) -> None:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


_install_dummy_module(
    "crawl4ai",
    {
        "AsyncWebCrawler": object,
        "BrowserConfig": object,
        "CacheMode": object,
        "CrawlerRunConfig": object,
        "LLMConfig": object,
        "WebCrawler": object,
    },
)
_install_dummy_module("crawl4ai.extraction_strategy", {"LLMExtractionStrategy": object})
_install_dummy_module("openai", {"OpenAI": object})

from tools.builtin import _normalize_web_url, _resolve_search_url, search_web


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


def test_resolve_search_url_handles_google_redirect() -> None:
    raw_redirect = "/url?q=https://example.com/path%3Fa%3D1&sa=U&ved=2ahUKEwj"
    assert _resolve_search_url(raw_redirect) == "https://example.com/path?a=1"


def test_search_web_uses_googlesearch(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int, str]] = []

    class DummyResult:
        def __init__(self, title: str, link: str, description: str) -> None:
            self.title = title
            self.link = link
            self.description = description

    def fake_search(query: str, num_results: int = 5, lang: str = "en") -> list[object]:
        calls.append((query, num_results, lang))
        return [
            DummyResult("Title A", "https://example.com/a", "Snippet A"),
            {
                "title": "Title B",
                "link": "/url?q=https://example.com/b&sa=U",
                "description": "Snippet B",
            },
            "https://example.com/c",
        ]

    _install_dummy_module("googlesearch", {"search": fake_search})

    results = search_web("widgets", limit=2)

    assert calls == [("widgets", 2, "en")]
    assert results == {
        "results": [
            {
                "title": "Title A",
                "url": "https://example.com/a",
                "snippet": "Snippet A",
            },
            {
                "title": "Title B",
                "url": "https://example.com/b",
                "snippet": "Snippet B",
            },
        ]
    }
