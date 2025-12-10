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


# Stub out optional dependencies so importing the tool module remains lightweight
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

from tools.builtin import _resolve_search_url, business_action


def test_search_and_crawl_uses_google_results(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, int]] = []
    scraped: list[tuple[str, str, str]] = []

    def fake_search(query: str, limit: int = 5, timeout: int = 8):
        calls.append((query, limit))
        url = _resolve_search_url("/url?q=https://example.com/path%3Fa%3D1&sa=U&ved=example")
        return {
            "results": [
                {
                    "title": "Example title",
                    "url": url,
                    "snippet": "Example snippet",
                }
            ]
        }

    def fake_scrape(
        url: str,
        user_request: str,
        *,
        llm_instruction=None,
        llm_provider: str = "",
    ) -> dict[str, str]:
        scraped.append((url, user_request, llm_provider))
        return {"status": "ok", "extracted_content": f"content for {url}"}

    monkeypatch.setattr("tools.builtin.search_web", fake_search)
    monkeypatch.setattr("tools.builtin._scrape_single_url", fake_scrape)

    output = business_action("widgets", search_limit=3, llm_provider="provider-x")

    assert calls == [("widgets", 3)]
    assert scraped == [("https://example.com/path?a=1", "widgets", "provider-x")]

    assert output["status"] == "ok"
    assert output["results"] == [
        {
            "title": "Example title",
            "url": "https://example.com/path?a=1",
            "snippet": "Example snippet",
            "scrape": {"status": "ok", "extracted_content": "content for https://example.com/path?a=1"},
        }
    ]
