from pathlib import Path
import sys
import types

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

fake_crawl4ai = types.ModuleType("crawl4ai")


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass


fake_crawl4ai.AsyncWebCrawler = _Dummy
fake_crawl4ai.BrowserConfig = _Dummy
fake_crawl4ai.CacheMode = types.SimpleNamespace(BYPASS="bypass")
fake_crawl4ai.CrawlerRunConfig = _Dummy
fake_crawl4ai.LLMConfig = _Dummy
fake_crawl4ai.LLMExtractionStrategy = _Dummy

sys.modules.setdefault("crawl4ai", fake_crawl4ai)
fake_extraction_strategy = types.ModuleType("crawl4ai.extraction_strategy")
fake_extraction_strategy.LLMExtractionStrategy = _Dummy
sys.modules.setdefault("crawl4ai.extraction_strategy", fake_extraction_strategy)

from tools import builtin


def test_scrape_web_page_accepts_multiple_urls(monkeypatch):
    urls = ["http://example.com/a", "http://example.com/b"]
    captured = []

    def fake_scrape(url, *args, **kwargs):
        captured.append(url)
        return {
            "status": "ok",
            "extracted_content": f"content for {url}",
        }

    monkeypatch.setattr(builtin, "_scrape_single_url", fake_scrape)

    result = builtin.scrape_web_page(urls, "summarize")

    assert captured == urls
    assert result["status"] == "ok"
    assert result["extracted_content"] == "\n".join(
        ["content for http://example.com/a", "content for http://example.com/b"]
    )


def test_scrape_web_page_partial_status(monkeypatch):
    calls = []

    def fake_scrape(url, *args, **kwargs):
        calls.append(url)
        return {
            "status": "ok" if "good" in url else "error",
            "extracted_content": f"analysis for {url}",
        }

    monkeypatch.setattr(builtin, "_scrape_single_url", fake_scrape)

    result = builtin.scrape_web_page(["http://good", "http://bad"], "check")

    assert calls == ["http://good", "http://bad"]
    assert result["status"] == "partial"
    assert "analysis for http://bad" in result["extracted_content"]


def test_scrape_web_page_validates_inputs():
    with pytest.raises(ValueError):
        builtin.scrape_web_page([], "query")

    with pytest.raises(ValueError):
        builtin.scrape_web_page([""], "query")

