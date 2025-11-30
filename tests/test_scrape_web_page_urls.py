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
            "url": url,
            "analysis": f"analysis for {url}",
            "markdown": "",
            "attempts": [],
            "llm_used": False,
            "llm_provider": None,
            "user_request": "test",
        }

    monkeypatch.setattr(builtin, "_scrape_single_url", fake_scrape)

    result = builtin.scrape_web_page(urls, "summarize")

    assert captured == urls
    assert result["status"] == "ok"
    assert result["urls"] == urls
    assert len(result["results"]) == 2
    assert "Aggregate summary" in result["aggregate_summary"]


def test_scrape_web_page_partial_status(monkeypatch):
    calls = []

    def fake_scrape(url, *args, **kwargs):
        calls.append(url)
        return {
            "status": "ok" if "good" in url else "error",
            "url": url,
            "analysis": f"analysis for {url}",
            "markdown": "",
            "attempts": [],
            "llm_used": False,
            "llm_provider": None,
            "user_request": "test",
        }

    monkeypatch.setattr(builtin, "_scrape_single_url", fake_scrape)

    result = builtin.scrape_web_page(["http://good", "http://bad"], "check")

    assert calls == ["http://good", "http://bad"]
    assert result["status"] == "partial"
    assert result["results"][1]["status"] == "error"


def test_scrape_web_page_validates_inputs():
    with pytest.raises(ValueError):
        builtin.scrape_web_page([], "query")

    with pytest.raises(ValueError):
        builtin.scrape_web_page([""], "query")

