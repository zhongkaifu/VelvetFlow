# Author: Zhongkai Fu (fuzhongkai@gmail.com)
# License: BSD 3-Clause License

from pathlib import Path
import sys
import types

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

fake_scrapegraphai = types.ModuleType("scrapegraphai")
fake_graphs = types.ModuleType("scrapegraphai.graphs")


class _Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def run(self, *args, **kwargs):
        return {}


fake_graphs.SmartScraperGraph = _Dummy

sys.modules.setdefault("scrapegraphai", fake_scrapegraphai)
sys.modules.setdefault("scrapegraphai.graphs", fake_graphs)

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


def test_scrape_single_url_stays_on_domain(monkeypatch):
    visits = []

    def fake_crawl(url, *_args, **_kwargs):
        visits.append(url)
        if url.endswith("/start"):
            return {
                "status": "ok",
                "extracted_content": "root content",
                "links": ["/internal", "https://other-site.test/offsite"],
            }
        return {
            "status": "ok",
            "extracted_content": "internal content",
            "links": [],
        }

    def fake_decider(_req, pages, remaining_links, *_args, **_kwargs):
        # stop after the internal page is crawled
        should_stop = any(page.get("url", "").endswith("/internal") for page in pages)
        # never follow offsite links because they are filtered out
        assert all("other-site" not in link for link in remaining_links)
        return should_stop, remaining_links

    monkeypatch.setattr(builtin, "_crawl_page_once", fake_crawl)
    monkeypatch.setattr(builtin, "_should_continue_crawling", fake_decider)

    result = builtin._scrape_single_url(
        "https://example.com/start",
        "find info",
        run_coroutine=lambda factory: factory(),
    )

    assert visits == ["https://example.com/start", "https://example.com/internal"]
    assert result["status"] == "ok"
    assert "internal content" in result["extracted_content"]

