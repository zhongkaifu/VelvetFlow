from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

import tools.builtin as builtin


def test_business_action_runs_search_and_scrape(monkeypatch):
    searches = []
    scrapes = []

    def fake_search(query, limit=5, timeout=8):
        searches.append({"query": query, "limit": limit, "timeout": timeout})
        return {
            "results": [
                {
                    "title": "Example",
                    "url": "https://example.com/start",
                    "snippet": "sample",
                }
            ]
        }

    def fake_scrape(url, user_request, **kwargs):
        scrapes.append({"url": url, "user_request": user_request, **kwargs})
        return {"status": "ok", "extracted_content": f"content for {url}"}

    monkeypatch.setattr(builtin, "search_web", fake_search)
    monkeypatch.setattr(builtin, "_scrape_single_url", fake_scrape)

    result = builtin.business_action("find info", search_limit=3, llm_provider="test-model")

    assert result["status"] == "ok"
    assert searches[0]["query"] == "find info"
    assert scrapes[0]["url"] == "https://example.com/start"
    assert "Aggregate summary" in result["summary"]


def test_business_action_handles_no_results(monkeypatch):
    def fake_search(query, limit=5, timeout=8):
        return {"results": []}

    monkeypatch.setattr(builtin, "search_web", fake_search)
    result = builtin.business_action("missing topic")

    assert result["status"] == "not_found"
    assert result["results"] == []
