from pathlib import Path
import sys

import types

sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools import builtin


def test_crawl_passes_query_to_crawl4ai(monkeypatch):
    calls = {}

    class DummyResult:
        answer = "query matched"
        markdown = "Dummy content"

    class DummyCrawler:
        def __init__(self, *, verbose: bool = False):
            self.verbose = verbose

        def run(self, *, url: str, query: str):
            calls["url"] = url
            calls["query"] = query
            return DummyResult()

    monkeypatch.setattr(builtin, "_load_crawl4ai_module", lambda: types.SimpleNamespace())

    def _resolver(name: str):
        if name == "WebCrawler":
            return DummyCrawler
        return None

    monkeypatch.setattr(builtin, "_resolve_crawl4ai_class", _resolver)

    result = builtin.crawl_and_summarize("example.com", "test query")

    assert calls["url"] == "https://example.com"
    assert calls["query"] == "test query"
    assert result["answer"] == "query matched"
