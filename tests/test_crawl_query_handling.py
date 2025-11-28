from pathlib import Path
import sys

import json
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

        def run(self, *, url: str, query: str, config=None):
            calls["url"] = url
            calls["query"] = query
            calls["config_passed"] = config is not None
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
    assert calls["config_passed"] is False
    assert result["answer"] == "query matched"


def test_crawl_uses_llm_extraction_when_available(monkeypatch):
    calls = {}

    class DummyRunConfig:
        def __init__(self, **kwargs):
            calls["config_kwargs"] = kwargs

    class DummyLLMExtractionStrategy:
        def __init__(self, **kwargs):
            calls["strategy_kwargs"] = kwargs

    class DummyLLMConfig:
        def __init__(self, **kwargs):
            calls["llm_config_kwargs"] = kwargs

    class DummyCacheMode:
        BYPASS = "bypass"

    module = types.SimpleNamespace(
        CrawlerRunConfig=DummyRunConfig,
        LLMExtractionStrategy=DummyLLMExtractionStrategy,
        LLMConfig=DummyLLMConfig,
        CacheMode=DummyCacheMode,
    )

    class DummyResult:
        extracted_content = json.dumps(
            {
                "answer": "structured answer",
                "summary": "structured summary",
                "evidence": ["snippet"],
            }
        )
        markdown = "Body"

    class DummyCrawler:
        def __init__(self, *, verbose: bool = False):
            self.verbose = verbose

        def run(self, *, url: str, query: str, config=None):
            calls["url"] = url
            calls["query"] = query
            calls["config"] = config
            return DummyResult()

    monkeypatch.setattr(builtin, "_load_crawl4ai_module", lambda: module)

    def _resolver(name: str):
        if name == "WebCrawler":
            return DummyCrawler
        return None

    monkeypatch.setattr(builtin, "_resolve_crawl4ai_class", _resolver)

    result = builtin.crawl_and_summarize("example.com", "What is inside?")

    assert isinstance(calls["config"], DummyRunConfig)
    assert "extraction_strategy" in calls["config_kwargs"]
    assert calls["config_kwargs"].get("cache_mode") == DummyCacheMode.BYPASS
    assert calls["url"] == "https://example.com"
    assert calls["query"] == "What is inside?"
    assert result["answer"] == "structured answer"
    assert result["summary"] == "structured summary"
    assert result["evidence"] == ["snippet"]
