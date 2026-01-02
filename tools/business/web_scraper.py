"""LLM-driven web scraping business tool built on the stateful crawler."""
from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from openai import OpenAI

from tools.base import Tool
from tools.registry import call_registered_tool, get_registered_tool, register_tool
from velvetflow.openai_utils import safe_chat_completion


def run_llm_web_scraper(
    urls: List[str],
    goal: str,
    *,
    max_pages: int = 40,
    same_domain_only: bool = True,
    concurrency: int = 2,
    politeness_delay: float = 0.8,
    llm_model: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None,
    page_md_chars: int = 9000,
    goal_check_interval: int = 10,
    min_enqueue_score: int = 55,
    timeout_ms: int = 20000,
    state_path: Optional[str] = "summary_state.json",
) -> Dict[str, Any]:
    """Run the LLM-guided crawler starting from a list of URLs and return the answer."""

    from tools.web_scraper import crawl_and_answer

    cleaned_urls = [u.strip() for u in urls if u.strip()]
    if not cleaned_urls:
        raise ValueError("`urls` must contain at least one non-empty URL")

    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API key. Provide `openai_api_key` or set env OPENAI_API_KEY.")

    result = asyncio.run(
        crawl_and_answer(
            seed_urls=cleaned_urls,
            goal=goal,
            max_pages=max_pages,
            concurrency=concurrency,
            same_domain=same_domain_only,
            politeness_delay=politeness_delay,
            page_md_chars=page_md_chars,
            goal_check_interval=goal_check_interval,
            min_enqueue_score=min_enqueue_score,
            llm_model=llm_model,
            api_key=api_key,
            timeout_ms=timeout_ms,
            state_path=state_path,
        )
    )

    return {
        "satisfied": result["satisfied"],
        "answer": result["answer"],
        "pages_crawled": result["pages_crawled"],
        "state_path": result["state_path"],
        "pages_path": result["pages_path"],
    }


def search_and_crawl_with_goal(
    goal: Any,
    *,
    search_limit: int = 5,
    max_rewrite_attempts: int = 3,
    max_pages: int = 40,
    same_domain_only: bool = True,
    concurrency: int = 2,
    politeness_delay: float = 0.8,
    llm_model: str = "gpt-4o-mini",
    openai_api_key: Optional[str] = None,
    page_md_chars: int = 9000,
    goal_check_interval: int = 10,
    min_enqueue_score: int = 55,
    timeout_ms: int = 20000,
) -> Dict[str, Any]:
    """Search the web for a goal, then crawl results with the LLM scraper."""

    cleaned_goal = _normalize_goal(goal)
    if not cleaned_goal:
        raise ValueError("goal must be a non-empty string")

    search_tool = get_registered_tool("search_web")
    if search_tool:
        search_fn = search_tool
    else:  # pragma: no cover - fallback when registry is not initialized
        from tools.builtin import search_web as builtin_search_web

        search_fn = builtin_search_web

    query = cleaned_goal
    search_results: Optional[List[Any]] = None
    for attempt in range(max_rewrite_attempts + 1):
        search_output = search_fn(query=query, limit=search_limit)
        search_results = search_output.get("results") if isinstance(search_output, dict) else None
        if not isinstance(search_results, list):
            raise RuntimeError("web search did not return a valid results list")
        if search_results:
            break
        if attempt >= max_rewrite_attempts:
            break
        query = _rewrite_search_query(
            query=query,
            goal=cleaned_goal,
            llm_model=llm_model,
            openai_api_key=openai_api_key,
        )

    seed_urls: List[str] = []
    seen: set[str] = set()
    for item in search_results:
        url_value = "" if not isinstance(item, dict) else str(item.get("url", "")).strip()
        if url_value and url_value not in seen:
            seed_urls.append(url_value)
            seen.add(url_value)

    if not seed_urls:
        raise RuntimeError("web search did not yield any URLs to crawl")

    crawl_result = call_registered_tool(
        "web_scraper.llm_crawl",
        urls=seed_urls,
        goal=cleaned_goal,
        max_pages=max_pages,
        same_domain_only=same_domain_only,
        concurrency=concurrency,
        politeness_delay=politeness_delay,
        llm_model=llm_model,
        openai_api_key=openai_api_key,
        page_md_chars=page_md_chars,
        goal_check_interval=goal_check_interval,
        min_enqueue_score=min_enqueue_score,
        timeout_ms=timeout_ms,
        state_path="",
    )

    return {
        "query": query,
        "seed_urls": seed_urls,
        "search_results": search_results,
        "crawl_result": crawl_result,
    }


def _normalize_goal(goal: Any) -> str:
    if isinstance(goal, str):
        cleaned_goal = goal.strip()
    elif isinstance(goal, (dict, list)):
        cleaned_goal = json.dumps(goal, ensure_ascii=False)
    else:
        cleaned_goal = str(goal).strip()
    return cleaned_goal


def _rewrite_search_query(
    *,
    query: str,
    goal: str,
    llm_model: str,
    openai_api_key: Optional[str],
) -> str:
    api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenAI API key. Provide `openai_api_key` or set env OPENAI_API_KEY.")
    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are a search query rewriter.\n"
        "Given the user goal and current query, produce a rewritten query that uses alternative phrasing, synonyms, "
        "or narrower terms to improve search results.\n"
        "Return JSON only: {\"rewritten_query\": \"...\"}."
    )
    response = safe_chat_completion(
        client,
        model=llm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": json.dumps(
                    {"goal": goal, "current_query": query, "language": "zh-CN"},
                    ensure_ascii=False,
                ),
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {}
    rewritten = str(parsed.get("rewritten_query", "")).strip()
    if not rewritten:
        return query
    return rewritten


def register_web_scraper_tools() -> None:
    register_tool(
        Tool(
            name="web_scraper.llm_crawl",
            description=(
                "LLM-driven web scraper that starts from one or more URLs, explores in-domain pages, "
                "and returns an answer to the user's request."
            ),
            function=run_llm_web_scraper,
            args_schema={
                "type": "object",
                "properties": {
                    "urls": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                    },
                    "goal": {"type": "string"},
                    "max_pages": {"type": "integer", "default": 40},
                    "same_domain_only": {"type": "boolean", "default": True},
                    "concurrency": {"type": "integer", "default": 2},
                    "politeness_delay": {"type": "number", "default": 0.8},
                    "llm_model": {"type": "string", "default": "gpt-4o-mini"},
                    "openai_api_key": {"type": "string"},
                    "page_md_chars": {"type": "integer", "default": 9000},
                    "goal_check_interval": {"type": "integer", "default": 10},
                    "min_enqueue_score": {"type": "integer", "default": 55},
                    "timeout_ms": {"type": "integer", "default": 20000},
                    "state_path": {"type": "string", "default": "summary_state.json"},
                },
                "required": ["urls", "goal"],
            },
        )
    )

    register_tool(
        Tool(
            name="web_scraper.search_and_crawl",
            description=(
                "根据自然语言需求先进行网页搜索，再把搜索得到的链接作为种子调用 LLM 爬虫返回最终答案。"
            ),
            function=search_and_crawl_with_goal,
            args_schema={
                "type": "object",
                "properties": {
                    "goal": {"type": "string"},
                    "search_limit": {"type": "integer", "default": 5},
                    "max_rewrite_attempts": {"type": "integer", "default": 3},
                    "max_pages": {"type": "integer", "default": 40},
                    "same_domain_only": {"type": "boolean", "default": True},
                    "concurrency": {"type": "integer", "default": 2},
                    "politeness_delay": {"type": "number", "default": 0.8},
                    "llm_model": {"type": "string", "default": "gpt-4o-mini"},
                    "openai_api_key": {"type": "string"},
                    "page_md_chars": {"type": "integer", "default": 9000},
                    "goal_check_interval": {"type": "integer", "default": 10},
                    "min_enqueue_score": {"type": "integer", "default": 55},
                    "timeout_ms": {"type": "integer", "default": 20000},
                },
                "required": ["goal"],
            },
        )
    )


__all__ = [
    "run_llm_web_scraper",
    "search_and_crawl_with_goal",
    "register_web_scraper_tools",
]
