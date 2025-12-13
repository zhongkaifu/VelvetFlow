"""LLM-driven web scraping business tool built on the stateful crawler."""
from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from tools.base import Tool
from tools.registry import register_tool
from tools.web_scraper import crawl_and_answer


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
) -> Dict[str, Any]:
    """Run the LLM-guided crawler starting from a list of URLs and return the answer."""

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
        )
    )

    return {
        "satisfied": result["satisfied"],
        "answer": result["answer"],
        "pages_crawled": result["pages_crawled"],
        "state_path": result["state_path"],
        "pages_path": result["pages_path"],
    }


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
                },
                "required": ["urls", "goal"],
            },
        )
    )


__all__ = [
    "run_llm_web_scraper",
    "register_web_scraper_tools",
]
