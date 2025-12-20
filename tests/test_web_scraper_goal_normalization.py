import json

from tools.base import Tool
from tools.business.web_scraper import search_and_crawl_with_goal
from tools.registry import GLOBAL_TOOL_REGISTRY, register_tool


def test_search_and_crawl_with_goal_accepts_dict_goal() -> None:
    original_tools = dict(GLOBAL_TOOL_REGISTRY._tools)
    GLOBAL_TOOL_REGISTRY._tools = {}

    try:
        register_tool(
            Tool(
                name="search_web",
                description="test search",
                function=lambda query, limit=5: {"results": [{"url": "https://example.com"}]},
            )
        )
        register_tool(
            Tool(
                name="web_scraper.llm_crawl",
                description="test crawl",
                function=lambda urls, goal, **kwargs: {"urls": urls, "goal": goal},
            )
        )

        goal_payload = {"city": "Kyoto", "budget": "mid-range"}
        expected_goal = json.dumps(goal_payload, ensure_ascii=False)

        result = search_and_crawl_with_goal(goal_payload, search_limit=1, max_pages=1)

        assert result["query"] == expected_goal
        assert result["crawl_result"]["goal"] == expected_goal
    finally:
        GLOBAL_TOOL_REGISTRY._tools = original_tools
