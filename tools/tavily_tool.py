from __future__ import annotations

import os
from typing import Any

import requests


class TavilySearchError(RuntimeError):
    pass


def search_web(query: str, max_results: int = 5) -> list[dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise TavilySearchError("TAVILY_API_KEY is not configured in the environment or .env file.")
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "max_results": min(max(max_results, 1), 10),
                "search_depth": "advanced",
                "include_answer": False,
            },
            timeout=25,
        )
    except requests.RequestException as exc:
        raise TavilySearchError(f"Tavily search request failed: {exc}") from exc
    if not response.ok:
        raise TavilySearchError(f"Tavily search failed with HTTP status {response.status_code}.")
    try:
        payload = response.json()
    except ValueError as exc:
        raise TavilySearchError("Tavily returned an unreadable response.") from exc
    results = payload.get("results", [])
    return results if isinstance(results, list) else []