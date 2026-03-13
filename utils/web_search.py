"""
utils/web_search.py
Real-time web search via Serper.dev (Google Search API).
Falls back gracefully if the key is missing.
"""

import logging
import requests
from config.config import SERPER_API_KEY

logger = logging.getLogger(__name__)

SERPER_URL = "https://google.serper.dev/search"


def web_search(query: str, num_results: int = 5) -> list[dict]:
    """
    Returns a list of dicts: [{"title": ..., "snippet": ..., "link": ...}, ...]
    Returns [] if key is missing or request fails.
    """
    if not SERPER_API_KEY:
        logger.warning("SERPER_API_KEY not set — web search skipped.")
        return []
    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=10,
        )
        resp.raise_for_status()
        data   = resp.json()
        organic = data.get("organic", [])
        return [
            {
                "title":   r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "link":    r.get("link", ""),
            }
            for r in organic[:num_results]
        ]
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []


def format_search_results(results: list[dict]) -> str:
    """Convert result list to a context block for the LLM."""
    if not results:
        return ""
    lines = ["Web search results:\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['snippet']}\n   Source: {r['link']}\n")
    return "\n".join(lines)


def should_search(query: str, history: list[dict]) -> bool:
    """
    Simple heuristic: search when the query looks like it needs fresh/external data.
    You could swap this for an LLM-based router later.
    """
    triggers = [
        "latest", "recent", "news", "today", "current", "2024", "2025",
        "price", "update", "law", "regulation", "case", "verdict",
    ]
    q = query.lower()
    return any(t in q for t in triggers)
