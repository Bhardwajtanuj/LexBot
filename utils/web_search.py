# utils/web_search.py
# serper.dev gives 2500 free searches/month which is fine for a demo
# if the key isn't set, search just silently returns nothing

import logging
import requests
from config.config import get_serper_key

logger = logging.getLogger(__name__)

SERPER_URL = "https://google.serper.dev/search"


def web_search(query: str, num_results: int = 5) -> list[dict]:
    # returns list of {title, snippet, link} or [] on failure
    key = get_serper_key()
    if not key:
        logger.warning("SERPER_API_KEY not set, skipping web search")
        return []
    try:
        resp = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
            json={"q": query, "num": num_results},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
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
        logger.error(f"web search failed: {e}")
        return []


def format_search_results(results: list[dict]) -> str:
    if not results:
        return ""
    lines = ["Web search results:\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"{i}. {r['title']}\n   {r['snippet']}\n   Source: {r['link']}\n")
    return "\n".join(lines)


def should_search(query: str, history: list[dict]) -> bool:
    # crude keyword check — good enough, could swap for an llm router later
    triggers = [
        "latest", "recent", "news", "today", "current", "2024", "2025",
        "price", "update", "law", "regulation", "case", "verdict",
    ]
    q = query.lower()
    return any(t in q for t in triggers)
