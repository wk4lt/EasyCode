"""Web search skill implementation using DuckDuckGo Instant Answer API.

Real HTTP-based search. No API key required — uses DuckDuckGo's public endpoint.

Layer: Skill layer (first layer).
"""

import json
from typing import Optional

import requests

from liteagent.core.base_skill import BaseSkill


class WebSearchImpl(BaseSkill):
    """Search the web using DuckDuckGo Instant Answer API."""

    API_URL = "https://api.duckduckgo.com/"

    def execute(self, query: str, max_results: int = 5) -> dict:
        """Execute a web search.

        Args:
            query: Search query string.
            max_results: Maximum number of results (1-10).

        Returns:
            dict with 'status' and 'results' or 'error'.
        """
        try:
            max_results = min(max(max_results, 1), 10)

            params = {
                "q": query,
                "format": "json",
                "no_html": 1,
                "skip_disambig": 1,
            }

            response = requests.get(self.API_URL, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = []

            heading = data.get("Heading", "")
            abstract = data.get("AbstractText", "")
            abstract_url = data.get("AbstractURL", "")

            if abstract:
                results.append({
                    "title": heading or query,
                    "snippet": abstract,
                    "url": abstract_url,
                    "source": "DuckDuckGo Abstract",
                })

            answer = data.get("Answer", "")
            answer_type = data.get("AnswerType", "")
            if answer and answer_type != "calc":
                results.append({
                    "title": f"Instant Answer: {heading or query}",
                    "snippet": answer,
                    "url": abstract_url or "",
                    "source": "DuckDuckGo Instant Answer",
                })

            related_topics = data.get("RelatedTopics", [])
            for topic in related_topics[:max_results - len(results)]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("FirstURL", "").rsplit("/", 1)[-1].replace("_", " ").title(),
                        "snippet": topic.get("Text", ""),
                        "url": topic.get("FirstURL", ""),
                        "source": "DuckDuckGo Related",
                    })
                elif isinstance(topic, str):
                    results.append({"title": "", "snippet": topic, "url": "", "source": "DuckDuckGo"})

            if not results:
                results = [{"title": "No results", "snippet": f"No information found for '{query}'.", "url": "", "source": ""}]

            return {
                "status": "ok",
                "query": query,
                "results_count": len(results),
                "results": results[:max_results],
            }

        except requests.RequestException as e:
            return {"status": "error", "error": f"Web search request failed: {e}"}
        except Exception as e:
            return {"status": "error", "error": f"Web search failed: {e}"}
