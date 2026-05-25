from __future__ import annotations

from typing import Any

from tools.exa_tools import ExaSearchClient


class SentimentAgent:
    name = "Sentiment Agent"

    def __init__(self) -> None:
        self.exa = ExaSearchClient()

    def analyze(self, ticker: str, company_name: str) -> dict[str, Any]:
        news = self.exa.search_news(company_name=company_name, ticker=ticker)
        return {
            "articles": news,
            "article_count": len(news),
        }
