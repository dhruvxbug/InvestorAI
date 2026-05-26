from __future__ import annotations

from typing import Any

from exa_py import Exa

from config.settings import EXA_API_KEY, MANAGEMENT_RESULT_COUNT, NEWS_RESULT_COUNT


class ExaSearchClient:
    def __init__(self) -> None:
        if not EXA_API_KEY:
            raise ValueError("EXA_API_KEY is not configured")
        self.client = Exa(api_key=EXA_API_KEY)

    def search(self, query: str, num_results: int) -> list[dict[str, Any]]:
        response = self.client.search_and_contents(
            query=query,
            type="neural",
            num_results=num_results,
            text=True,
            highlights=True,
        )
        return [
            {
                "title": result.title,
                "url": result.url,
                "published_date": result.published_date,
                "summary": (result.text or "")[:1200],
                "highlights": result.highlights or [],
            }
            for result in response.results
        ]

    def search_news(self, company_name: str, ticker: str, num_results: int = NEWS_RESULT_COUNT) -> list[dict[str, Any]]:
        query = f"{company_name} ({ticker}) India stock latest news sentiment"
        return self.search(query=query, num_results=num_results)

    def search_management_signals(self, company_name: str, ticker: str, num_results: int = MANAGEMENT_RESULT_COUNT) -> list[dict[str, Any]]:
        query = f"{company_name} ({ticker}) CEO board changes insider trading corporate governance India"
        return self.search(query=query, num_results=num_results)
