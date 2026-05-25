from __future__ import annotations

from typing import Any

from tools.exa_tools import ExaSearchClient


class ManagementAgent:
    name = "Management Agent"

    def __init__(self) -> None:
        self.exa = ExaSearchClient()

    def analyze(self, ticker: str, company_name: str) -> dict[str, Any]:
        results = self.exa.search_management_signals(company_name=company_name, ticker=ticker)
        return {
            "management_signals": results,
            "signal_count": len(results),
        }
