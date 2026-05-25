from __future__ import annotations

from typing import Any

from tools.yfinance_tools import extract_fundamentals, fetch_ticker_info


class FundamentalAgent:
    name = "Fundamental Agent"

    def analyze(self, ticker: str) -> dict[str, Any]:
        info = fetch_ticker_info(ticker)
        fundamentals = extract_fundamentals(info)
        fundamentals["sector"] = info.get("sector")
        fundamentals["industry"] = info.get("industry")
        fundamentals["long_business_summary"] = (info.get("longBusinessSummary") or "")[:1000]
        return fundamentals
