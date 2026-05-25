from __future__ import annotations

from typing import Any

from tools.technical_tools import compute_indicators
from tools.yfinance_tools import fetch_historical_ohlcv


class TechnicalAgent:
    name = "Technical Agent"

    def analyze(self, ticker: str) -> dict[str, Any]:
        ohlcv = fetch_historical_ohlcv(ticker)
        return compute_indicators(ohlcv)
