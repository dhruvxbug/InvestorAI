from __future__ import annotations

from typing import Any

from tools.yfinance_tools import build_stock_snapshot, fetch_historical_ohlcv, fetch_ticker_info


class StockDataAgent:
    name = "Stock Data Agent"

    def analyze(self, ticker: str) -> dict[str, Any]:
        info = fetch_ticker_info(ticker)
        snapshot = build_stock_snapshot(ticker, info)
        ohlcv = fetch_historical_ohlcv(ticker)
        return {
            "ticker": ticker,
            "company_name": info.get("longName") or info.get("shortName") or ticker,
            "exchange": info.get("exchange") or "NSE/BSE",
            "snapshot": snapshot.__dict__,
            "recent_close": float(ohlcv["Close"].iloc[-1]),
            "avg_volume_20d": float(ohlcv["Volume"].tail(20).mean()),
            "52w_high": float(ohlcv["High"].max()),
            "52w_low": float(ohlcv["Low"].min()),
        }
