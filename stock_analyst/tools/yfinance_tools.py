from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import yfinance as yf

from config.settings import HISTORICAL_INTERVAL, HISTORICAL_PERIOD


@dataclass
class StockSnapshot:
    ticker: str
    current_price: float | None
    previous_close: float | None
    day_high: float | None
    day_low: float | None
    volume: int | None
    market_cap: int | None


def fetch_ticker_info(ticker: str) -> dict[str, Any]:
    stock = yf.Ticker(ticker)
    return stock.info or {}


def fetch_historical_ohlcv(ticker: str, period: str = HISTORICAL_PERIOD, interval: str = HISTORICAL_INTERVAL) -> pd.DataFrame:
    data = yf.download(
        tickers=ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )
    if data.empty:
        raise ValueError(f"No historical data found for {ticker}")
    return data


def build_stock_snapshot(ticker: str, info: dict[str, Any]) -> StockSnapshot:
    return StockSnapshot(
        ticker=ticker,
        current_price=info.get("currentPrice") or info.get("regularMarketPrice"),
        previous_close=info.get("previousClose"),
        day_high=info.get("dayHigh"),
        day_low=info.get("dayLow"),
        volume=info.get("volume"),
        market_cap=info.get("marketCap"),
    )


def extract_fundamentals(info: dict[str, Any]) -> dict[str, Any]:
    fields = [
        "trailingPE",
        "forwardPE",
        "trailingEps",
        "returnOnEquity",
        "debtToEquity",
        "profitMargins",
        "revenueGrowth",
        "earningsGrowth",
        "bookValue",
        "priceToBook",
        "dividendYield",
    ]
    return {field: info.get(field) for field in fields}
