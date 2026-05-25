from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf


class StockDataAgent:
    name = "Stock Data Agent"

    @staticmethod
    def _normalize_ticker(ticker: str, exchange: str = "NSE") -> str:
        cleaned = ticker.strip().upper()
        if not cleaned:
            raise ValueError("Ticker cannot be empty")
        if cleaned.endswith((".NS", ".BO")):
            return cleaned
        suffix = ".BO" if exchange.strip().upper().startswith("B") else ".NS"
        return f"{cleaned}{suffix}"

    @staticmethod
    def _pct_change(close: pd.Series, lookback_days: int) -> float | None:
        series = close.dropna()
        if len(series) <= lookback_days:
            return None
        previous = float(series.iloc[-lookback_days - 1])
        current = float(series.iloc[-1])
        if previous == 0:
            return None
        return round(((current - previous) / previous) * 100, 2)

    def analyze(self, ticker: str, exchange: str = "NSE") -> dict[str, Any]:
        normalized_ticker = self._normalize_ticker(ticker, exchange=exchange)
        stock = yf.Ticker(normalized_ticker)
        info = stock.info or {}

        raw_df = yf.download(
            tickers=normalized_ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        hourly_df = yf.download(
            tickers=normalized_ticker,
            period="3mo",
            interval="1h",
            auto_adjust=False,
            progress=False,
        )

        if raw_df.empty:
            raise ValueError(f"No daily OHLCV data found for {normalized_ticker}")

        raw_df = raw_df.dropna(how="all")
        hourly_df = hourly_df.dropna(how="all") if not hourly_df.empty else hourly_df

        close = raw_df["Close"].dropna()
        high = raw_df["High"].dropna()
        low = raw_df["Low"].dropna()

        current_price = float(close.iloc[-1])
        week_52_high = float(high.tail(252).max()) if len(high) >= 1 else current_price
        week_52_low = float(low.tail(252).min()) if len(low) >= 1 else current_price

        price_changes = {
            "1W": self._pct_change(close, 5),
            "1M": self._pct_change(close, 21),
            "3M": self._pct_change(close, 63),
            "6M": self._pct_change(close, 126),
            "1Y": self._pct_change(close, 252),
            "3Y": self._pct_change(close, 756),
        }

        avg_volume = info.get("averageVolume")
        if avg_volume is None and "Volume" in raw_df:
            avg_volume = float(raw_df["Volume"].tail(20).mean())

        near_52_high = current_price >= (week_52_high * 0.95) if week_52_high else False
        near_52_low = current_price <= (week_52_low * 1.05) if week_52_low else False

        return {
            "ticker": normalized_ticker,
            "company_name": info.get("longName") or info.get("shortName") or normalized_ticker,
            "sector": info.get("sector"),
            "current_price": current_price,
            "week_52_high": week_52_high,
            "week_52_low": week_52_low,
            "market_cap": info.get("marketCap"),
            "beta": info.get("beta"),
            "avg_volume": avg_volume,
            "price_changes": price_changes,
            "raw_df": raw_df,
            "hourly_df": hourly_df,
            "near_52_high": near_52_high,
            "near_52_low": near_52_low,
        }
