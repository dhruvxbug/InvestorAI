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
    def _to_scalar(val: object) -> float:
        """Safely extract a Python float from a scalar, Series, or 0-d array."""
        import numpy as np  # noqa: PLC0415

        if isinstance(val, pd.Series):
            return float(val.iloc[0])
        if isinstance(val, np.generic):
            return float(val)
        return float(val)  # type: ignore[arg-type]

    @classmethod
    def _pct_change(cls, close: pd.Series, lookback_days: int) -> float | None:
        series = close.squeeze().dropna()  # ensure 1-D
        if len(series) <= lookback_days:
            return None
        previous = cls._to_scalar(series.iloc[-lookback_days - 1])
        current = cls._to_scalar(series.iloc[-1])
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

        # Flatten MultiIndex columns that yfinance returns for single-ticker downloads
        if isinstance(raw_df.columns, pd.MultiIndex):
            raw_df.columns = raw_df.columns.get_level_values(0)
        if not hourly_df.empty and isinstance(hourly_df.columns, pd.MultiIndex):
            hourly_df.columns = hourly_df.columns.get_level_values(0)

        raw_df = raw_df.dropna(how="all")
        hourly_df = hourly_df.dropna(how="all") if not hourly_df.empty else hourly_df

        close = raw_df["Close"].squeeze().dropna()
        high = raw_df["High"].squeeze().dropna()
        low = raw_df["Low"].squeeze().dropna()

        current_price = self._to_scalar(close.iloc[-1])
        week_52_high = (
            self._to_scalar(high.tail(252).max()) if len(high) >= 1 else current_price
        )
        week_52_low = (
            self._to_scalar(low.tail(252).min()) if len(low) >= 1 else current_price
        )

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
            avg_volume = self._to_scalar(raw_df["Volume"].squeeze().tail(20).mean())

        near_52_high = current_price >= (week_52_high * 0.95) if week_52_high else False
        near_52_low = current_price <= (week_52_low * 1.05) if week_52_low else False

        return {
            "ticker": normalized_ticker,
            "company_name": info.get("longName")
            or info.get("shortName")
            or normalized_ticker,
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
