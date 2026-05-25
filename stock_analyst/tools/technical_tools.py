from __future__ import annotations

from typing import Any

import pandas as pd
import pandas_ta as ta


REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(c).title() for c in cleaned.columns]
    missing = REQUIRED_COLUMNS.difference(cleaned.columns)
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    return cleaned


def compute_indicators(df: pd.DataFrame) -> dict[str, Any]:
    data = _ensure_ohlcv(df)
    close = data["Close"]

    rsi_series = ta.rsi(close, length=14)
    macd = ta.macd(close, fast=12, slow=26, signal=9)
    bbands = ta.bbands(close, length=20, std=2.0)

    sma20 = ta.sma(close, length=20)
    sma50 = ta.sma(close, length=50)
    ema20 = ta.ema(close, length=20)

    latest = {
        "close": float(close.iloc[-1]),
        "rsi_14": float(rsi_series.iloc[-1]) if rsi_series is not None else None,
        "macd": float(macd["MACD_12_26_9"].iloc[-1]) if macd is not None else None,
        "macd_signal": float(macd["MACDs_12_26_9"].iloc[-1]) if macd is not None else None,
        "macd_hist": float(macd["MACDh_12_26_9"].iloc[-1]) if macd is not None else None,
        "bb_lower": float(bbands["BBL_20_2.0"].iloc[-1]) if bbands is not None else None,
        "bb_middle": float(bbands["BBM_20_2.0"].iloc[-1]) if bbands is not None else None,
        "bb_upper": float(bbands["BBU_20_2.0"].iloc[-1]) if bbands is not None else None,
        "sma_20": float(sma20.iloc[-1]) if sma20 is not None else None,
        "sma_50": float(sma50.iloc[-1]) if sma50 is not None else None,
        "ema_20": float(ema20.iloc[-1]) if ema20 is not None else None,
    }

    trend = "BULLISH" if latest["sma_20"] and latest["sma_50"] and latest["sma_20"] > latest["sma_50"] else "BEARISH"
    latest["trend"] = trend
    return latest
