from __future__ import annotations

from typing import Any

import pandas as pd
import ta as ta_lib

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

    rsi_series = ta_lib.momentum.RSIIndicator(close=close, window=14).rsi()

    _macd = ta_lib.trend.MACD(
        close=close, window_slow=26, window_fast=12, window_sign=9
    )
    macd_line = _macd.macd()
    macd_signal = _macd.macd_signal()
    macd_hist = _macd.macd_diff()

    _bb = ta_lib.volatility.BollingerBands(close=close, window=20, window_dev=2)
    bb_lower = _bb.bollinger_lband()
    bb_middle = _bb.bollinger_mavg()
    bb_upper = _bb.bollinger_hband()

    sma20 = ta_lib.trend.SMAIndicator(close=close, window=20).sma_indicator()
    sma50 = ta_lib.trend.SMAIndicator(close=close, window=50).sma_indicator()
    ema20 = ta_lib.trend.EMAIndicator(close=close, window=20).ema_indicator()

    def _last(s: pd.Series) -> float | None:
        valid = s.dropna()
        return float(valid.iloc[-1]) if not valid.empty else None

    latest: dict[str, Any] = {
        "close": float(close.iloc[-1]),
        "rsi_14": _last(rsi_series),
        "macd": _last(macd_line),
        "macd_signal": _last(macd_signal),
        "macd_hist": _last(macd_hist),
        "bb_lower": _last(bb_lower),
        "bb_middle": _last(bb_middle),
        "bb_upper": _last(bb_upper),
        "sma_20": _last(sma20),
        "sma_50": _last(sma50),
        "ema_20": _last(ema20),
    }

    latest["trend"] = (
        "BULLISH"
        if latest["sma_20"] and latest["sma_50"] and latest["sma_20"] > latest["sma_50"]
        else "BEARISH"
    )
    return latest
