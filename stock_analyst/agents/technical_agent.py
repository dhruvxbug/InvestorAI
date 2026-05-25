from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta


class TechnicalAgent:
    name = "Technical Analysis Agent"

    @staticmethod
    def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data.columns = [str(col).title() for col in data.columns]
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(f"Missing OHLCV columns: {sorted(missing)}")
        return data.dropna(subset=["Close"]).copy()

    @staticmethod
    def _latest(series: pd.Series | None) -> float | None:
        if series is None:
            return None
        valid = series.dropna()
        if valid.empty:
            return None
        return float(valid.iloc[-1])

    @staticmethod
    def _cross_up(a: pd.Series, b: pd.Series) -> bool:
        merged = pd.concat([a, b], axis=1).dropna().tail(2)
        if len(merged) < 2:
            return False
        return bool(merged.iloc[0, 0] <= merged.iloc[0, 1] and merged.iloc[1, 0] > merged.iloc[1, 1])

    @staticmethod
    def _cross_down(a: pd.Series, b: pd.Series) -> bool:
        merged = pd.concat([a, b], axis=1).dropna().tail(2)
        if len(merged) < 2:
            return False
        return bool(merged.iloc[0, 0] >= merged.iloc[0, 1] and merged.iloc[1, 0] < merged.iloc[1, 1])

    def analyze(self, raw_df: pd.DataFrame, week_52_high: float | None = None) -> dict[str, Any]:
        data = self._normalize_ohlcv(raw_df)

        close = data["Close"]
        high = data["High"]
        low = data["Low"]
        volume = data["Volume"]

        rsi_series = ta.rsi(close, length=14)
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        bbands_df = ta.bbands(close, length=20, std=2)

        sma_20_s = ta.sma(close, length=20)
        sma_50_s = ta.sma(close, length=50)
        sma_200_s = ta.sma(close, length=200)
        ema_9_s = ta.ema(close, length=9)
        ema_21_s = ta.ema(close, length=21)

        atr_s = ta.atr(high, low, close, length=14)
        stoch_df = ta.stoch(high, low, close, k=14, d=3)
        adx_df = ta.adx(high, low, close, length=14)

        typical_price = (high + low + close) / 3
        vwap = ((typical_price * volume).cumsum() / volume.replace(0, np.nan).cumsum()).replace([np.inf, -np.inf], np.nan)

        macd_line = macd_df["MACD_12_26_9"] if macd_df is not None else pd.Series(dtype=float)
        macd_signal_line = macd_df["MACDs_12_26_9"] if macd_df is not None else pd.Series(dtype=float)

        bb_upper_s = bbands_df["BBU_20_2.0"] if bbands_df is not None else pd.Series(dtype=float)
        bb_lower_s = bbands_df["BBL_20_2.0"] if bbands_df is not None else pd.Series(dtype=float)

        adx_s = adx_df["ADX_14"] if adx_df is not None else pd.Series(dtype=float)

        current_price = float(close.iloc[-1])
        rsi = self._latest(rsi_series)
        macd = self._latest(macd_line)
        macd_signal = self._latest(macd_signal_line)
        bb_upper = self._latest(bb_upper_s)
        bb_lower = self._latest(bb_lower_s)
        sma_20 = self._latest(sma_20_s)
        sma_50 = self._latest(sma_50_s)
        sma_200 = self._latest(sma_200_s)
        ema_9 = self._latest(ema_9_s)
        ema_21 = self._latest(ema_21_s)
        atr = self._latest(atr_s)
        adx = self._latest(adx_s)

        golden_cross = self._cross_up(sma_50_s, sma_200_s)
        death_cross = self._cross_down(sma_50_s, sma_200_s)
        macd_bullish = self._cross_up(macd_line, macd_signal_line)
        macd_bearish = self._cross_down(macd_line, macd_signal_line)

        recent_price = close.tail(60)
        recent_rsi = rsi_series.tail(60) if rsi_series is not None else pd.Series(dtype=float)
        rsi_divergence = False
        if len(recent_price.dropna()) >= 30 and len(recent_rsi.dropna()) >= 30:
            first_half_price_high = float(recent_price.head(30).max())
            second_half_price_high = float(recent_price.tail(30).max())
            first_half_rsi_high = float(recent_rsi.head(30).max())
            second_half_rsi_high = float(recent_rsi.tail(30).max())
            rsi_divergence = second_half_price_high > first_half_price_high and second_half_rsi_high < first_half_rsi_high

        price_above_all_ma = bool(
            sma_20 is not None
            and sma_50 is not None
            and sma_200 is not None
            and current_price > sma_20
            and current_price > sma_50
            and current_price > sma_200
        )
        price_below_all_ma = bool(
            sma_20 is not None
            and sma_50 is not None
            and sma_200 is not None
            and current_price < sma_20
            and current_price < sma_50
            and current_price < sma_200
        )

        bb_width = ((bb_upper_s - bb_lower_s) / close).dropna()
        bb_squeeze = False
        if len(bb_width) >= 30:
            bb_squeeze = float(bb_width.iloc[-1]) <= float(bb_width.tail(120).quantile(0.2))

        recent_window = data.tail(30)
        immediate_support = float(recent_window["Low"].min()) if not recent_window.empty else current_price
        immediate_resistance = float(recent_window["High"].max()) if not recent_window.empty else current_price

        support_levels = [level for level in [immediate_support, sma_50, sma_200] if level is not None]
        resistance_levels = [
            level
            for level in [immediate_resistance, week_52_high or float(data["High"].tail(252).max())]
            if level is not None
        ]

        if current_price <= immediate_support * 1.01:
            entry_price = round(current_price, 2)
        else:
            entry_price = round(immediate_support, 2)

        atr_value = atr or 0.0
        stop_loss = round(entry_price - (1.5 * atr_value), 2) if atr_value else round(entry_price * 0.95, 2)

        sorted_resistances = sorted({round(level, 2) for level in resistance_levels if level > entry_price})
        target_1 = sorted_resistances[0] if sorted_resistances else round(entry_price + (2 * atr_value or entry_price * 0.05), 2)
        target_2 = (
            sorted_resistances[1]
            if len(sorted_resistances) > 1
            else round(target_1 + (1.5 * atr_value or entry_price * 0.05), 2)
        )
        target_3 = round(target_2 + (2 * atr_value or entry_price * 0.08), 2)

        bullish_checks = [
            rsi is not None and rsi > 50,
            macd is not None and macd_signal is not None and macd > macd_signal,
            price_above_all_ma,
            ema_9 is not None and ema_21 is not None and ema_9 > ema_21,
            adx is not None and adx >= 20,
            current_price > (vwap.dropna().iloc[-1] if not vwap.dropna().empty else current_price),
            macd_bullish,
            golden_cross,
            not price_below_all_ma,
            not macd_bearish,
        ]
        bearish_checks = [
            rsi is not None and rsi < 45,
            macd is not None and macd_signal is not None and macd < macd_signal,
            price_below_all_ma,
            ema_9 is not None and ema_21 is not None and ema_9 < ema_21,
            adx is not None and adx >= 20 and price_below_all_ma,
            current_price < (vwap.dropna().iloc[-1] if not vwap.dropna().empty else current_price),
            macd_bearish,
            death_cross,
            rsi_divergence,
            bb_squeeze and current_price < (bb_lower if bb_lower is not None else current_price),
        ]

        bullish_count = sum(1 for item in bullish_checks if item)
        bearish_count = sum(1 for item in bearish_checks if item)
        signal_score = f"{bullish_count}/10 indicators bullish"

        if bullish_count >= 8:
            technical_signal = "STRONG BUY"
        elif bullish_count >= 6:
            technical_signal = "BUY"
        elif bearish_count >= 7:
            technical_signal = "STRONG SELL"
        elif bearish_count >= 5:
            technical_signal = "SELL"
        else:
            technical_signal = "NEUTRAL"

        patterns_detected = []
        if golden_cross:
            patterns_detected.append("Golden Cross")
        if death_cross:
            patterns_detected.append("Death Cross")
        if rsi_divergence:
            patterns_detected.append("RSI Divergence")
        if macd_bullish:
            patterns_detected.append("MACD Bullish Crossover")
        if macd_bearish:
            patterns_detected.append("MACD Bearish Crossover")
        if price_above_all_ma:
            patterns_detected.append("Price Above All Major Moving Averages")
        if price_below_all_ma:
            patterns_detected.append("Price Below All Major Moving Averages")
        if bb_squeeze:
            patterns_detected.append("Bollinger Band Squeeze")

        return {
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "bb_upper": bb_upper,
            "bb_lower": bb_lower,
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "adx": adx,
            "atr": atr,
            "patterns_detected": patterns_detected,
            "support_levels": [round(level, 2) for level in support_levels],
            "resistance_levels": [round(level, 2) for level in sorted(set(resistance_levels))],
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "target_1": round(target_1, 2),
            "target_2": round(target_2, 2),
            "target_3": round(target_3, 2),
            "technical_signal": technical_signal,
            "signal_score": signal_score,
            "signal_reasoning": (
                f"Bullish indicators: {bullish_count}/10, bearish indicators: {bearish_count}/10. "
                f"RSI={rsi}, MACD spread={(None if macd is None or macd_signal is None else round(macd - macd_signal, 4))}, "
                f"price_vs_ma={'above' if price_above_all_ma else 'below' if price_below_all_ma else 'mixed'}."
            ),
        }
