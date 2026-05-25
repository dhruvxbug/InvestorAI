from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf


SECTOR_PE_AVERAGES = {
    "IT": 25.0,
    "BANKING": 15.0,
    "FMCG": 45.0,
    "AUTO": 20.0,
    "PHARMA": 30.0,
    "ENERGY": 12.0,
}


class FundamentalAgent:
    name = "Fundamental Analysis Agent"

    @staticmethod
    def _match_row(df: pd.DataFrame, candidates: list[str]) -> pd.Series | None:
        if df is None or df.empty:
            return None
        index_map = {str(idx).strip().lower(): idx for idx in df.index}
        for candidate in candidates:
            key = candidate.strip().lower()
            if key in index_map:
                series = df.loc[index_map[key]]
                return series if isinstance(series, pd.Series) else None
        return None

    @staticmethod
    def _last_non_null(values: list[Any]) -> float | None:
        for value in values:
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                return float(value)
        return None

    @staticmethod
    def _safe_pct(value: float | None, scale: float = 1.0) -> float | None:
        if value is None:
            return None
        return round(float(value) * scale, 2)

    @staticmethod
    def _valuation_label(pe_ratio: float | None, sector: str | None) -> str:
        if pe_ratio is None:
            return "FAIRLY VALUED"
        sector_key = (sector or "").upper()
        sector_average = next((v for k, v in SECTOR_PE_AVERAGES.items() if k in sector_key), None)
        if not sector_average:
            return "FAIRLY VALUED"
        if pe_ratio <= sector_average * 0.85:
            return "UNDERVALUED"
        if pe_ratio >= sector_average * 1.15:
            return "OVERVALUED"
        return "FAIRLY VALUED"

    @staticmethod
    def _score_metric(value: float | None, thresholds: tuple[float, float, float, float], reverse: bool = False) -> int:
        if value is None:
            return 5
        t1, t2, t3, t4 = thresholds
        if reverse:
            if value <= t1:
                return 10
            if value <= t2:
                return 8
            if value <= t3:
                return 6
            if value <= t4:
                return 4
            return 2
        if value >= t4:
            return 10
        if value >= t3:
            return 8
        if value >= t2:
            return 6
        if value >= t1:
            return 4
        return 2

    def analyze(self, ticker: str) -> dict[str, Any]:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow
        quarterly_financials = stock.quarterly_financials

        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        pe_ratio = trailing_pe if trailing_pe is not None else forward_pe
        pb_ratio = info.get("priceToBook")
        ev_ebitda = info.get("enterpriseToEbitda")

        revenue_series = self._match_row(quarterly_financials, ["Total Revenue", "Revenue", "Operating Revenue"])
        eps_series = self._match_row(quarterly_financials, ["Diluted EPS", "Basic EPS"])

        revenue_values = revenue_series.dropna().head(4).tolist() if revenue_series is not None else []
        eps_values = eps_series.dropna().head(4).tolist() if eps_series is not None else []

        revenue_growth = None
        if len(revenue_values) >= 4 and revenue_values[3] != 0:
            revenue_growth = round(((revenue_values[0] - revenue_values[3]) / abs(revenue_values[3])) * 100, 2)
        elif info.get("revenueGrowth") is not None:
            revenue_growth = self._safe_pct(info.get("revenueGrowth"), 100)

        eps_growth = None
        if len(eps_values) >= 4 and eps_values[3] != 0:
            eps_growth = round(((eps_values[0] - eps_values[3]) / abs(eps_values[3])) * 100, 2)
        elif info.get("earningsGrowth") is not None:
            eps_growth = self._safe_pct(info.get("earningsGrowth"), 100)

        profit_margin = self._safe_pct(info.get("profitMargins"), 100)
        roe = self._safe_pct(info.get("returnOnEquity"), 100)
        roa = self._safe_pct(info.get("returnOnAssets"), 100)
        debt_to_equity = info.get("debtToEquity")
        current_ratio = info.get("currentRatio")
        free_cashflow = info.get("freeCashflow")
        if free_cashflow is None:
            fcf_series = self._match_row(cashflow, ["Free Cash Flow"])
            free_cashflow = self._last_non_null(fcf_series.dropna().tolist() if fcf_series is not None else [])

        dividend_yield = self._safe_pct(info.get("dividendYield"), 100)
        promoter_holding = self._safe_pct(info.get("heldPercentInsiders"), 100)

        valuation_label = self._valuation_label(pe_ratio=pe_ratio, sector=info.get("sector"))

        metric_scores = {
            "revenue_growth": self._score_metric(revenue_growth, (-5, 0, 8, 15)),
            "profit_margin": self._score_metric(profit_margin, (5, 10, 15, 20)),
            "eps_growth": self._score_metric(eps_growth, (-5, 0, 8, 15)),
            "roe": self._score_metric(roe, (8, 12, 16, 20)),
            "roa": self._score_metric(roa, (3, 5, 8, 10)),
            "debt_to_equity": self._score_metric(debt_to_equity, (0.5, 1.0, 1.5, 2.0), reverse=True),
            "current_ratio": self._score_metric(current_ratio, (0.8, 1.0, 1.3, 1.8)),
            "free_cashflow": self._score_metric(free_cashflow, (0, 1e8, 5e8, 1e9)),
        }
        average_score = round(sum(metric_scores.values()) / len(metric_scores), 1)

        if average_score >= 8.5:
            financial_health = "EXCELLENT"
        elif average_score >= 7:
            financial_health = "GOOD"
        elif average_score >= 5:
            financial_health = "FAIR"
        else:
            financial_health = "POOR"

        sorted_metrics = sorted(metric_scores.items(), key=lambda item: item[1], reverse=True)
        strengths = [name.replace("_", " ").title() for name, _ in sorted_metrics[:3]]
        concerns = [name.replace("_", " ").title() for name, _ in sorted(metric_scores.items(), key=lambda i: i[1])[:2]]

        fcf_history_series = self._match_row(cashflow, ["Free Cash Flow"])
        fcf_history = []
        if fcf_history_series is not None:
            fcf_history = [float(x) for x in fcf_history_series.dropna().head(5).tolist() if float(x) > 0]

        discount_rate = 0.12
        terminal_growth = 0.04
        growth_rate = 0.05
        if len(fcf_history) >= 2 and fcf_history[-1] > 0:
            growth_rate = max(min((fcf_history[0] / fcf_history[-1]) ** (1 / (len(fcf_history) - 1)) - 1, 0.2), -0.1)

        base_fcf = fcf_history[0] if fcf_history else (free_cashflow if free_cashflow and free_cashflow > 0 else 0)
        projected_fcfs = []
        running_fcf = float(base_fcf)
        for year in range(1, 6):
            running_fcf *= 1 + growth_rate
            projected_fcfs.append(running_fcf / ((1 + discount_rate) ** year))

        terminal_value = 0.0
        if running_fcf > 0 and discount_rate > terminal_growth:
            terminal_value = (running_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
            terminal_value /= (1 + discount_rate) ** 5

        intrinsic_equity_value = sum(projected_fcfs) + terminal_value
        shares_outstanding = info.get("sharesOutstanding") or 0
        intrinsic_value = round(intrinsic_equity_value / shares_outstanding, 2) if shares_outstanding else None

        current_price = info.get("currentPrice") or info.get("regularMarketPrice")
        upside_pct = None
        if intrinsic_value is not None and current_price:
            upside_pct = round(((intrinsic_value - float(current_price)) / float(current_price)) * 100, 2)

        if valuation_label == "UNDERVALUED" and financial_health in {"EXCELLENT", "GOOD"}:
            fundamental_signal = "BUY"
        elif valuation_label == "OVERVALUED" and financial_health in {"FAIR", "POOR"}:
            fundamental_signal = "SELL"
        else:
            fundamental_signal = "HOLD"

        return {
            "pe_ratio": {"trailing": trailing_pe, "forward": forward_pe},
            "pb_ratio": pb_ratio,
            "ev_ebitda": ev_ebitda,
            "revenue_growth": revenue_growth,
            "profit_margin": profit_margin,
            "eps_growth": eps_growth,
            "roe": roe,
            "roa": roa,
            "debt_to_equity": debt_to_equity,
            "current_ratio": current_ratio,
            "free_cashflow": free_cashflow,
            "dividend_yield": dividend_yield,
            "promoter_holding": promoter_holding,
            "valuation_label": valuation_label,
            "intrinsic_value": intrinsic_value,
            "upside_pct": upside_pct,
            "financial_health": financial_health,
            "strengths": strengths,
            "concerns": concerns,
            "fundamental_signal": fundamental_signal,
            "fundamental_reasoning": (
                f"Valuation: {valuation_label}; Financial health: {financial_health} ({average_score}/10). "
                f"DCF intrinsic value={intrinsic_value}, upside={upside_pct}%"
            ),
        }
