from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import SECTOR_PE_AVERAGES


class FundamentalAgent:
    name = "Fundamental Analysis Agent"

    RISK_FREE_RATE = 0.072
    INDIA_ERP = 0.065
    COST_OF_DEBT = 0.085
    TAX_RATE = 0.25
    SCENARIO_ADJUSTMENT = 0.03
    SCENARIO_DISCOUNT_ADJUSTMENT = 0.005
    MIN_BEAR_GROWTH = -0.15
    MAX_BULL_GROWTH = 0.30

    TERMINAL_GROWTH_MAP: dict[str, float] = {
        "TECH": 0.07,
        "INFORMATION TECHNOLOGY": 0.07,
        "BANK": 0.05,
        "BANKING": 0.05,
        "FMCG": 0.06,
        "CONSUMER STAPLES": 0.06,
        "ENERGY": 0.03,
        "OIL & GAS": 0.03,
        "INFRA": 0.04,
        "INFRASTRUCTURE": 0.04,
    }

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
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _valuation_label(pe_ratio: float | None, sector: str | None) -> str:
        if pe_ratio is None:
            return "FAIRLY VALUED"
        sector_key = (sector or "").upper()
        sector_average = next(
            (v for k, v in SECTOR_PE_AVERAGES.items() if k in sector_key), None
        )
        if not sector_average:
            return "FAIRLY VALUED"
        if pe_ratio <= sector_average * 0.85:
            return "UNDERVALUED"
        if pe_ratio >= sector_average * 1.15:
            return "OVERVALUED"
        return "FAIRLY VALUED"

    @staticmethod
    def _score_metric(
        value: float | None,
        thresholds: tuple[float, float, float, float],
        reverse: bool = False,
    ) -> int:
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

    def _terminal_growth_for_sector(self, sector: str | None) -> float:
        sector_upper = (sector or "").upper()
        for key, value in self.TERMINAL_GROWTH_MAP.items():
            if key in sector_upper:
                return value
        return 0.04

    def _dynamic_wacc(self, beta: float | None, debt_to_equity: float | None) -> float:
        beta_value = 1.0 if beta is None or beta <= 0 else beta
        cost_of_equity = self.RISK_FREE_RATE + (beta_value * self.INDIA_ERP)

        if debt_to_equity is None or debt_to_equity <= 0.5:
            return round(cost_of_equity, 4)

        weight_debt = debt_to_equity / (1 + debt_to_equity)
        weight_equity = 1 - weight_debt
        wacc = (cost_of_equity * weight_equity) + (
            self.COST_OF_DEBT * (1 - self.TAX_RATE) * weight_debt
        )
        return round(wacc, 4)

    @staticmethod
    def _growth_rate_from_history(fcf_history: list[float]) -> float:
        if len(fcf_history) < 2 or fcf_history[-1] <= 0:
            return 0.05
        cagr = (fcf_history[-1] / fcf_history[0]) ** (1 / (len(fcf_history) - 1)) - 1
        return max(min(cagr, 0.25), -0.12)

    @staticmethod
    def _dcf_value_per_share(
        base_fcf: float,
        shares_outstanding: float,
        growth_rate: float,
        wacc: float,
        terminal_growth: float,
    ) -> float | None:
        # Gordon Growth terminal value is only valid when WACC > terminal growth.
        if base_fcf <= 0 or shares_outstanding <= 0 or wacc <= terminal_growth:
            return None

        running_fcf = float(base_fcf)
        pv_sum = 0.0
        for year in range(1, 6):
            running_fcf *= 1 + growth_rate
            pv_sum += running_fcf / ((1 + wacc) ** year)

        terminal_value = (running_fcf * (1 + terminal_growth)) / (wacc - terminal_growth)
        terminal_value /= (1 + wacc) ** 5
        return round((pv_sum + terminal_value) / shares_outstanding, 2)

    @staticmethod
    def _pct_change(new: float | None, old: float | None) -> float | None:
        if new is None or old is None or old == 0:
            return None
        return round(((new - old) / abs(old)) * 100, 2)

    def _build_historical_context(
        self,
        pe_ratio: float | None,
        revenue_growth: float | None,
        eps_growth: float | None,
        roe: float | None,
        debt_to_equity: float | None,
    ) -> dict[str, Any]:
        return {
            "pe_ratio": {
                "current": pe_ratio,
                "comment": "Context limited by yfinance point-in-time PE availability.",
            },
            "revenue_growth": {
                "current": revenue_growth,
                "comment": "Compared against EPS growth to detect quality of growth.",
            },
            "eps_growth": {
                "current": eps_growth,
                "comment": "EPS growth below revenue growth can indicate margin pressure.",
            },
            "roe": {
                "current": roe,
                "comment": "ROE below 10% is usually weak for long-term compounding.",
            },
            "debt_to_equity": {
                "current": debt_to_equity,
                "comment": "Higher leverage increases fragility in down cycles.",
            },
        }

    def analyze(self, ticker: str) -> dict[str, Any]:
        stock = yf.Ticker(ticker)
        info = stock.info or {}
        cashflow = stock.cashflow
        quarterly_financials = stock.quarterly_financials

        trailing_pe = info.get("trailingPE")
        forward_pe = info.get("forwardPE")
        pe_ratio = trailing_pe if trailing_pe is not None else forward_pe
        pb_ratio = info.get("priceToBook")
        ev_ebitda = info.get("enterpriseToEbitda")

        revenue_series = self._match_row(
            quarterly_financials, ["Total Revenue", "Revenue", "Operating Revenue"]
        )
        eps_series = self._match_row(quarterly_financials, ["Diluted EPS", "Basic EPS"])

        revenue_values = (
            revenue_series.dropna().head(4).tolist()
            if revenue_series is not None
            else []
        )
        eps_values = (
            eps_series.dropna().head(4).tolist() if eps_series is not None else []
        )

        revenue_growth = None
        if len(revenue_values) >= 4 and revenue_values[3] != 0:
            revenue_growth = round(
                ((revenue_values[0] - revenue_values[3]) / abs(revenue_values[3]))
                * 100,
                2,
            )
        elif info.get("revenueGrowth") is not None:
            revenue_growth = self._safe_pct(info.get("revenueGrowth"), 100)

        eps_growth = None
        if len(eps_values) >= 4 and eps_values[3] != 0:
            eps_growth = round(
                ((eps_values[0] - eps_values[3]) / abs(eps_values[3])) * 100, 2
            )
        elif info.get("earningsGrowth") is not None:
            eps_growth = self._safe_pct(info.get("earningsGrowth"), 100)

        profit_margin = self._safe_pct(info.get("profitMargins"), 100)
        roe = self._safe_pct(info.get("returnOnEquity"), 100)
        roa = self._safe_pct(info.get("returnOnAssets"), 100)
        debt_to_equity = self._safe_float(info.get("debtToEquity"))
        current_ratio = self._safe_float(info.get("currentRatio"))
        free_cashflow = self._safe_float(info.get("freeCashflow"))
        if free_cashflow is None:
            fcf_series = self._match_row(cashflow, ["Free Cash Flow"])
            free_cashflow = self._last_non_null(
                fcf_series.dropna().tolist() if fcf_series is not None else []
            )

        dividend_yield = self._safe_pct(info.get("dividendYield"), 100)
        promoter_holding = self._safe_pct(info.get("heldPercentInsiders"), 100)
        beta = self._safe_float(info.get("beta"))

        sector = str(info.get("sector") or "")
        valuation_label = self._valuation_label(pe_ratio=pe_ratio, sector=sector)

        metric_scores = {
            "revenue_growth": self._score_metric(revenue_growth, (-5, 0, 8, 15)),
            "profit_margin": self._score_metric(profit_margin, (5, 10, 15, 20)),
            "eps_growth": self._score_metric(eps_growth, (-5, 0, 8, 15)),
            "roe": self._score_metric(roe, (8, 12, 16, 20)),
            "roa": self._score_metric(roa, (3, 5, 8, 10)),
            "debt_to_equity": self._score_metric(
                debt_to_equity, (0.5, 1.0, 1.5, 2.0), reverse=True
            ),
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

        sorted_metrics = sorted(
            metric_scores.items(), key=lambda item: item[1], reverse=True
        )
        strengths = [name.replace("_", " ").title() for name, _ in sorted_metrics[:3]]
        concerns = [
            name.replace("_", " ").title()
            for name, _ in sorted(metric_scores.items(), key=lambda i: i[1])[:2]
        ]

        fcf_history_series = self._match_row(cashflow, ["Free Cash Flow"])
        fcf_history: list[float] = []
        if fcf_history_series is not None:
            fcf_chronological = fcf_history_series.sort_index().dropna().tail(5)
            fcf_history = [
                float(x) for x in fcf_chronological.tolist() if float(x) > 0
            ]

        shares_outstanding = self._safe_float(info.get("sharesOutstanding") or 0) or 0
        current_price = self._safe_float(info.get("currentPrice") or info.get("regularMarketPrice"))

        wacc = self._dynamic_wacc(beta=beta, debt_to_equity=debt_to_equity)
        terminal_growth = self._terminal_growth_for_sector(sector)
        base_growth = self._growth_rate_from_history(fcf_history)
        bear_growth = max(base_growth - self.SCENARIO_ADJUSTMENT, self.MIN_BEAR_GROWTH)
        bull_growth = min(base_growth + self.SCENARIO_ADJUSTMENT, self.MAX_BULL_GROWTH)

        base_fcf = (
            fcf_history[-1]
            if fcf_history
            else (free_cashflow if free_cashflow and free_cashflow > 0 else 0.0)
        )

        bear_value = self._dcf_value_per_share(
            base_fcf=base_fcf,
            shares_outstanding=shares_outstanding,
            growth_rate=bear_growth,
            wacc=wacc,
            terminal_growth=max(
                terminal_growth - self.SCENARIO_DISCOUNT_ADJUSTMENT, 0.02
            ),
        )
        base_value = self._dcf_value_per_share(
            base_fcf=base_fcf,
            shares_outstanding=shares_outstanding,
            growth_rate=base_growth,
            wacc=wacc,
            terminal_growth=terminal_growth,
        )
        bull_value = self._dcf_value_per_share(
            base_fcf=base_fcf,
            shares_outstanding=shares_outstanding,
            growth_rate=bull_growth,
            wacc=max(
                wacc - self.SCENARIO_DISCOUNT_ADJUSTMENT, terminal_growth + 0.01
            ),
            terminal_growth=min(
                terminal_growth + self.SCENARIO_DISCOUNT_ADJUSTMENT, 0.08
            ),
        )

        intrinsic_value = base_value
        upside_pct = None
        if intrinsic_value is not None and current_price:
            upside_pct = round(
                ((intrinsic_value - float(current_price)) / float(current_price)) * 100,
                2,
            )

        dcf_scenarios = {
            "bear": {
                "growth": round(bear_growth * 100, 2),
                "terminal_growth": round(
                    max(terminal_growth - self.SCENARIO_DISCOUNT_ADJUSTMENT, 0.02)
                    * 100,
                    2,
                ),
                "intrinsic_value": bear_value,
                "upside_pct": self._pct_change(bear_value, current_price),
            },
            "base": {
                "growth": round(base_growth * 100, 2),
                "terminal_growth": round(terminal_growth * 100, 2),
                "intrinsic_value": base_value,
                "upside_pct": self._pct_change(base_value, current_price),
            },
            "bull": {
                "growth": round(bull_growth * 100, 2),
                "terminal_growth": round(
                    min(terminal_growth + self.SCENARIO_DISCOUNT_ADJUSTMENT, 0.08)
                    * 100,
                    2,
                ),
                "intrinsic_value": bull_value,
                "upside_pct": self._pct_change(bull_value, current_price),
            },
        }

        if valuation_label == "UNDERVALUED" and financial_health in {
            "EXCELLENT",
            "GOOD",
        }:
            fundamental_signal = "BUY"
        elif valuation_label == "OVERVALUED" and financial_health in {"FAIR", "POOR"}:
            fundamental_signal = "SELL"
        else:
            fundamental_signal = "HOLD"

        if (
            dcf_scenarios["base"]["upside_pct"] is not None
            and dcf_scenarios["base"]["upside_pct"] >= 20
        ):
            dcf_label = "BUY"
        elif (
            dcf_scenarios["bear"]["upside_pct"] is not None
            and dcf_scenarios["bear"]["upside_pct"] < 0
        ):
            dcf_label = "SPECULATIVE"
        else:
            dcf_label = "NEUTRAL"

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
            "dcf_scenarios": dcf_scenarios,
            "dcf_label": dcf_label,
            "wacc_pct": round(wacc * 100, 2),
            "terminal_growth_pct": round(terminal_growth * 100, 2),
            "historical_context": self._build_historical_context(
                pe_ratio=pe_ratio,
                revenue_growth=revenue_growth,
                eps_growth=eps_growth,
                roe=roe,
                debt_to_equity=debt_to_equity,
            ),
            "fundamental_reasoning": (
                f"Valuation: {valuation_label}; Financial health: {financial_health} ({average_score}/10). "
                f"Dynamic DCF base={intrinsic_value}, base upside={upside_pct}%, WACC={round(wacc * 100, 2)}%."
            ),
        }
