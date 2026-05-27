from __future__ import annotations

import json
from typing import Any

from utils.llm_client import LLMClient, build_llm_client

SYSTEM_PROMPT = """
You are a senior equity research analyst with 20 years of experience
in Indian stock markets. You synthesize multi-agent signals into precise,
actionable recommendations.

Rules:
- Every major claim should be grounded in provided data.
- Prioritize downside risk and thesis invalidation points.
- Resolve contradictions explicitly before giving final signals.
- Return strict JSON when asked.
""".strip()

PASS1_TEMPLATE = """
Review agent outputs for {ticker} ({company_name}).

Stock data: {stock_data_json}
Technical: {technical_data_json}
Fundamental: {fundamental_data_json}
Sentiment: {sentiment_data_json}
Management: {management_data_json}
Red team: {red_team_json}

Return strict JSON with keys:
- technical: {{strengths: [2], weaknesses: [2]}}
- fundamental: {{strengths: [2], weaknesses: [2]}}
- sentiment: {{strengths: [2], weaknesses: [2]}}
- management: {{strengths: [2], weaknesses: [2]}}
- valuation: {{strengths: [2], weaknesses: [2]}}
- thesis_killers: [top 3]
""".strip()

PASS2_TEMPLATE = """
Given this agent critique JSON:
{pass1_json}

And raw red-team findings:
{red_team_json}

Find contradictions and resolve signal hierarchy.
Return strict JSON:
- contradictions: [max 6]
- resolution: [for each contradiction, which signal dominates and why]
- dominant_risks: [top 5]
- conviction_adjustments: {{technical: -2..2, fundamental: -2..2, sentiment: -2..2, management: -2..2, valuation: -2..2}}
""".strip()

FINAL_TEMPLATE = """
Generate final investment report JSON for {ticker} ({company_name}).

Inputs:
Stock data: {stock_data_json}
Technical: {technical_data_json}
Fundamental: {fundamental_data_json}
Sentiment: {sentiment_data_json}
Management: {management_data_json}
Red team: {red_team_json}
Pass1 critique: {pass1_json}
Pass2 conflict resolution: {pass2_json}

Return JSON with EXACTLY these keys:
- key_summary: array of 5 bullet points
- long_term: object with signal, entry_price, entry_condition, target_1_year, target_3_year, stop_loss_price, expected_cagr, risk_level, exit_triggers (array), key_risks (array), key_catalysts (array), confidence_pct (int), reasoning
- short_term: object with signal, entry_price, entry_condition, target_1, target_2, target_3, stop_loss, risk_reward_ratio, trade_setup_type, holding_period, confidence_pct (int), reasoning
- scores: object with technical, fundamental, sentiment, management, valuation, overall, agents_bullish (int), agents_bearish (int), agents_neutral (int)
""".strip()


class SynthesisAgent:
    name = "Report Synthesis Agent"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client: LLMClient | None = llm_client or build_llm_client()

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _signal_to_numeric(signal: str | None) -> int:
        mapping = {
            "STRONG BUY": 2,
            "BUY": 1,
            "ACCUMULATE": 1,
            "HOLD": 0,
            "NEUTRAL": 0,
            "WAIT": 0,
            "REDUCE": -1,
            "SELL": -1,
            "AVOID": -2,
            "STRONG SELL": -2,
        }
        return mapping.get((signal or "").upper(), 0)

    @staticmethod
    def _normalize_signal(signal: str, short_term: bool = False) -> str:
        normalized = (signal or "HOLD").upper()
        if short_term:
            allowed = {"STRONG BUY", "BUY", "WAIT", "AVOID", "SELL"}
            if normalized == "STRONG SELL":
                return "SELL"
            if normalized == "HOLD":
                return "WAIT"
            return normalized if normalized in allowed else "WAIT"
        allowed = {"STRONG BUY", "BUY", "ACCUMULATE", "HOLD", "REDUCE", "SELL"}
        if normalized == "WAIT":
            return "HOLD"
        if normalized == "AVOID":
            return "SELL"
        return normalized if normalized in allowed else "HOLD"

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any] | None:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        try:
            payload = json.loads(cleaned)
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _agent_weights(market_cap: float | None, avg_volume: float | None) -> dict[str, float]:
        cap = market_cap or 0.0
        if cap > 1_000_000_000_000:
            return {
                "technical": 0.25,
                "fundamental": 0.30,
                "sentiment": 0.20,
                "management": 0.10,
                "valuation": 0.15,
            }
        if cap > 200_000_000_000:
            return {
                "technical": 0.20,
                "fundamental": 0.25,
                "sentiment": 0.20,
                "management": 0.20,
                "valuation": 0.15,
            }
        if (avg_volume or 0) < 1_000_000:
            return {
                "technical": 0.12,
                "fundamental": 0.20,
                "sentiment": 0.13,
                "management": 0.35,
                "valuation": 0.20,
            }
        return {
            "technical": 0.15,
            "fundamental": 0.20,
            "sentiment": 0.15,
            "management": 0.30,
            "valuation": 0.20,
        }

    def _heuristic_synthesis(
        self, ticker: str, company_name: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        stock_data = context.get("stock_data", {})
        technical = context.get("technical", {})
        fundamental = context.get("fundamental", {})
        sentiment = context.get("sentiment", {})
        management = context.get("management", {})
        red_team = context.get("red_team", {})

        current_price = self._safe_float(stock_data.get("current_price"), 0.0)
        entry_price = self._safe_float(technical.get("entry_price"), current_price)
        stop_loss = self._safe_float(
            technical.get("stop_loss"), round(entry_price * 0.95, 2)
        )
        target_1 = self._safe_float(
            technical.get("target_1"), round(entry_price * 1.05, 2)
        )
        target_2 = self._safe_float(
            technical.get("target_2"), round(entry_price * 1.1, 2)
        )
        target_3 = self._safe_float(
            technical.get("target_3"), round(entry_price * 1.18, 2)
        )

        signals = {
            "technical": technical.get("technical_signal", "NEUTRAL"),
            "fundamental": fundamental.get("fundamental_signal", "HOLD"),
            "sentiment": sentiment.get("sentiment_signal", "HOLD"),
            "management": management.get("management_signal", "HOLD"),
        }
        signal_sum = sum(self._signal_to_numeric(value) for value in signals.values())
        bullish = sum(1 for value in signals.values() if self._signal_to_numeric(value) > 0)
        bearish = sum(1 for value in signals.values() if self._signal_to_numeric(value) < 0)
        neutral = 4 - bullish - bearish

        if signal_sum >= 4:
            long_signal = "STRONG BUY"
        elif signal_sum >= 2:
            long_signal = "BUY"
        elif signal_sum >= 1:
            long_signal = "ACCUMULATE"
        elif signal_sum <= -3:
            long_signal = "SELL"
        elif signal_sum <= -1:
            long_signal = "REDUCE"
        else:
            long_signal = "HOLD"

        technical_signal = self._normalize_signal(
            technical.get("technical_signal", "WAIT"), short_term=True
        )

        valuation_label = str(fundamental.get("valuation_label", "FAIRLY VALUED"))
        if valuation_label == "UNDERVALUED":
            entry_condition = "Strong buy at current levels"
        elif valuation_label == "OVERVALUED":
            entry_condition = f"Wait for pullback to ₹{entry_price:.2f}"
        else:
            entry_condition = "Current price is a good entry"

        intrinsic_value = self._safe_float(fundamental.get("intrinsic_value"), target_2)
        target_1_year = round(intrinsic_value, 2)
        target_3_year = round(max(target_1_year * 1.35, target_2), 2)
        expected_cagr = (
            round((((target_3_year / current_price) ** (1 / 3)) - 1) * 100, 2)
            if current_price
            else 0.0
        )

        risk_level = (
            "LOW"
            if bearish == 0
            else "MEDIUM"
            if bearish == 1
            else "HIGH"
            if bearish == 2
            else "VERY HIGH"
        )

        key_risks = list(
            dict.fromkeys(
                (fundamental.get("concerns") or [])
                + (management.get("red_flags") or [])
                + (red_team.get("dominant_risks") or [])
                + (red_team.get("thesis_killers") or [])
            )
        )[:5]
        if not key_risks:
            key_risks = ["Market volatility", "Earnings miss risk", "Sector slowdown risk"]

        key_catalysts = [
            item.get("event")
            for item in (management.get("upcoming_catalysts") or [])
            if isinstance(item, dict) and item.get("event")
        ]
        key_catalysts.extend(sentiment.get("key_events") or [])
        key_catalysts = list(dict.fromkeys(key_catalysts))[:5] or [
            "Positive earnings surprise",
            "Sector rerating",
            "Improving technical trend",
        ]

        risk_amount = max(entry_price - stop_loss, 0.01)
        reward_amount = max(target_2 - entry_price, 0.01)
        rr_ratio = round(reward_amount / risk_amount, 2)

        technical_score = min(
            10.0,
            max(0.0, float((self._signal_to_numeric(signals["technical"]) + 2) * 2.5)),
        )
        fundamental_score = min(
            10.0,
            max(0.0, float((self._signal_to_numeric(signals["fundamental"]) + 2) * 2.5)),
        )
        sentiment_score = min(
            10.0,
            max(0.0, float((self._signal_to_numeric(signals["sentiment"]) + 2) * 2.5)),
        )
        management_score = min(
            10.0,
            max(0.0, float((self._signal_to_numeric(signals["management"]) + 2) * 2.5)),
        )
        valuation_score = (
            8.0
            if valuation_label == "UNDERVALUED"
            else 5.0
            if valuation_label == "FAIRLY VALUED"
            else 3.0
        )

        weights = self._agent_weights(
            market_cap=self._safe_float(stock_data.get("market_cap"), 0.0),
            avg_volume=self._safe_float(stock_data.get("avg_volume"), 0.0),
        )
        overall_score = round(
            (technical_score * weights["technical"])
            + (fundamental_score * weights["fundamental"])
            + (sentiment_score * weights["sentiment"])
            + (management_score * weights["management"])
            + (valuation_score * weights["valuation"]),
            2,
        )

        contradictions = red_team.get("contradiction_map") or []
        contradiction_note = (
            f"Key contradiction: {contradictions[0]}" if contradictions else "No major contradiction detected."
        )

        return {
            "key_summary": [
                f"Strongest bullish reason: technical setup indicates {technical.get('technical_signal', 'NEUTRAL')} bias.",
                f"Second bullish reason: valuation appears {valuation_label.lower()}.",
                f"Main risk to watch: **{key_risks[0]}**.",
                f"{contradiction_note}",
                f"Exit strategy: close if price breaches ₹{stop_loss:.2f} on closing basis.",
            ],
            "long_term": {
                "signal": self._normalize_signal(long_signal),
                "entry_price": round(entry_price, 2),
                "entry_condition": entry_condition,
                "target_1_year": target_1_year,
                "target_3_year": target_3_year,
                "expected_cagr": expected_cagr,
                "stop_loss_price": round(stop_loss, 2),
                "exit_triggers": [
                    "Exit if quarterly revenue growth falls below 5% for 2 consecutive quarters",
                    "Exit if debt-to-equity crosses 2.0",
                    "Exit if promoter holding trend turns decreasing",
                    f"Exit if monthly close falls below ₹{stop_loss:.2f}",
                ],
                "risk_level": risk_level,
                "key_risks": key_risks[:5],
                "key_catalysts": key_catalysts[:5],
                "confidence_pct": int(max(35, min(90, 50 + signal_sum * 8))),
                "reasoning": f"Long-term view weights dynamic signals by market-cap/liquidity profile; agreement {bullish}/4 bullish.",
            },
            "short_term": {
                "signal": technical_signal,
                "entry_price": round(entry_price, 2),
                "entry_condition": f"Enter only if price holds above ₹{entry_price:.2f} or breaks above ₹{target_1:.2f} with volume.",
                "target_1": round(target_1, 2),
                "target_2": round(target_2, 2),
                "target_3": round(target_3, 2),
                "stop_loss": round(stop_loss, 2),
                "risk_reward_ratio": f"Risk ₹{risk_amount:.2f}, Reward ₹{reward_amount:.2f} = 1:{rr_ratio}",
                "trade_setup_type": "Breakout" if target_1 > current_price else "Pullback",
                "holding_period": "1-12 weeks",
                "confidence_pct": int(
                    max(30, min(85, 45 + self._signal_to_numeric(technical_signal) * 12))
                ),
                "reasoning": "Short-term setup is driven by trend, momentum, and conflict-adjusted risk controls.",
            },
            "scores": {
                "technical": round(technical_score, 2),
                "fundamental": round(fundamental_score, 2),
                "sentiment": round(sentiment_score, 2),
                "management": round(management_score, 2),
                "valuation": round(valuation_score, 2),
                "overall": overall_score,
                "agents_bullish": bullish,
                "agents_bearish": bearish,
                "agents_neutral": neutral,
            },
        }

    @staticmethod
    def _clean_context_for_prompt(context: dict[str, Any]) -> dict[str, Any]:
        stock_data = {
            k: v
            for k, v in dict(context.get("stock_data", {})).items()
            if k not in {"raw_df", "hourly_df"}
        }
        return {
            "stock_data": stock_data,
            "technical": context.get("technical", {}),
            "fundamental": context.get("fundamental", {}),
            "sentiment": context.get("sentiment", {}),
            "management": context.get("management", {}),
            "red_team": context.get("red_team", {}),
        }

    def _run_multi_pass(
        self, ticker: str, company_name: str, context: dict[str, Any], fallback: dict[str, Any]
    ) -> dict[str, Any]:
        if self.llm_client is None:
            return fallback

        pass1_prompt = PASS1_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            stock_data_json=json.dumps(context.get("stock_data", {}), default=str),
            technical_data_json=json.dumps(context.get("technical", {}), default=str),
            fundamental_data_json=json.dumps(context.get("fundamental", {}), default=str),
            sentiment_data_json=json.dumps(context.get("sentiment", {}), default=str),
            management_data_json=json.dumps(context.get("management", {}), default=str),
            red_team_json=json.dumps(context.get("red_team", {}), default=str),
        )
        try:
            pass1_text = self.llm_client.complete(
                system=SYSTEM_PROMPT,
                user=pass1_prompt,
                max_tokens=1600,
                temperature=0.1,
            )
            pass1 = self._extract_json(pass1_text) or {}
        except Exception:
            pass1 = {}

        pass2_prompt = PASS2_TEMPLATE.format(
            pass1_json=json.dumps(pass1, default=str),
            red_team_json=json.dumps(context.get("red_team", {}), default=str),
        )
        try:
            pass2_text = self.llm_client.complete(
                system=SYSTEM_PROMPT,
                user=pass2_prompt,
                max_tokens=1400,
                temperature=0.1,
            )
            pass2 = self._extract_json(pass2_text) or {}
        except Exception:
            pass2 = {}

        final_prompt = FINAL_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            stock_data_json=json.dumps(context.get("stock_data", {}), default=str),
            technical_data_json=json.dumps(context.get("technical", {}), default=str),
            fundamental_data_json=json.dumps(context.get("fundamental", {}), default=str),
            sentiment_data_json=json.dumps(context.get("sentiment", {}), default=str),
            management_data_json=json.dumps(context.get("management", {}), default=str),
            red_team_json=json.dumps(context.get("red_team", {}), default=str),
            pass1_json=json.dumps(pass1, default=str),
            pass2_json=json.dumps(pass2, default=str),
        )

        try:
            final_text = self.llm_client.complete(
                system=SYSTEM_PROMPT,
                user=final_prompt,
                max_tokens=2600,
                temperature=0.15,
            )
            parsed = self._extract_json(final_text)
            if not parsed:
                return fallback
        except Exception:
            return fallback

        parsed.setdefault("key_summary", fallback["key_summary"])
        if (
            not isinstance(parsed.get("key_summary"), list)
            or len(parsed["key_summary"]) != 5
        ):
            parsed["key_summary"] = fallback["key_summary"]
        parsed.setdefault("long_term", fallback["long_term"])
        parsed.setdefault("short_term", fallback["short_term"])
        parsed.setdefault("scores", fallback["scores"])
        return parsed

    def synthesize(
        self, ticker: str, company_name: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        clean_context = self._clean_context_for_prompt(context)
        heuristic = self._heuristic_synthesis(
            ticker=ticker, company_name=company_name, context=clean_context
        )
        return self._run_multi_pass(
            ticker=ticker,
            company_name=company_name,
            context=clean_context,
            fallback=heuristic,
        )
