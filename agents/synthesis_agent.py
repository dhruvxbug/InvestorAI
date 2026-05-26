from __future__ import annotations

import json
from typing import Any

from utils.llm_client import LLMClient, build_llm_client

SYSTEM_PROMPT = """
You are a senior equity research analyst with 20 years of experience
in Indian stock markets. You have deep expertise in fundamental
analysis, technical analysis, and market sentiment. Your job is to
synthesize data from multiple analysis agents and produce clear,
actionable investment reports.

Your reports are used by retail investors on Groww. They need:
1. EXACT price levels — not vague ranges
2. CLEAR reasoning — not jargon
3. HONEST risk assessment — never hide risks
4. DECISIVE signals — no fence-sitting

Rules:
- Never say "it depends" without giving the specific condition
- Always give a specific entry price or a specific condition to enter
- Always give a specific stop loss with exact ₹ level
- Always distinguish between long-term and short-term views
- If data is contradictory, explain which signal you weight more and why
- Use ₹ symbol for all Indian prices
- Flag any red flags in bold
- If you cannot confidently recommend, say AVOID with clear reasoning
""".strip()

USER_TEMPLATE = """
Analyze the following data for {ticker} ({company_name}) and generate
a complete investment report.

STOCK DATA: {stock_data_json}
TECHNICAL ANALYSIS: {technical_data_json}
FUNDAMENTAL ANALYSIS: {fundamental_data_json}
NEWS & SENTIMENT: {sentiment_data_json}
MANAGEMENT INTELLIGENCE: {management_data_json}

Generate the full report following the exact structure specified.
Include specific ₹ price levels for every entry, target, and stop loss.
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

    def _heuristic_synthesis(
        self, ticker: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        stock_data = context.get("stock_data", {})
        technical = context.get("technical", {})
        fundamental = context.get("fundamental", {})
        sentiment = context.get("sentiment", {})
        management = context.get("management", {})

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
        score = sum(self._signal_to_numeric(value) for value in signals.values())
        bullish = sum(
            1 for value in signals.values() if self._signal_to_numeric(value) > 0
        )
        bearish = sum(
            1 for value in signals.values() if self._signal_to_numeric(value) < 0
        )
        neutral = 4 - bullish - bearish

        if score >= 4:
            long_signal = "STRONG BUY"
        elif score >= 2:
            long_signal = "BUY"
        elif score >= 1:
            long_signal = "ACCUMULATE"
        elif score <= -3:
            long_signal = "SELL"
        elif score <= -1:
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
            )
        )[:5]
        if not key_risks:
            key_risks = [
                "Market volatility",
                "Earnings miss risk",
                "Sector slowdown risk",
            ]

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
            max(
                0.0, float((self._signal_to_numeric(signals["fundamental"]) + 2) * 2.5)
            ),
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
        overall_score = round(
            (
                technical_score
                + fundamental_score
                + sentiment_score
                + management_score
                + valuation_score
            )
            / 5,
            2,
        )

        return {
            "key_summary": [
                f"Strongest bullish reason: technical setup indicates {technical.get('technical_signal', 'NEUTRAL')} bias.",
                f"Second bullish reason: valuation appears {valuation_label.lower()}.",
                f"Main risk to watch: **{key_risks[0]}**.",
                f"Entry recommendation: accumulate near ₹{entry_price:.2f} with staggered buying.",
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
                "confidence_pct": int(max(35, min(90, 50 + score * 8))),
                "reasoning": f"Long-term view weights fundamentals and management over short-term volatility; signal agreement {bullish}/4 bullish.",
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
                "trade_setup_type": "Breakout"
                if target_1 > current_price
                else "Pullback",
                "holding_period": "1-12 weeks",
                "confidence_pct": int(
                    max(
                        30, min(85, 45 + self._signal_to_numeric(technical_signal) * 12)
                    )
                ),
                "reasoning": "Short-term setup is driven primarily by trend, momentum, and support-resistance positioning.",
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

    def _build_user_prompt(
        self, ticker: str, company_name: str, context: dict[str, Any]
    ) -> str:
        return USER_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            stock_data_json=json.dumps(context.get("stock_data", {}), default=str),
            technical_data_json=json.dumps(context.get("technical", {}), default=str),
            fundamental_data_json=json.dumps(
                context.get("fundamental", {}), default=str
            ),
            sentiment_data_json=json.dumps(context.get("sentiment", {}), default=str),
            management_data_json=json.dumps(context.get("management", {}), default=str),
        )

    def synthesize(
        self, ticker: str, company_name: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        clean_context = {
            "stock_data": {
                k: v
                for k, v in dict(context.get("stock_data", {})).items()
                if k not in {"raw_df", "hourly_df"}
            },
            "technical": context.get("technical", {}),
            "fundamental": context.get("fundamental", {}),
            "sentiment": context.get("sentiment", {}),
            "management": context.get("management", {}),
        }

        heuristic = self._heuristic_synthesis(ticker=ticker, context=clean_context)
        if self.llm_client is None:
            return heuristic

        user_prompt = self._build_user_prompt(
            ticker=ticker, company_name=company_name, context=clean_context
        )
        text_output = self.llm_client.complete(
            system=SYSTEM_PROMPT,
            user=user_prompt,
            max_tokens=2500,
            temperature=0.2,
        )
        parsed = self._extract_json(text_output)
        if not parsed:
            return heuristic

        parsed.setdefault("key_summary", heuristic["key_summary"])
        if (
            not isinstance(parsed.get("key_summary"), list)
            or len(parsed["key_summary"]) != 5
        ):
            parsed["key_summary"] = heuristic["key_summary"]
        parsed.setdefault("long_term", heuristic["long_term"])
        parsed.setdefault("short_term", heuristic["short_term"])
        parsed.setdefault("scores", heuristic["scores"])
        return parsed
