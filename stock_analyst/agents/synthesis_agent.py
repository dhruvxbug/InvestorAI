from __future__ import annotations

import json
from typing import Any

from crewai import Agent, Crew, LLM, Task

from config.settings import ANTHROPIC_API_KEY


class SynthesisAgent:
    name = "Report Synthesis Agent"

    def __init__(self) -> None:
        self.llm = None
        if ANTHROPIC_API_KEY:
            self.llm = LLM(model="anthropic/claude-sonnet-4-20250514", api_key=ANTHROPIC_API_KEY, temperature=0.2)

    @staticmethod
    def _parse_json_payload(raw_output: str) -> dict[str, Any]:
        cleaned = raw_output.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()
        return json.loads(cleaned)

    @staticmethod
    def _safe_float(value: Any) -> float | None:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

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
            "AVOID": -1,
            "STRONG SELL": -2,
        }
        return mapping.get((signal or "").upper(), 0)

    def _prepare_context(self, context: dict[str, Any]) -> dict[str, Any]:
        stock_data = dict(context.get("stock_data", {}))
        stock_data.pop("raw_df", None)
        stock_data.pop("hourly_df", None)
        return {
            "stock_data": stock_data,
            "technical": context.get("technical", {}),
            "fundamental": context.get("fundamental", {}),
            "sentiment": context.get("sentiment", {}),
            "management": context.get("management", {}),
        }

    def _heuristic_synthesis(self, ticker: str, context: dict[str, Any]) -> dict[str, Any]:
        stock_data = context.get("stock_data", {})
        technical = context.get("technical", {})
        fundamental = context.get("fundamental", {})
        sentiment = context.get("sentiment", {})
        management = context.get("management", {})

        current_price = self._safe_float(stock_data.get("current_price")) or 0.0
        entry_price = self._safe_float(technical.get("entry_price")) or current_price
        stop_loss = self._safe_float(technical.get("stop_loss")) or round(entry_price * 0.95, 2)
        target_1 = self._safe_float(technical.get("target_1")) or round(entry_price * 1.05, 2)
        target_2 = self._safe_float(technical.get("target_2")) or round(entry_price * 1.1, 2)
        target_3 = self._safe_float(technical.get("target_3")) or round(entry_price * 1.18, 2)

        agent_signal_map = {
            "technical": technical.get("technical_signal", "NEUTRAL"),
            "fundamental": fundamental.get("fundamental_signal", "HOLD"),
            "sentiment": sentiment.get("sentiment_signal", "HOLD"),
            "management": management.get("management_signal", "HOLD"),
        }

        numeric_score = sum(self._signal_to_numeric(signal) for signal in agent_signal_map.values())
        bullish = sum(1 for signal in agent_signal_map.values() if self._signal_to_numeric(signal) > 0)
        neutral = sum(1 for signal in agent_signal_map.values() if self._signal_to_numeric(signal) == 0)
        bearish = sum(1 for signal in agent_signal_map.values() if self._signal_to_numeric(signal) < 0)

        if numeric_score >= 4:
            long_term_signal = "STRONG BUY"
        elif numeric_score >= 2:
            long_term_signal = "BUY"
        elif numeric_score >= 1:
            long_term_signal = "ACCUMULATE"
        elif numeric_score <= -4:
            long_term_signal = "SELL"
        elif numeric_score <= -2:
            long_term_signal = "REDUCE"
        else:
            long_term_signal = "HOLD"

        short_term_signal = technical.get("technical_signal", "WAIT")
        if short_term_signal not in {"STRONG BUY", "BUY", "WAIT", "AVOID", "SELL"}:
            short_term_signal = "WAIT"

        valuation_label = fundamental.get("valuation_label", "FAIRLY VALUED")
        if valuation_label == "UNDERVALUED":
            entry_strategy = f"Strong buy at current levels around ₹{current_price:.2f}; consider SIP in 3 tranches over 3 months."
        elif valuation_label == "OVERVALUED":
            entry_strategy = f"Wait for pullback to ₹{entry_price:.2f} before fresh entry."
        else:
            entry_strategy = f"Current price near fair value; start with staggered entries around ₹{entry_price:.2f}."

        intrinsic_value = self._safe_float(fundamental.get("intrinsic_value"))
        upside_pct = self._safe_float(fundamental.get("upside_pct"))
        one_year_target = intrinsic_value if intrinsic_value else target_2
        three_year_target = round(one_year_target * 1.35, 2)
        cagr = round((((three_year_target / current_price) ** (1 / 3)) - 1) * 100, 2) if current_price else 0.0

        long_term_confidence = max(35, min(90, 50 + numeric_score * 8 + bullish * 4 - bearish * 3))
        short_term_confidence = max(30, min(90, 45 + (2 if short_term_signal in {"BUY", "STRONG BUY"} else -2 if short_term_signal in {"SELL", "AVOID"} else 0) * 10))

        risk_rating = "LOW" if bearish == 0 else "MEDIUM" if bearish <= 1 else "HIGH" if bearish == 2 else "VERY HIGH"

        key_risks = list(dict.fromkeys((fundamental.get("concerns") or []) + (management.get("red_flags") or []) + ["Volatility risk near key support"]))[:5]
        key_catalysts = list(
            dict.fromkeys(
                [c.get("event") for c in management.get("upcoming_catalysts", []) if c.get("event")]
                + (sentiment.get("key_events") or [])
                + ["Improving technical momentum"]
            )
        )[:5]

        risk_amount = max(entry_price - stop_loss, 0.01)
        reward_amount = max(target_2 - entry_price, 0.01)
        rr_ratio = round(reward_amount / risk_amount, 2)

        technical_score = min(10, max(1, int((self._signal_to_numeric(technical.get("technical_signal")) + 2) * 2.5)))
        fundamental_score = min(10, max(1, int((self._signal_to_numeric(fundamental.get("fundamental_signal")) + 2) * 2.5)))
        sentiment_score = min(10, max(1, int((self._signal_to_numeric(sentiment.get("sentiment_signal")) + 2) * 2.5)))
        management_score = min(10, max(1, int((self._signal_to_numeric(management.get("management_signal")) + 2) * 2.5)))
        valuation_score = 8 if valuation_label == "UNDERVALUED" else 5 if valuation_label == "FAIRLY VALUED" else 3

        overall_score_out_of_50 = technical_score + fundamental_score + sentiment_score + management_score + valuation_score

        return {
            "ticker": ticker,
            "long_term_analysis": {
                "signal": long_term_signal,
                "entry_strategy": entry_strategy,
                "targets": {
                    "1_year_target": round(one_year_target, 2),
                    "3_year_target": round(three_year_target, 2),
                    "expected_cagr_pct": cagr,
                },
                "stop_loss_exit_triggers": [
                    "Exit if quarterly revenue growth falls below 5% for 2 consecutive quarters",
                    "Exit if debt-to-equity rises above 2.0",
                    "Exit if promoter holding trend turns persistently decreasing",
                    f"Exit if price closes below ₹{stop_loss:.2f} on monthly chart",
                ],
                "risk_rating": risk_rating,
                "key_risks": key_risks,
                "key_catalysts": key_catalysts,
                "confidence_pct": long_term_confidence,
            },
            "short_term_analysis": {
                "signal": short_term_signal,
                "entry": f"Enter near ₹{entry_price:.2f}; enter on breakout above ₹{target_1:.2f} with volume confirmation.",
                "targets": {
                    "target_1": target_1,
                    "target_2": target_2,
                    "target_3": target_3,
                },
                "stop_loss": {
                    "hard_stop": stop_loss,
                    "reasoning": "ATR-based stop to protect downside if support fails on closing basis.",
                },
                "risk_reward_ratio": f"Risk ₹{risk_amount:.2f}, Reward ₹{reward_amount:.2f} = 1:{rr_ratio}",
                "trade_setup_type": "Pullback" if entry_price < current_price else "Breakout",
                "time_in_trade": "2-8 weeks",
                "confidence_pct": short_term_confidence,
            },
            "overall_composite_score": {
                "technical_score": technical_score,
                "fundamental_score": fundamental_score,
                "sentiment_score": sentiment_score,
                "management_score": management_score,
                "valuation_score": valuation_score,
                "overall_score_out_of_50": overall_score_out_of_50,
                "overall_score_out_of_10": round(overall_score_out_of_50 / 5, 1),
            },
            "signal_agreement": f"{bullish}/4 agents bullish, {neutral} neutral, {bearish} bearish",
            "key_decision_summary": [
                f"Strongest bullish factor: technical setup with {technical.get('signal_score', 'mixed momentum')}",
                f"Second bullish factor: valuation appears {valuation_label.lower()} with estimated upside {upside_pct}%",
                f"Main risk to watch: {(key_risks[0] if key_risks else 'execution and market volatility')}",
                f"Entry recommendation: build positions near ₹{entry_price:.2f} with staged deployment",
                f"Exit strategy: respect hard stop at ₹{stop_loss:.2f} and review on deteriorating fundamentals",
            ],
            "agent_signals": agent_signal_map,
        }

    def synthesize(self, ticker: str, context: dict[str, Any]) -> dict[str, Any]:
        compact_context = self._prepare_context(context)
        heuristic = self._heuristic_synthesis(ticker=ticker, context=compact_context)

        if self.llm is None:
            return heuristic

        analyst_agent = Agent(
            role="Senior Indian Equity Strategist",
            goal="Generate robust long-term and short-term recommendations from multi-agent evidence",
            backstory=(
                "You synthesize technical, fundamental, sentiment, and management inputs into actionable Indian stock decisions "
                "with explicit entry, targets, risks, and confidence."
            ),
            llm=self.llm,
            verbose=False,
        )

        description = (
            "Use the provided JSON context and return valid JSON only with keys: long_term_analysis, short_term_analysis, "
            "overall_composite_score, signal_agreement, key_decision_summary (exactly 5 bullet strings), agent_signals. "
            "Long-term signal must be one of STRONG BUY/BUY/ACCUMULATE/HOLD/REDUCE/SELL. "
            "Short-term signal must be one of STRONG BUY/BUY/WAIT/AVOID/SELL."
            f"\n\nTicker: {ticker}\nContext: {json.dumps(compact_context, default=str)}"
        )
        task = Task(description=description, expected_output="Valid JSON only.", agent=analyst_agent)

        try:
            crew = Crew(agents=[analyst_agent], tasks=[task], verbose=False)
            output = crew.kickoff()
            payload = self._parse_json_payload(str(output))
            if not isinstance(payload, dict) or "long_term_analysis" not in payload:
                return heuristic
            payload.setdefault("ticker", ticker)
            payload.setdefault("agent_signals", heuristic.get("agent_signals", {}))
            if not isinstance(payload.get("key_decision_summary"), list) or len(payload["key_decision_summary"]) != 5:
                payload["key_decision_summary"] = heuristic["key_decision_summary"]
            return payload
        except Exception:
            return heuristic
