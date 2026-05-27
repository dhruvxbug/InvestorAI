from __future__ import annotations

import json
from typing import Any

from utils.llm_client import LLMClient, build_llm_client

RED_TEAM_SYSTEM = """
You are a forensic equity red-team analyst. Your job is to challenge bullish narratives,
find failure modes, and highlight thesis-breakers.

Rules:
- Be specific and use concrete figures/signals from the provided agent outputs.
- Focus on downside and permanent capital loss risk.
- If data is missing, explicitly state the uncertainty.
- Return strict JSON only.
""".strip()

RED_TEAM_TEMPLATE = """
Challenge this stock thesis for {ticker} ({company_name}).

Technical output:
{technical_json}

Fundamental output:
{fundamental_json}

Sentiment output:
{sentiment_json}

Management output:
{management_json}

Return JSON with keys:
- contradiction_map: array of strings
- failure_scenarios: array of exactly 5 concise downside scenarios
- thesis_killers: array of top 3 hard invalidation points
- dominant_risks: array of top 5 risks ordered by impact
- confidence_breakdown: object with technical, fundamental, sentiment, management (LOW|MEDIUM|HIGH)
""".strip()


class RedTeamAgent:
    name = "Red Team Agent"

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm_client = llm_client or build_llm_client()

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
    def _fallback(context: dict[str, Any]) -> dict[str, Any]:
        technical = context.get("technical", {})
        fundamental = context.get("fundamental", {})
        sentiment = context.get("sentiment", {})
        management = context.get("management", {})

        contradictions: list[str] = []
        if technical.get("technical_signal") in {"BUY", "STRONG BUY"} and fundamental.get(
            "valuation_label"
        ) == "OVERVALUED":
            contradictions.append(
                "Technical momentum is bullish while valuation appears stretched versus sector."
            )
        if sentiment.get("sentiment_signal") == "BUY" and management.get(
            "management_signal"
        ) == "SELL":
            contradictions.append(
                "Positive news flow conflicts with management/governance risk signals."
            )

        risks = list(dict.fromkeys((fundamental.get("concerns") or []) + (management.get("red_flags") or [])))
        if not risks:
            risks = [
                "Earnings miss risk",
                "Multiple de-rating risk",
                "Governance visibility risk",
                "Sector slowdown risk",
                "High volatility risk",
            ]

        return {
            "contradiction_map": contradictions[:5],
            "failure_scenarios": [
                "Revenue growth decelerates while valuation remains expensive.",
                "Margin compression drives EPS miss and sharp de-rating.",
                "Promoter/insider selling accelerates and sentiment reverses.",
                "Technical breakdown below major support triggers momentum unwind.",
                "Regulatory or governance issue invalidates forward guidance.",
            ],
            "thesis_killers": risks[:3],
            "dominant_risks": risks[:5],
            "confidence_breakdown": {
                "technical": "MEDIUM",
                "fundamental": "MEDIUM",
                "sentiment": "LOW",
                "management": "MEDIUM",
            },
        }

    def analyze(
        self, ticker: str, company_name: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        fallback = self._fallback(context)
        if self.llm_client is None:
            return fallback

        prompt = RED_TEAM_TEMPLATE.format(
            ticker=ticker,
            company_name=company_name,
            technical_json=json.dumps(context.get("technical", {}), default=str),
            fundamental_json=json.dumps(context.get("fundamental", {}), default=str),
            sentiment_json=json.dumps(context.get("sentiment", {}), default=str),
            management_json=json.dumps(context.get("management", {}), default=str),
        )
        try:
            output = self.llm_client.complete(
                system=RED_TEAM_SYSTEM,
                user=prompt,
                max_tokens=1800,
                temperature=0.1,
            )
            parsed = self._extract_json(output)
            if not parsed:
                return fallback
            parsed.setdefault("contradiction_map", fallback["contradiction_map"])
            parsed.setdefault("failure_scenarios", fallback["failure_scenarios"])
            parsed.setdefault("thesis_killers", fallback["thesis_killers"])
            parsed.setdefault("dominant_risks", fallback["dominant_risks"])
            parsed.setdefault("confidence_breakdown", fallback["confidence_breakdown"])
            return parsed
        except Exception:
            return fallback
