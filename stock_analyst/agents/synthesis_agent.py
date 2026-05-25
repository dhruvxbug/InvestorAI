from __future__ import annotations

import json
from typing import Any

from crewai import Agent, LLM, Task

from config.settings import ANTHROPIC_API_KEY, ANTHROPIC_MODEL
from models.report_schema import FinalRecommendation, TradeLevels


class SynthesisAgent:
    def __init__(self) -> None:
        if not ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is not configured")
        self.llm = LLM(model=f"anthropic/{ANTHROPIC_MODEL}", api_key=ANTHROPIC_API_KEY, temperature=0.2)

    def _build_agent(self) -> Agent:
        return Agent(
            role="Senior Indian Equity Strategist",
            goal="Generate precise stock recommendations with clear levels and risk-aware reasoning",
            backstory=(
                "You are a professional NSE/BSE strategist. You combine technicals, fundamentals, news sentiment, "
                "and management quality to create strict BUY/SELL/HOLD recommendations with explicit trade levels."
            ),
            llm=self.llm,
            verbose=False,
        )

    def synthesize(self, ticker: str, context: dict[str, Any]) -> FinalRecommendation:
        analyst_agent = self._build_agent()
        description = (
            "Given the JSON context, produce output in strict JSON with keys: "
            "signal (BUY/SELL/HOLD), confidence (0-100 integer), rationale, risks (list), "
            "entry_price (number|null), target_price (number|null), stop_loss (number|null), time_horizon. "
            "Return only valid JSON.\n\n"
            f"Ticker: {ticker}\n"
            f"Context: {json.dumps(context, default=str)}"
        )
        task = Task(
            description=description,
            expected_output="Valid JSON only.",
            agent=analyst_agent,
        )
        output = task.execute_sync()

        payload = json.loads(str(output))
        return FinalRecommendation(
            signal=payload["signal"],
            confidence=int(payload["confidence"]),
            rationale=payload["rationale"],
            risks=payload.get("risks", []),
            trade_levels=TradeLevels(
                entry_price=payload.get("entry_price"),
                target_price=payload.get("target_price"),
                stop_loss=payload.get("stop_loss"),
                time_horizon=payload.get("time_horizon", "Swing (2-8 weeks)"),
            ),
        )
