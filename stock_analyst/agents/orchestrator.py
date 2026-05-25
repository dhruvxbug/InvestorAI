from __future__ import annotations

import asyncio
from typing import Any

from agents.fundamental_agent import FundamentalAgent
from agents.management_agent import ManagementAgent
from agents.sentiment_agent import SentimentAgent
from agents.stock_data_agent import StockDataAgent
from agents.synthesis_agent import SynthesisAgent
from agents.technical_agent import TechnicalAgent
from config.settings import REPORTS_DIR, normalize_ticker
from models.report_schema import AgentSection, FinalRecommendation, StockAnalysisReport, TradeLevels
from utils.formatter import report_to_markdown, save_report_files
from utils.logger import get_logger


class AnalysisOrchestrator:
    def __init__(self) -> None:
        self.logger = get_logger()
        self.stock_agent = StockDataAgent()
        self.technical_agent = TechnicalAgent()
        self.fundamental_agent = FundamentalAgent()
        self.sentiment_agent = SentimentAgent()
        self.management_agent = ManagementAgent()
        self.synthesis_agent = SynthesisAgent()

    async def _run_parallel(self, ticker: str, company_name: str, sector_name: str, stock_data: dict[str, Any]) -> dict[str, Any]:
        technical_task = asyncio.to_thread(
            self.technical_agent.analyze,
            stock_data["raw_df"],
            stock_data.get("week_52_high"),
        )
        fundamental_task = asyncio.to_thread(self.fundamental_agent.analyze, ticker)
        sentiment_task = asyncio.to_thread(self.sentiment_agent.analyze, ticker, company_name, sector_name)
        management_task = asyncio.to_thread(self.management_agent.analyze, ticker, company_name)

        technical, fundamental, sentiment, management = await asyncio.gather(
            technical_task,
            fundamental_task,
            sentiment_task,
            management_task,
        )
        return {
            "technical": technical,
            "fundamental": fundamental,
            "sentiment": sentiment,
            "management": management,
        }

    @staticmethod
    def _build_final_recommendation(synthesis_output: dict[str, Any]) -> FinalRecommendation:
        long_term = synthesis_output.get("long_term_analysis", {})
        short_term = synthesis_output.get("short_term_analysis", {})

        signal_map = {
            "STRONG BUY": "BUY",
            "BUY": "BUY",
            "ACCUMULATE": "BUY",
            "HOLD": "HOLD",
            "REDUCE": "SELL",
            "SELL": "SELL",
        }

        mapped_signal = signal_map.get(str(long_term.get("signal", "HOLD")).upper(), "HOLD")
        confidence = int(long_term.get("confidence_pct", 50) or 50)

        targets = short_term.get("targets", {}) if isinstance(short_term.get("targets"), dict) else {}
        stop_loss_data = short_term.get("stop_loss", {}) if isinstance(short_term.get("stop_loss"), dict) else {}

        return FinalRecommendation(
            signal=mapped_signal,
            confidence=max(0, min(100, confidence)),
            rationale="\n".join(synthesis_output.get("key_decision_summary", [])) or "Composite multi-agent synthesis generated.",
            risks=long_term.get("key_risks", []) if isinstance(long_term.get("key_risks"), list) else [],
            trade_levels=TradeLevels(
                entry_price=TechnicalValueParser.to_float(short_term.get("entry")),
                target_price=TechnicalValueParser.to_float(targets.get("target_2")),
                stop_loss=TechnicalValueParser.to_float(stop_loss_data.get("hard_stop")),
                time_horizon=short_term.get("time_in_trade", "2-8 weeks"),
            ),
        )

    async def analyze_stock(self, raw_ticker: str) -> tuple[StockAnalysisReport, str, str]:
        ticker = normalize_ticker(raw_ticker)
        self.logger.info("Running analysis for %s", ticker)

        stock_data = await asyncio.to_thread(self.stock_agent.analyze, ticker)
        company_name = stock_data.get("company_name", ticker)
        sector_name = stock_data.get("sector") or "Indian equity"

        parallel_outputs = await self._run_parallel(
            ticker=ticker,
            company_name=company_name,
            sector_name=sector_name,
            stock_data=stock_data,
        )

        merged_context = {
            "stock_data": stock_data,
            **parallel_outputs,
        }
        synthesis_output = await asyncio.to_thread(self.synthesis_agent.synthesize, ticker, merged_context)
        recommendation = self._build_final_recommendation(synthesis_output)

        report = StockAnalysisReport(
            ticker=ticker,
            sections=[
                AgentSection(name="Stock Data", summary="Price and volume snapshot", key_points=[], raw_data=stock_data),
                AgentSection(name="Technical", summary="Indicator and trend read", key_points=[], raw_data=parallel_outputs["technical"]),
                AgentSection(name="Fundamental", summary="Valuation and quality signals", key_points=[], raw_data=parallel_outputs["fundamental"]),
                AgentSection(name="Sentiment", summary="Recent media/news sentiment", key_points=[], raw_data=parallel_outputs["sentiment"]),
                AgentSection(name="Management", summary="Leadership and governance signals", key_points=[], raw_data=parallel_outputs["management"]),
                AgentSection(name="Synthesis", summary="Cross-agent long/short horizon recommendation", key_points=[], raw_data=synthesis_output),
            ],
            recommendation=recommendation,
        )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        markdown = report_to_markdown(report)
        json_path, md_path = save_report_files(report=report, markdown=markdown, output_dir=REPORTS_DIR)
        self.logger.info("Saved report files: %s, %s", json_path, md_path)
        return report, str(json_path), str(md_path)


class TechnicalValueParser:
    @staticmethod
    def to_float(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            import re

            match = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
            if match:
                try:
                    return float(match.group(0))
                except ValueError:
                    return None
        return None
