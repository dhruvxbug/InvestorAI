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
from models.report_schema import AgentSection, StockAnalysisReport
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

    async def _run_parallel(self, ticker: str, company_name: str) -> dict[str, Any]:
        return {
            "technical": await asyncio.to_thread(self.technical_agent.analyze, ticker),
            "fundamental": await asyncio.to_thread(self.fundamental_agent.analyze, ticker),
            "sentiment": await asyncio.to_thread(self.sentiment_agent.analyze, ticker, company_name),
            "management": await asyncio.to_thread(self.management_agent.analyze, ticker, company_name),
        }

    async def analyze_stock(self, raw_ticker: str) -> tuple[StockAnalysisReport, str, str]:
        ticker = normalize_ticker(raw_ticker)
        self.logger.info("Running analysis for %s", ticker)

        stock_data = await asyncio.to_thread(self.stock_agent.analyze, ticker)
        company_name = stock_data.get("company_name", ticker)
        parallel_outputs = await self._run_parallel(ticker=ticker, company_name=company_name)

        merged_context = {
            "stock_data": stock_data,
            **parallel_outputs,
        }
        recommendation = await asyncio.to_thread(self.synthesis_agent.synthesize, ticker, merged_context)

        report = StockAnalysisReport(
            ticker=ticker,
            sections=[
                AgentSection(name="Stock Data", summary="Price and volume snapshot", key_points=[], raw_data=stock_data),
                AgentSection(name="Technical", summary="Indicator and trend read", key_points=[], raw_data=parallel_outputs["technical"]),
                AgentSection(name="Fundamental", summary="Valuation and quality signals", key_points=[], raw_data=parallel_outputs["fundamental"]),
                AgentSection(name="Sentiment", summary="Recent media/news sentiment", key_points=[], raw_data=parallel_outputs["sentiment"]),
                AgentSection(name="Management", summary="Leadership and governance signals", key_points=[], raw_data=parallel_outputs["management"]),
            ],
            recommendation=recommendation,
        )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        markdown = report_to_markdown(report)
        json_path, md_path = save_report_files(report=report, markdown=markdown, output_dir=REPORTS_DIR)
        self.logger.info("Saved report files: %s, %s", json_path, md_path)
        return report, str(json_path), str(md_path)
