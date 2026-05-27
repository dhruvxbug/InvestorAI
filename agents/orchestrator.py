from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Awaitable, Callable

import yfinance as yf

from agents.fundamental_agent import FundamentalAgent
from agents.management_agent import ManagementAgent
from agents.sentiment_agent import SentimentAgent
from agents.stock_data_agent import StockDataAgent
from agents.synthesis_agent import SynthesisAgent
from agents.technical_agent import TechnicalAgent
from config.settings import REPORTS_DIR, normalize_ticker
from models.report_schema import (
    CompositeScores,
    LongTermRecommendation,
    RiskLevel,
    ShortTermRecommendation,
    SignalType,
    StockReport,
)
from utils.formatter import report_to_markdown, save_report_files
from utils.llm_client import LLMClient, build_llm_client
from utils.logger import get_logger


class StockDataError(Exception):
    pass


class AnalysisOrchestrator:
    def __init__(
        self, provider: str | None = None, model_id: str | None = None
    ) -> None:
        self.logger = get_logger()
        self.stock_agent = StockDataAgent()
        self.technical_agent = TechnicalAgent()
        self.fundamental_agent = FundamentalAgent()
        self.sentiment_agent = SentimentAgent()
        self.management_agent = ManagementAgent()
        _llm = build_llm_client(provider=provider, model_id=model_id)
        self._active_model = _llm.display_name if _llm else "heuristic fallback"
        self.synthesis_agent = SynthesisAgent(llm_client=_llm)

    async def _fetch_info_with_retry(self, ticker: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return await asyncio.to_thread(lambda: yf.Ticker(ticker).info or {})
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self.logger.warning(
                    "yfinance info fetch failed for %s (attempt %s/2): %s",
                    ticker,
                    attempt + 1,
                    exc,
                )
                if attempt == 0:
                    await asyncio.sleep(1)
        raise StockDataError(f"Failed to fetch data for {ticker}: {last_error}")

    async def _resolve_ticker(self, raw_ticker: str) -> tuple[str, dict[str, Any]]:
        cleaned = raw_ticker.strip().upper().replace(" ", "")
        if not cleaned:
            raise StockDataError("Ticker cannot be empty")

        if cleaned.endswith((".NS", ".BO")):
            candidates = [cleaned]
        else:
            candidates = [cleaned, f"{cleaned}.NS", f"{cleaned}.BO"]

        failures: list[str] = []
        for candidate in candidates:
            try:
                info = await self._fetch_info_with_retry(candidate)
                if (
                    info.get("longName")
                    or info.get("shortName")
                    or info.get("regularMarketPrice")
                    or info.get("currentPrice")
                ):
                    return candidate, info
                failures.append(f"{candidate}: no market metadata")
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{candidate}: {exc}")

        raise StockDataError(
            f"Ticker validation failed for '{raw_ticker}'. Tried {candidates}. Details: {' | '.join(failures)}"
        )

    async def _run_with_timeout(
        self,
        agent_name: str,
        func: Callable[..., Any],
        *args: Any,
        exa_fallback: bool = False,
    ) -> dict[str, Any]:
        try:
            result = await asyncio.wait_for(asyncio.to_thread(func, *args), timeout=60)
            return result if isinstance(result, dict) else {}
        except asyncio.TimeoutError:
            self.logger.error("%s timed out after 60 seconds", agent_name)
        except Exception as exc:  # noqa: BLE001
            if exa_fallback:
                self.logger.warning(
                    "%s failed (continuing with empty data): %s", agent_name, exc
                )
            else:
                self.logger.error("%s failed: %s", agent_name, exc)
        return {}

    async def _run_synthesis_with_retry(
        self, ticker: str, company_name: str, context: dict[str, Any]
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(2):
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(
                        self.synthesis_agent.synthesize, ticker, company_name, context
                    ),
                    timeout=60,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self.logger.warning(
                    "Claude synthesis failed (attempt %s/2): %s", attempt + 1, exc
                )
                if attempt == 0:
                    await asyncio.sleep(2**attempt)
        self.logger.error("Claude synthesis failed after retries: %s", last_error)
        return {}

    @staticmethod
    def _to_signal(value: str | None, default: SignalType) -> SignalType:
        normalized = (value or default.value).upper()
        aliases = {
            "NEUTRAL": "HOLD",
            "STRONG SELL": "SELL",
        }
        normalized = aliases.get(normalized, normalized)
        try:
            return SignalType(normalized)
        except ValueError:
            return default

    @staticmethod
    def _to_risk(value: str | None) -> RiskLevel:
        normalized = (value or "MEDIUM").upper()
        try:
            return RiskLevel(normalized)
        except ValueError:
            return RiskLevel.MEDIUM

    @staticmethod
    def _to_float(value: Any, default: float = 0.0) -> float:
        try:
            if value is None:
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    def _build_report(
        self,
        ticker: str,
        company_name: str,
        current_price: float,
        technical: dict[str, Any],
        fundamental: dict[str, Any],
        sentiment: dict[str, Any],
        management: dict[str, Any],
        synthesis: dict[str, Any],
    ) -> StockReport:
        long_data = (
            synthesis.get("long_term", {})
            if isinstance(synthesis.get("long_term"), dict)
            else {}
        )
        short_data = (
            synthesis.get("short_term", {})
            if isinstance(synthesis.get("short_term"), dict)
            else {}
        )
        score_data = (
            synthesis.get("scores", {})
            if isinstance(synthesis.get("scores"), dict)
            else {}
        )

        default_entry = self._to_float(technical.get("entry_price"), current_price)
        default_sl = self._to_float(
            technical.get("stop_loss"), round(default_entry * 0.95, 2)
        )
        default_t1 = self._to_float(
            technical.get("target_1"), round(default_entry * 1.05, 2)
        )
        default_t2 = self._to_float(
            technical.get("target_2"), round(default_entry * 1.1, 2)
        )
        default_t3 = self._to_float(
            technical.get("target_3"), round(default_entry * 1.2, 2)
        )

        summary = (
            synthesis.get("key_summary")
            if isinstance(synthesis.get("key_summary"), list)
            else []
        )
        summary = [str(item) for item in summary][:5]
        while len(summary) < 5:
            filler = [
                f"Technical signal: {technical.get('technical_signal', 'N/A')}",
                f"Fundamental signal: {fundamental.get('fundamental_signal', 'N/A')}",
                f"Sentiment signal: {sentiment.get('sentiment_signal', 'N/A')}",
                f"Management signal: {management.get('management_signal', 'N/A')}",
                f"Current price reference: ₹{current_price:.2f}",
            ]
            summary.append(filler[len(summary)])

        long_term = LongTermRecommendation(
            signal=self._to_signal(long_data.get("signal"), SignalType.HOLD),
            entry_price=self._to_float(long_data.get("entry_price"), default_entry),
            entry_condition=str(
                long_data.get("entry_condition")
                or f"Accumulate near ₹{default_entry:.2f}"
            ),
            target_1_year=self._to_float(long_data.get("target_1_year"), default_t2),
            target_3_year=self._to_float(
                long_data.get("target_3_year"), max(default_t2, default_t3)
            ),
            expected_cagr=self._to_float(long_data.get("expected_cagr"), 12.0),
            stop_loss_price=self._to_float(
                long_data.get("stop_loss_price"), default_sl
            ),
            exit_triggers=[
                str(x)
                for x in (
                    long_data.get("exit_triggers")
                    or [f"Exit if monthly close below ₹{default_sl:.2f}"]
                )
            ],
            risk_level=self._to_risk(long_data.get("risk_level")),
            key_risks=[str(x) for x in (long_data.get("key_risks") or ["Market risk"])][
                :5
            ],
            key_catalysts=[
                str(x)
                for x in (long_data.get("key_catalysts") or ["Earnings surprise"])
            ][:5],
            confidence_pct=int(
                max(0, min(100, self._to_float(long_data.get("confidence_pct"), 55)))
            ),
            reasoning=str(
                long_data.get("reasoning")
                or "Balanced long-term view based on multi-agent synthesis."
            ),
        )

        short_term = ShortTermRecommendation(
            signal=self._to_signal(short_data.get("signal"), SignalType.WAIT),
            entry_price=self._to_float(short_data.get("entry_price"), default_entry),
            entry_condition=str(
                short_data.get("entry_condition")
                or f"Enter if price holds above ₹{default_entry:.2f}"
            ),
            target_1=self._to_float(short_data.get("target_1"), default_t1),
            target_2=self._to_float(short_data.get("target_2"), default_t2),
            target_3=self._to_float(short_data.get("target_3"), default_t3),
            stop_loss=self._to_float(short_data.get("stop_loss"), default_sl),
            risk_reward_ratio=str(short_data.get("risk_reward_ratio") or "1:2"),
            trade_setup_type=str(short_data.get("trade_setup_type") or "Positional"),
            holding_period=str(short_data.get("holding_period") or "1-12 weeks"),
            confidence_pct=int(
                max(0, min(100, self._to_float(short_data.get("confidence_pct"), 50)))
            ),
            reasoning=str(
                short_data.get("reasoning")
                or "Short-term setup based on momentum and support/resistance."
            ),
        )

        scores = CompositeScores(
            technical=self._to_float(score_data.get("technical"), 5.0),
            fundamental=self._to_float(score_data.get("fundamental"), 5.0),
            sentiment=self._to_float(score_data.get("sentiment"), 5.0),
            management=self._to_float(score_data.get("management"), 5.0),
            valuation=self._to_float(score_data.get("valuation"), 5.0),
            overall=self._to_float(score_data.get("overall"), 5.0),
            agents_bullish=int(self._to_float(score_data.get("agents_bullish"), 0.0)),
            agents_bearish=int(self._to_float(score_data.get("agents_bearish"), 0.0)),
            agents_neutral=int(self._to_float(score_data.get("agents_neutral"), 0.0)),
        )

        return StockReport(
            ticker=ticker,
            company_name=company_name,
            current_price=current_price,
            analysis_date=datetime.utcnow().strftime("%Y-%m-%d"),
            key_summary=summary[:5],
            long_term=long_term,
            short_term=short_term,
            scores=scores,
            raw_technical=technical,
            raw_fundamental=fundamental,
            raw_sentiment=sentiment,
            raw_management=management,
        )

    async def analyze_stock(self, raw_ticker: str) -> tuple[StockReport, str, str]:
        normalized_input = normalize_ticker(raw_ticker)
        ticker, info = await self._resolve_ticker(normalized_input)
        company_name = str(info.get("longName") or info.get("shortName") or ticker)
        sector_name = str(info.get("sector") or "Indian equity")
        current_price = self._to_float(
            info.get("currentPrice") or info.get("regularMarketPrice"), 0.0
        )

        self.logger.info(
            "Running analysis for %s (%s) | model=%s",
            ticker,
            company_name,
            self._active_model,
        )

        stock_task = asyncio.create_task(
            self._run_with_timeout("Stock Data Agent", self.stock_agent.analyze, ticker)
        )

        async def run_technical() -> dict[str, Any]:
            stock_data = await stock_task
            raw_df = stock_data.get("raw_df") if isinstance(stock_data, dict) else None
            if raw_df is None:
                self.logger.error(
                    "Technical Analysis Agent skipped due to missing stock raw_df"
                )
                return {}
            return await self._run_with_timeout(
                "Technical Analysis Agent",
                self.technical_agent.analyze,
                raw_df,
                stock_data.get("week_52_high"),
            )

        technical_task = asyncio.create_task(run_technical())
        fundamental_task = asyncio.create_task(
            self._run_with_timeout(
                "Fundamental Analysis Agent", self.fundamental_agent.analyze, ticker
            )
        )
        sentiment_task = asyncio.create_task(
            self._run_with_timeout(
                "News & Sentiment Agent",
                self.sentiment_agent.analyze,
                ticker,
                company_name,
                sector_name,
                exa_fallback=True,
            )
        )
        management_task = asyncio.create_task(
            self._run_with_timeout(
                "Management & Insider Intelligence Agent",
                self.management_agent.analyze,
                ticker,
                company_name,
                exa_fallback=True,
            )
        )

        (
            stock_data,
            technical,
            fundamental,
            sentiment,
            management,
        ) = await asyncio.gather(
            stock_task,
            technical_task,
            fundamental_task,
            sentiment_task,
            management_task,
        )

        merged_context = {
            "stock_data": stock_data,
            "technical": technical,
            "fundamental": fundamental,
            "sentiment": sentiment,
            "management": management,
        }

        synthesis = await self._run_synthesis_with_retry(
            ticker=ticker, company_name=company_name, context=merged_context
        )
        report = self._build_report(
            ticker=ticker,
            company_name=company_name,
            current_price=self._to_float(
                stock_data.get("current_price"), current_price
            ),
            technical=technical,
            fundamental=fundamental,
            sentiment=sentiment,
            management=management,
            synthesis=synthesis,
        )

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        markdown = report_to_markdown(report)
        json_path, md_path = save_report_files(
            report=report, markdown=markdown, output_dir=REPORTS_DIR
        )
        self.logger.info("Saved report files: %s, %s", json_path, md_path)
        return report, str(json_path), str(md_path)
