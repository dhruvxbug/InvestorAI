from __future__ import annotations

import argparse
import asyncio
import sys

from agents.orchestrator import AnalysisOrchestrator
from config.settings import validate_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Agent Indian Stock Analysis System")
    parser.add_argument("ticker", help="Ticker symbol (e.g., RELIANCE, HDFCBANK, INFY)")
    return parser.parse_args()


async def _run(ticker: str) -> int:
    env = validate_environment()
    if not env["anthropic_key_configured"]:
        print("Error: ANTHROPIC_API_KEY is missing in stock_analyst/.env")
        return 1
    if not env["exa_key_configured"]:
        print("Error: EXA_API_KEY is missing in stock_analyst/.env")
        return 1

    orchestrator = AnalysisOrchestrator()
    report, json_path, md_path = await orchestrator.analyze_stock(ticker)

    print(f"\nTicker: {report.ticker}")
    print(f"Signal: {report.recommendation.signal}")
    print(f"Confidence: {report.recommendation.confidence}%")
    print(f"Entry: {report.recommendation.trade_levels.entry_price}")
    print(f"Target: {report.recommendation.trade_levels.target_price}")
    print(f"Stop Loss: {report.recommendation.trade_levels.stop_loss}")
    print(f"JSON Report: {json_path}")
    print(f"Markdown Report: {md_path}")
    return 0


def main() -> None:
    args = parse_args()
    code = asyncio.run(_run(args.ticker))
    sys.exit(code)


if __name__ == "__main__":
    main()
