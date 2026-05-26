from __future__ import annotations

import argparse
import asyncio
import sys

from agents.orchestrator import AnalysisOrchestrator
from config.settings import validate_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Agent Indian Stock Analysis System"
    )
    parser.add_argument("ticker", help="Ticker symbol (e.g., RELIANCE, HDFCBANK, INFY)")
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider: anthropic | openrouter | openai  (auto-detected from .env if omitted)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model ID, e.g. claude-sonnet-4-20250514, openai/gpt-4o  (uses provider default if omitted)",
    )
    return parser.parse_args()


async def _run(
    ticker: str, provider: str | None = None, model_id: str | None = None
) -> int:
    env = validate_environment()
    has_llm = (
        env["anthropic_key_configured"]
        or env["openrouter_key_configured"]
        or env["openai_key_configured"]
    )
    if not has_llm:
        print(
            "Error: No LLM API key found. Set ANTHROPIC_API_KEY, OPENROUTER_API_KEY, or OPENAI_API_KEY in .env"
        )
        return 1
    if not env["exa_key_configured"]:
        print(
            "Warning: EXA_API_KEY not set — news/sentiment agents will return empty data"
        )

    orchestrator = AnalysisOrchestrator(provider=provider, model_id=model_id)
    report, json_path, md_path = await orchestrator.analyze_stock(ticker)

    print(f"\nTicker: {report.ticker}")
    print(f"Company: {report.company_name}")
    print(f"Current Price: ₹{report.current_price:.2f}")
    print(f"Long-Term Signal: {report.long_term.signal.value}")
    print(f"Long-Term Confidence: {report.long_term.confidence_pct}%")
    print(f"Short-Term Signal: {report.short_term.signal.value}")
    print(f"Short-Term Entry: ₹{report.short_term.entry_price:.2f}")
    print(f"Short-Term Stop Loss: ₹{report.short_term.stop_loss:.2f}")
    print(f"JSON Report: {json_path}")
    print(f"Markdown Report: {md_path}")
    print(f"Model used: {orchestrator._active_model}")
    return 0


def main() -> None:
    args = parse_args()
    code = asyncio.run(_run(args.ticker, provider=args.provider, model_id=args.model))
    sys.exit(code)


if __name__ == "__main__":
    main()
