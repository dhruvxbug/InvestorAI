from __future__ import annotations

from datetime import datetime
from pathlib import Path

from models.report_schema import StockReport


def report_to_markdown(report: StockReport) -> str:
    lines = [
        f"# Stock Analysis Report — {report.ticker} ({report.company_name})",
        "",
        f"**Analysis Date:** {report.analysis_date}",
        f"**Current Price:** ₹{report.current_price:.2f}",
        f"**Long-Term Signal:** {report.long_term.signal.value}",
        f"**Short-Term Signal:** {report.short_term.signal.value}",
        f"**Overall Score:** {report.scores.overall}/10",
        "",
        "## Key Decision Summary",
    ]
    lines.extend([f"- {item}" for item in report.key_summary])

    lines.extend(
        [
            "",
            "## Long-Term Recommendation",
            f"- **Signal:** {report.long_term.signal.value}",
            f"- **Entry Price:** ₹{report.long_term.entry_price:.2f}",
            f"- **Entry Condition:** {report.long_term.entry_condition}",
            f"- **1-Year Target:** ₹{report.long_term.target_1_year:.2f}",
            f"- **3-Year Target:** ₹{report.long_term.target_3_year:.2f}",
            f"- **Expected CAGR:** {report.long_term.expected_cagr:.2f}%",
            f"- **Stop Loss:** ₹{report.long_term.stop_loss_price:.2f}",
            f"- **Risk Level:** {report.long_term.risk_level.value}",
            f"- **Confidence:** {report.long_term.confidence_pct}%",
            "",
            "### Exit Triggers",
        ]
    )
    lines.extend([f"- {item}" for item in report.long_term.exit_triggers])

    lines.extend(["", "### Key Risks"])
    lines.extend([f"- {item}" for item in report.long_term.key_risks])
    lines.extend(["", "### Key Catalysts"])
    lines.extend([f"- {item}" for item in report.long_term.key_catalysts])

    lines.extend(
        [
            "",
            "## Short-Term Recommendation",
            f"- **Signal:** {report.short_term.signal.value}",
            f"- **Entry Price:** ₹{report.short_term.entry_price:.2f}",
            f"- **Entry Condition:** {report.short_term.entry_condition}",
            f"- **Target 1:** ₹{report.short_term.target_1:.2f}",
            f"- **Target 2:** ₹{report.short_term.target_2:.2f}",
            f"- **Target 3:** ₹{report.short_term.target_3:.2f}" if report.short_term.target_3 is not None else "- **Target 3:** N/A",
            f"- **Stop Loss:** ₹{report.short_term.stop_loss:.2f}",
            f"- **Risk/Reward:** {report.short_term.risk_reward_ratio}",
            f"- **Trade Setup:** {report.short_term.trade_setup_type}",
            f"- **Holding Period:** {report.short_term.holding_period}",
            f"- **Confidence:** {report.short_term.confidence_pct}%",
            "",
            "## Composite Scores",
            f"- Technical: {report.scores.technical}/10",
            f"- Fundamental: {report.scores.fundamental}/10",
            f"- Sentiment: {report.scores.sentiment}/10",
            f"- Management: {report.scores.management}/10",
            f"- Valuation: {report.scores.valuation}/10",
            f"- Overall: {report.scores.overall}/10",
            f"- Agent Agreement: {report.scores.agents_bullish} bullish, {report.scores.agents_neutral} neutral, {report.scores.agents_bearish} bearish",
        ]
    )

    return "\n".join(lines)


def save_report_files(report: StockReport, markdown: str, output_dir: Path) -> tuple[Path, Path]:
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    stem = f"{report.ticker}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path
