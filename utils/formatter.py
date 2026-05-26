"""
Report formatting utilities — matches the Section 10 output format exactly.

Sample output structure:
  # 📈 TICKER — Investment Analysis Report
  **Date:** DD Mon YYYY | **Price:** ₹X,XXX | **Score:** X.X/10
  ---
  ## ⚡ QUICK SUMMARY   (5 bullets with ✅ / ⚠️ / 📍 emoji)
  ## 📅 LONG-TERM VIEW  (table + narrative)
  ## ⚡ SHORT-TERM VIEW (table + narrative)
  ## 📊 SIGNAL SCORECARD (table)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from models.report_schema import StockReport

# ─── Helpers ─────────────────────────────────────────────────────────────────


def _fmt_date(iso: str) -> str:
    """Convert 'YYYY-MM-DD' → '25 May 2025' (cross-platform, no zero-padding)."""
    try:
        dt = datetime.strptime(iso, "%Y-%m-%d")
        return f"{dt.day} {dt.strftime('%b %Y')}"
    except (ValueError, TypeError):
        return iso


def _inr(value: float) -> str:
    """Format a float as an Indian-style ₹ price string (e.g. ₹1,642.00)."""
    return f"₹{value:,.2f}"


def _score_emoji(score: float) -> str:
    if score > 7:
        return "🟢"
    if score >= 4:
        return "🟡"
    return "🔴"


def _score_label(score: float) -> str:
    if score > 7:
        return "Bullish"
    if score >= 4:
        return "Neutral-Positive"
    return "Bearish"


def _bullet_emoji(text: str) -> str:
    """Pick an appropriate emoji for a key-summary bullet."""
    lower = text.lower()
    if any(w in lower for w in ("entry", "buy at", "accumulate at")):
        return "📍"
    if any(w in lower for w in ("exit", "stop", "stop-loss", "sl")):
        return "📍"
    if any(
        w in lower
        for w in (
            "risk",
            "watch",
            "concern",
            "headwind",
            "pressure",
            "weak",
            "negative",
            "caution",
            "beware",
        )
    ):
        return "⚠️"
    return "✅"


# ─── Main formatter ───────────────────────────────────────────────────────────


def report_to_markdown(report: StockReport) -> str:  # noqa: PLR0914
    lt = report.long_term
    short = report.short_term
    sc = report.scores

    lines: list[str] = []

    # ── Header ──────────────────────────────────────────────────────────────
    lines += [
        f"# 📈 {report.ticker} — Investment Analysis Report",
        "",
        (
            f"**Date:** {_fmt_date(report.analysis_date)}"
            f" | **Price:** {_inr(report.current_price)}"
            f" | **Score:** {sc.overall:.1f}/10"
        ),
        "",
        "---",
        "",
    ]

    # ── Quick Summary ────────────────────────────────────────────────────────
    lines.append("## ⚡ QUICK SUMMARY")
    for item in report.key_summary:
        emoji = _bullet_emoji(item)
        lines.append(f"- {emoji} {item}")
    lines += ["", "---", ""]

    # ── Long-Term View ───────────────────────────────────────────────────────
    lines += [
        "## 📅 LONG-TERM VIEW (1–3 Years)",
        f"**Signal: {lt.signal.value}** | Confidence: {lt.confidence_pct}%",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Recommended Entry | {_inr(lt.entry_price)} |",
        f"| 1-Year Target | {_inr(lt.target_1_year)} |",
        f"| 3-Year Target | {_inr(lt.target_3_year)} |",
        f"| Expected CAGR | {lt.expected_cagr:.1f}% |",
        f"| Stop Loss / Exit Trigger | Below {_inr(lt.stop_loss_price)} on monthly close |",
        f"| Risk Level | {lt.risk_level.value} |",
        "",
        f"**Entry Condition:** {lt.entry_condition}",
        "",
        "**Exit if:**",
    ]
    for trigger in lt.exit_triggers:
        lines.append(f"- {trigger}")

    lines += ["", "**Key Catalysts:**"]
    for catalyst in lt.key_catalysts:
        lines.append(f"- {catalyst}")

    lines += ["", "**Key Risks:**"]
    for risk in lt.key_risks:
        lines.append(f"- {risk}")

    if lt.reasoning:
        lines += ["", f"*{lt.reasoning}*"]

    lines += ["", "---", ""]

    # ── Short-Term View ──────────────────────────────────────────────────────
    lines += [
        "## ⚡ SHORT-TERM VIEW (2–6 Weeks)",
        f"**Signal: {short.signal.value}** | Confidence: {short.confidence_pct}%",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Entry Price | {_inr(short.entry_price)} |",
        f"| Entry Condition | {short.entry_condition} |",
        f"| Target 1 | {_inr(short.target_1)} (book 40%) |",
        f"| Target 2 | {_inr(short.target_2)} (book 40%) |",
    ]
    if short.target_3 is not None:
        lines.append(f"| Target 3 | {_inr(short.target_3)} (trail remaining 20%) |")

    lines += [
        f"| Stop Loss | {_inr(short.stop_loss)} (hard stop — exit 100%) |",
        f"| Risk/Reward | {short.risk_reward_ratio} |",
        f"| Setup Type | {short.trade_setup_type} |",
        f"| Expected Hold | {short.holding_period} |",
        "",
        f"**Why this setup:** {short.reasoning}",
        "",
        "---",
        "",
    ]

    # ── Signal Scorecard ─────────────────────────────────────────────────────
    overall_signal_str = (
        f"🟢 **{lt.signal.value}**"
        if sc.overall > 7
        else (
            f"🟡 **{lt.signal.value}**"
            if sc.overall >= 4
            else f"🔴 **{lt.signal.value}**"
        )
    )
    total_agents = sc.agents_bullish + sc.agents_bearish + sc.agents_neutral

    lines += [
        "## 📊 SIGNAL SCORECARD",
        "| Dimension | Score | Signal |",
        "|---|---|---|",
        f"| Technical | {sc.technical:.1f}/10 | {_score_emoji(sc.technical)} {_score_label(sc.technical)} |",
        f"| Fundamental | {sc.fundamental:.1f}/10 | {_score_emoji(sc.fundamental)} {_score_label(sc.fundamental)} |",
        f"| Sentiment | {sc.sentiment:.1f}/10 | {_score_emoji(sc.sentiment)} {_score_label(sc.sentiment)} |",
        f"| Management | {sc.management:.1f}/10 | {_score_emoji(sc.management)} {_score_label(sc.management)} |",
        f"| Valuation | {sc.valuation:.1f}/10 | {_score_emoji(sc.valuation)} {_score_label(sc.valuation)} |",
        f"| **OVERALL** | **{sc.overall:.1f}/10** | {overall_signal_str} |",
        "",
        f"Signal Agreement: **{sc.agents_bullish}/{total_agents} agents bullish,"
        f" {sc.agents_neutral} neutral**",
        "",
    ]

    return "\n".join(lines)


# ─── File persistence ─────────────────────────────────────────────────────────


def save_report_files(
    report: StockReport,
    markdown: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist JSON + Markdown reports and return their paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d")
    stem = f"{report.ticker}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path
