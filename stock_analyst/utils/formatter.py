from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from models.report_schema import StockAnalysisReport


def report_to_markdown(report: StockAnalysisReport) -> str:
    levels = report.recommendation.trade_levels
    lines = [
        f"# Stock Analysis Report — {report.ticker}",
        "",
        f"**Generated At:** {report.generated_at.isoformat()}Z",
        f"**Signal:** {report.recommendation.signal}",
        f"**Confidence:** {report.recommendation.confidence}%",
        "",
        "## Trade Setup",
        f"- **Entry:** {levels.entry_price}",
        f"- **Target:** {levels.target_price}",
        f"- **Stop Loss:** {levels.stop_loss}",
        f"- **Time Horizon:** {levels.time_horizon}",
        "",
        "## Rationale",
        report.recommendation.rationale,
        "",
        "## Key Risks",
    ]
    lines.extend([f"- {risk}" for risk in report.recommendation.risks] or ["- Not specified"])

    for section in report.sections:
        lines.extend(
            [
                "",
                f"## {section.name}",
                section.summary,
                "",
                "```json",
                json.dumps(section.raw_data, indent=2, default=str),
                "```",
            ]
        )

    return "\n".join(lines)


def save_report_files(report: StockAnalysisReport, markdown: str, output_dir: Path) -> tuple[Path, Path]:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    stem = f"{report.ticker}_{timestamp}"
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"

    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    return json_path, md_path
