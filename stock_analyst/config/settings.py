from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
DEFAULT_EXCHANGE = os.getenv("DEFAULT_EXCHANGE", ".NS")
REPORTS_DIR = BASE_DIR / "reports"

HISTORICAL_PERIOD = "1y"
HISTORICAL_INTERVAL = "1d"
NEWS_RESULT_COUNT = 8
MANAGEMENT_RESULT_COUNT = 5


def normalize_ticker(ticker: str) -> str:
    cleaned = ticker.strip().upper()
    if not cleaned:
        raise ValueError("Ticker cannot be empty")
    if cleaned.endswith(".NS") or cleaned.endswith(".BO"):
        return cleaned
    return f"{cleaned}{DEFAULT_EXCHANGE}"


def validate_environment() -> dict[str, Any]:
    return {
        "anthropic_key_configured": bool(ANTHROPIC_API_KEY),
        "exa_key_configured": bool(EXA_API_KEY),
        "reports_dir": str(REPORTS_DIR),
    }
