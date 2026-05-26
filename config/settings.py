from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BASE_DIR / ".env")

# ─── LLM ─────────────────────────────────────────────────────────────────────
ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
DEFAULT_EXCHANGE = os.getenv("DEFAULT_EXCHANGE", ".NS")
REPORTS_DIR = BASE_DIR / "reports"

# ─── Data fetch defaults ──────────────────────────────────────────────────────
HISTORICAL_PERIOD = "5y"
HISTORICAL_INTERVAL = "1d"
NEWS_RESULT_COUNT = 8
MANAGEMENT_RESULT_COUNT = 5
DEFAULT_LOOKBACK_DAYS: int = 365 * 5  # 5 years of daily bars
SHORT_TERM_LOOKBACK_DAYS: int = 90  # 3 months for short-term view

# ─── Risk & valuation constants ───────────────────────────────────────────────
STOP_LOSS_ATR_MULTIPLIER: float = 1.5  # stop-loss = entry - 1.5 × ATR
DCF_DISCOUNT_RATE: float = 0.12  # 12 % WACC for Indian equities
DCF_TERMINAL_GROWTH: float = 0.04  # 4 % long-run GDP terminal growth

# ─── Sector P/E averages (15 major Indian sectors) ────────────────────────────
# Used by FundamentalAgent to label stocks as UNDERVALUED / FAIRLY VALUED / OVERVALUED
SECTOR_PE_AVERAGES: dict[str, float] = {
    "IT": 25.0,
    "INFORMATION TECHNOLOGY": 25.0,
    "BANKING": 15.0,
    "FINANCIAL SERVICES": 15.0,
    "FMCG": 45.0,
    "CONSUMER STAPLES": 45.0,
    "AUTO": 20.0,
    "AUTOMOBILE": 20.0,
    "PHARMA": 30.0,
    "HEALTHCARE": 30.0,
    "ENERGY": 12.0,
    "OIL & GAS": 12.0,
    "METALS": 10.0,
    "MATERIALS": 10.0,
    "REAL ESTATE": 25.0,
    "TELECOM": 18.0,
    "COMMUNICATION SERVICES": 18.0,
    "INFRASTRUCTURE": 20.0,
    "INDUSTRIALS": 20.0,
    "CONSUMER DURABLES": 35.0,
    "CONSUMER DISCRETIONARY": 35.0,
    "CHEMICALS": 22.0,
    "BASIC MATERIALS": 22.0,
    "POWER": 15.0,
    "UTILITIES": 15.0,
    "PSU": 10.0,
    "RETAIL": 30.0,
}


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
