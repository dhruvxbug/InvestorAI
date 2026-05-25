from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SignalType(str, Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    ACCUMULATE = "ACCUMULATE"
    HOLD = "HOLD"
    WAIT = "WAIT"
    REDUCE = "REDUCE"
    SELL = "SELL"
    AVOID = "AVOID"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY HIGH"


class LongTermRecommendation(BaseModel):
    signal: SignalType
    entry_price: float
    entry_condition: str
    target_1_year: float
    target_3_year: float
    expected_cagr: float
    stop_loss_price: float
    exit_triggers: list[str]
    risk_level: RiskLevel
    key_risks: list[str]
    key_catalysts: list[str]
    confidence_pct: int
    reasoning: str


class ShortTermRecommendation(BaseModel):
    signal: SignalType
    entry_price: float
    entry_condition: str
    target_1: float
    target_2: float
    target_3: Optional[float]
    stop_loss: float
    risk_reward_ratio: str
    trade_setup_type: str
    holding_period: str
    confidence_pct: int
    reasoning: str


class CompositeScores(BaseModel):
    technical: float
    fundamental: float
    sentiment: float
    management: float
    valuation: float
    overall: float
    agents_bullish: int
    agents_bearish: int
    agents_neutral: int


class StockReport(BaseModel):
    ticker: str
    company_name: str
    current_price: float
    analysis_date: str
    key_summary: list[str] = Field(min_length=5, max_length=5)  # exactly 5 bullets
    long_term: LongTermRecommendation
    short_term: ShortTermRecommendation
    scores: CompositeScores
    raw_technical: dict
    raw_fundamental: dict
    raw_sentiment: dict
    raw_management: dict
