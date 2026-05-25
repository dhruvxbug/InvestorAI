from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


Signal = Literal["BUY", "SELL", "HOLD"]


class AgentSection(BaseModel):
    name: str
    summary: str
    key_points: list[str] = Field(default_factory=list)
    raw_data: dict[str, Any] = Field(default_factory=dict)


class TradeLevels(BaseModel):
    entry_price: float | None = None
    target_price: float | None = None
    stop_loss: float | None = None
    time_horizon: str = "Swing (2-8 weeks)"


class FinalRecommendation(BaseModel):
    signal: Signal
    confidence: int = Field(ge=0, le=100)
    rationale: str
    risks: list[str] = Field(default_factory=list)
    trade_levels: TradeLevels


class StockAnalysisReport(BaseModel):
    ticker: str
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    sections: list[AgentSection]
    recommendation: FinalRecommendation
