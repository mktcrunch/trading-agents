"""Pydantic schemas for ADK structured output."""
from typing import List, Literal

from pydantic import BaseModel, Field


class TradingDecisionSchema(BaseModel):
    action: Literal["BUY", "SELL", "HOLD", "CLOSE"]
    ticker: str
    size_pct: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    rationale: str = ""
    invalidation: str = ""
    competitive_note: str = ""


class TradingDecisionsResponse(BaseModel):
    decisions: List[TradingDecisionSchema]
