"""Pydantic schemas for ADK structured output."""
from typing import List, Literal

from pydantic import BaseModel, Field


class TradingDecisionSchema(BaseModel):
    action: Literal["BUY", "SELL", "HOLD", "CLOSE", "SHORT", "COVER"]
    ticker: str
    size_pct: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    rationale: str = ""
    invalidation: str = ""
    competitive_note: str = ""


class TradingDecisionsResponse(BaseModel):
    decisions: List[TradingDecisionSchema] = Field(default_factory=list)
    no_action_rationale: str = Field(
        default="",
        description=(
            "Required when there are no BUY/SELL/CLOSE/SHORT/COVER decisions: "
            "2-4 sentences on why no edge tonight (not merely leaderboard rank), market read, "
            "and why no trades tonight."
        ),
    )
