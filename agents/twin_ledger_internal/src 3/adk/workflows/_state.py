"""Shared workflow state for Twin Ledger ADK pipelines."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.models.trading_decision import TradingDecision


class TwinLedgerWorkflowState(BaseModel):
    system: str = "baseline"
    account_info: Optional[Dict[str, Any]] = None
    competition: Optional[Dict[str, Any]] = None
    technical_data: Dict[str, Any] = Field(default_factory=dict)
    mc_predictions: Dict[str, Any] = Field(default_factory=dict)
    databento_sources: Dict[str, Any] = Field(default_factory=dict)
    decisions: List[Dict[str, Any]] = Field(default_factory=list)
    success: bool = False
    error: Optional[str] = None
