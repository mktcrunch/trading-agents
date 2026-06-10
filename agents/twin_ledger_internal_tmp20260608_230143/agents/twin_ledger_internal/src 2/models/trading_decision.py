"""
Trading decision model — Twin Ledger structured output.
"""
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


VALID_ACTIONS = {"BUY", "SELL", "HOLD", "CLOSE"}


@dataclass
class TradingDecision:
    action: str
    ticker: str
    size_pct: float = 0.0
    confidence: float = 0.5
    rationale: str = ""
    invalidation: str = ""
    competitive_note: str = ""
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["TradingDecision"]:
        action = str(data.get("action", "HOLD")).upper()
        ticker = str(data.get("ticker", "")).strip().upper()
        if not ticker or action not in VALID_ACTIONS:
            return None

        return cls(
            action=action,
            ticker=ticker,
            size_pct=float(data.get("size_pct", 0) or 0),
            confidence=float(data.get("confidence", 0.5) or 0.5),
            rationale=str(data.get("rationale", "")),
            invalidation=str(data.get("invalidation", "")),
            competitive_note=str(data.get("competitive_note", "")),
            stop_loss_pct=_optional_float(data.get("stop_loss_pct")),
            take_profit_pct=_optional_float(data.get("take_profit_pct")),
        )


def _optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    return float(value)
