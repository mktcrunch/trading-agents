"""SHORT/COVER allocation."""
from datetime import datetime

from src.models.position import Position
from src.models.trading_decision import TradingDecision
from src.strategies.allocator import PositionAllocator


def _position(ticker: str, qty: float) -> Position:
    return Position(
        ticker=ticker,
        qty=qty,
        avg_entry_price=100.0,
        current_price=100.0,
        entry_date=datetime.now(),
    )


def test_short_opens_negative_qty():
    decisions = [
        TradingDecision(action="SHORT", ticker="QQQ", size_pct=0.10, confidence=0.8),
    ]
    changes = PositionAllocator.allocate_from_decisions(
        decisions,
        portfolio_value=100_000,
        current_positions={},
        prices={"QQQ": 400.0},
    )
    assert changes["QQQ"] == -25


def test_cover_reduces_short():
    decisions = [
        TradingDecision(action="COVER", ticker="QQQ", size_pct=0.50, confidence=0.8),
    ]
    changes = PositionAllocator.allocate_from_decisions(
        decisions,
        portfolio_value=100_000,
        current_positions={"QQQ": _position("QQQ", -100)},
        prices={"QQQ": 400.0},
    )
    assert changes["QQQ"] == 50
