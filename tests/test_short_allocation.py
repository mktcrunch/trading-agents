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


def test_cover_uses_portfolio_weight_capped_at_short():
    # 10% of 100k = 10k → 25 shares @ 400; held -100 → cap at 25
    decisions = [
        TradingDecision(action="COVER", ticker="QQQ", size_pct=0.10, confidence=0.8),
    ]
    changes = PositionAllocator.allocate_from_decisions(
        decisions,
        portfolio_value=100_000,
        current_positions={"QQQ": _position("QQQ", -100)},
        prices={"QQQ": 400.0},
    )
    assert changes["QQQ"] == 25


def test_cover_portfolio_weight_full_short_overnight_regression():
    """Jun 15 internal: COVER QQQ 8.1% ≈ full -11 short at ~8.1% portfolio weight."""
    pv = 100_672.35
    decisions = [
        TradingDecision(action="COVER", ticker="QQQ", size_pct=0.081, confidence=0.95),
    ]
    changes = PositionAllocator.allocate_from_decisions(
        decisions,
        portfolio_value=pv,
        current_positions={"QQQ": _position("QQQ", -11)},
        prices={"QQQ": 740.28},
    )
    assert changes["QQQ"] == 11


def test_sell_uses_portfolio_weight_capped_at_long():
    decisions = [
        TradingDecision(action="SELL", ticker="SPY", size_pct=0.05, confidence=0.8),
    ]
    changes = PositionAllocator.allocate_from_decisions(
        decisions,
        portfolio_value=100_000,
        current_positions={"SPY": _position("SPY", 50)},
        prices={"SPY": 500.0},
    )
    assert changes["SPY"] == -10


def test_close_exits_short_as_full_cover():
    decisions = [
        TradingDecision(action="CLOSE", ticker="TLT", size_pct=0.09, confidence=0.9),
    ]
    changes = PositionAllocator.allocate_from_decisions(
        decisions,
        portfolio_value=100_000,
        current_positions={"TLT": _position("TLT", -105)},
        prices={"TLT": 85.0},
    )
    assert changes["TLT"] == 105


def test_close_exits_long_as_full_sell():
    decisions = [
        TradingDecision(action="CLOSE", ticker="SPY", size_pct=0.10, confidence=0.9),
    ]
    changes = PositionAllocator.allocate_from_decisions(
        decisions,
        portfolio_value=100_000,
        current_positions={"SPY": _position("SPY", 13)},
        prices={"SPY": 500.0},
    )
    assert changes["SPY"] == -13
