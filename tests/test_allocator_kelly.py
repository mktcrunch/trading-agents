"""Kelly sizing for internal BUY and SHORT entries."""
from datetime import datetime

from src.models.position import Position
from src.models.signal import Signal
from src.models.trading_decision import TradingDecision
from src.strategies.allocator import PositionAllocator


def _signal(ticker: str, predicted_return: float, confidence: float = 0.6) -> Signal:
    return Signal(
        ticker=ticker,
        timestamp=datetime.now(),
        predicted_return=predicted_return,
        confidence=confidence,
    )


def test_kelly_edge_for_side():
    assert PositionAllocator.kelly_edge_for_side(0.02, "long") == 0.02
    assert PositionAllocator.kelly_edge_for_side(0.02, "short") == 0.0
    assert PositionAllocator.kelly_edge_for_side(-0.02, "long") == 0.0
    assert PositionAllocator.kelly_edge_for_side(-0.02, "short") == 0.02


def test_kelly_criterion_respects_side():
    long_kelly = PositionAllocator.kelly_criterion(0.02, 0.6, side="long")
    short_kelly = PositionAllocator.kelly_criterion(0.02, 0.6, side="short")
    assert long_kelly > 0
    assert short_kelly == 0

    long_kelly_neg = PositionAllocator.kelly_criterion(-0.02, 0.6, side="long")
    short_kelly_neg = PositionAllocator.kelly_criterion(-0.02, 0.6, side="short")
    assert long_kelly_neg == 0
    assert short_kelly_neg > 0


def test_internal_entry_target_weights_mixed_sides():
    signals = {
        "XLF": _signal("XLF", -0.01),
        "TLT": _signal("TLT", 0.0051),
    }
    entry_sides = {"XLF": "long", "TLT": "short"}
    weights = PositionAllocator.internal_entry_target_weights(signals, entry_sides)
    assert weights["XLF"] == 0.0
    assert weights["TLT"] == 0.0

    signals = {
        "XLF": _signal("XLF", 0.02),
        "TLT": _signal("TLT", -0.02),
    }
    entry_sides = {"XLF": "long", "TLT": "short"}
    weights = PositionAllocator.internal_entry_target_weights(signals, entry_sides)
    assert weights["XLF"] > 0
    assert weights["TLT"] > 0
    assert weights["XLF"] <= 0.10
    assert weights["TLT"] <= 0.10


def test_allocate_internal_short_uses_kelly_not_size_pct():
    """Positive MC target → short Kelly 0 even when LLM size_pct is large."""
    decisions = [
        TradingDecision(action="SHORT", ticker="TLT", size_pct=0.10, confidence=0.8),
    ]
    entry_signals = {"TLT": _signal("TLT", 0.0051)}
    changes = PositionAllocator.allocate_internal_from_decisions(
        decisions,
        entry_signals,
        portfolio_value=100_000,
        current_positions={},
        prices={"TLT": 90.0},
        entry_sides={"TLT": "short"},
    )
    assert "TLT" not in changes

    decisions = [
        TradingDecision(action="SHORT", ticker="TLT", size_pct=0.10, confidence=0.8),
    ]
    entry_signals = {"TLT": _signal("TLT", -0.02)}
    changes = PositionAllocator.allocate_internal_from_decisions(
        decisions,
        entry_signals,
        portfolio_value=100_000,
        current_positions={},
        prices={"TLT": 90.0},
        entry_sides={"TLT": "short"},
    )
    assert changes["TLT"] < 0
