"""Overnight RiskAgent validation."""
import asyncio
from datetime import datetime
from types import SimpleNamespace

from src.agents.risk_agent import RiskAgent, entry_sides_from_decisions
from src.models.position import Position
from src.models.trading_decision import TradingDecision


def _pos(ticker: str, qty: float, price: float = 100.0) -> Position:
    return Position(
        ticker=ticker,
        qty=qty,
        avg_entry_price=price,
        current_price=price,
        entry_date=datetime.now(),
    )


def _open_sell(symbol: str, qty: float, price: float, filled: float = 0.0):
    return SimpleNamespace(
        symbol=symbol,
        side=SimpleNamespace(value="sell"),
        qty=qty,
        filled_qty=filled,
        limit_price=price,
        time_in_force=SimpleNamespace(value="day"),
        id="order-1",
    )


class _RiskAgentNoFetch(RiskAgent):
    """RiskAgent that does not call Alpaca or LLM when open_orders_raw is omitted."""

    def __init__(self):
        super().__init__(system="baseline")
        self._alpaca = None
        self._overnight_planner = None


def test_rejects_short_when_long_held():
    agent = _RiskAgentNoFetch()
    result = asyncio.run(agent.validate_positions(
        proposed_positions={"QQQ": 0.05},
        portfolio_value=100_000,
        current_positions={"QQQ": _pos("QQQ", 50, 400.0)},
        entry_sides={"QQQ": "short"},
        open_orders_raw=[],
    ))
    assert result["QQQ"] is False


def test_rejects_short_stacking_with_pending_sell():
    agent = _RiskAgentNoFetch()
    # 5% short held + 4% pending sell + 3% proposed = 12%
    result = asyncio.run(agent.validate_positions(
        proposed_positions={"QQQ": 0.03},
        portfolio_value=100_000,
        current_positions={"QQQ": _pos("QQQ", -12, 400.0)},
        entry_sides={"QQQ": "short"},
        open_orders_raw=[_open_sell("QQQ", 10, 400.0)],
    ))
    assert result["QQQ"] is False


def test_accepts_short_when_room_available():
    agent = _RiskAgentNoFetch()
    result = asyncio.run(agent.validate_positions(
        proposed_positions={"QQQ": 0.05},
        portfolio_value=100_000,
        current_positions={},
        entry_sides={"QQQ": "short"},
        open_orders_raw=[],
    ))
    assert result["QQQ"] is True


def test_rejects_when_gross_exposure_exceeded():
    agent = _RiskAgentNoFetch()
    positions = {
        f"T{i}": _pos(f"T{i}", 100, 100.0)
        for i in range(13)
    }
    result = asyncio.run(agent.validate_positions(
        proposed_positions={"NEW": 0.05},
        portfolio_value=100_000,
        current_positions=positions,
        entry_sides={"NEW": "long"},
        open_orders_raw=[],
    ))
    assert result["NEW"] is False


def test_entry_sides_from_decisions():
    decisions = [
        TradingDecision(action="BUY", ticker="SPY", size_pct=0.05),
        TradingDecision(action="SHORT", ticker="QQQ", size_pct=0.04),
    ]
    assert entry_sides_from_decisions(decisions) == {"SPY": "long", "QQQ": "short"}
