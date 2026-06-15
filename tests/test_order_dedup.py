"""Order dedup and actionable-order filtering."""
import asyncio
from types import SimpleNamespace

from src.agents.risk_agent import RiskAgent
from src.strategies.order_dedup import (
    filter_actionable_open_orders,
    filter_orders_for_placement,
    is_actionable_open_order,
)


def _open_buy(symbol: str, qty: float, price: float, status: str = "accepted"):
    return SimpleNamespace(
        symbol=symbol,
        side=SimpleNamespace(value="buy"),
        qty=qty,
        filled_qty=0.0,
        limit_price=price,
        time_in_force=SimpleNamespace(value="day"),
        status=SimpleNamespace(value=status),
        id=f"order-{symbol}",
    )


class _RiskAgentNoFetch(RiskAgent):
    def __init__(self):
        super().__init__(system="baseline")
        self._alpaca = None
        self._overnight_planner = None


def test_cancelled_orders_are_not_actionable():
    cancelled = _open_buy("SPY", 10, 738.0, status="canceled")
    assert is_actionable_open_order(cancelled) is False
    assert filter_actionable_open_orders([cancelled]) == []


def test_cancelled_orders_do_not_block_dedup():
    cancelled = _open_buy("SPY", 13, 738.04, status="canceled")
    filtered, skipped = filter_orders_for_placement(
        position_changes={"SPY": 13},
        reference_prices={"SPY": 742.0},
        open_orders_raw=[cancelled],
        buying_power=100_000,
        current_positions={},
    )
    assert filtered == {"SPY": 13}
    assert skipped == []


def test_cancelled_orders_do_not_count_for_risk_pending():
    agent = _RiskAgentNoFetch()
    cancelled = _open_buy("SPY", 13, 738.04, status="canceled")
    result = asyncio.run(agent.validate_positions(
        proposed_positions={"SPY": 0.05, "TLT": 0.05},
        portfolio_value=100_000,
        current_positions={},
        entry_sides={"SPY": "long", "TLT": "short"},
        open_orders_raw=[cancelled],
    ))
    assert result["SPY"] is True
    assert result["TLT"] is True
