"""Risk monitor cancels open limit orders before market exits."""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

from src.models.position import Position
from src.risk.risk_monitor import RiskMonitor


def _limit_order(order_id: str, symbol: str, side: str, qty: float, limit: float):
    return SimpleNamespace(
        id=order_id,
        symbol=symbol,
        side=SimpleNamespace(value=side),
        qty=qty,
        filled_qty=0,
        limit_price=limit,
        time_in_force=SimpleNamespace(value="day"),
        status=SimpleNamespace(value="accepted"),
    )


def test_close_position_cancels_open_limits_on_symbol():
    monitor = RiskMonitor(system="internal")
    monitor.alpaca = MagicMock()
    monitor.alpaca.get_orders.return_value = [
        _limit_order("overnight-slv", "SLV", "buy", 114, 63.15),
        _limit_order("other-qqq", "QQQ", "buy", 11, 740.0),
    ]
    monitor.alpaca.cancel_order.return_value = True
    monitor.alpaca.place_market_order.return_value = "market-1"

    pos = Position(
        ticker="SLV",
        qty=-114,
        avg_entry_price=64.0,
        current_price=63.0,
        entry_date=datetime.now(),
    )

    result = monitor._close_position("SLV", pos, dry_run=False)

    monitor.alpaca.cancel_order.assert_called_once_with("overnight-slv")
    monitor.alpaca.place_market_order.assert_called_once_with(
        ticker="SLV",
        qty=114,
        side="buy",
        time_in_force="day",
    )
    assert result["status"] == "submitted"
    assert result["cancelled_limit_orders"][0]["order_id"] == "overnight-slv"


def test_close_position_leaves_other_symbols_untouched():
    monitor = RiskMonitor(system="baseline")
    monitor.alpaca = MagicMock()
    monitor.alpaca.get_orders.return_value = [
        _limit_order("qqq-only", "QQQ", "buy", 11, 740.0),
    ]
    monitor.alpaca.place_market_order.return_value = "market-2"

    pos = Position(
        ticker="IWM",
        qty=34,
        avg_entry_price=290.0,
        current_price=285.0,
        entry_date=datetime.now(),
    )

    monitor._close_position("IWM", pos, dry_run=False)

    monitor.alpaca.cancel_order.assert_not_called()


def test_cancel_open_limit_orders_skips_market_orders():
    monitor = RiskMonitor(system="internal")
    monitor.alpaca = MagicMock()
    monitor.alpaca.get_orders.return_value = [
        SimpleNamespace(
            id="mkt-1",
            symbol="SLV",
            side=SimpleNamespace(value="buy"),
            qty=114,
            filled_qty=0,
            limit_price=None,
            time_in_force=SimpleNamespace(value="day"),
            status=SimpleNamespace(value="accepted"),
        ),
    ]

    cancelled = monitor._cancel_open_limit_orders("SLV")

    assert cancelled == []
    monitor.alpaca.cancel_order.assert_not_called()
