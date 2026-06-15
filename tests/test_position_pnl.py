"""Position P&L must match Alpaca semantics for long and short."""
from datetime import datetime

from src.models.position import Position


def test_long_unrealized_return():
    pos = Position("SPY", qty=13, avg_entry_price=752.91, current_price=756.47, entry_date=datetime.now())
    assert abs(pos.unrealized_pnl - 46.28) < 0.1
    assert abs(pos.unrealized_return * 100 - 0.47) < 0.05


def test_short_unrealized_return_matches_alpaca_style():
    # Internal QQQ short: price up → loss on short
    pos = Position("QQQ", qty=-11, avg_entry_price=739.69, current_price=744.24, entry_date=datetime.now())
    assert abs(pos.unrealized_pnl - (-50.05)) < 0.1
    assert pos.unrealized_return < 0
    assert abs(pos.unrealized_return * 100 - (-0.61)) < 0.05


def test_short_profit_when_price_falls():
    pos = Position("SLV", qty=-114, avg_entry_price=64.22, current_price=63.93, entry_date=datetime.now())
    assert pos.unrealized_pnl > 0
    assert pos.unrealized_return > 0
    assert pos.is_profitable
