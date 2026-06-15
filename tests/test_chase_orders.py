"""Post-open chase eligibility: open limits, expired OPG, never cancelled."""
from datetime import datetime
from types import SimpleNamespace

import pytz

from src.agents.execution_agent import (
    _overnight_limit_chase_mode,
    collect_overnight_chase_candidates,
    overnight_order_already_chased,
)
from src.strategies.order_dedup import normalize_open_order

ET = pytz.timezone("US/Eastern")


def _limit_order(
    symbol: str,
    *,
    order_id: str,
    side: str = "buy",
    qty: float = 10,
    filled_qty: float = 0,
    status: str = "accepted",
    created_at: datetime,
    tif: str = "day",
):
    return SimpleNamespace(
        symbol=symbol,
        side=SimpleNamespace(value=side),
        qty=qty,
        filled_qty=filled_qty,
        limit_price=100.0,
        time_in_force=SimpleNamespace(value=tif),
        status=SimpleNamespace(value=status),
        id=order_id,
        created_at=created_at,
    )


def _market_order(symbol: str, *, order_id: str, side: str, created_at: datetime):
    return SimpleNamespace(
        symbol=symbol,
        side=SimpleNamespace(value=side),
        qty=10,
        filled_qty=0,
        limit_price=None,
        time_in_force=SimpleNamespace(value="day"),
        status=SimpleNamespace(value="filled"),
        id=order_id,
        created_at=created_at,
    )


def test_chase_mode_open_day_limit():
    order = _limit_order("SPY", order_id="d1", status="accepted", created_at=ET.localize(datetime(2026, 6, 15, 5, 0)))
    norm = normalize_open_order(order)
    assert _overnight_limit_chase_mode(order, norm) == "open_limit"


def test_chase_mode_expired_opg_limit():
    order = _limit_order(
        "SPY",
        order_id="o1",
        status="expired",
        tif="opg",
        created_at=ET.localize(datetime(2026, 6, 15, 5, 0)),
    )
    norm = normalize_open_order(order)
    assert _overnight_limit_chase_mode(order, norm) == "expired_opg"


def test_chase_mode_cancelled_day_never_chased():
    order = _limit_order(
        "TLT",
        order_id="c1",
        status="canceled",
        tif="day",
        created_at=ET.localize(datetime(2026, 6, 15, 5, 0)),
    )
    norm = normalize_open_order(order)
    assert _overnight_limit_chase_mode(order, norm) is None


def test_chase_mode_expired_day_not_chased():
    order = _limit_order(
        "TLT",
        order_id="e1",
        status="expired",
        tif="day",
        created_at=ET.localize(datetime(2026, 6, 15, 5, 0)),
    )
    norm = normalize_open_order(order)
    assert _overnight_limit_chase_mode(order, norm) is None


def test_collect_candidates_skips_cancelled_picks_open_and_expired_opg():
    cutoff = ET.localize(datetime(2026, 6, 14, 16, 0))
    created = ET.localize(datetime(2026, 6, 14, 17, 0))
    orders = [
        _limit_order("TLT", order_id="cancelled", status="canceled", created_at=created),
        _limit_order("SPY", order_id="open-spy", status="accepted", created_at=created),
        _limit_order("IWM", order_id="exp-opg", status="expired", tif="opg", created_at=created),
    ]
    candidates = collect_overnight_chase_candidates(
        orders,
        cutoff,
        normalize_order=normalize_open_order,
    )
    assert set(candidates) == {"SPY", "IWM"}
    assert candidates["SPY"][3] == "open_limit"
    assert candidates["IWM"][3] == "expired_opg"


def test_overnight_order_already_chased_detects_later_market():
    base = ET.localize(datetime(2026, 6, 15, 4, 30))
    later = ET.localize(datetime(2026, 6, 15, 9, 46))
    history = [
        (
            normalize_open_order(_limit_order("SPY", order_id="limit-1", created_at=base)),
            base,
            None,
        ),
        (
            normalize_open_order(_market_order("SPY", order_id="mkt-1", side="buy", created_at=later)),
            later,
            None,
        ),
    ]
    assert overnight_order_already_chased(history, "limit-1", base, "buy") is True
