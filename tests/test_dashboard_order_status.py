"""Dashboard audit ↔ Alpaca order status reconciliation."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.adk.tools.dashboard_tools import (
    annotate_orders_with_alpaca_status,
    enrich_order_placed_audit_events,
)
from src.strategies.order_dedup import order_live_snapshot, static_order_snapshot


def _alpaca_order(order_id: str, status: str, qty: float = 10, filled: float = 0.0):
    return SimpleNamespace(
        id=order_id,
        symbol="SPY",
        side=SimpleNamespace(value="buy"),
        qty=qty,
        filled_qty=filled,
        limit_price=738.04,
        time_in_force=SimpleNamespace(value="day"),
        status=SimpleNamespace(value=status),
    )


def test_static_snapshot_for_dry_run():
    snap = static_order_snapshot("dry-run")
    assert snap["alpaca_status"] == "simulated"
    assert snap["alpaca_is_active"] is False


def test_order_live_snapshot_open():
    snap = order_live_snapshot(_alpaca_order("abc", "accepted"))
    assert snap["alpaca_status"] == "open"
    assert snap["alpaca_is_active"] is True


def test_order_live_snapshot_canceled():
    snap = order_live_snapshot(_alpaca_order("abc", "canceled"))
    assert snap["alpaca_status"] == "canceled"
    assert snap["alpaca_is_active"] is False
    assert "not counted" in snap["alpaca_status_note"].lower()


@patch("src.apis.alpaca_client.AlpacaClient")
def test_annotate_orders_with_live_status(mock_client_cls):
    client = MagicMock()
    mock_client_cls.return_value = client
    client.get_orders.return_value = [_alpaca_order("oid-1", "canceled")]
    client.get_order.return_value = None

    rows = [{
        "order_id": "oid-1",
        "ticker": "SPY",
        "side": "buy",
        "qty": 10,
    }]
    annotated = annotate_orders_with_alpaca_status("internal", rows)

    assert annotated[0]["alpaca_status"] == "canceled"
    assert annotated[0]["alpaca_is_active"] is False


@patch("src.apis.alpaca_client.AlpacaClient")
def test_enrich_audit_events_adds_payload_status(mock_client_cls):
    client = MagicMock()
    mock_client_cls.return_value = client
    client.get_orders.return_value = [_alpaca_order("oid-2", "filled", filled=10)]
    client.get_order.return_value = None

    events = [{
        "event_type": "order_placed",
        "system": "internal",
        "payload": {
            "order_id": "oid-2",
            "ticker": "SPY",
            "side": "buy",
            "qty": 10,
        },
    }]
    enriched = enrich_order_placed_audit_events(events)

    payload = enriched[0]["payload"]
    assert payload["alpaca_status"] == "filled"
    assert payload["alpaca_filled_qty"] == 10.0
