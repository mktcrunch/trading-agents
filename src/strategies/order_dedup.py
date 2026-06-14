"""
Filter proposed orders against open Alpaca orders and available buying power.

Problems addressed:
- Exact duplicate skip (same symbol/side/qty/price)
- Near-duplicate stacking (same symbol/side/price, different qty from repeated LLM runs)
- Delta-only placement (only buy/sell the gap vs pending + held)
- Optional consolidation (cancel redundant open OPG orders per symbol)
"""
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src import config
from src.apis.alpaca_client import round_limit_price
from src.logger import setup_logger

logger = setup_logger(__name__)


def _order_side(order: Any) -> str:
    side = getattr(order, "side", None)
    if side is None:
        return str(order.get("side", "")).lower()
    return side.value.lower() if hasattr(side, "value") else str(side).lower()


def _order_tif(order: Any) -> str:
    tif = getattr(order, "time_in_force", None)
    if tif is None:
        return str(order.get("time_in_force", "day")).lower()
    return tif.value.lower() if hasattr(tif, "value") else str(tif).lower()


TERMINAL_ORDER_STATUSES = frozenset({
    "filled",
    "canceled",
    "cancelled",
    "expired",
    "rejected",
    "done_for_day",
})


def _order_status(order: Any) -> str:
    status = getattr(order, "status", None)
    if status is None:
        if hasattr(order, "get"):
            return str(order.get("status", "")).lower()
        return ""
    return status.value.lower() if hasattr(status, "value") else str(status).lower()


def is_actionable_open_order(order: Any) -> bool:
    """
    True if an order should count toward pending exposure or dedup.

    Cancelled, expired, rejected, and fully filled orders are mistakes/history —
    they must not block new placements.
    """
    if _order_status(order) in TERMINAL_ORDER_STATUSES:
        return False
    qty = float(getattr(order, "qty", 0) or 0)
    filled = float(getattr(order, "filled_qty", 0) or 0)
    return max(0.0, qty - filled) > 0


def filter_actionable_open_orders(orders: Optional[List[Any]]) -> List[Any]:
    """Drop terminal or zero-remaining orders before risk checks or dedup."""
    if not orders:
        return []
    actionable = [o for o in orders if is_actionable_open_order(o)]
    dropped = len(orders) - len(actionable)
    if dropped:
        logger.info(
            f"Ignoring {dropped} non-actionable orders "
            f"(cancelled/expired/filled/empty)"
        )
    return actionable


_OPEN_ALPACA_STATUSES = frozenset({
    "",
    "accepted",
    "new",
    "pending_new",
    "accepted_for_bidding",
    "stopped",
    "pending_cancel",
    "pending_replace",
    "partially_filled",
})

_ALPACA_STATUS_NOTES = {
    "open": "Still open on Alpaca — counts toward risk and dedup",
    "canceled": "Cancelled on Alpaca — not counted for risk or dedup",
    "filled": "Fully filled on Alpaca",
    "expired": "Expired on Alpaca — not counted for risk or dedup",
    "rejected": "Rejected by Alpaca",
    "partially_filled": "Partially filled on Alpaca",
    "simulated": "Dry run — no live Alpaca order",
    "unknown": "Not found in Alpaca (may be purged or outside the query window)",
}


def _display_alpaca_status(raw_status: str, is_active: bool) -> str:
    status = raw_status or ""
    if status == "cancelled":
        status = "canceled"
    if is_active and status not in TERMINAL_ORDER_STATUSES:
        if status in _OPEN_ALPACA_STATUSES:
            return "open"
        return status or "open"
    return status or "unknown"


def order_live_snapshot(order: Any) -> Dict[str, Any]:
    """Normalize an Alpaca order for dashboard / audit display."""
    norm = normalize_open_order(order)
    raw_status = _order_status(order)
    active = is_actionable_open_order(order)
    display = _display_alpaca_status(raw_status, active)
    return {
        "alpaca_status": display,
        "alpaca_filled_qty": norm["filled_qty"],
        "alpaca_remaining_qty": norm["remaining_qty"],
        "alpaca_is_active": active,
        "alpaca_status_note": _ALPACA_STATUS_NOTES.get(
            display,
            "Live Alpaca status",
        ),
    }


def static_order_snapshot(order_id: Optional[str]) -> Optional[Dict[str, Any]]:
    """Snapshots for audit rows that never hit Alpaca."""
    if not order_id:
        return None
    if order_id == "dry-run":
        return {
            "alpaca_status": "simulated",
            "alpaca_filled_qty": 0.0,
            "alpaca_remaining_qty": 0.0,
            "alpaca_is_active": False,
            "alpaca_status_note": _ALPACA_STATUS_NOTES["simulated"],
        }
    return None


def normalize_open_order(order: Any) -> Dict:
    """Normalize an Alpaca order object to a comparable dict."""
    qty = float(getattr(order, "qty", 0) or 0)
    filled = float(getattr(order, "filled_qty", 0) or 0)
    limit = getattr(order, "limit_price", None)
    return {
        "symbol": getattr(order, "symbol", ""),
        "side": _order_side(order),
        "qty": qty,
        "filled_qty": filled,
        "remaining_qty": max(0.0, qty - filled),
        "limit_price": round_limit_price(float(limit)) if limit else None,
        "time_in_force": _order_tif(order),
        "order_id": str(getattr(order, "id", "")),
    }


def _proposed_order(
    ticker: str,
    qty_change: float,
    ref_price: float,
    offset_pct: float,
) -> Dict:
    if qty_change > 0:
        side = "buy"
        limit_price = round_limit_price(ref_price * (1 - offset_pct))
    else:
        side = "sell"
        limit_price = round_limit_price(ref_price * (1 + offset_pct))
    return {
        "symbol": ticker,
        "side": side,
        "qty": abs(qty_change),
        "limit_price": limit_price,
        "time_in_force": "opg",
    }


def pending_qty_by_symbol_side(open_orders: List[Dict]) -> Dict[Tuple[str, str], float]:
    """Sum remaining qty per (symbol, side) across all open orders."""
    pending: Dict[Tuple[str, str], float] = defaultdict(float)
    for o in open_orders:
        pending[(o["symbol"], o["side"])] += o["remaining_qty"]
    return dict(pending)


def pending_qty_at_price(
    open_orders: List[Dict],
    symbol: str,
    side: str,
    limit_price: float,
) -> float:
    """Sum remaining qty for matching symbol/side/limit price."""
    total = 0.0
    for o in open_orders:
        if o["symbol"] == symbol and o["side"] == side and o["limit_price"] == limit_price:
            total += o["remaining_qty"]
    return total


def is_duplicate_order(proposed: Dict, open_orders: List[Dict]) -> bool:
    """True if an open order matches symbol, side, qty, limit price, and TIF."""
    for open_o in open_orders:
        if proposed["symbol"] != open_o["symbol"]:
            continue
        if proposed["side"] != open_o["side"]:
            continue
        if proposed["time_in_force"] != open_o["time_in_force"]:
            continue
        if proposed["limit_price"] != open_o["limit_price"]:
            continue
        open_qty = open_o["qty"] if open_o["filled_qty"] == 0 else open_o["remaining_qty"]
        if float(proposed["qty"]) == float(open_qty):
            return True
    return False


def compute_order_delta(
    proposed_qty: float,
    symbol: str,
    side: str,
    limit_price: float,
    open_orders: List[Dict],
    held_qty: float = 0.0,
) -> Tuple[float, str]:
    """
    Compute net shares still needed after pending open orders.

    For BUY: proposed is shares to buy this run; subtract pending buys at same price.
    Returns (delta_qty, reason_if_zero).
    """
    pending_at_price = pending_qty_at_price(open_orders, symbol, side, limit_price)
    pending_total = pending_qty_by_symbol_side(open_orders).get((symbol, side), 0.0)

    if side == "buy":
        if pending_at_price >= proposed_qty:
            return 0.0, "pending_covers_proposed"
        if pending_total >= proposed_qty:
            return 0.0, "pending_symbol_covers_proposed"
        delta = proposed_qty - pending_at_price
        return max(0.0, delta), ""

    # SELL: subtract pending sell orders
    if pending_at_price >= proposed_qty:
        return 0.0, "pending_sell_covers_proposed"
    delta = proposed_qty - pending_at_price
    return max(0.0, delta), ""


def reconcile_duplicate_open_orders(
    alpaca_client,
    open_orders_raw: Optional[List[Any]] = None,
) -> List[Dict]:
    """
    Cancel redundant OPG orders: multiple open orders for same symbol+side+limit_price.

    Keeps the single largest remaining-qty order; cancels the rest.
    Prevents 3× IWM @ $282.69 all filling at open.
    """
    if not config.ORDER_CONFIG.get("reconcile_open_orders", True):
        return []

    open_orders_raw = open_orders_raw or alpaca_client.get_orders(status="open")
    open_orders = [
        normalize_open_order(o)
        for o in filter_actionable_open_orders(open_orders_raw)
    ]

    groups: Dict[Tuple[str, str, float], List[Dict]] = defaultdict(list)
    for o in open_orders:
        if o["time_in_force"] != "opg" or not o["limit_price"]:
            continue
        key = (o["symbol"], o["side"], o["limit_price"])
        groups[key].append(o)

    cancelled = []
    for key, orders in groups.items():
        if len(orders) <= 1:
            continue

        symbol, side, price = key
        orders.sort(key=lambda x: x["remaining_qty"], reverse=True)
        keeper = orders[0]
        extras = orders[1:]

        logger.warning(
            f"Reconciling {len(orders)} duplicate {side} {symbol} @ ${price:.2f} OPG — "
            f"keeping {keeper['remaining_qty']:.0f} sh (id={keeper['order_id'][:8]}…), "
            f"cancelling {len(extras)}"
        )

        for extra in extras:
            if alpaca_client.cancel_order(extra["order_id"]):
                cancelled.append({
                    "symbol": symbol,
                    "side": side,
                    "limit_price": price,
                    "cancelled_qty": extra["remaining_qty"],
                    "order_id": extra["order_id"],
                    "kept_order_id": keeper["order_id"],
                })

    return cancelled


def _open_buy_commitment(open_orders: List[Dict]) -> float:
    total = 0.0
    for o in open_orders:
        if o["side"] != "buy" or not o["limit_price"]:
            continue
        total += o["remaining_qty"] * o["limit_price"]
    return total


def filter_orders_for_placement(
    position_changes: Dict[str, float],
    reference_prices: Dict[str, float],
    open_orders_raw: List[Any],
    buying_power: float,
    current_positions: Optional[Dict[str, Any]] = None,
    offset_pct: Optional[float] = None,
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Delta placement + duplicate skip + buying power check.

    Returns:
        (filtered_changes, skipped_records)
    """
    offset_pct = offset_pct or config.ORDER_CONFIG.get("overnight_limit_offset_pct", 0.005)
    min_order_value = config.ORDER_CONFIG.get("min_order_value", 100)
    current_positions = current_positions or {}

    open_orders = [
        normalize_open_order(o)
        for o in filter_actionable_open_orders(open_orders_raw)
    ]
    available_bp = max(0.0, float(buying_power) - _open_buy_commitment(open_orders))

    filtered: Dict[str, float] = {}
    skipped: List[Dict] = []

    sell_items = [(t, q) for t, q in position_changes.items() if q < 0]
    buy_items = [(t, q) for t, q in position_changes.items() if q > 0]

    for ticker, qty_change in sell_items + buy_items:
        if qty_change == 0:
            continue

        ref_price = reference_prices.get(ticker, 0)
        if not ref_price:
            skipped.append({
                "ticker": ticker,
                "reason": "missing_reference_price",
                "qty_change": qty_change,
            })
            continue

        proposed = _proposed_order(ticker, qty_change, ref_price, offset_pct)
        held = 0.0
        pos = current_positions.get(ticker)
        if pos:
            held = float(getattr(pos, "qty", 0) or 0)

        if is_duplicate_order(proposed, open_orders):
            skipped.append({
                "ticker": ticker,
                "reason": "duplicate_open_order",
                "proposed": proposed,
            })
            continue

        delta_qty, zero_reason = compute_order_delta(
            proposed_qty=proposed["qty"],
            symbol=ticker,
            side=proposed["side"],
            limit_price=proposed["limit_price"],
            open_orders=open_orders,
            held_qty=held,
        )

        if delta_qty <= 0:
            pending_at_price = pending_qty_at_price(
                open_orders, ticker, proposed["side"], proposed["limit_price"]
            )
            logger.info(
                f"Skipping {proposed['side']} {ticker}: {zero_reason} "
                f"(proposed={proposed['qty']:.0f}, pending={pending_at_price:.0f} @ ${proposed['limit_price']:.2f})"
            )
            skipped.append({
                "ticker": ticker,
                "reason": zero_reason,
                "proposed": proposed,
                "pending_at_price": pending_at_price,
            })
            continue

        if delta_qty < proposed["qty"]:
            logger.info(
                f"Delta {proposed['side']} {ticker}: {proposed['qty']:.0f} → {delta_qty:.0f} "
                f"(pending orders already cover part)"
            )
            proposed["qty"] = delta_qty
            qty_change = delta_qty if proposed["side"] == "buy" else -delta_qty

        if proposed["side"] == "buy":
            notional = proposed["qty"] * proposed["limit_price"]
            if notional > available_bp:
                if available_bp < min_order_value:
                    skipped.append({
                        "ticker": ticker,
                        "reason": "insufficient_buying_power",
                        "proposed": proposed,
                        "available_buying_power": round(available_bp, 2),
                    })
                    continue
                max_shares = int(available_bp / proposed["limit_price"])
                if max_shares <= 0:
                    skipped.append({
                        "ticker": ticker,
                        "reason": "insufficient_buying_power",
                        "proposed": proposed,
                        "available_buying_power": round(available_bp, 2),
                    })
                    continue
                logger.info(
                    f"Scaling BUY {ticker}: {proposed['qty']} → {max_shares} shares "
                    f"(buying power ${available_bp:,.2f})"
                )
                qty_change = float(max_shares)
                proposed["qty"] = max_shares
                notional = max_shares * proposed["limit_price"]

            available_bp -= notional

        signed_qty = qty_change if proposed["side"] == "buy" else -abs(qty_change)
        filtered[ticker] = signed_qty

    return filtered, skipped
