"""
Execution Agent (SHARED by both systems)
Places orders on Alpaca:
- Overnight: LIMIT orders at MOC ± 0.5%
- Post-open: Chase unfilled with MARKET orders
- EOD: Hold/trim based on next-day predictions
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from src import config
from src.agents.base_agent import BaseAgent
from src.apis.alpaca_client import AlpacaClient, round_limit_price
from src.models.order import Order, OrderType, OrderTimeInForce
from src.strategies.order_dedup import (
    filter_orders_for_placement,
    reconcile_duplicate_open_orders,
)
from src.logger import setup_logger

logger = setup_logger(__name__)

_OPEN_ORDER_STATUSES = frozenset({
    "new",
    "partially_filled",
    "accepted",
    "pending_new",
    "accepted_for_bidding",
    "held",
})

_CANCELLED_ORDER_STATUSES = frozenset({"canceled", "cancelled"})

# ``open_limit``: cancel the live DAY/OPG limit, then market the remainder.
# ``expired_opg``: OPG limit expired at the open — market only (never re-chase cancels).
ChaseMode = str  # "open_limit" | "expired_opg"


def _order_status_str(order_obj: Any) -> str:
    status_val = getattr(order_obj, "status", None)
    if status_val is None and isinstance(order_obj, dict):
        status_val = order_obj.get("status")
    if hasattr(status_val, "value"):
        status_val = status_val.value
    return str(status_val).lower() if status_val else ""


def _order_created_at(order_obj: Any):
    from datetime import datetime

    created_at = getattr(order_obj, "created_at", None)
    if created_at is None and isinstance(order_obj, dict):
        created_at = order_obj.get("created_at")
    if isinstance(created_at, str):
        try:
            if created_at.endswith("Z"):
                created_at = created_at[:-1] + "+00:00"
            return datetime.fromisoformat(created_at)
        except Exception:
            return None
    return created_at


def _overnight_lookback_cutoff_et(now_et, system: str = "baseline"):
    from src.market.calendar import prior_session_close_cutoff_et

    return prior_session_close_cutoff_et(now_et, system=system)


async def _await_calm_market_for_chase(
    alpaca_client: AlpacaClient,
    symbol: str,
    *,
    log_action,
    log_error,
    max_retries: int = 3,
    retry_delay_sec: int = 60,
) -> bool:
    """Return True when recent volatility is below chase thresholds."""
    import asyncio

    for attempt in range(max_retries + 1):
        vol_metrics = alpaca_client.get_recent_volatility(symbol, minutes=30)
        if not vol_metrics:
            if attempt < max_retries:
                log_action(
                    f"Could not calculate volatility for {symbol} (insufficient data) — "
                    f"retrying in {retry_delay_sec}s..."
                )
                await asyncio.sleep(retry_delay_sec)
            else:
                log_error(
                    f"Failed to calculate volatility for {symbol} after {max_retries} retries — "
                    f"skipping chase to prevent unsafe execution"
                )
            continue

        std_dev = vol_metrics["std_dev_pct"]
        hl_range = vol_metrics["range_pct"]

        if std_dev < 0.15 and hl_range < 1.2:
            log_action(
                f"Market conditions for {symbol} are calm: "
                f"30-min std dev is {std_dev:.3f}% (< 0.15%), range is {hl_range:.2f}% (< 1.2%). "
                f"Proceeding with chase."
            )
            return True

        if attempt < max_retries:
            log_action(
                f"{symbol} is highly volatile right now (attempt {attempt + 1}/{max_retries + 1}): "
                f"30-min std dev is {std_dev:.3f}% (threshold: 0.15%), range is {hl_range:.2f}% "
                f"(threshold: 1.2%). Waiting {retry_delay_sec}s for market to calm down before chasing..."
            )
            await asyncio.sleep(retry_delay_sec)
        else:
            log_action(
                f"{symbol} remained highly volatile after {max_retries} retries "
                f"(std dev: {std_dev:.3f}%, range: {hl_range:.2f}%). "
                f"Skipping chase for this session to protect against bad execution fills."
            )

    return False


def _overnight_limit_chase_mode(order_obj: Any, norm: Dict[str, Any]) -> Optional[ChaseMode]:
    """
    Whether an overnight limit is eligible for post-open chase, and how.

    - Open DAY/OPG limits → cancel then market.
    - Expired OPG limits → market only (auction miss).
    - Cancelled limits (any TIF) → never chase (explicit exit from the book).
    - Expired DAY limits → not chased (distinct from auction expiry).
    """
    status = _order_status_str(order_obj)
    if status in _CANCELLED_ORDER_STATUSES:
        return None
    if norm["time_in_force"] not in ("day", "opg") or norm["limit_price"] is None:
        return None
    if status in _OPEN_ORDER_STATUSES:
        return "open_limit"
    if status == "expired" and norm["time_in_force"] == "opg":
        return "expired_opg"
    return None


def collect_overnight_chase_candidates(
    orders: List[Any],
    lookback_cutoff_et,
    *,
    normalize_order,
) -> Dict[str, tuple]:
    """
    Newest chase-eligible overnight limit per symbol placed after ``lookback_cutoff_et``.

    Returns ``symbol -> (norm, created_at_et, order_obj, chase_mode)``.
    """
    symbol_orders: Dict[str, list] = {}
    for order_obj in orders:
        norm = normalize_order(order_obj)
        chase_mode = _overnight_limit_chase_mode(order_obj, norm)
        if not chase_mode:
            continue
        created_at = _order_created_at(order_obj)
        if not created_at:
            continue
        created_at_et = created_at.astimezone(lookback_cutoff_et.tzinfo)
        if created_at_et < lookback_cutoff_et:
            continue
        symbol = norm["symbol"]
        symbol_orders.setdefault(symbol, []).append(
            (norm, created_at_et, order_obj, chase_mode)
        )

    candidates: Dict[str, tuple] = {}
    for symbol, rows in symbol_orders.items():
        rows.sort(key=lambda x: x[1], reverse=True)
        candidates[symbol] = rows[0]
    return candidates


def collect_open_overnight_chase_candidates(
    open_orders: List[Any],
    lookback_cutoff_et,
    *,
    normalize_order,
) -> Dict[str, tuple]:
    """Backward-compatible wrapper — open orders only."""
    return collect_overnight_chase_candidates(
        open_orders,
        lookback_cutoff_et,
        normalize_order=normalize_order,
    )


def overnight_order_already_chased(
    symbol_orders: List[tuple],
    overnight_order_id: str,
    overnight_created_at,
    overnight_side: str,
) -> bool:
    """True when a market order for the same side was placed after the overnight limit."""
    for norm, created_at_et, _ in symbol_orders:
        if norm["order_id"] == overnight_order_id:
            continue
        if created_at_et <= overnight_created_at:
            continue
        is_market = norm.get("limit_price") is None
        if is_market and norm["side"] == overnight_side:
            return True
    return False


class ExecutionAgent(BaseAgent):
    """
    Executes trades on Alpaca
    - Manages order placement (limit overnight, market chase)
    - Tracks execution results
    - Handles position sizing based on signals
    """

    def __init__(self, system: str = "baseline"):
        super().__init__(system=system)
        self.alpaca_client = AlpacaClient(system=system)
        self.orders_placed = []

    async def place_overnight_orders(
        self,
        position_changes: Dict[str, float],
        reference_prices: Dict[str, float],
        current_positions: Optional[Dict] = None,
        dry_run: bool | None = None,
    ) -> Dict[str, Optional[str]]:
        """
        Place LIMIT orders overnight (OPG time-in-force)
        Buy: MOC - 0.5%
        Sell: MOC + 0.5%

        Args:
            position_changes: Dict mapping ticker -> qty change
            reference_prices: Dict mapping ticker -> reference price (close)

        Returns:
            Dict mapping ticker -> order ID (or None if failed)
        """
        order_ids: Dict[str, Optional[str]] = {}
        offset_pct = config.ORDER_CONFIG.get("overnight_limit_offset_pct", 0.005)
        simulate = dry_run if dry_run is not None else config.is_dry_run()

        if simulate:
            self.log_action(
                "DRY RUN — overnight orders will be simulated (no reconcile, no placement)",
                data={"dry_run": True},
                event_type="agent_action",
            )

        cancelled = [] if simulate else reconcile_duplicate_open_orders(self.alpaca_client)
        for c in cancelled:
            self.log_action(
                f"Cancelled duplicate {c['side']} {c['symbol']} "
                f"{c['cancelled_qty']:.0f} sh @ ${c['limit_price']:.2f}",
                data=c,
                event_type="order_cancelled_duplicate",
            )

        open_orders = self.alpaca_client.get_orders(status="open")
        account = self.alpaca_client.get_account() or {}
        buying_power = float(account.get("buying_power", 0))
        positions = current_positions or self.alpaca_client.get_positions()

        filtered_changes, skipped = filter_orders_for_placement(
            position_changes=position_changes,
            reference_prices=reference_prices,
            open_orders_raw=open_orders,
            buying_power=buying_power,
            current_positions=positions,
            offset_pct=offset_pct,
        )

        for skip in skipped:
            self.log_action(
                f"Skipped order {skip['ticker']}: {skip['reason']}",
                data=skip,
                event_type="order_skipped",
            )

        if skipped:
            self.log_action(
                f"Order filter: {len(filtered_changes)} to place, "
                f"{len(skipped)} skipped ({len(open_orders)} open orders, "
                f"${buying_power:,.2f} buying power)",
            )

        for ticker, qty_change in filtered_changes.items():
            if qty_change == 0:
                continue

            ref_price = reference_prices.get(ticker, 0)
            if not ref_price:
                self.log_error(f"Missing reference price for {ticker}")
                continue

            if qty_change > 0:
                limit_price = round_limit_price(ref_price * (1 - offset_pct))
                side = "buy"
            else:
                limit_price = round_limit_price(ref_price * (1 + offset_pct))
                side = "sell"
                qty_change = abs(qty_change)

            if simulate:
                order_ids[ticker] = "dry-run"
                self.log_action(
                    f"DRY RUN would place overnight {side}: {qty_change} {ticker} @ ${limit_price:.2f}",
                    data={
                        "ticker": ticker,
                        "side": side,
                        "qty": qty_change,
                        "limit_price": limit_price,
                        "time_in_force": "day",
                        "dry_run": True,
                        "reason": "dry_run",
                    },
                    event_type="order_skipped",
                )
                continue

            try:
                order_id = self.alpaca_client.place_limit_order(
                    ticker=ticker,
                    qty=qty_change,
                    limit_price=limit_price,
                    side=side,
                    time_in_force="day"
                )
                order_ids[ticker] = order_id
                if order_id:
                    self.log_action(
                        f"Placed overnight {side} order: {qty_change} {ticker} @ ${limit_price:.2f}",
                        data={
                            "ticker": ticker,
                            "side": side,
                            "qty": qty_change,
                            "limit_price": limit_price,
                            "order_id": order_id,
                            "time_in_force": "day",
                        },
                        event_type="order_placed",
                    )
                else:
                    self.log_error(
                        f"Failed to place overnight {side} order: "
                        f"{qty_change} {ticker} @ ${limit_price:.2f}"
                    )
            except Exception as e:
                self.log_error(f"Failed to place order for {ticker}: {e}")
                order_ids[ticker] = None

        return order_ids

    async def chase_unfilled_orders(self, fill_threshold: float = 0.7) -> Dict[str, Optional[str]]:
        """
        Post-market-open chase for unfilled overnight limits.

        - **Open DAY/OPG limits** → volatility gate, cancel, then market the remainder.
        - **Expired OPG limits** (opening auction miss) → volatility gate, then market only.
        - **Cancelled limits** (any TIF, including DAY) → never chased.
        """
        import pytz
        from src.market.calendar import check_chase_trading_session
        from src.strategies.order_dedup import normalize_open_order

        session_ok, session_reason = check_chase_trading_session(system=self.system)
        if not session_ok:
            self.log_action(f"Chase skipped: {session_reason}")
            return {}

        new_orders = {}
        ET = pytz.timezone("US/Eastern")
        now_et = datetime.now(ET)
        lookback_cutoff = _overnight_lookback_cutoff_et(now_et, system=self.system)

        try:
            all_orders = self.alpaca_client.get_orders(status="all")
            candidates = collect_overnight_chase_candidates(
                all_orders,
                lookback_cutoff,
                normalize_order=normalize_open_order,
            )
            if not candidates:
                self.log_action("No overnight limit orders eligible for chase")
                return new_orders

            symbol_history: Dict[str, list] = {}
            for order_obj in all_orders:
                norm = normalize_open_order(order_obj)
                symbol = norm["symbol"]
                created_at = _order_created_at(order_obj)
                if not created_at:
                    continue
                created_at_et = created_at.astimezone(ET)
                if created_at_et >= lookback_cutoff:
                    symbol_history.setdefault(symbol, []).append(
                        (norm, created_at_et, order_obj)
                    )

            for symbol, (on_norm, on_created_at, on_original, chase_mode) in candidates.items():
                qty = on_norm["qty"]
                filled_qty = on_norm["filled_qty"]
                fill_rate = (filled_qty / qty) if qty > 0 else 0.0
                if fill_rate >= fill_threshold:
                    self.log_action(
                        f"Skipping chase for {symbol}: overnight order {on_norm['order_id']} "
                        f"already {fill_rate * 100:.0f}% filled"
                    )
                    continue

                history = symbol_history.get(symbol, [])
                if overnight_order_already_chased(
                    history,
                    on_norm["order_id"],
                    on_created_at,
                    on_norm["side"],
                ):
                    self.log_action(
                        f"Skipping chase for {symbol}: already chased after overnight order "
                        f"{on_norm['order_id']}"
                    )
                    continue

                remaining_qty = round(qty - filled_qty, 4)
                if remaining_qty <= 0:
                    continue

                side = on_norm["side"]
                if not await _await_calm_market_for_chase(
                    self.alpaca_client,
                    symbol,
                    log_action=self.log_action,
                    log_error=self.log_error,
                ):
                    continue

                if chase_mode == "open_limit":
                    self.log_action(
                        f"Cancelling open overnight limit order {on_norm['order_id']} for {symbol} "
                        f"to chase with market order"
                    )
                    cancelled = self.alpaca_client.cancel_order(on_norm["order_id"])
                    if not cancelled:
                        self.log_error(
                            f"Failed to cancel order {on_norm['order_id']} for {symbol} — skipping market chase"
                        )
                        continue
                else:
                    self.log_action(
                        f"Chasing expired OPG limit {on_norm['order_id']} for {symbol} with market order"
                    )

                try:
                    order_id = self.alpaca_client.place_market_order(
                        ticker=symbol,
                        qty=remaining_qty,
                        side=side,
                    )
                    if not order_id:
                        self.log_error(
                            f"Failed to place market chase for {symbol} "
                            f"(remaining {remaining_qty} {side.upper()})"
                        )
                        continue
                    new_orders[symbol] = order_id
                    self.log_action(
                        f"Chased unfilled overnight {on_norm['time_in_force'].upper()} order "
                        f"{on_norm['order_id']} for {symbol}: "
                        f"{fill_rate*100:.0f}% filled, placed market order {order_id} "
                        f"for remaining {remaining_qty} {side.upper()}",
                        event_type="order_chased",
                    )
                except Exception as e:
                    self.log_error(f"Failed to chase order for {symbol}: {e}")

        except Exception as e:
            self.log_error(f"Failed to run chase unfilled orders: {e}")

        return new_orders

    async def execute(self) -> bool:
        """Execute order management workflow"""
        self.log_action("Starting execution agent")
        return True
