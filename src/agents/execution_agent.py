"""
Execution Agent (SHARED by both systems)
Places orders on Alpaca:
- Overnight: LIMIT orders at MOC ± 0.5%
- Post-open: Chase unfilled with MARKET orders
- EOD: Hold/trim based on next-day predictions
"""
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
        Post-market-open: chase unfilled overnight orders (both DAY and OPG) with MARKET orders.
        - For active DAY limit orders (still "open"), we cancel them first and place a market order for the remaining qty.
        - For expired/cancelled OPG limit orders, we directly place a market order for the remaining qty.
        To avoid double-chasing, we check if a market order has already been placed for the symbol after the overnight order.

        Args:
            fill_threshold: Threshold % for chasing (default 70%)

        Returns:
            Dict mapping ticker -> new order ID
        """
        from datetime import datetime, timezone, timedelta
        import pytz
        from src.strategies.order_dedup import normalize_open_order

        new_orders = {}
        ET = pytz.timezone("US/Eastern")
        now_et = datetime.now(ET)
        
        # Determine the start of the current trading session's order placement window.
        # Orders placed after the previous trading day's close (4:00 PM ET) are for today's session.
        # This ensures we do not check or chase expired/cancelled orders from previous trading days.
        if now_et.weekday() == 0:  # Monday
            # Look back to Friday 4:00 PM ET
            lookback_cutoff = (now_et - timedelta(days=3)).replace(hour=16, minute=0, second=0, microsecond=0)
        else:
            # Look back to yesterday 4:00 PM ET
            lookback_cutoff = (now_et - timedelta(days=1)).replace(hour=16, minute=0, second=0, microsecond=0)

        def _order_created_at(order_obj: Any) -> Optional[datetime]:
            created_at = getattr(order_obj, "created_at", None)
            if created_at is None:
                created_at = order_obj.get("created_at")
            if isinstance(created_at, str):
                try:
                    if created_at.endswith("Z"):
                        created_at = created_at[:-1] + "+00:00"
                    return datetime.fromisoformat(created_at)
                except Exception:
                    return None
            return created_at

        try:
            # Query all orders (both open and closed/cancelled/expired)
            all_orders = self.alpaca_client.get_orders(status="all")
            
            # Group normalized orders by symbol
            symbol_orders = {}
            for o in all_orders:
                norm = normalize_open_order(o)
                symbol = norm["symbol"]
                created_at = _order_created_at(o)
                if not created_at:
                    continue
                created_at_et = created_at.astimezone(ET)
                
                # Only consider orders within our lookback window
                if created_at_et >= lookback_cutoff:
                    if symbol not in symbol_orders:
                        symbol_orders[symbol] = []
                    symbol_orders[symbol].append((norm, created_at_et, o))

            # Process each symbol
            for symbol, orders in symbol_orders.items():
                # Sort orders by creation time descending (most recent first)
                orders.sort(key=lambda x: x[1], reverse=True)
                
                # Find the most recent overnight limit order (TIF is "day" or "opg", and has limit price)
                overnight_order_info = None
                for norm, created_at_et, original_obj in orders:
                    if norm["time_in_force"] in ("day", "opg") and norm["limit_price"] is not None:
                        overnight_order_info = (norm, created_at_et, original_obj)
                        break
                
                if not overnight_order_info:
                    continue
                
                on_norm, on_created_at, on_original = overnight_order_info
                qty = on_norm["qty"]
                filled_qty = on_norm["filled_qty"]
                fill_rate = (filled_qty / qty) if qty > 0 else 0.0
                
                if fill_rate < fill_threshold:
                    # Check if we already placed a market order for this symbol after the overnight order
                    already_chased = False
                    for norm, created_at_et, _ in orders:
                        if norm["order_id"] != on_norm["order_id"] and created_at_et > on_created_at:
                            is_market = norm.get("limit_price") is None
                            if is_market and norm["side"] == on_norm["side"]:
                                already_chased = True
                                break
                    
                    if already_chased:
                        self.log_action(
                            f"Skipping chase for {symbol}: already chased after overnight order {on_norm['order_id']}"
                        )
                        continue
                    
                    remaining_qty = round(qty - filled_qty, 4)
                    if remaining_qty <= 0:
                        continue
                        
                    side = on_norm["side"]
                    
                    # If the overnight order is still open (e.g. a DAY order), cancel it first
                    status_val = getattr(on_original, "status", None)
                    if status_val is None:
                        status_val = on_original.get("status")
                    if hasattr(status_val, "value"):
                        status_val = status_val.value
                    status_str = str(status_val).lower() if status_val else ""
                    
                    is_open = status_str in ("new", "partially_filled", "accepted", "pending_new", "accepted_for_bidding", "held")
                    
                    if is_open:
                        self.log_action(f"Cancelling open overnight limit order {on_norm['order_id']} for {symbol} to chase with market order")
                        cancelled = self.alpaca_client.cancel_order(on_norm["order_id"])
                        if not cancelled:
                            self.log_error(f"Failed to cancel order {on_norm['order_id']} for {symbol} — skipping market chase")
                            continue
                    
                    # Place a market order for the remaining quantity
                    try:
                        # Adaptive Execution: Check recent market conditions (volatility in the last 30 minutes)
                        # We define "calm" as:
                        # - Standard deviation of 1-minute returns is < 0.15% (per minute)
                        # - Total high-low range over the last 30 minutes is < 1.2%
                        # If the ticker is too volatile, we wait up to 3 retries (1 minute apart) for it to calm down.
                        max_retries = 3
                        retry_delay_sec = 60
                        is_calm = False
                        
                        for attempt in range(max_retries + 1):
                            vol_metrics = self.alpaca_client.get_recent_volatility(symbol, minutes=30)
                            if not vol_metrics:
                                if attempt < max_retries:
                                    self.log_action(
                                        f"⚠️ Could not calculate volatility for {symbol} (insufficient data) — "
                                        f"retrying in {retry_delay_sec}s..."
                                    )
                                    import asyncio
                                    await asyncio.sleep(retry_delay_sec)
                                else:
                                    self.log_error(
                                        f"❌ Failed to calculate volatility for {symbol} after {max_retries} retries — "
                                        f"skipping chase to prevent unsafe execution"
                                    )
                                continue
                                
                            std_dev = vol_metrics["std_dev_pct"]
                            hl_range = vol_metrics["range_pct"]
                            
                            # Thresholds: std_dev < 0.15% AND range < 1.2%
                            if std_dev < 0.15 and hl_range < 1.2:
                                self.log_action(
                                    f"Market conditions for {symbol} are calm: "
                                    f"30-min std dev is {std_dev:.3f}% (< 0.15%), range is {hl_range:.2f}% (< 1.2%). "
                                    f"Proceeding with chase."
                                )
                                is_calm = True
                                break
                            else:
                                if attempt < max_retries:
                                    self.log_action(
                                        f"⚠️ {symbol} is highly volatile right now (attempt {attempt+1}/{max_retries+1}): "
                                        f"30-min std dev is {std_dev:.3f}% (threshold: 0.15%), range is {hl_range:.2f}% (threshold: 1.2%). "
                                        f"Waiting {retry_delay_sec}s for market to calm down before chasing..."
                                    )
                                    import asyncio
                                    await asyncio.sleep(retry_delay_sec)
                                else:
                                    self.log_action(
                                        f"❌ {symbol} remained highly volatile after {max_retries} retries "
                                        f"(std dev: {std_dev:.3f}%, range: {hl_range:.2f}%). "
                                        f"Skipping chase for this session to protect against bad execution fills."
                                    )
                                    
                        if not is_calm:
                            continue

                        order_id = self.alpaca_client.place_market_order(
                            ticker=symbol,
                            qty=remaining_qty,
                            side=side
                        )
                        new_orders[symbol] = order_id
                        self.log_action(
                            f"Chased unfilled overnight {on_norm['time_in_force'].upper()} order {on_norm['order_id']} for {symbol}: "
                            f"{fill_rate*100:.0f}% filled, placed market order {order_id} "
                            f"for remaining {remaining_qty} {side.upper()}",
                            event_type="order_chased"
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
