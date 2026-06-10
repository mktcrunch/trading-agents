"""
Order Manager (SHARED)
Manages overnight LIMIT orders + post-open market chase logic
Tracks order fill rates and timing
"""
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from src.apis.alpaca_client import round_limit_price
from src.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class OrderRequest:
    """Order placement request"""
    ticker: str
    qty: int
    side: str  # "buy" or "sell"
    order_type: str  # "limit" or "market"
    limit_price: Optional[float] = None
    time_in_force: str = "day"  # "opg", "cls", "day", "ioc", "fok"
    timestamp: Optional[str] = None


class OrderManager:
    """
    Manages order lifecycle:
    - Place overnight LIMIT orders (OPG)
    - Track fill rates
    - Chase with MARKET orders post-open
    - Cancel unfilled at EOD
    """

    def __init__(self):
        self.orders_placed = []
        self.orders_filled = []
        self.orders_failed = []

    def build_overnight_orders(
        self,
        position_changes: Dict[str, int],
        reference_prices: Dict[str, float],
        spread_pct: float = 0.5
    ) -> List[OrderRequest]:
        """
        Build LIMIT order requests for overnight (OPG)

        Args:
            position_changes: Dict mapping ticker -> qty change
            reference_prices: Dict mapping ticker -> reference price
            spread_pct: Buy spread below price, sell spread above

        Returns:
            List of OrderRequest objects
        """
        orders = []

        for ticker, qty in position_changes.items():
            if qty == 0:
                continue

            ref_price = reference_prices.get(ticker, 0)
            if not ref_price:
                logger.warning(f"Missing reference price for {ticker}")
                continue

            # Calculate limit price
            if qty > 0:  # BUY
                limit_price = ref_price * (1 - spread_pct / 100)
                side = "buy"
            else:  # SELL
                limit_price = ref_price * (1 + spread_pct / 100)
                side = "sell"
                qty = abs(qty)

            order = OrderRequest(
                ticker=ticker,
                qty=qty,
                side=side,
                order_type="limit",
                limit_price=round_limit_price(limit_price),
                time_in_force="opg",
                timestamp=datetime.now().isoformat()
            )
            orders.append(order)

        logger.info(f"Built {len(orders)} overnight LIMIT orders")
        return orders

    def build_chase_orders(
        self,
        open_orders: List[Dict],
        fill_threshold: float = 0.70
    ) -> List[OrderRequest]:
        """
        Build MARKET order requests to chase unfilled overnight orders

        Args:
            open_orders: List of open order dicts from Alpaca
            fill_threshold: Chase if fill rate < threshold

        Returns:
            List of OrderRequest objects for market chase
        """
        chase_orders = []

        for order in open_orders:
            fill_rate = order.get('filled_qty', 0) / order.get('qty', 1) if order.get('qty', 0) > 0 else 0

            if fill_rate < fill_threshold:
                remaining_qty = order.get('qty', 0) - order.get('filled_qty', 0)
                side = "buy" if order.get('side') == "buy" else "sell"

                chase_order = OrderRequest(
                    ticker=order.get('symbol'),
                    qty=remaining_qty,
                    side=side,
                    order_type="market",
                    time_in_force="opg"
                )
                chase_orders.append(chase_order)

        logger.info(f"Built {len(chase_orders)} chase MARKET orders")
        return chase_orders

    def log_order_request(self, order: OrderRequest):
        """Log order request"""
        logger.info(
            f"Order: {order.side.upper()} {order.qty} {order.ticker} "
            f"@ ${order.limit_price:.2f}" if order.limit_price else
            f"@ MARKET"
        )
        self.orders_placed.append(order)

    def track_fill(self, ticker: str, filled_qty: int, total_qty: int, fill_price: float):
        """Track order fill"""
        fill_rate = filled_qty / total_qty if total_qty > 0 else 0
        logger.info(f"Fill: {ticker} {filled_qty}/{total_qty} ({fill_rate*100:.0f}%) @ ${fill_price:.2f}")

    def build_eod_cleanup_orders(
        self,
        open_orders: List[Dict],
        next_day_signals: Dict = None
    ) -> List[OrderRequest]:
        """
        Build EOD cleanup orders
        Trim losing positions if next-day confidence < 0.55, else hold

        Args:
            open_orders: Current open orders
            next_day_signals: Tomorrow's signals (optional)

        Returns:
            List of OrderRequest for exits
        """
        cleanup_orders = []

        # Placeholder: will be implemented with real next-day signal logic
        logger.info("EOD cleanup: checking positions for trim/hold")

        return cleanup_orders
