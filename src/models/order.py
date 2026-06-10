"""
Order data model - represents trading orders
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class OrderType(Enum):
    """Order types"""
    BUY = "buy"
    SELL = "sell"


class OrderTimeInForce(Enum):
    """Time in force"""
    DAY = "day"
    GTC = "gtc"  # Good till canceled
    OPG = "opg"  # At open
    CLS = "cls"  # At close


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    ACCEPTED = "accepted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """
    Represents a trading order
    """
    ticker: str
    qty: float
    order_type: OrderType  # BUY or SELL
    limit_price: Optional[float] = None  # For limit orders
    timestamp: Optional[datetime] = None  # When order was placed

    # Order attributes
    time_in_force: OrderTimeInForce = OrderTimeInForce.DAY
    status: OrderStatus = OrderStatus.PENDING

    # Execution details
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    filled_timestamp: Optional[datetime] = None

    # Alpaca order ID
    alpaca_order_id: Optional[str] = None

    # Metadata
    system: str = "baseline"  # "baseline" or "internal"
    reason: Optional[str] = None  # Why this order was placed

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        d = asdict(self)
        # Convert enums to strings for serialization
        d['order_type'] = self.order_type.value
        d['time_in_force'] = self.time_in_force.value
        d['status'] = self.status.value
        return d

    def __repr__(self) -> str:
        return (
            f"Order({self.order_type.value} {self.qty} {self.ticker} @ "
            f"{self.limit_price or 'market'}, status={self.status.value})"
        )

    @property
    def is_filled(self) -> bool:
        """Order is completely filled"""
        return self.status == OrderStatus.FILLED

    @property
    def fill_rate(self) -> float:
        """Percentage of order filled"""
        if self.qty == 0:
            return 0.0
        return self.filled_qty / self.qty

    @property
    def order_value(self) -> float:
        """Total order value (qty * price)"""
        price = self.limit_price or self.filled_avg_price or 0
        return self.qty * price

    @property
    def filled_value(self) -> float:
        """Total filled value"""
        if self.filled_avg_price:
            return self.filled_qty * self.filled_avg_price
        return 0.0
