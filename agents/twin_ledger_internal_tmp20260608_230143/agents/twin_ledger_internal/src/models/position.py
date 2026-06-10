"""
Position data model - represents a current trading position
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any


@dataclass
class Position:
    """
    Represents a current open position
    """
    ticker: str
    qty: float  # Quantity of shares
    avg_entry_price: float  # Average entry price
    current_price: float  # Current market price
    entry_date: datetime

    # P&L metrics
    unrealized_pnl: Optional[float] = None  # Dollar P&L
    unrealized_return: Optional[float] = None  # P&L as percentage

    # Metadata
    source_signal: Optional[str] = None  # Which signal generated this position

    def __post_init__(self):
        """Calculate P&L metrics"""
        if self.current_price and self.avg_entry_price:
            self.unrealized_pnl = (self.current_price - self.avg_entry_price) * self.qty
            self.unrealized_return = (self.current_price - self.avg_entry_price) / self.avg_entry_price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"Position(ticker={self.ticker}, qty={self.qty}, "
            f"entry={self.avg_entry_price:.2f}, current={self.current_price:.2f}, "
            f"return={self.unrealized_return*100:.2f}%)"
        )

    @property
    def market_value(self) -> float:
        """Total market value of position"""
        return self.qty * self.current_price

    @property
    def cost_basis(self) -> float:
        """Total cost basis"""
        return self.qty * self.avg_entry_price

    @property
    def is_profitable(self) -> bool:
        """Position is in profit"""
        return self.unrealized_return > 0 if self.unrealized_return else False

    @property
    def days_held(self) -> int:
        """Number of days position held"""
        return (datetime.now() - self.entry_date).days
