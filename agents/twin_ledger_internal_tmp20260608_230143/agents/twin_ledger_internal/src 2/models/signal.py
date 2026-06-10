"""
Signal data model - represents a trading signal for a ticker
"""
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum


class SignalSource(Enum):
    """Sources that contribute to a signal"""
    MARKETCRUNCH_AI = "marketcrunch_ai_estimates"
    TECHNICAL_INDICATORS = "technical_indicators"
    DATABENTO_DISCOVERY = "databento_discovery"
    HISTORICAL_FEATURES = "historical_features"
    LLM_JUDGMENT = "llm_judgment"


@dataclass
class Signal:
    """
    Complete trading signal for a ticker
    Combines multiple data sources
    """
    ticker: str
    timestamp: datetime

    # Core MarketCrunch prediction (System B)
    predicted_return: float  # Expected return as decimal (e.g., 0.0175 for 1.75%)
    confidence: float  # Win rate / confidence (0-1)

    # Technical indicators (both systems)
    bollinger_zscore: Optional[float] = None
    macd_histogram: Optional[float] = None
    rsi_14: Optional[float] = None

    # DataBento discovered features (System B only)
    databento_features: Optional[Dict[str, float]] = None

    # Historical context
    historical_features: Optional[Dict[str, float]] = None

    # LLM judgment
    llm_reasoning: Optional[str] = None

    # Metadata
    source: Optional[SignalSource] = SignalSource.MARKETCRUNCH_AI
    system: str = "baseline"  # "baseline" or "internal"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage"""
        return asdict(self)

    def __repr__(self) -> str:
        return (
            f"Signal(ticker={self.ticker}, predicted_return={self.predicted_return:.4f}, "
            f"confidence={self.confidence:.2f}, system={self.system})"
        )

    @property
    def is_bullish(self) -> bool:
        """Signal direction is bullish"""
        return self.predicted_return > 0

    @property
    def is_bearish(self) -> bool:
        """Signal direction is bearish"""
        return self.predicted_return < 0

    @property
    def signal_strength(self) -> float:
        """Combined signal strength (0-1)"""
        base = self.confidence

        # Boost from DataBento if available
        if self.databento_features:
            db_boost = min(0.1, len(self.databento_features) * 0.02)  # Max +10%
            base += db_boost

        # Cap at 1.0
        return min(1.0, base)
