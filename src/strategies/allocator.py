"""
Position Allocator (SHARED)
- Kelly Criterion sizing for System B (internal)
- Equal weight or equal-risk sizing for System A (baseline)
- Adjusts for confidence and signal strength
"""
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np
from src.models.signal import Signal
from src.logger import setup_logger

if TYPE_CHECKING:
    from src.models.trading_decision import TradingDecision
    from src.models.position import Position

logger = setup_logger(__name__)


class PositionAllocator:
    """
    Calculates position sizes based on signal strength and risk constraints
    """

    @staticmethod
    def kelly_criterion(
        predicted_return: float,
        confidence: float,
        estimated_win_rate: float = 0.55,
        max_kelly: float = 0.25
    ) -> float:
        """
        Kelly Criterion: f* = (bp - q) / b
        where:
        - b = odds (return multiple)
        - p = win probability
        - q = loss probability (1-p)

        Args:
            predicted_return: Expected return (e.g., 0.02 for 2%)
            confidence: Confidence score (0-1)
            estimated_win_rate: Historical win rate
            max_kelly: Cap kelly allocation (default 25%)

        Returns:
            Position size as fraction of portfolio (0-1)
        """
        try:
            # Map confidence to win rate adjustment
            # High confidence = higher win rate
            adjusted_win_rate = estimated_win_rate + (confidence - 0.5) * 0.2  # ±10%
            adjusted_win_rate = np.clip(adjusted_win_rate, 0.35, 0.65)

            # Risk/reward based on predicted return
            if predicted_return <= 0:
                return 0.0

            # Simple kelly: assume 1:1 risk/reward
            # f = (p - q) / 1
            kelly = adjusted_win_rate - (1 - adjusted_win_rate)
            kelly = np.clip(kelly, 0, max_kelly)

            return kelly

        except Exception as e:
            logger.error(f"Kelly calculation error: {e}")
            return 0.0

    @staticmethod
    def baseline_target_weights(signals: Dict[str, Signal]) -> Dict[str, float]:
        """Equal-weight targets capped at 10% per position."""
        if not signals:
            return {}
        weight = min(1.0 / len(signals), 0.10)
        return {ticker: weight for ticker in signals}

    @staticmethod
    def internal_target_weights(signals: Dict[str, Signal]) -> Dict[str, float]:
        """Kelly-normalized targets capped at 10% per position."""
        if not signals:
            return {}

        kelly_weights = {}
        total_kelly = 0.0
        for ticker, signal in signals.items():
            kelly = PositionAllocator.kelly_criterion(
                predicted_return=signal.predicted_return,
                confidence=signal.confidence,
            )
            kelly_weights[ticker] = kelly
            total_kelly += kelly

        weights = {}
        for ticker, kelly in kelly_weights.items():
            if kelly <= 0:
                weights[ticker] = 0.0
                continue
            weight = kelly / total_kelly if total_kelly > 0 else 1.0 / len(signals)
            weights[ticker] = min(weight, 0.10)
        return weights

    @staticmethod
    def decision_target_weights(decisions: List["TradingDecision"]) -> Dict[str, float]:
        """Portfolio weights implied by Twin Ledger BUY/SHORT entry decisions."""
        weights = {}
        for decision in decisions:
            if decision.action in ("BUY", "SHORT") and decision.size_pct > 0:
                weights[decision.ticker] = min(decision.size_pct, 0.10)
        return weights

    @staticmethod
    def allocate_from_decisions(
        decisions: List["TradingDecision"],
        portfolio_value: float,
        current_positions: Dict[str, "Position"],
        prices: Dict[str, float],
    ) -> Dict[str, int]:
        """
        Convert Twin Ledger decisions into signed share quantities.

        Positive qty = buy, negative qty = sell.
        """
        position_changes: Dict[str, int] = {}

        for decision in decisions:
            price = prices.get(decision.ticker, 0)
            if price <= 0:
                logger.warning(f"No price for {decision.ticker}, skipping {decision.action}")
                continue

            if decision.action == "BUY":
                weight = min(max(decision.size_pct, 0), 0.10)
                dollars = portfolio_value * weight
                qty = int(dollars / price)
                if qty > 0:
                    position_changes[decision.ticker] = qty

            elif decision.action == "SHORT":
                position = current_positions.get(decision.ticker)
                if position and position.qty > 0:
                    continue
                weight = min(max(decision.size_pct, 0), 0.10)
                dollars = portfolio_value * weight
                qty = int(dollars / price)
                if qty > 0:
                    position_changes[decision.ticker] = -qty

            elif decision.action in ("SELL", "CLOSE"):
                position = current_positions.get(decision.ticker)
                if not position or position.qty <= 0:
                    continue
                if decision.action == "CLOSE":
                    sell_qty = int(position.qty)
                else:
                    sell_pct = min(max(decision.size_pct, 0.01), 1.0)
                    sell_qty = max(1, int(position.qty * sell_pct))
                    sell_qty = min(sell_qty, int(position.qty))
                position_changes[decision.ticker] = -sell_qty

            elif decision.action == "COVER":
                position = current_positions.get(decision.ticker)
                if not position or position.qty >= 0:
                    continue
                short_qty = abs(int(position.qty))
                cover_pct = min(max(decision.size_pct, 0.01), 1.0)
                cover_qty = max(1, int(short_qty * cover_pct))
                cover_qty = min(cover_qty, short_qty)
                position_changes[decision.ticker] = cover_qty

        logger.info(f"Twin Ledger allocation: {len(position_changes)} order(s) from decisions")
        return position_changes

    @staticmethod
    def allocate_internal_from_decisions(
        decisions: List["TradingDecision"],
        buy_signals: Dict[str, Signal],
        portfolio_value: float,
        current_positions: Dict[str, "Position"],
        prices: Dict[str, float],
    ) -> Dict[str, int]:
        """
        Internal Twin Ledger sizing: Kelly for BUYs, decision-based for SELL/CLOSE.
        """
        position_changes: Dict[str, int] = {}

        exit_decisions = [
            d for d in decisions if d.action in ("SELL", "CLOSE", "SHORT", "COVER")
        ]
        if exit_decisions:
            position_changes.update(
                PositionAllocator.allocate_from_decisions(
                    exit_decisions, portfolio_value, current_positions, prices
                )
            )

        buy_tickers = {d.ticker for d in decisions if d.action == "BUY"}
        kelly_signals = {t: s for t, s in buy_signals.items() if t in buy_tickers}
        if kelly_signals:
            allocations = PositionAllocator.allocate_internal(kelly_signals, portfolio_value)
            for ticker, dollars in allocations.items():
                price = prices.get(ticker, 0)
                if price > 0:
                    qty = int(dollars / price)
                    if qty > 0:
                        position_changes[ticker] = qty

        logger.info(
            f"Internal Twin Ledger allocation: {len(position_changes)} order(s) "
            f"({len(kelly_signals)} Kelly BUY, {len(exit_decisions)} exit/short/cover)"
        )
        return position_changes

    @staticmethod
    def allocate_baseline(signals: Dict[str, Signal], portfolio_value: float) -> Dict[str, float]:
        """
        Baseline allocation: equal weight across signals

        Args:
            signals: Dict mapping ticker -> Signal
            portfolio_value: Total portfolio value

        Returns:
            Dict mapping ticker -> position size ($)
        """
        if not signals:
            return {}

        weights = PositionAllocator.baseline_target_weights(signals)
        allocations = {
            ticker: portfolio_value * weight
            for ticker, weight in weights.items()
        }

        position_size = next(iter(allocations.values()), 0)
        logger.info(f"Baseline allocation: {len(allocations)} positions @ ${position_size:,.2f} each")
        return allocations

    @staticmethod
    def allocate_internal(signals: Dict[str, Signal], portfolio_value: float) -> Dict[str, float]:
        """
        Internal allocation: Kelly Criterion with confidence weighting

        Args:
            signals: Dict mapping ticker -> Signal
            portfolio_value: Total portfolio value

        Returns:
            Dict mapping ticker -> position size ($)
        """
        if not signals:
            return {}

        weights = PositionAllocator.internal_target_weights(signals)
        allocations = {
            ticker: portfolio_value * weight
            for ticker, weight in weights.items()
            if weight > 0
        }

        total_kelly = sum(weights.values())
        logger.info(
            f"Internal allocation: {len(allocations)} positions, "
            f"total Kelly weight: {total_kelly:.4f}"
        )
        return allocations

    @staticmethod
    def calculate_quantities(
        allocations: Dict[str, float],
        prices: Dict[str, float]
    ) -> Dict[str, int]:
        """
        Convert dollar allocations to share quantities

        Args:
            allocations: Dict mapping ticker -> position size ($)
            prices: Dict mapping ticker -> current price ($)

        Returns:
            Dict mapping ticker -> shares to buy
        """
        quantities = {}

        for ticker, allocation in allocations.items():
            price = prices.get(ticker, 0)
            if price > 0:
                qty = int(allocation / price)
                quantities[ticker] = qty

        logger.info(f"Calculated quantities for {len(quantities)} tickers")
        return quantities
