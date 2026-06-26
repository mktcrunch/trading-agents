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


def _max_position_weight() -> float:
    from src import config
    return float(config.MAX_POSITION_SIZE_PCT)


class PositionAllocator:
    """
    Calculates position sizes based on signal strength and risk constraints
    """

    @staticmethod
    def kelly_edge_for_side(predicted_return: float, side: str) -> float:
        """MC edge magnitude for Kelly: positive target → long, negative → short."""
        side = (side or "long").lower()
        ret = float(predicted_return or 0)
        if side == "short":
            return max(0.0, -ret)
        return max(0.0, ret)

    @staticmethod
    def kelly_criterion(
        predicted_return: float,
        confidence: float,
        estimated_win_rate: float = 0.55,
        max_kelly: float = 0.25,
        side: str = "long",
    ) -> float:
        """
        Kelly Criterion: f* = (bp - q) / b
        where:
        - b = odds (return multiple)
        - p = win probability
        - q = loss probability (1-p)

        Args:
            predicted_return: MC expected return (e.g., 0.02 for +2%, -0.01 for -1%)
            confidence: Confidence score (0-1)
            estimated_win_rate: Historical win rate
            max_kelly: Cap kelly allocation (default 25%)
            side: ``long`` uses positive MC edge; ``short`` uses negative MC edge

        Returns:
            Position size as fraction of portfolio (0-1)
        """
        try:
            edge = PositionAllocator.kelly_edge_for_side(predicted_return, side)
            if edge <= 0:
                return 0.0

            adjusted_win_rate = estimated_win_rate + (confidence - 0.5) * 0.2  # ±10%
            adjusted_win_rate = np.clip(adjusted_win_rate, 0.35, 0.65)

            kelly = adjusted_win_rate - (1 - adjusted_win_rate)
            kelly = np.clip(kelly, 0, max_kelly)

            return kelly

        except Exception as e:
            logger.error(f"Kelly calculation error: {e}")
            return 0.0

    @staticmethod
    def _normalize_kelly_weights(
        kelly_weights: Dict[str, float],
        signals: Dict[str, Signal],
    ) -> Dict[str, float]:
        total_kelly = sum(kelly_weights.values())
        weights: Dict[str, float] = {}
        for ticker, kelly in kelly_weights.items():
            if kelly <= 0:
                weights[ticker] = 0.0
                continue
            weight = kelly / total_kelly if total_kelly > 0 else 1.0 / len(signals)
            weights[ticker] = min(weight, _max_position_weight())
        return weights

    @staticmethod
    def internal_entry_target_weights(
        signals: Dict[str, Signal],
        entry_sides: Dict[str, str],
        max_kelly: Optional[float] = None,
    ) -> Dict[str, float]:
        """Kelly-normalized portfolio weights for BUY (long) and SHORT entries."""
        if not signals:
            return {}

        from src import config

        cap = max_kelly if max_kelly is not None else config.INTERNAL_CONFIG.get(
            "kelly_fraction", 0.25
        )
        kelly_weights: Dict[str, float] = {}
        for ticker, signal in signals.items():
            side = entry_sides.get(ticker, "long")
            kelly_weights[ticker] = PositionAllocator.kelly_criterion(
                predicted_return=signal.predicted_return,
                confidence=signal.confidence,
                max_kelly=cap,
                side=side,
            )
        return PositionAllocator._normalize_kelly_weights(kelly_weights, signals)

    @staticmethod
    def internal_target_weights(signals: Dict[str, Signal]) -> Dict[str, float]:
        """Kelly-normalized long-only targets (legacy helper)."""
        if not signals:
            return {}
        entry_sides = {ticker: "long" for ticker in signals}
        return PositionAllocator.internal_entry_target_weights(signals, entry_sides)

    @staticmethod
    def baseline_target_weights(signals: Dict[str, Signal]) -> Dict[str, float]:
        """Equal-weight targets capped at max position weight."""
        if not signals:
            return {}
        cap = _max_position_weight()
        weight = min(1.0 / len(signals), cap)
        return {ticker: weight for ticker in signals}

    @staticmethod
    def decision_target_weights(decisions: List["TradingDecision"]) -> Dict[str, float]:
        """Portfolio weights implied by Twin Ledger BUY/SHORT entry decisions."""
        weights = {}
        for decision in decisions:
            if decision.action in ("BUY", "SHORT") and decision.size_pct > 0:
                weights[decision.ticker] = min(decision.size_pct, _max_position_weight())
        return weights

    @staticmethod
    def _shares_from_portfolio_weight(
        size_pct: float,
        portfolio_value: float,
        price: float,
        max_shares: int,
    ) -> int:
        """size_pct is portfolio weight; return share count capped at max_shares."""
        if portfolio_value <= 0 or price <= 0 or max_shares <= 0:
            return 0
        weight = min(max(float(size_pct), 0.0), 1.0)
        if weight <= 0:
            return 0
        qty = int(portfolio_value * weight / price)
        if qty <= 0:
            return 0
        return min(qty, max_shares)

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
        cap = _max_position_weight()

        for decision in decisions:
            price = prices.get(decision.ticker, 0)
            if price <= 0:
                logger.warning(f"No price for {decision.ticker}, skipping {decision.action}")
                continue

            if decision.action == "BUY":
                weight = min(max(decision.size_pct, 0), cap)
                dollars = portfolio_value * weight
                qty = int(dollars / price)
                if qty > 0:
                    position_changes[decision.ticker] = qty

            elif decision.action == "SHORT":
                position = current_positions.get(decision.ticker)
                if position and position.qty > 0:
                    continue
                weight = min(max(decision.size_pct, 0), cap)
                dollars = portfolio_value * weight
                qty = int(dollars / price)
                if qty > 0:
                    position_changes[decision.ticker] = -qty

            elif decision.action == "CLOSE":
                position = current_positions.get(decision.ticker)
                if not position or position.qty == 0:
                    continue
                if position.qty > 0:
                    position_changes[decision.ticker] = -int(position.qty)
                else:
                    position_changes[decision.ticker] = abs(int(position.qty))

            elif decision.action == "SELL":
                position = current_positions.get(decision.ticker)
                if not position or position.qty <= 0:
                    continue
                sell_qty = PositionAllocator._shares_from_portfolio_weight(
                    decision.size_pct,
                    portfolio_value,
                    price,
                    int(position.qty),
                )
                if sell_qty > 0:
                    position_changes[decision.ticker] = -sell_qty

            elif decision.action == "COVER":
                position = current_positions.get(decision.ticker)
                if not position or position.qty >= 0:
                    continue
                short_qty = abs(int(position.qty))
                cover_qty = PositionAllocator._shares_from_portfolio_weight(
                    decision.size_pct,
                    portfolio_value,
                    price,
                    short_qty,
                )
                if cover_qty > 0:
                    position_changes[decision.ticker] = cover_qty

        logger.info(f"Twin Ledger allocation: {len(position_changes)} order(s) from decisions")
        return position_changes

    @staticmethod
    def allocate_internal_from_decisions(
        decisions: List["TradingDecision"],
        entry_signals: Dict[str, Signal],
        portfolio_value: float,
        current_positions: Dict[str, "Position"],
        prices: Dict[str, float],
        entry_sides: Dict[str, str],
    ) -> Dict[str, int]:
        """
        Internal Twin Ledger sizing: Kelly for BUY and SHORT entries; decision-based exits.
        """
        position_changes: Dict[str, int] = {}

        exit_decisions = [
            d for d in decisions if d.action in ("SELL", "CLOSE", "COVER")
        ]
        if exit_decisions:
            position_changes.update(
                PositionAllocator.allocate_from_decisions(
                    exit_decisions, portfolio_value, current_positions, prices
                )
            )

        entry_tickers = {
            d.ticker for d in decisions if d.action in ("BUY", "SHORT")
        }
        kelly_signals = {
            t: s for t, s in entry_signals.items() if t in entry_tickers
        }
        sides = {
            t: entry_sides[t]
            for t in kelly_signals
            if t in entry_sides
        }
        weights = PositionAllocator.internal_entry_target_weights(
            kelly_signals, sides
        )

        kelly_buys = 0
        kelly_shorts = 0
        for ticker, weight in weights.items():
            if weight <= 0:
                continue
            side = sides.get(ticker, "long")
            price = prices.get(ticker, 0)
            if price <= 0:
                logger.warning(f"No price for {ticker}, skipping Kelly {side}")
                continue
            if side == "short":
                position = current_positions.get(ticker)
                if position and position.qty > 0:
                    continue
                kelly_shorts += 1
            else:
                kelly_buys += 1
            dollars = portfolio_value * weight
            qty = int(dollars / price)
            if qty > 0:
                position_changes[ticker] = qty if side == "long" else -qty

        logger.info(
            f"Internal Twin Ledger allocation: {len(position_changes)} order(s) "
            f"({kelly_buys} Kelly BUY, {kelly_shorts} Kelly SHORT, "
            f"{len(exit_decisions)} exit/cover)"
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
