"""
Risk Agent (SHARED by both systems)
Validates positions against portfolio constraints
- Max position size (long and short)
- Max gross exposure including held + pending + proposed
- Position count limits
- Conflicting long/short on same ticker
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Union

from src.agents.base_agent import BaseAgent
from src.logger import setup_logger
from src.strategies.order_dedup import normalize_open_order

logger = setup_logger(__name__)

PositionLike = Union[Dict[str, Any], Any]
ACTIONABLE_TICKERS = frozenset({"BUY", "SELL", "CLOSE", "SHORT", "COVER"})


def entry_sides_from_decisions(decisions: List[Any]) -> Dict[str, str]:
    """Map ticker -> 'long' | 'short' from BUY/SHORT decisions."""
    sides: Dict[str, str] = {}
    for decision in decisions:
        action = (
            decision.action
            if hasattr(decision, "action")
            else str(decision.get("action", "")).upper()
        )
        ticker = (
            decision.ticker
            if hasattr(decision, "ticker")
            else str(decision.get("ticker", "")).strip().upper()
        )
        if not ticker:
            continue
        if action == "SHORT":
            sides[ticker] = "short"
        elif action == "BUY":
            sides[ticker] = "long"
    return sides


class RiskAgent(BaseAgent):
    """
    Validates trading decisions against risk constraints.
    """

    def __init__(self, system: str = "baseline"):
        super().__init__(system=system)
        from src import config
        from src.apis.alpaca_client import AlpacaClient

        if self.system == "internal":
            cfg = config.INTERNAL_CONFIG
        else:
            cfg = config.BASELINE_CONFIG

        self.max_position_weight = cfg.get("position_size_pct", 0.10)
        self.max_total_exposure = 1.25
        self.max_positions = cfg.get("max_positions", 8)
        self._alpaca = AlpacaClient(system=system)

    @staticmethod
    def _position_qty(pos: Optional[PositionLike]) -> float:
        if not pos:
            return 0.0
        if hasattr(pos, "qty"):
            return float(pos.qty)
        return float(pos.get("qty", 0) or 0)

    @staticmethod
    def _position_price(pos: Optional[PositionLike]) -> float:
        if not pos:
            return 0.0
        if hasattr(pos, "current_price"):
            price = pos.current_price
        else:
            price = pos.get("current_price") or pos.get("avg_entry_price", 0)
        return float(price or 0)

    @classmethod
    def _position_weight(cls, pos: PositionLike, portfolio_value: float) -> float:
        if portfolio_value <= 0:
            return 0.0
        qty = cls._position_qty(pos)
        price = cls._position_price(pos)
        if qty == 0 or price <= 0:
            return 0.0
        return abs(qty * price) / portfolio_value

    @classmethod
    def _split_book(
        cls,
        current_positions: Mapping[str, PositionLike],
        portfolio_value: float,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        long_weights: Dict[str, float] = {}
        short_weights: Dict[str, float] = {}
        for ticker, pos in (current_positions or {}).items():
            qty = cls._position_qty(pos)
            weight = cls._position_weight(pos, portfolio_value)
            if weight <= 0:
                continue
            if qty > 0:
                long_weights[ticker] = weight
            elif qty < 0:
                short_weights[ticker] = weight
        return long_weights, short_weights

    @classmethod
    def _pending_weights(
        cls,
        open_orders_raw: List[Any],
        portfolio_value: float,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        pending_long: Dict[str, float] = {}
        pending_short: Dict[str, float] = {}
        if portfolio_value <= 0:
            return pending_long, pending_short

        for raw in open_orders_raw or []:
            order = normalize_open_order(raw)
            remaining = float(order.get("remaining_qty", 0) or 0)
            if remaining <= 0:
                continue
            limit_price = order.get("limit_price")
            if not limit_price:
                continue
            weight = (remaining * float(limit_price)) / portfolio_value
            symbol = order["symbol"]
            if order["side"] == "buy":
                pending_long[symbol] = pending_long.get(symbol, 0.0) + weight
            else:
                pending_short[symbol] = pending_short.get(symbol, 0.0) + weight
        return pending_long, pending_short

    @classmethod
    def _gross_exposure(
        cls,
        long_weights: Mapping[str, float],
        short_weights: Mapping[str, float],
    ) -> float:
        tickers = set(long_weights) | set(short_weights)
        return sum(
            float(long_weights.get(t, 0.0)) + float(short_weights.get(t, 0.0))
            for t in tickers
        )

    async def validate_positions(
        self,
        proposed_positions: Dict[str, float],
        portfolio_value: float,
        current_positions: Optional[Mapping[str, PositionLike]] = None,
        entry_sides: Optional[Dict[str, str]] = None,
        open_orders_raw: Optional[List[Any]] = None,
    ) -> Dict[str, bool]:
        """
        Validate proposed entry weights against book + pending orders.

        Args:
            proposed_positions: ticker -> entry weight (0-0.10)
            portfolio_value: total portfolio value
            current_positions: open Alpaca positions
            entry_sides: ticker -> 'long' | 'short' for proposed entries
            open_orders_raw: raw Alpaca open orders (fetched if omitted)
        """
        if open_orders_raw is None:
            open_orders_raw = self._alpaca.get_orders(status="open")

        current_positions = current_positions or {}
        entry_sides = entry_sides or {}

        held_long, held_short = self._split_book(current_positions, portfolio_value)
        pending_long, pending_short = self._pending_weights(
            open_orders_raw, portfolio_value
        )

        baseline_gross = self._gross_exposure(
            {t: held_long.get(t, 0.0) + pending_long.get(t, 0.0) for t in set(held_long) | set(pending_long)},
            {t: held_short.get(t, 0.0) + pending_short.get(t, 0.0) for t in set(held_short) | set(pending_short)},
        )

        active_tickers = set(held_long) | set(held_short) | set(pending_long) | set(pending_short)

        sorted_proposals = sorted(
            proposed_positions.items(),
            key=lambda item: abs(item[1]),
            reverse=True,
        )

        validation_results: Dict[str, bool] = {}
        running_long = dict(held_long)
        running_short = dict(held_short)
        for ticker, pending in pending_long.items():
            running_long[ticker] = running_long.get(ticker, 0.0) + pending
        for ticker, pending in pending_short.items():
            running_short[ticker] = running_short.get(ticker, 0.0) + pending
        running_gross = self._gross_exposure(running_long, running_short)
        accepted_tickers = set(active_tickers)

        for ticker, weight in sorted_proposals:
            weight = min(max(float(weight), 0.0), self.max_position_weight)
            if weight <= 0:
                validation_results[ticker] = False
                continue

            side = entry_sides.get(ticker, "long")
            if side not in ("long", "short"):
                side = "long"

            if side == "long":
                if running_short.get(ticker, 0.0) > 0:
                    self.log_risk_rejection(
                        f"{ticker}: BUY rejected — existing short exposure "
                        f"{running_short[ticker]:.2%}"
                    )
                    validation_results[ticker] = False
                    continue
                combined = running_long.get(ticker, 0.0) + weight
                if combined > self.max_position_weight + 1e-9:
                    self.log_risk_rejection(
                        f"{ticker}: long exposure {combined:.2%} would exceed max "
                        f"{self.max_position_weight:.2%} "
                        f"(held+pending={running_long.get(ticker, 0.0):.2%}, "
                        f"proposed={weight:.2%})"
                    )
                    validation_results[ticker] = False
                    continue
            else:
                if running_long.get(ticker, 0.0) > 0:
                    self.log_risk_rejection(
                        f"{ticker}: SHORT rejected — existing long exposure "
                        f"{running_long[ticker]:.2%}"
                    )
                    validation_results[ticker] = False
                    continue
                combined = running_short.get(ticker, 0.0) + weight
                if combined > self.max_position_weight + 1e-9:
                    self.log_risk_rejection(
                        f"{ticker}: short exposure {combined:.2%} would exceed max "
                        f"{self.max_position_weight:.2%} "
                        f"(held+pending={running_short.get(ticker, 0.0):.2%}, "
                        f"proposed={weight:.2%})"
                    )
                    validation_results[ticker] = False
                    continue

            new_ticker = ticker not in accepted_tickers
            if new_ticker and len(accepted_tickers) >= self.max_positions:
                self.log_risk_rejection(
                    f"{ticker}: rejected — would exceed max positions "
                    f"({self.max_positions})"
                )
                validation_results[ticker] = False
                continue

            projected_gross = running_gross + weight
            if projected_gross > self.max_total_exposure + 1e-9:
                self.log_risk_rejection(
                    f"{ticker}: rejected — gross exposure would be "
                    f"{projected_gross:.2%} > {self.max_total_exposure:.2%}"
                )
                validation_results[ticker] = False
                continue

            validation_results[ticker] = True
            if side == "long":
                running_long[ticker] = running_long.get(ticker, 0.0) + weight
            else:
                running_short[ticker] = running_short.get(ticker, 0.0) + weight
            running_gross = self._gross_exposure(running_long, running_short)
            accepted_tickers.add(ticker)

        self.log_action(
            f"Validated {sum(validation_results.values())}/{len(validation_results)} "
            f"entry proposals",
            data={
                "validation_results": validation_results,
                "baseline_gross_exposure": round(baseline_gross, 4),
                "projected_gross_exposure": round(running_gross, 4),
                "max_total_exposure": self.max_total_exposure,
                "held_long": {k: round(v, 4) for k, v in held_long.items()},
                "held_short": {k: round(v, 4) for k, v in held_short.items()},
                "pending_long": {k: round(v, 4) for k, v in pending_long.items()},
                "pending_short": {k: round(v, 4) for k, v in pending_short.items()},
            },
        )
        return validation_results

    async def calculate_position_sizes(
        self,
        signals: Dict[str, "Signal"],
        portfolio_value: float,
        capital_available: float,
    ) -> Dict[str, float]:
        """Placeholder: sizing lives in the allocator."""
        self.log_action("Computing position sizes (to be implemented in allocator)")
        return {}

    async def execute(self) -> bool:
        self.log_action("Starting risk agent")
        return True
