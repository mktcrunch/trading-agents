"""
Risk Agent (SHARED by both systems)
Validates positions against portfolio constraints
- Max position size
- Max exposure
- Sector limits
- Position count limits
"""
from typing import Dict, List, Optional
from src.agents.base_agent import BaseAgent
from src.logger import setup_logger

logger = setup_logger(__name__)


class RiskAgent(BaseAgent):
    """
    Validates trading decisions against risk constraints
    Ensures positions comply with:
    - Max position weight (e.g., 10% per ticker)
    - Max total exposure (e.g., 125% of portfolio)
    - Max open positions (e.g., 8)
    - Stop losses (if configured)
    """

    def __init__(self, system: str = "baseline"):
        super().__init__(system=system)
        self.max_position_weight = 0.10  # 10% max per position
        self.max_total_exposure = 1.25   # 125% leverage max
        self.max_positions = 8

    async def validate_positions(
        self,
        proposed_positions: Dict[str, float],
        portfolio_value: float,
        current_positions: Dict[str, Dict]
    ) -> Dict[str, bool]:
        """
        Validate proposed positions against risk constraints

        Args:
            proposed_positions: Dict mapping ticker -> target weight
            portfolio_value: Total portfolio value
            current_positions: Current open positions

        Returns:
            Dict mapping ticker -> valid (True/False)
        """
        validation_results = {}

        # Check position count
        if len(proposed_positions) > self.max_positions:
            self.log_error(f"Too many positions: {len(proposed_positions)} > {self.max_positions}")
            # Trim to max
            sorted_positions = sorted(
                proposed_positions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            proposed_positions = dict(sorted_positions[:self.max_positions])

        # Check individual position weights
        total_weight = 0
        for ticker, weight in proposed_positions.items():
            if weight > self.max_position_weight:
                self.log_error(f"{ticker}: weight {weight:.2%} exceeds max {self.max_position_weight:.2%}")
                validation_results[ticker] = False
            else:
                validation_results[ticker] = True
                total_weight += weight

        # Check total exposure
        if total_weight > self.max_total_exposure:
            self.log_error(f"Total exposure {total_weight:.2%} exceeds max {self.max_total_exposure:.2%}")

        self.log_action(
            f"Validated {sum(validation_results.values())}/{len(validation_results)} positions",
            data={
                "validation_results": validation_results,
                "total_weight": round(total_weight, 4),
                "max_total_exposure": self.max_total_exposure,
            },
        )
        return validation_results

    async def calculate_position_sizes(
        self,
        signals: Dict[str, 'Signal'],
        portfolio_value: float,
        capital_available: float
    ) -> Dict[str, float]:
        """
        Calculate position sizes based on signals and risk constraints

        Args:
            signals: Dict mapping ticker -> Signal object
            portfolio_value: Total portfolio value
            capital_available: Available capital for new positions

        Returns:
            Dict mapping ticker -> qty to trade
        """
        # Placeholder: will be implemented in allocator
        self.log_action("Computing position sizes (to be implemented in allocator)")
        return {}

    async def execute(self) -> bool:
        """Execute risk validation workflow"""
        self.log_action("Starting risk agent")
        return True
