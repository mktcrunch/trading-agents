"""
Monitor Agent (SHARED by both systems)
Tracks P&L, position performance, and system health
Real-time monitoring during trading hours
"""
from typing import Dict, Optional
from datetime import datetime
from src.agents.base_agent import BaseAgent
from src.apis.alpaca_client import AlpacaClient
from src.logger import setup_logger

logger = setup_logger(__name__)


class MonitorAgent(BaseAgent):
    """
    Monitors portfolio performance in real-time
    - Tracks P&L
    - Updates position metrics
    - Logs daily performance
    - Detects anomalies
    """

    def __init__(self, system: str = "baseline"):
        super().__init__(system=system)
        self.alpaca_client = AlpacaClient(system=system)
        self.start_time = datetime.now()

    async def get_portfolio_metrics(self) -> Dict:
        """
        Get current portfolio metrics

        Returns:
            Dict with portfolio stats
        """
        try:
            account = self.alpaca_client.get_account()
            positions = self.alpaca_client.get_positions()

            # Calculate metrics
            portfolio_value = float(account.get('portfolio_value', 0))
            cash = float(account.get('cash', 0))
            total_return = float(account.get('total_return', 0))
            buying_power = float(account.get('buying_power', 0))

            metrics = {
                'portfolio_value': portfolio_value,
                'cash': cash,
                'total_return': total_return,
                'total_return_pct': (total_return / portfolio_value * 100) if portfolio_value > 0 else 0,
                'buying_power': buying_power,
                'position_count': len(positions),
                'timestamp': datetime.now().isoformat()
            }

            self.log_action(
                f"Portfolio metrics: ${portfolio_value:,.2f} | "
                f"Return: {metrics['total_return_pct']:.2f}% | "
                f"Positions: {len(positions)}"
            )
            return metrics

        except Exception as e:
            self.log_error(f"Failed to get portfolio metrics: {e}")
            return {}

    async def get_position_performance(self) -> Dict[str, Dict]:
        """
        Get individual position performance

        Returns:
            Dict mapping ticker -> performance metrics
        """
        try:
            positions = self.alpaca_client.get_positions()
            performance = {}

            for position in positions:
                ticker = position.get('symbol')
                qty = float(position.get('qty', 0))
                entry_price = float(position.get('avg_fill_price', 0))
                current_price = float(position.get('current_price', 0))
                market_value = float(position.get('market_value', 0))
                unrealized_pl = float(position.get('unrealized_pl', 0))
                unrealized_plpc = float(position.get('unrealized_plpc', 0))

                performance[ticker] = {
                    'qty': qty,
                    'entry_price': entry_price,
                    'current_price': current_price,
                    'market_value': market_value,
                    'unrealized_pl': unrealized_pl,
                    'unrealized_plpc': unrealized_plpc * 100,  # Convert to %
                    'return_pct': ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                }

            self.log_action(f"Tracked {len(performance)} position(s)")
            return performance

        except Exception as e:
            self.log_error(f"Failed to get position performance: {e}")
            return {}

    async def detect_anomalies(self, position_performance: Dict) -> Dict[str, str]:
        """
        Detect unusual position behavior

        Returns:
            Dict mapping ticker -> anomaly description
        """
        anomalies = {}

        for ticker, perf in position_performance.items():
            return_pct = perf.get('return_pct', 0)

            # Large drawdown
            if return_pct < -10:
                anomalies[ticker] = f"Significant drawdown: {return_pct:.2f}%"

            # Large gain
            if return_pct > 20:
                anomalies[ticker] = f"Exceptional gain: {return_pct:.2f}%"

        if anomalies:
            self.log_action(f"Detected {len(anomalies)} anomalie(s)")
            for ticker, anomaly in anomalies.items():
                self.logger.warning(f"  {ticker}: {anomaly}")

        return anomalies

    async def log_daily_performance(self, metrics: Dict):
        """Log daily performance summary and persist snapshot for dashboard."""
        from src.agents.competition_context import STARTING_EQUITY

        pv = metrics.get("portfolio_value", 0)
        pnl_pct = ((pv - STARTING_EQUITY) / STARTING_EQUITY * 100) if STARTING_EQUITY else 0
        snapshot = {
            **metrics,
            "pnl_pct": round(pnl_pct, 2),
            "pnl_usd": round(pv - STARTING_EQUITY, 2),
        }
        self.log_action(
            f"EOD Summary | PV: ${pv:,.2f} | P&L: {pnl_pct:+.2f}% | "
            f"Positions: {metrics.get('position_count', 0)}",
            data=snapshot,
            event_type="portfolio_snapshot",
        )

    async def execute(self) -> bool:
        """Execute monitoring workflow"""
        self.log_action("Starting monitor agent")
        return True
