"""
Baseline Data Agent (System A)
Fetches market data from Alpaca (OHLCV, positions, account).
No MarketCrunch predictions.
"""
from typing import Dict, List, Optional
import pandas as pd
from src.agents.base_agent import BaseAgent
from src.audit.serialize import account_snapshot, positions_snapshot
from src.apis.alpaca_client import AlpacaClient
from src.apis.price_fetcher import fetch_ohlcv_for_tickers
from src.logger import setup_logger

logger = setup_logger(__name__)


class BaselineDataAgent(BaseAgent):
    """
    System A Data Agent
    Fetches:
    - OHLCV data from Alpaca
    - Current positions from Alpaca
    """

    def __init__(self):
        super().__init__(system="baseline")
        self.alpaca_client = AlpacaClient(system="baseline")

    async def fetch_price_data(self, tickers: List[str], lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data for tickers from Alpaca.

        Args:
            tickers: List of tickers
            lookback_days: Days of history

        Returns:
            Dict mapping ticker -> OHLCV DataFrame
        """
        price_data = fetch_ohlcv_for_tickers(
            self.alpaca_client,
            tickers,
            lookback_days=lookback_days,
        )
        self.log_action(
            f"Fetched price data for {len(price_data)}/{len(tickers)} tickers "
            f"({lookback_days}-day lookback)",
            data={
                "tickers_requested": tickers,
                "tickers_fetched": list(price_data.keys()),
                "lookback_days": lookback_days,
            },
        )
        return price_data

    async def get_current_positions(self) -> Dict[str, Dict]:
        """
        Get current positions from Alpaca

        Returns:
            Dict mapping ticker -> position info
        """
        try:
            positions = self.alpaca_client.get_positions()
            self.log_action(
                f"Retrieved {len(positions)} open positions",
                data={"positions": positions_snapshot(positions)},
            )
            return positions
        except Exception as e:
            self.log_error(f"Failed to get positions: {e}")
            return {}

    async def get_account_info(self) -> Optional[Dict]:
        """Get account cash, buying power, etc."""
        try:
            account = self.alpaca_client.get_account()
            self.log_action(
                "Retrieved account info",
                data={"account": account_snapshot(account)},
            )
            return account
        except Exception as e:
            self.log_error(f"Failed to get account info: {e}")
            return None

    async def execute(self) -> bool:
        """Execute data fetching workflow"""
        self.log_action("Starting baseline data fetch")
        return True
