"""
Internal Data Agent (System B)
Fetches data from: MarketCrunch + Alpaca + DataBento
Multi-source data enrichment for signal generation
"""
from typing import Dict, List, Optional
import pandas as pd
from src.agents.base_agent import BaseAgent
from src.audit.serialize import account_snapshot, discovery_snapshot, mc_analysis_snapshot, positions_snapshot
from src.apis.marketcrunch_client import MarketCrunchClient
from src.apis.alpaca_client import AlpacaClient
from src.apis.price_fetcher import fetch_ohlcv_for_tickers
from src.discovery.approved_sources import enrich_tickers, load_approved_sources
from src import config
from src.logger import setup_logger

logger = setup_logger(__name__)


class InternalDataAgent(BaseAgent):
    """
    System B Data Agent
    Fetches from 3 sources:
    1. MarketCrunch API (predictions + analysis)
    2. Alpaca (OHLCV, positions, account)
    3. DataBento (discovered signals)
    """

    def __init__(self):
        super().__init__(system="internal")
        self.mc_client = MarketCrunchClient()
        self.alpaca_client = AlpacaClient(system="internal")

    async def fetch_mc_predictions(self, tickers: List[str]) -> Dict[str, Dict]:
        """
        Fetch MarketCrunch complete analysis for tickers

        Returns:
            Dict mapping ticker -> analysis dict
        """
        analyses = {}

        for ticker in tickers:
            try:
                analysis = self.mc_client.get_ai_estimates(ticker)
                if analysis:
                    analyses[ticker] = analysis
                    ai_est = analysis.get('ai_estimate', {})
                    conf = ai_est.get('confidence', 'N/A')
                    delta = ai_est.get('target_delta_numeric', 0)
                    self.log_action(
                        f"Fetched MC analysis for {ticker}: {conf}, {delta:.2f}%",
                        data=mc_analysis_snapshot(analysis, ticker=ticker),
                    )
            except Exception as e:
                self.log_error(f"Failed to fetch MC data for {ticker}: {e}")

        return analyses

    async def fetch_price_data(self, tickers: List[str], lookback_days: int = 90) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV data from Alpaca.

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

    async def enrich_with_databento(
        self,
        tickers: List[str],
        approved_sources: Dict = None,
    ) -> Dict[str, Dict]:
        """
        Load DataBento-discovered features from approved_datasources.json.
        Populated by DiscoveryAgent.run_daily_discovery().
        """
        if not config.INTERNAL_CONFIG.get("use_databento", True):
            self.log_action("DataBento enrichment disabled in config")
            return {}

        approved = approved_sources or load_approved_sources()
        if not approved or not approved.get("ticker_features"):
            self.log_action("No DataBento discovery output available — run: python main.py --discovery")
            return {}

        enriched = enrich_tickers(tickers, approved)
        approved_count = len(approved.get("sources", []))
        self.log_action(
            f"Enriched {len(enriched)}/{len(tickers)} tickers "
            f"({approved_count} gate-approved sources)",
            data={
                "approved_count": approved_count,
                "enriched_tickers": list(enriched.keys()),
                "discovery": discovery_snapshot(approved),
            },
        )
        return enriched

    async def get_current_positions(self) -> Dict[str, Dict]:
        """Get current positions"""
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
        """Get account info"""
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
        self.log_action("Starting internal multi-source data fetch")
        return True
