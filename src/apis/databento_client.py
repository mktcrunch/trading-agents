"""
DataBento historical data client for discovery pipeline.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

try:
    import databento as db
    DATABENTO_AVAILABLE = True
except ImportError:
    DATABENTO_AVAILABLE = False

from src import config
from src.discovery.catalog import probe_lookback_days
from src.logger import setup_logger

logger = setup_logger(__name__)

DEFAULT_DATASET = "EQUS.MINI"
DEFAULT_SCHEMA = "ohlcv-1d"


class DataBentoClient:
    """Fetch bars from DataBento for discovery and enrichment."""

    def __init__(self, api_key: Optional[str] = None):
        if not DATABENTO_AVAILABLE:
            raise ImportError("databento package not installed")
        self.api_key = api_key or config.DATABENTO_API_KEY
        self.client = db.Historical(self.api_key)
        self.dataset = config.DISCOVERY_CONFIG.get("dataset", DEFAULT_DATASET)
        self.schema = config.DISCOVERY_CONFIG.get("schema", DEFAULT_SCHEMA)

    def list_datasets(self) -> List[str]:
        try:
            return list(self.client.metadata.list_datasets())
        except Exception as e:
            logger.error(f"DataBento list_datasets failed: {e}")
            return []

    def list_schemas(self, dataset: str) -> List[str]:
        try:
            return list(self.client.metadata.list_schemas(dataset=dataset))
        except Exception as e:
            logger.warning(f"DataBento list_schemas failed for {dataset}: {e}")
            return []

    def _date_range(self, lookback_days: int) -> tuple[str, str]:
        # EQUS.MINI ohlcv-1d is published with a multi-day lag
        lag_days = config.DISCOVERY_CONFIG.get("end_lag_days", 3)
        end = datetime.now(timezone.utc) - timedelta(days=lag_days)
        start = end - timedelta(days=lookback_days)
        return start.date().isoformat(), end.date().isoformat()

    def fetch_range(
        self,
        symbols: List[str],
        dataset: Optional[str] = None,
        schema: Optional[str] = None,
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Fetch bars for symbols from a specific dataset/schema.

        Returns:
            DataFrame with symbol column and OHLCV fields when applicable
        """
        if not symbols:
            return pd.DataFrame()

        dataset = dataset or self.dataset
        schema = schema or self.schema
        lookback_days = lookback_days or probe_lookback_days(schema or DEFAULT_SCHEMA)
        start, end = self._date_range(lookback_days)

        try:
            store = self.client.timeseries.get_range(
                dataset=dataset,
                symbols=symbols,
                schema=schema,
                stype_in="raw_symbol",
                start=start,
                end=end,
            )
            df = store.to_df()
            if df is None or df.empty:
                logger.warning(f"DataBento returned empty data for {dataset}/{schema}")
                return pd.DataFrame()

            logger.info(
                f"✓ DataBento {dataset}/{schema}: {len(df)} rows, "
                f"{df['symbol'].nunique() if 'symbol' in df.columns else '?'} symbols "
                f"({start} → {end})"
            )
            return df
        except Exception as e:
            logger.error(f"DataBento fetch failed for {dataset}/{schema}: {e}")
            return pd.DataFrame()

    def fetch_daily_ohlcv(
        self,
        symbols: List[str],
        lookback_days: Optional[int] = None,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV using configured default dataset/schema."""
        return self.fetch_range(
            symbols=symbols,
            dataset=self.dataset,
            schema=self.schema,
            lookback_days=lookback_days,
        )
