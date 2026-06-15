"""
DataBento historical data client for discovery pipeline.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

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
_NON_OHLCV_SCHEMAS = frozenset({"statistics", "status", "definition"})


class DataBentoClient:
    """Fetch bars from DataBento for discovery and enrichment."""

    def __init__(self, api_key: Optional[str] = None):
        if not DATABENTO_AVAILABLE:
            raise ImportError("databento package not installed")
        self.api_key = api_key or config.DATABENTO_API_KEY
        self.client = db.Historical(self.api_key)
        self.dataset = config.DISCOVERY_CONFIG.get("dataset", DEFAULT_DATASET)
        self.schema = config.DISCOVERY_CONFIG.get("schema", DEFAULT_SCHEMA)
        self.last_fetch_skip_reason: Optional[str] = None

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

    def _probe_limits(self) -> Tuple[int, float]:
        cfg = config.DISCOVERY_CONFIG
        max_bytes = int(float(cfg.get("max_probe_download_mb", 10)) * 1024 * 1024)
        max_cost = float(cfg.get("max_sample_cost_usd", 1.0))
        return max_bytes, max_cost

    @staticmethod
    def _is_high_risk_probe(dataset: str, schema: str) -> bool:
        if "SUMMARY" in (dataset or "").upper():
            return True
        return (schema or "").lower() in _NON_OHLCV_SCHEMAS

    def check_probe_download_allowed(
        self,
        symbols: List[str],
        dataset: str,
        schema: str,
        lookback_days: int,
    ) -> Tuple[bool, str, Optional[int]]:
        """Pre-flight billable size + cost checks before streaming a probe."""
        if not symbols:
            return False, "no_symbols", None

        max_bytes, max_cost = self._probe_limits()
        start, end = self._date_range(lookback_days)
        estimated: Optional[int] = None

        try:
            estimated = int(
                self.client.metadata.get_billable_size(
                    dataset=dataset,
                    symbols=symbols,
                    schema=schema,
                    stype_in="raw_symbol",
                    start=start,
                    end=end,
                )
            )
        except Exception as e:
            logger.warning(
                f"DataBento billable size estimate failed for {dataset}/{schema}: {e}"
            )
            if self._is_high_risk_probe(dataset, schema):
                return False, f"size_estimate_failed:{schema}", None

        if estimated is not None and estimated > max_bytes:
            cap_mb = max_bytes / (1024 * 1024)
            est_mb = estimated / (1024 * 1024)
            return (
                False,
                f"download_size_exceeded:{est_mb:.1f}MB>{cap_mb:.0f}MB",
                estimated,
            )

        try:
            cost = float(
                self.client.metadata.get_cost(
                    dataset=dataset,
                    symbols=symbols,
                    schema=schema,
                    stype_in="raw_symbol",
                    start=start,
                    end=end,
                )
            )
            if cost > max_cost:
                return (
                    False,
                    f"download_cost_exceeded:${cost:.2f}>${max_cost:.2f}",
                    estimated,
                )
        except Exception as e:
            logger.warning(
                f"DataBento cost estimate failed for {dataset}/{schema}: {e}"
            )

        return True, "", estimated

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
        self.last_fetch_skip_reason = None
        if not symbols:
            return pd.DataFrame()

        dataset = dataset or self.dataset
        schema = schema or self.schema
        lookback_days = lookback_days or probe_lookback_days(schema or DEFAULT_SCHEMA)
        allowed, reason, estimated = self.check_probe_download_allowed(
            symbols=symbols,
            dataset=dataset,
            schema=schema,
            lookback_days=lookback_days,
        )
        if not allowed:
            self.last_fetch_skip_reason = reason
            est_note = f" (estimated {estimated} bytes)" if estimated else ""
            logger.warning(
                f"Skipping DataBento download for {dataset}/{schema}: {reason}{est_note}"
            )
            return pd.DataFrame()

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
