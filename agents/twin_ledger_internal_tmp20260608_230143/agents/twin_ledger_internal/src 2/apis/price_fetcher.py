"""
Shared OHLCV fetcher: Alpaca market data only.
"""
from typing import Dict, List, Optional

import pandas as pd

from src.logger import setup_logger

logger = setup_logger(__name__)

MIN_BARS_FOR_INDICATORS = 26
OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize to standard OHLCV columns sorted by date."""
    normalized = df.copy()
    if "date" not in normalized.columns and normalized.index.name == "date":
        normalized = normalized.reset_index()

    for col in OHLCV_COLUMNS[1:]:
        if col in normalized.columns:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    if "date" in normalized.columns:
        normalized["date"] = pd.to_datetime(normalized["date"])

    normalized = (
        normalized[OHLCV_COLUMNS]
        .dropna(subset=["close"])
        .sort_values("date")
        .reset_index(drop=True)
    )
    return normalized


def is_sufficient(df: Optional[pd.DataFrame]) -> bool:
    return df is not None and len(df) >= MIN_BARS_FOR_INDICATORS


def fetch_ticker_ohlcv(
    alpaca_client,
    ticker: str,
    lookback_days: int = 90,
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV for one ticker from Alpaca."""
    try:
        df = alpaca_client.get_historical_bars(ticker, lookback_days=lookback_days)
    except Exception as e:
        logger.warning(f"Alpaca fetch failed for {ticker}: {e}")
        return None

    if not is_sufficient(df):
        rows = 0 if df is None else len(df)
        logger.warning(
            f"Insufficient OHLCV for {ticker}: {rows} rows "
            f"(need {MIN_BARS_FOR_INDICATORS})"
        )
        return None

    logger.info(f"✓ {ticker}: {len(df)} bars from Alpaca")
    return normalize_ohlcv(df)


def fetch_ohlcv_for_tickers(
    alpaca_client,
    tickers: List[str],
    lookback_days: int = 90,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for multiple tickers from Alpaca."""
    if not tickers:
        return {}

    price_data: Dict[str, pd.DataFrame] = {}

    try:
        alpaca_data = alpaca_client.get_historical_bars_batch(
            tickers, lookback_days=lookback_days
        )
        for ticker, df in alpaca_data.items():
            if is_sufficient(df):
                price_data[ticker] = normalize_ohlcv(df)
                logger.info(f"✓ {ticker}: {len(price_data[ticker])} bars from Alpaca")
            else:
                rows = 0 if df is None else len(df)
                logger.warning(
                    f"Insufficient OHLCV for {ticker}: {rows} rows "
                    f"(need {MIN_BARS_FOR_INDICATORS})"
                )
    except Exception as e:
        logger.error(f"Alpaca batch fetch failed: {e}")

    still_missing = [t for t in tickers if t not in price_data]
    if still_missing:
        logger.warning(f"No price data for: {', '.join(still_missing)}")

    return price_data
