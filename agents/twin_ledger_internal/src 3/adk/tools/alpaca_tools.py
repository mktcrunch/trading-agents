"""ADK FunctionTools wrapping Alpaca market data and account APIs."""
import json
from typing import Any, Dict, List

from src import config
from src.apis.alpaca_client import AlpacaClient
from src.apis.price_fetcher import fetch_ohlcv_for_tickers
from src.audit.serialize import account_snapshot, positions_snapshot
from src.strategies.signal_generator import SignalGenerator


def _client(system: str) -> AlpacaClient:
    return AlpacaClient(system=system)


def get_account_info(system: str = "baseline") -> Dict[str, Any]:
    """Return Alpaca account snapshot (cash, portfolio value, buying power).

    Args:
        system: 'baseline' or 'internal' — selects the Twin Ledger paper account.
    """
    account = _client(system).get_account()
    return account_snapshot(account)


def get_open_positions(system: str = "baseline") -> Dict[str, Any]:
    """Return open positions for the given Twin Ledger system account."""
    positions = _client(system).get_positions()
    return {"positions": positions_snapshot(positions), "count": len(positions)}


def get_technical_indicators(
    system: str = "baseline",
    tickers: str | None = None,
    lookback_days: int = 90,
) -> Dict[str, Any]:
    """Fetch OHLCV from Alpaca and compute RSI, MACD, Bollinger indicators.

    Args:
        system: 'baseline' or 'internal'.
        tickers: Comma-separated ticker list; defaults to full trading universe.
        lookback_days: History window for indicator calculation.
    """
    universe = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers
        else list(config.TICKER_UNIVERSE)
    )
    client = _client(system)
    price_data = fetch_ohlcv_for_tickers(client, universe, lookback_days=lookback_days)
    technical = SignalGenerator.build_technical_data(price_data)
    return {
        "tickers": list(technical.keys()),
        "technical_data": technical,
    }


def get_latest_prices(
    system: str = "baseline",
    tickers: str | None = None,
) -> Dict[str, float]:
    """Return latest close prices for tickers from Alpaca OHLCV."""
    result = get_technical_indicators(system=system, tickers=tickers, lookback_days=30)
    technical = result.get("technical_data", {})
    return {
        t: float(data.get("close", 0) or 0)
        for t, data in technical.items()
        if data.get("close")
    }
