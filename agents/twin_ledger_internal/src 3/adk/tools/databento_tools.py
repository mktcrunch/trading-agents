"""ADK FunctionTools for DataBento discovery enrichment."""
from typing import Any, Dict, List

from src import config
from src.discovery.approved_sources import enrich_tickers


def get_databento_features(tickers: str | None = None) -> Dict[str, Any]:
    """Load approved DataBento feature sources for tickers in the trading universe.

    Args:
        tickers: Comma-separated symbols; defaults to full universe.
    """
    universe = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers
        else list(config.TICKER_UNIVERSE)
    )
    enriched = enrich_tickers(universe)
    return {"sources": enriched, "count": len(enriched)}
