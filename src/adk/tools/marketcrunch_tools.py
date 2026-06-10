"""ADK FunctionTools wrapping MarketCrunch prediction APIs."""
from typing import Any, Dict, List

from src import config
from src.agents.ledger_utils import mc_confidence_score
from src.apis.marketcrunch_client import MarketCrunchClient
from src.strategies.allocator import PositionAllocator


def get_marketcrunch_predictions(tickers: str | None = None) -> Dict[str, Any]:
    """Fetch MarketCrunch AI estimates and Kelly sizing guidance per ticker.

    Args:
        tickers: Comma-separated symbols; defaults to full trading universe.
    """
    universe = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers
        else list(config.TICKER_UNIVERSE)
    )
    client = MarketCrunchClient()
    kelly_fraction = config.INTERNAL_CONFIG.get("kelly_fraction", 0.25)
    out: Dict[str, Any] = {}

    for ticker in universe:
        analysis = client.get_ai_estimates(ticker)
        if not analysis:
            continue
        ai_est = analysis.get("ai_estimate", {})
        conf_label = ai_est.get("confidence", "Low")
        target_delta = float(ai_est.get("target_delta_numeric", 0) or 0)
        confidence = mc_confidence_score(conf_label)
        predicted_return = target_delta / 100
        kelly_raw = PositionAllocator.kelly_criterion(
            predicted_return=predicted_return,
            confidence=confidence,
            max_kelly=kelly_fraction,
        )
        out[ticker] = {
            "ai_estimate": ai_est,
            "mc_confidence_score": confidence,
            "kelly_suggested_weight": round(kelly_raw, 4),
            "kelly_max_fraction": kelly_fraction,
        }

    return {"predictions": out, "count": len(out)}
