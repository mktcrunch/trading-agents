"""
Competition context for Twin Ledger head-to-head trading.
Compares Baseline (System A) vs Internal (System B) paper accounts.
"""
from datetime import datetime, timezone
from typing import Dict, List

from src.apis.alpaca_client import AlpacaClient
from src.models.position import Position

STARTING_EQUITY = 100_000.0

COMPETITOR_NOTES = {
    "baseline": (
        "Baseline Trader uses Alpaca technical indicators and Twin Ledger LLM strategy only. "
        "No MarketCrunch predictions or Kelly sizing."
    ),
    "internal": (
        "Internal Trader uses MarketCrunch predictions, Kelly Criterion sizing, "
        "and optional DataBento enrichment."
    ),
}

COMPETITION_INFO_BOUNDARY = (
    "Competitor snapshot shows filled Alpaca positions and account values only — "
    "NOT pending or unfilled overnight orders. Both agents submit overnight orders "
    "at the same time (~4:10 PM ET). The competitor may deploy new exposure tonight "
    "even if their current book looks static; do not infer their intent from cash alone."
)


def _position_rows(positions: Dict[str, Position]) -> List[Dict]:
    rows = []
    for ticker, pos in positions.items():
        rows.append({
            "ticker": ticker,
            "qty": pos.qty,
            "avg_entry_price": round(pos.avg_entry_price, 2),
            "current_price": round(pos.current_price, 2),
            "market_value": round(pos.market_value, 2),
            "unrealized_return_pct": round((pos.unrealized_return or 0) * 100, 2),
        })
    return rows


def _account_snapshot(system: str, label: str) -> Dict:
    client = AlpacaClient(system=system)
    account = client.get_account() or {}
    positions = client.get_positions()

    portfolio_value = float(account.get("portfolio_value", 0))
    cash = float(account.get("cash", 0))
    equity = float(account.get("equity", portfolio_value))
    pnl = equity - STARTING_EQUITY
    pnl_pct = (pnl / STARTING_EQUITY) * 100 if STARTING_EQUITY else 0

    return {
        "system": system,
        "label": label,
        "portfolio_value": round(portfolio_value, 2),
        "cash": round(cash, 2),
        "equity": round(equity, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct, 2),
        "position_count": len(positions),
        "positions": _position_rows(positions),
    }


def build_competition_context(perspective: str = "baseline") -> Dict:
    """
    Build portfolio + leaderboard context from a given agent's perspective.

    Args:
        perspective: "baseline" or "internal" — whose portfolio is "yours"
    """
    baseline = _account_snapshot("baseline", "Baseline Trader")
    internal = _account_snapshot("internal", "Internal Trader")

    if perspective == "internal":
        your_portfolio = {**internal, "label": "Internal Trader (you)"}
        competitor = {**baseline, "label": "Baseline Trader (competitor)"}
        competitor_system = "baseline"
    else:
        your_portfolio = {**baseline, "label": "Baseline Trader (you)"}
        competitor = {**internal, "label": "Internal Trader (competitor)"}
        competitor_system = "internal"

    if your_portfolio["portfolio_value"] >= competitor["portfolio_value"]:
        rank = 1
        gap = your_portfolio["portfolio_value"] - competitor["portfolio_value"]
        status = "ahead"
    else:
        rank = 2
        gap = competitor["portfolio_value"] - your_portfolio["portfolio_value"]
        status = "behind"

    return {
        "perspective": perspective,
        "starting_equity": STARTING_EQUITY,
        "your_portfolio": your_portfolio,
        "competitor": competitor,
        "leaderboard": {
            "your_rank": rank,
            "total_agents": 2,
            "status": status,
            "value_gap_usd": round(gap, 2),
            "competitor_profile": COMPETITOR_NOTES[competitor_system],
            "information_boundary": COMPETITION_INFO_BOUNDARY,
        },
    }


def get_competition_snapshot() -> Dict:
    """Neutral side-by-side snapshot for the performance dashboard."""
    baseline = _account_snapshot("baseline", "Baseline")
    internal = _account_snapshot("internal", "Internal")

    if baseline["portfolio_value"] >= internal["portfolio_value"]:
        leader = "baseline"
        gap = baseline["portfolio_value"] - internal["portfolio_value"]
    else:
        leader = "internal"
        gap = internal["portfolio_value"] - baseline["portfolio_value"]

    return {
        "starting_equity": STARTING_EQUITY,
        "baseline": baseline,
        "internal": internal,
        "leaderboard": {
            "leader": leader,
            "gap_usd": round(gap, 2),
            "baseline_rank": 1 if leader == "baseline" else 2,
            "internal_rank": 1 if leader == "internal" else 2,
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
