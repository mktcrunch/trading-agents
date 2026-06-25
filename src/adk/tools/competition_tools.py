"""ADK FunctionTools for Twin Ledger competition context."""
from typing import Any, Dict

from src.agents.competition_context import build_competition_context, get_competition_snapshot


def get_leaderboard(system: str = "baseline") -> Dict[str, Any]:
    """Return live Twin Ledger leaderboard snapshot for baseline or internal.

    Args:
        system: 'baseline' or 'internal' — your account vs the competitor.
    """
    return build_competition_context(system)


def get_competition_status() -> Dict[str, Any]:
    """Return portfolio values and ranks for both competing agents."""
    return get_competition_snapshot()


def get_performance_metrics(hours: int = 720, perspective: str = "") -> Dict[str, Any]:
    """Twin Ledger quant head-to-head metrics (dashboard Performance tab).

    Returns excess return, daily delta, Sharpe/drawdown differences, significance
    tests, projected days to 95% significance, and calculation methodology.

    Args:
        hours: Alpaca equity lookback (720 ≈ 30d, 2160 ≈ 90d).
        perspective: ``baseline`` or ``internal`` — returns ``for_you`` where positive
            values favor that desk. Leave empty for neutral Internal − Baseline view.
    """
    from src.adk.tools.dashboard_tools import get_performance_metrics as _impl

    kwargs: Dict[str, Any] = {"hours": int(hours)}
    if perspective in ("baseline", "internal"):
        kwargs["perspective"] = perspective
    return _impl(**kwargs)
