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
