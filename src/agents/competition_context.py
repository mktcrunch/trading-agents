"""
Competition context for Twin Ledger head-to-head trading.
Compares Baseline (System A) vs Internal (System B) paper accounts.
"""
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.analytics.performance_metrics import MIN_OBS_FOR_STATS, SIGN_CONVENTION
from src.apis.alpaca_client import AlpacaClient
from src.models.position import Position

STARTING_EQUITY = 100_000.0
DEFAULT_QUANT_LOOKBACK_HOURS = 720

PERFORMANCE_METRICS_METHODOLOGY = {
    "sign_convention": SIGN_CONVENTION,
    "formula": "internal_value - baseline_value",
    "agent_guidance": (
        "Raw comparison.* metrics are always Internal − Baseline. "
        "Baseline agents MUST read quant_head_to_head.for_you or perspectives.baseline "
        "(positive = favorable to you). Never treat positive comparison.daily_delta_pct "
        "as a Baseline win."
    ),
    "source": "Aligned daily Alpaca portfolio equity snapshots for Baseline and Internal.",
    "paired_window": (
        "Metrics use calendar days where both accounts have equity points. "
        "Daily returns are simple close-to-close % changes on those aligned dates."
    ),
    "dashboard_cards": {
        "excess_return": (
            "Internal − Baseline total return vs $100k starting equity."
        ),
        "daily_delta": (
            "Internal − Baseline return today (live equity vs prior close). "
            "Mean daily alpha uses completed paired days only."
        ),
        "sharpe_difference": (
            "Annualized Sharpe (mean/std × √252) per desk; card = Internal − Baseline."
        ),
        "drawdown_difference": (
            "Max peak-to-trough drawdown % per desk; card = Internal − Baseline "
            "(negative means Internal shallower)."
        ),
    },
    "for_you_fields": {
        "daily_delta_pct": "Your return minus competitor on the latest paired day.",
        "excess_return_pct": "Your total return minus competitor vs $100k start.",
        "sharpe_difference": "Your Sharpe minus competitor Sharpe.",
        "drawdown_advantage_pp": (
            "Positive when your max drawdown is shallower than competitor (pp)."
        ),
    },
    "significance": {
        "daily_alpha": "Two-sided paired t-test on daily excess returns.",
        "excess_return": "Paired bootstrap on compounded return paths (4,000 resamples).",
        "sharpe_difference": "Paired bootstrap on daily return pairs.",
        "drawdown_difference": "Paired bootstrap on reconstructed equity paths.",
        "days_to_significance": (
            "daily alpha: Cohen's d power (80%); others: √n projection from p-value. "
            "Omitted when effect is zero/flat or paired days < min_paired_days."
        ),
    },
    "min_paired_days": MIN_OBS_FOR_STATS,
}

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


def fetch_quant_head_to_head(since_hours: int = DEFAULT_QUANT_LOOKBACK_HOURS) -> Dict[str, Any]:
    """Aligned quant metrics from Alpaca equity history (same engine as the dashboard)."""
    from src.analytics.performance_metrics import (
        collect_live_daily_returns,
        compute_head_to_head_metrics,
    )

    history: Dict[str, List[Dict[str, Any]]] = {"baseline": [], "internal": []}
    for system in ("baseline", "internal"):
        try:
            history[system] = AlpacaClient(system=system).get_portfolio_history_series(
                since_hours=since_hours
            )
        except Exception:
            history[system] = []

    live_daily = collect_live_daily_returns(history)
    metrics = compute_head_to_head_metrics(
        history["baseline"],
        history["internal"],
        starting_equity=STARTING_EQUITY,
        live_daily_returns=live_daily,
    )
    return {
        "since_hours": since_hours,
        "history_points": {k: len(v) for k, v in history.items()},
        "data_quality": _quant_data_quality(history, metrics),
        "metrics": metrics,
        "methodology": PERFORMANCE_METRICS_METHODOLOGY,
        "perspectives": {
            "baseline": build_perspective_quant_view(metrics, "baseline"),
            "internal": build_perspective_quant_view(metrics, "internal"),
        },
    }


def _internal_minus_baseline_block(comparison: Dict[str, Any]) -> Dict[str, Any]:
    nested = comparison.get("internal_minus_baseline")
    if isinstance(nested, dict) and nested:
        return nested
    return comparison


def _flip_higher_is_better(
    internal_minus: Optional[float],
    perspective: str,
) -> Optional[float]:
    if internal_minus is None:
        return None
    v = float(internal_minus)
    return round(-v, 4) if perspective == "baseline" else round(v, 4)


def _drawdown_advantage_pp(
    internal_minus_dd: Optional[float],
    perspective: str,
) -> Optional[float]:
    """Positive when your max drawdown is shallower than competitor."""
    if internal_minus_dd is None:
        return None
    v = float(internal_minus_dd)
    return round(-v, 3) if perspective == "internal" else round(v, 3)


def _you_vs_competitor(value_for_you: Optional[float]) -> str:
    if value_for_you is None:
        return "unavailable"
    if abs(value_for_you) < 1e-9:
        return "tied"
    return "you_ahead" if value_for_you > 0 else "you_behind"


def build_perspective_quant_view(
    metrics: Dict[str, Any],
    perspective: str,
) -> Dict[str, Any]:
    """Perspective-relative quant view: positive for_you values favor the given desk."""
    if perspective not in ("baseline", "internal"):
        raise ValueError(f"perspective must be baseline or internal, got {perspective!r}")

    paired = int(metrics.get("observation_days") or 0)
    cmp = metrics.get("comparison") or {}
    imb = _internal_minus_baseline_block(cmp)

    if paired < 1:
        return {
            "perspective": perspective,
            "you_are": "Baseline Trader" if perspective == "baseline" else "Internal Trader",
            "competitor": "Internal Trader" if perspective == "baseline" else "Baseline Trader",
            "sign_convention": cmp.get("sign_convention", SIGN_CONVENTION),
            "paired_days": paired,
            "sufficient_for_stats": False,
            "status": "unavailable",
            "for_you": None,
            "interpretation": {},
            "read_guidance": (
                "Insufficient paired Alpaca equity history. "
                "Do not infer head-to-head edge from dollar leaderboard alone."
            ),
        }

    daily_you = _flip_higher_is_better(
        imb.get("daily_delta_pct") or cmp.get("daily_delta_pct"),
        perspective,
    )
    excess_you = _flip_higher_is_better(
        imb.get("excess_return_pct") or cmp.get("total_return_diff_pct"),
        perspective,
    )
    sharpe_you = _flip_higher_is_better(
        imb.get("sharpe_diff") or cmp.get("sharpe_diff"),
        perspective,
    )
    dd_adv = _drawdown_advantage_pp(
        imb.get("max_drawdown_diff_pct") or cmp.get("max_drawdown_diff_pct"),
        perspective,
    )
    alpha_you = _flip_higher_is_better(
        imb.get("mean_daily_alpha_pct") or cmp.get("mean_daily_alpha_pct"),
        perspective,
    )

    return {
        "perspective": perspective,
        "you_are": "Baseline Trader" if perspective == "baseline" else "Internal Trader",
        "competitor": "Internal Trader" if perspective == "baseline" else "Baseline Trader",
        "sign_convention": cmp.get("sign_convention", SIGN_CONVENTION),
        "paired_days": paired,
        "sufficient_for_stats": paired >= MIN_OBS_FOR_STATS,
        "status": "ok" if paired >= MIN_OBS_FOR_STATS else "insufficient_paired_days",
        "for_you": {
            "daily_delta_pct": daily_you,
            "excess_return_pct": excess_you,
            "sharpe_difference": sharpe_you,
            "drawdown_advantage_pp": dd_adv,
            "mean_daily_alpha_pct": alpha_you,
        },
        "interpretation": {
            "daily_delta": _you_vs_competitor(daily_you),
            "excess_return": _you_vs_competitor(excess_you),
            "sharpe": _you_vs_competitor(sharpe_you),
            "drawdown": _you_vs_competitor(dd_adv),
        },
        "read_guidance": (
            "Use for_you.* only: positive = favorable to you, negative = favorable to competitor. "
            "comparison.* and internal_minus_baseline.* are Internal − Baseline (dashboard neutral view)."
        ),
    }


def _quant_data_quality(
    history: Dict[str, List[Dict[str, Any]]],
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    b_pts = len(history.get("baseline") or [])
    i_pts = len(history.get("internal") or [])
    paired = int(metrics.get("observation_days") or 0)
    if b_pts == 0 or i_pts == 0:
        status = "missing_history"
    elif paired < 2:
        status = "no_overlap"
    elif paired < MIN_OBS_FOR_STATS:
        status = "insufficient_paired_days"
    else:
        status = "ok"
    return {
        "baseline_history_points": b_pts,
        "internal_history_points": i_pts,
        "paired_days": paired,
        "min_days_for_stats": MIN_OBS_FOR_STATS,
        "sufficient_for_stats": paired >= MIN_OBS_FOR_STATS,
        "status": status,
    }


def _perspective_excess_return_pct(
    comparison: Dict[str, Any],
    perspective: str,
) -> Optional[float]:
    diff = comparison.get("total_return_diff_pct")
    if diff is None:
        return None
    # Stored as Internal − Baseline; flip sign when Baseline is "you".
    return round(-float(diff), 3) if perspective == "baseline" else round(float(diff), 3)


def format_quant_learning_block(
    perspective: str = "baseline",
    since_hours: int = DEFAULT_QUANT_LOOKBACK_HOURS,
) -> str:
    """Compact quant summary for signal-agent learning prompts."""
    qh = fetch_quant_head_to_head(since_hours=since_hours)
    view = (qh.get("perspectives") or {}).get(perspective) or {}
    if view.get("status") == "unavailable":
        return ""

    fy = view.get("for_you") or {}
    interp = view.get("interpretation") or {}
    metrics = qh.get("metrics") or {}
    cmp = metrics.get("comparison") or {}
    sig = cmp.get("significance") or {}
    excess_you = fy.get("excess_return_pct")
    excess_lbl = interp.get("excess_return") or "unavailable"

    def _sig_note(key: str) -> str:
        block = sig.get(key) or {}
        if block.get("significant_95"):
            return "significant at 95%"
        if block.get("zero_effect"):
            return "not significant (flat effect — days-to-significance n/a)"
        rem = block.get("days_remaining_95")
        if rem:
            return f"not significant (~{rem} more paired days at current pace)"
        if block.get("insufficient_data"):
            return f"not significant (<{MIN_OBS_FOR_STATS} paired days)"
        return "not significant"

    daily_you = fy.get("daily_delta_pct")
    daily_txt = f"{daily_you:+.3f}%" if daily_you is not None else "n/a"
    excess_txt = f"{excess_you:+.2f}%" if excess_you is not None else "n/a"

    return f"""HEAD-TO-HEAD QUANT (Performance dashboard · {view.get('paired_days')} paired days):
- Your excess return vs competitor: {excess_txt} ({excess_lbl})
- Your daily delta vs competitor (latest day): {daily_txt} ({interp.get('daily_delta', 'n/a')})
- Your Sharpe difference vs competitor: {fy.get('sharpe_difference')} ({interp.get('sharpe', 'n/a')})
- Your drawdown advantage: {fy.get('drawdown_advantage_pp')} pp ({interp.get('drawdown', 'n/a')})
- Excess return: {_sig_note('total_return_diff')} · daily alpha: {_sig_note('daily_alpha')}
Sign rule: comparison.* is Internal − Baseline; use for_you fields above for your desk.
Use risk-adjusted edge, not just dollar gap, when sizing overnight risk."""


def _attach_quant_head_to_head(
    ctx: Dict[str, Any],
    since_hours: int = DEFAULT_QUANT_LOOKBACK_HOURS,
) -> Dict[str, Any]:
    try:
        qh = fetch_quant_head_to_head(since_hours=since_hours)
        perspective = ctx.get("perspective")
        if perspective in ("baseline", "internal"):
            qh["for_you"] = (qh.get("perspectives") or {}).get(perspective) or {}
        ctx["quant_head_to_head"] = qh
    except Exception:
        pass
    return ctx


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

    ctx = {
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
    return _attach_quant_head_to_head(ctx)


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

    ctx = {
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
    return _attach_quant_head_to_head(ctx)
