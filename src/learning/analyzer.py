"""Score past audit events against market outcomes."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src import config
from src.apis.alpaca_client import AlpacaClient
from src.audit.store import load_events, _parse_ts
from src.logger import setup_logger

logger = setup_logger(__name__)


def _events_for_role(
    system: str,
    role: str,
    hours: int,
    limit: int = 3000,
) -> List[Dict[str, Any]]:
    from src.adk.tools.dashboard_tools import _event_matches_role, _trader_agent_roster

    cfg = next(r for r in _trader_agent_roster(system) if r["role"] == role)
    events = load_events(limit=limit, system=system, since_hours=hours)
    return [e for e in events if _event_matches_role(e, cfg)]


def _forward_return_pct(
    bars: Optional[pd.DataFrame],
    decision_ts: datetime,
    action: str,
    days: int = 1,
) -> Optional[float]:
    """Estimate signed outcome: positive = good for the action taken."""
    if bars is None or bars.empty:
        return None
    df = bars.copy()
    if "date" in df.columns:
        df["dt"] = pd.to_datetime(df["date"], utc=True)
    else:
        df["dt"] = pd.to_datetime(df.index, utc=True)
    df = df.sort_values("dt")
    decision_ts = decision_ts.astimezone(timezone.utc)
    past = df[df["dt"] <= decision_ts]
    if past.empty:
        past = df.head(1)
    idx = past.index[-1]
    loc = df.index.get_loc(idx)
    if isinstance(loc, slice):
        loc = loc.start
    future_loc = loc + days
    if future_loc >= len(df):
        return None
    entry = float(df.loc[df.index[loc], "close"])
    exit_ = float(df.loc[df.index[future_loc], "close"])
    if entry <= 0:
        return None
    raw = (exit_ - entry) / entry
    if action in ("SELL", "CLOSE"):
        return -raw
    return raw


def analyze_signal_outcomes(
    system: str,
    lookback_days: int = 7,
    max_decisions: int = 15,
) -> Dict[str, Any]:
    """Tier 1 digest: score recent ledger decisions with forward returns."""
    hours = max(24, lookback_days * 24)
    events = load_events(
        limit=5000,
        system=system,
        event_type="ledger_decision",
        since_hours=hours,
    )
    events.sort(key=lambda e: e.get("timestamp", ""))

    client = AlpacaClient(system=system)
    bar_cache: Dict[str, pd.DataFrame] = {}
    scored: List[Dict[str, Any]] = []
    pending: List[Dict[str, Any]] = []
    ticker_agg: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "total_return_pct": 0.0, "count": 0}
    )

    no_action_events = [
        ev for ev in events
        if (ev.get("payload") or {}).get("no_action")
        or (ev.get("payload") or {}).get("ticker") == "PORTFOLIO"
    ]
    actionable = [
        ev for ev in events
        if (ev.get("payload") or {}).get("action", "HOLD").upper() != "HOLD"
        and not (ev.get("payload") or {}).get("no_action")
    ]

    for ev in actionable[-max_decisions * 3 :]:
        payload = ev.get("payload") or {}
        ticker = (payload.get("ticker") or "").upper()
        action = (payload.get("action") or "HOLD").upper()
        if not ticker or action == "HOLD":
            continue
        try:
            ts = _parse_ts(ev["timestamp"])
        except (KeyError, ValueError):
            continue

        if ticker not in bar_cache:
            bar_cache[ticker] = client.get_historical_bars(
                ticker, lookback_days=lookback_days + 10
            )
        ret = _forward_return_pct(bar_cache[ticker], ts, action)
        outcome = None
        if ret is not None:
            outcome = "win" if ret > 0 else "loss"
            agg = ticker_agg[ticker]
            agg["count"] += 1
            agg["total_return_pct"] += ret * 100
            if ret > 0:
                agg["wins"] += 1
            else:
                agg["losses"] += 1
            scored.append({
                "timestamp": ev.get("timestamp"),
                "ticker": ticker,
                "action": action,
                "confidence": payload.get("confidence"),
                "rationale": (payload.get("rationale") or "")[:120],
                "forward_return_pct": round(ret * 100, 3),
                "outcome": outcome,
            })
        else:
            pending.append({
                "timestamp": ev.get("timestamp"),
                "ticker": ticker,
                "action": action,
                "confidence": payload.get("confidence"),
                "rationale": (payload.get("rationale") or "")[:120],
                "status": "pending_outcome",
            })

    scored = scored[-max_decisions:]
    pending = pending[-max_decisions:]
    recent_no_action = [
        {
            "timestamp": ev.get("timestamp"),
            "ticker": "PORTFOLIO",
            "action": "HOLD",
            "rationale": (ev.get("payload") or {}).get("rationale", ""),
            "no_action": True,
        }
        for ev in no_action_events[-5:]
    ]

    for ticker, agg in ticker_agg.items():
        if agg["count"]:
            agg["avg_return_pct"] = round(agg["total_return_pct"] / agg["count"], 3)
        else:
            agg["avg_return_pct"] = 0.0

    wins = sum(1 for s in scored if s.get("outcome") == "win")
    losses = sum(1 for s in scored if s.get("outcome") == "loss")
    total = wins + losses

    return {
        "lookback_days": lookback_days,
        "decisions_logged": len(actionable) + len(no_action_events),
        "scored_decisions": scored,
        "pending_decisions": pending,
        "no_action_sessions": recent_no_action,
        "recent_events": scored + recent_no_action,
        "ticker_stats": dict(ticker_agg),
        "scorecard": {
            "decisions_logged": len(actionable) + len(no_action_events),
            "no_action_logged": len(no_action_events),
            "decisions_pending": len(pending),
            "decisions_scored": total,
            "wins": wins,
            "losses": losses,
            "win_rate_pct": round(100 * wins / total, 1) if total else None,
        },
    }


def analyze_risk_outcomes(
    system: str,
    lookback_days: int = 7,
    max_exits: int = 20,
) -> Dict[str, Any]:
    """Tier 1 digest: aggregate stop/EOD exits and held positions."""
    hours = max(24, lookback_days * 24)
    events = load_events(limit=5000, system=system, since_hours=hours)

    exits: List[Dict[str, Any]] = []
    held: List[Dict[str, Any]] = []
    early_stops = 0
    stop_returns: List[float] = []

    for ev in events:
        et = ev.get("event_type")
        payload = ev.get("payload") or {}
        if et in ("risk_stop_exit", "risk_eod_exit"):
            ret = payload.get("return_pct")
            if ret is not None:
                stop_returns.append(float(ret))
                if float(ret) > 0 and et == "risk_stop_exit":
                    early_stops += 1
            exits.append({
                "timestamp": ev.get("timestamp"),
                "event_type": et,
                "ticker": payload.get("ticker"),
                "return_pct": ret,
                "reason": (payload.get("reason") or ev.get("action") or "")[:100],
            })
        elif et == "risk_held":
            held.append({
                "timestamp": ev.get("timestamp"),
                "ticker": payload.get("ticker"),
                "reason": payload.get("reason"),
            })

    exits.sort(key=lambda x: x.get("timestamp", ""))
    exits = exits[-max_exits:]

    avg_stop = round(sum(stop_returns) / len(stop_returns), 3) if stop_returns else None
    return {
        "lookback_days": lookback_days,
        "recent_exits": exits,
        "recent_held": held[-10:],
        "scorecard": {
            "stop_exits": len([e for e in exits if e["event_type"] == "risk_stop_exit"]),
            "eod_exits": len([e for e in exits if e["event_type"] == "risk_eod_exit"]),
            "held_count": len(held),
            "avg_exit_return_pct": avg_stop,
            "profitable_stops": early_stops,
        },
    }


def _hours(lookback_days: int) -> int:
    return max(24, lookback_days * 24)


def analyze_coordinator_outcomes(system: str, lookback_days: int = 7) -> Dict[str, Any]:
    hours = _hours(lookback_days)
    started = load_events(limit=2000, system=system, event_type="job_started", since_hours=hours)
    completed = load_events(limit=2000, system=system, event_type="job_completed", since_hours=hours)
    ok = [e for e in completed if e.get("status") == "ok"]
    failed = [e for e in completed if e.get("status") not in (None, "ok")]
    recent = [
        {
            "timestamp": e.get("timestamp"),
            "action": e.get("action"),
            "status": e.get("status"),
            "trace_id": e.get("trace_id"),
        }
        for e in completed[-10:]
    ]
    return {
        "lookback_days": lookback_days,
        "recent_events": recent,
        "scorecard": {
            "jobs_started": len(started),
            "jobs_completed": len(completed),
            "jobs_ok": len(ok),
            "jobs_failed": len(failed),
        },
    }


def analyze_data_outcomes(system: str, lookback_days: int = 7) -> Dict[str, Any]:
    hours = _hours(lookback_days)
    data_events = _events_for_role(system, "data", hours)
    enrichments = [
        e for e in data_events
        if "Enriched" in (e.get("action") or "")
        or (e.get("payload") or {}).get("discovery")
    ]
    mc_loads = [e for e in data_events if e.get("event_type") == "mc_context_loaded"]
    recent = [
        {
            "timestamp": e.get("timestamp"),
            "action": (e.get("action") or "")[:100],
            "event_type": e.get("event_type"),
        }
        for e in data_events[-10:]
    ]
    return {
        "lookback_days": lookback_days,
        "recent_events": recent,
        "scorecard": {
            "data_events": len(data_events),
            "enrichment_runs": len(enrichments),
            "mc_context_loads": len(mc_loads) if system == "internal" else 0,
        },
    }


def analyze_execution_outcomes(system: str, lookback_days: int = 7) -> Dict[str, Any]:
    hours = _hours(lookback_days)
    placed = load_events(limit=2000, system=system, event_type="order_placed", since_hours=hours)
    skipped = load_events(limit=500, system=system, event_type="order_skipped", since_hours=hours)
    chased = load_events(limit=500, system=system, event_type="order_chased", since_hours=hours)
    recent = [
        {
            "timestamp": e.get("timestamp"),
            "event_type": e.get("event_type"),
            "action": (e.get("action") or "")[:100],
            "ticker": (e.get("payload") or {}).get("ticker"),
        }
        for e in (placed + skipped + chased)
    ]
    recent.sort(key=lambda x: x.get("timestamp", ""))
    recent = recent[-12:]
    return {
        "lookback_days": lookback_days,
        "recent_events": recent,
        "scorecard": {
            "orders_placed": len(placed),
            "orders_skipped": len(skipped),
            "orders_chased": len(chased),
        },
    }


def analyze_monitor_outcomes(system: str, lookback_days: int = 7) -> Dict[str, Any]:
    hours = _hours(lookback_days)
    snaps = load_events(limit=500, system=system, event_type="portfolio_snapshot", since_hours=hours)
    values = []
    recent = []
    for e in snaps:
        payload = e.get("payload") or {}
        pv = payload.get("portfolio_value")
        if pv is not None:
            values.append(float(pv))
        recent.append({
            "timestamp": e.get("timestamp"),
            "portfolio_value": pv,
            "total_return_pct": payload.get("total_return_pct"),
        })
    recent = recent[-10:]
    delta = None
    if len(values) >= 2:
        delta = round((values[-1] - values[0]) / values[0] * 100, 3) if values[0] else None
    return {
        "lookback_days": lookback_days,
        "recent_events": recent,
        "scorecard": {
            "snapshots": len(snaps),
            "portfolio_delta_pct": delta,
        },
    }


def analyze_discovery_outcomes(lookback_days: int = 7) -> Dict[str, Any]:
    hours = _hours(lookback_days)
    probes = load_events(limit=500, system="discovery", event_type="discovery_probe", since_hours=hours)
    approved = rejected = 0
    recent = []
    for e in probes:
        payload = e.get("payload") or {}
        summary = payload.get("summary") or payload
        approved += int(summary.get("approved_count") or 0)
        rejected += int(summary.get("rejected_count") or 0)
        recent.append({
            "timestamp": e.get("timestamp"),
            "approved_count": summary.get("approved_count"),
            "rejected_count": summary.get("rejected_count"),
            "probes_today": summary.get("probes_today"),
        })
    recent = recent[-8:]
    return {
        "lookback_days": lookback_days,
        "system": "internal",
        "recent_events": recent,
        "scorecard": {
            "probe_runs": len(probes),
            "features_approved": approved,
            "features_rejected": rejected,
        },
    }


def analyze_agent_outcomes(
    system: str,
    role: str,
    lookback_days: int = 7,
) -> Dict[str, Any]:
    if role == "signal":
        return analyze_signal_outcomes(system, lookback_days=lookback_days)
    if role == "risk":
        return analyze_risk_outcomes(system, lookback_days=lookback_days)
    if role == "coordinator":
        return analyze_coordinator_outcomes(system, lookback_days=lookback_days)
    if role == "data":
        return analyze_data_outcomes(system, lookback_days=lookback_days)
    if role == "execution":
        return analyze_execution_outcomes(system, lookback_days=lookback_days)
    if role == "monitor":
        return analyze_monitor_outcomes(system, lookback_days=lookback_days)
    if role == "discovery":
        return analyze_discovery_outcomes(lookback_days=lookback_days)
    raise ValueError(f"Unknown learning role: {role}")
