"""Read-only tools for dashboard / coordinator chat (status, audit history)."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.adk.tools.alpaca_tools import get_account_info, get_open_positions
from src.adk.tools.competition_tools import get_competition_status, get_leaderboard
from src.audit.store import load_events


def get_trader_status(system: str = "baseline") -> Dict[str, Any]:
    """Return live portfolio snapshot for this trader (account, positions, leaderboard).

    Args:
        system: ``baseline`` or ``internal``.
    """
    if system not in ("baseline", "internal"):
        return {"success": False, "error": f"Invalid system: {system}"}

    account = get_account_info(system=system) or {}
    positions = get_open_positions(system=system) or {}
    leaderboard = get_leaderboard(system=system) or {}
    competition = get_competition_status() or {}

    return {
        "success": True,
        "system": system,
        "account": account,
        "positions": positions.get("positions", []),
        "position_count": positions.get("count", 0),
        "leaderboard": leaderboard,
        "competition": competition,
    }


def get_recent_trading_activity(
    system: str = "baseline",
    hours: int = 72,
    limit: int = 25,
) -> Dict[str, Any]:
    """Return recent audit events: decisions (with rationale), orders, and jobs.

    Args:
        system: ``baseline`` or ``internal``.
        hours: Lookback window in hours (default 72).
        limit: Max events per category.
    """
    if system not in ("baseline", "internal"):
        return {"success": False, "error": f"Invalid system: {system}"}

    hours = max(1, min(int(hours), 168))
    limit = max(1, min(int(limit), 50))

    events = load_events(limit=2000, system=system, since_hours=hours)

    decisions: List[Dict[str, Any]] = []
    orders: List[Dict[str, Any]] = []
    jobs: Dict[str, Dict[str, Any]] = {}

    for ev in events:
        et = ev.get("event_type")
        payload = ev.get("payload") or {}
        row = {
            "timestamp": ev.get("timestamp"),
            "trace_id": ev.get("trace_id"),
            "event_type": et,
            "action": ev.get("action"),
            "status": ev.get("status"),
        }

        if et in ("ledger_decision", "arena_decision"):
            decisions.append({
                **row,
                "ticker": payload.get("ticker"),
                "side": payload.get("action"),
                "size_pct": payload.get("size_pct"),
                "confidence": payload.get("confidence"),
                "rationale": payload.get("rationale"),
                "invalidation": payload.get("invalidation"),
                "competitive_note": payload.get("competitive_note"),
            })
        elif et == "order_placed":
            orders.append({
                **row,
                "ticker": payload.get("ticker"),
                "side": payload.get("side"),
                "qty": payload.get("qty"),
                "limit_price": payload.get("limit_price"),
                "time_in_force": payload.get("time_in_force"),
                "order_id": payload.get("order_id"),
            })
        elif et == "job_started":
            tid = ev.get("trace_id")
            if tid and tid not in jobs:
                jobs[tid] = {
                    "trace_id": tid,
                    "job_type": ev.get("action"),
                    "started_at": ev.get("timestamp"),
                    "completed_at": None,
                    "status": "running",
                }
        elif et == "job_completed":
            tid = ev.get("trace_id")
            if tid:
                jobs.setdefault(tid, {"trace_id": tid})
                jobs[tid].update({
                    "completed_at": ev.get("timestamp"),
                    "status": ev.get("status", "ok"),
                    "summary": {
                        k: payload.get(k)
                        for k in ("orders_placed", "message", "error")
                        if payload.get(k) is not None
                    },
                })

    job_list = sorted(
        jobs.values(),
        key=lambda j: j.get("started_at") or j.get("completed_at") or "",
        reverse=True,
    )[:limit]

    return {
        "success": True,
        "system": system,
        "since_hours": hours,
        "decisions": decisions[:limit],
        "orders": orders[:limit],
        "recent_jobs": job_list,
        "counts": {
            "decisions": len(decisions),
            "orders": len(orders),
            "jobs": len(job_list),
        },
    }
