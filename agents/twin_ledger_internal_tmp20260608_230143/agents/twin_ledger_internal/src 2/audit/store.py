"""
Read and aggregate audit events for the dashboard API.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src import config


def _parse_ts(ts: str) -> datetime:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _ensure_audit_hydrated() -> None:
    try:
        from src.gcs.store import get_gcs_store

        get_gcs_store().hydrate_audit_log()
    except Exception:
        pass


def load_events(
    limit: int = 500,
    system: Optional[str] = None,
    event_type: Optional[str] = None,
    trace_id: Optional[str] = None,
    since_hours: Optional[int] = None,
    log_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    if log_path is None:
        _ensure_audit_hydrated()
    path = log_path or config.AUDIT_LOG_PATH
    if not path.exists():
        return []

    cutoff = None
    if since_hours:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)

    events = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            if system and ev.get("system") not in (system, "both"):
                continue
            if event_type and ev.get("event_type") != event_type:
                continue
            if trace_id and ev.get("trace_id") != trace_id:
                continue
            if cutoff:
                try:
                    if _parse_ts(ev["timestamp"]) < cutoff:
                        continue
                except (KeyError, ValueError):
                    pass
            events.append(ev)

    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return events[:limit]


def get_summary(since_hours: int = 24) -> Dict[str, Any]:
    events = load_events(limit=5000, since_hours=since_hours)

    by_type = Counter(e.get("event_type") for e in events)
    by_system = Counter(e.get("system") for e in events)
    by_status = Counter(e.get("status") for e in events)

    traces = defaultdict(list)
    for e in events:
        tid = e.get("trace_id")
        if tid:
            traces[tid].append(e)

    recent_jobs = []
    for tid, trace_events in traces.items():
        starts = [e for e in trace_events if e.get("event_type") == "job_started"]
        ends = [e for e in trace_events if e.get("event_type") == "job_completed"]
        if starts:
            start = starts[0]
            end = ends[0] if ends else None
            recent_jobs.append({
                "trace_id": tid,
                "job_type": start.get("action"),
                "system": start.get("system"),
                "started_at": start.get("timestamp"),
                "completed_at": end.get("timestamp") if end else None,
                "status": end.get("status") if end else "running",
                "event_count": len(trace_events),
            })
    recent_jobs.sort(key=lambda j: j.get("started_at", ""), reverse=True)

    orders = [e for e in events if e.get("event_type") == "order_placed"]
    risk_exits = [e for e in events if e.get("event_type") in ("risk_stop_exit", "risk_eod_exit")]
    decisions = [
        e for e in events
        if e.get("event_type") in ("ledger_decision", "arena_decision")
    ]

    performance = None
    try:
        performance = get_performance(since_hours=since_hours)
    except Exception:
        pass

    return {
        "since_hours": since_hours,
        "total_events": len(events),
        "by_event_type": dict(by_type),
        "by_system": dict(by_system),
        "by_status": dict(by_status),
        "orders_placed": len(orders),
        "risk_exits": len(risk_exits),
        "ledger_decisions": len(decisions),
        "recent_jobs": recent_jobs[:20],
        "last_event_at": events[0]["timestamp"] if events else None,
        "performance": performance,
    }


def get_trace(trace_id: str) -> Dict[str, Any]:
    events = load_events(limit=1000, trace_id=trace_id)
    events.sort(key=lambda e: e.get("timestamp", ""))
    return {"trace_id": trace_id, "events": events, "count": len(events)}


def get_performance(since_hours: int = 168) -> Dict[str, Any]:
    """Live competition snapshot plus historical portfolio snapshots from audit log."""
    from src.agents.competition_context import get_competition_snapshot

    live = get_competition_snapshot()
    events = load_events(
        limit=2000,
        event_type="portfolio_snapshot",
        since_hours=since_hours,
    )
    events.sort(key=lambda e: e.get("timestamp", ""))

    history: Dict[str, List[Dict[str, Any]]] = {"baseline": [], "internal": []}
    for e in events:
        system = e.get("system")
        if system not in history:
            continue
        payload = e.get("payload") or {}
        history[system].append({
            "timestamp": e.get("timestamp"),
            "trace_id": e.get("trace_id"),
            "portfolio_value": payload.get("portfolio_value"),
            "pnl_pct": payload.get("pnl_pct"),
            "pnl_usd": payload.get("pnl_usd"),
            "position_count": payload.get("position_count"),
            "cash": payload.get("cash"),
        })

    from src.config import TRADING_UNIVERSE, UNIVERSE_RATIONALE

    return {
        "since_hours": since_hours,
        "live": live,
        "history": history,
        "universe": {
            "rationale": UNIVERSE_RATIONALE,
            "tickers": TRADING_UNIVERSE,
            "count": len(TRADING_UNIVERSE),
        },
    }
