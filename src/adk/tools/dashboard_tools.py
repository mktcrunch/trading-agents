"""Read-only tools for dashboard / coordinator chat (status, audit history)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from src import config
from src.adk.tools.alpaca_tools import get_account_info, get_open_positions
from src.adk.tools.competition_tools import get_competition_status, get_leaderboard
from src.audit.store import load_events
from src.discovery.approved_sources import load_approved_sources
from src.discovery.registry import load_registry
from src.logger import setup_logger
from src.strategies.order_dedup import (
    order_live_snapshot,
    static_order_snapshot,
)

logger = setup_logger(__name__)


def _hydrate_discovery_artifacts() -> None:
    try:
        from src.gcs.store import get_gcs_store

        get_gcs_store().hydrate_all_local_data()
    except Exception:
        pass


def _parse_ts(ts: str) -> Optional[datetime]:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _source_summary(source: Dict[str, Any]) -> Dict[str, Any]:
    metrics = source.get("metrics") or {}
    return {
        "id": source.get("id"),
        "dataset": source.get("dataset"),
        "feature": source.get("feature"),
        "status": source.get("status"),
        "proposed": source.get("proposed"),
        "ic": metrics.get("ic"),
        "mi": metrics.get("mi"),
        "t_stat": metrics.get("t_stat"),
        "incremental_alpha": metrics.get("incremental_alpha"),
    }


def _evaluation_summary(evaluation: Dict[str, Any]) -> Dict[str, Any]:
    """Compact per-feature approve/reject report for dashboard chat."""
    return {
        "feature_id": evaluation.get("feature_id"),
        "status": evaluation.get("status"),
        "failed_gate": evaluation.get("failed_gate"),
        "proposed": evaluation.get("proposed"),
        "metrics": evaluation.get("metrics"),
        "gates": evaluation.get("gates"),
        "thresholds": evaluation.get("thresholds"),
    }


def _probe_summary(probe: Dict[str, Any]) -> Dict[str, Any]:
    proposed = probe.get("proposed_features") or []
    approved_ids = probe.get("approved_feature_ids") or []
    evaluations = probe.get("feature_evaluations") or []
    rejected_count = probe.get("rejected_count")
    if rejected_count is None:
        rejected_count = max(0, len(evaluations) - len(approved_ids))
    return {
        "dataset": probe.get("dataset"),
        "schema": probe.get("schema"),
        "status": probe.get("status"),
        "approved_count": probe.get("approved_count"),
        "rejected_count": rejected_count,
        "approved_feature_ids": approved_ids,
        "proposed_feature_count": len(proposed) or len(evaluations),
        "best_ic": probe.get("best_ic"),
        "feature_strategy": probe.get("feature_strategy"),
        "rationale": probe.get("rationale"),
        "error": probe.get("error"),
        "feature_evaluations": [_evaluation_summary(e) for e in evaluations],
    }


def _feature_names_from_cache(ticker_features: Dict[str, Any]) -> List[str]:
    names: set[str] = set()
    for feats in (ticker_features or {}).values():
        if isinstance(feats, dict):
            names.update(feats.keys())
    return sorted(names)


def _cache_status(approved: Dict[str, Any], hours: int) -> Dict[str, Any]:
    from src.discovery.approved_sources import is_stale

    generated = _parse_ts(approved.get("generated_at"))
    now = datetime.now(timezone.utc)
    age_hours = (now - generated).total_seconds() / 3600 if generated else None
    sources = approved.get("sources") or []
    ticker_features = approved.get("ticker_features") or {}
    feature_names = _feature_names_from_cache(ticker_features)
    summary = approved.get("summary") or {}

    ran_within_lookback = age_hours is not None and age_hours <= hours
    interpretation: List[str] = []

    if not approved.get("generated_at"):
        interpretation.append("No discovery run has completed yet.")
    elif age_hours is not None and age_hours > 24:
        interpretation.append(
            f"No discovery run today (last run {age_hours:.0f}h ago at {approved['generated_at']})."
        )
    elif age_hours is not None and age_hours <= 24:
        interpretation.append(
            f"A discovery run completed within the last 24h ({approved['generated_at']})."
        )

    if ticker_features and not sources:
        interpretation.append(
            f"Cache has computed values for {len(ticker_features)} tickers "
            f"({', '.join(feature_names) or 'n/a'}) on dataset "
            f"{approved.get('dataset', 'unknown')}/{approved.get('schema', 'unknown')}, "
            "but 0 features passed the 3 approval gates — not gate-approved for trading."
        )
    elif sources:
        interpretation.append(f"{len(sources)} feature sources are gate-approved.")

    return {
        "last_run_at": approved.get("generated_at"),
        "age_hours": round(age_hours, 1) if age_hours is not None else None,
        "is_stale": is_stale(),
        "ran_within_lookback_hours": ran_within_lookback,
        "ran_in_last_24h": age_hours is not None and age_hours <= 24,
        "dataset": approved.get("dataset"),
        "schema": approved.get("schema"),
        "mode": approved.get("mode"),
        "approved_source_count": len(sources),
        "tickers_with_computed_features": list(ticker_features.keys()),
        "computed_feature_names": feature_names,
        "gate_approved_for_trading": bool(sources),
        "last_run_summary": summary,
        "interpretation": interpretation,
    }


def _legacy_probe_from_cache(approved: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Synthesize probe summary for older payloads without probes_today."""
    if approved.get("probes_today"):
        return None
    if not approved.get("dataset") and not approved.get("generated_at"):
        return None
    summary = approved.get("summary") or {}
    sources = approved.get("sources") or []
    return {
        "dataset": approved.get("dataset"),
        "schema": approved.get("schema"),
        "status": "approved" if sources else "rejected",
        "approved_count": summary.get("approved_count", len(sources)),
        "rejected_count": max(0, 4 - summary.get("approved_count", len(sources))),
        "approved_feature_ids": [s.get("id") for s in sources],
        "proposed_feature_count": 4,
        "best_ic": None,
        "feature_strategy": "legacy fixed dataset probe",
        "rationale": "Legacy discovery run (pre-agentic probes_today field)",
        "error": summary.get("error"),
    }


def _approval_criteria() -> Dict[str, Any]:
    cfg = config.DISCOVERY_CONFIG
    return {
        "process": (
            "Automated 3-gate statistical approval during each discovery probe. "
            "No human sign-off. Baseline formulas (volume_zscore, momentum_5d, etc.) "
            "and LLM-proposed formulas are evaluated the same way on vendor OHLCV bars."
        ),
        "gates": {
            "gate_1_mutual_information": {
                "threshold": cfg.get("gate_1_mi_threshold"),
                "description": "Feature must have MI >= threshold vs forward 1d return",
            },
            "gate_2_ic_and_t_stat": {
                "ic_threshold": cfg.get("gate_2_ic_threshold"),
                "t_stat_threshold": cfg.get("gate_2_t_stat_threshold"),
                "description": "Spearman IC and t-stat must exceed thresholds",
            },
            "gate_3_incremental_alpha": {
                "threshold": cfg.get("gate_3_incremental_alpha_threshold"),
                "description": "IC must beat baseline momentum by threshold",
            },
            "universe_coverage_pct": {
                "minimum": cfg.get("min_universe_coverage_pct"),
                "description": "Feature must cover enough tickers in the universe",
            },
        },
        "persistence": {
            "approved_file": "approved_datasources.json",
            "registry_file": "discovery_registry.json",
            "audit_event_type": "discovery_probe",
            "stale_after_hours": cfg.get("max_age_hours", 24),
        },
    }


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


def _fetch_live_order_statuses(
    system: str,
    order_ids: List[str],
) -> Dict[str, Dict[str, Any]]:
    """Build order_id -> live Alpaca snapshot for dashboard reconciliation."""
    unique_ids = sorted({oid for oid in order_ids if oid and oid != "dry-run"})
    if not unique_ids:
        return {}

    from src.apis.alpaca_client import AlpacaClient

    lookup: Dict[str, Dict[str, Any]] = {}
    try:
        client = AlpacaClient(system=system)
    except Exception as exc:
        logger.warning(f"Could not initialize Alpaca client for {system}: {exc}")
        return {}

    try:
        for order in client.get_orders(status="all"):
            oid = str(getattr(order, "id", "") or "")
            if oid in unique_ids:
                lookup[oid] = order_live_snapshot(order)
    except Exception as exc:
        logger.warning(f"Failed to list Alpaca orders for {system}: {exc}")

    for oid in unique_ids:
        if oid in lookup:
            continue
        try:
            order = client.get_order(oid)
        except Exception:
            order = None
        if order:
            lookup[oid] = order_live_snapshot(order)

    return lookup


def _unknown_order_snapshot() -> Dict[str, Any]:
    return {
        "alpaca_status": "unknown",
        "alpaca_filled_qty": None,
        "alpaca_remaining_qty": None,
        "alpaca_is_active": False,
        "alpaca_status_note": (
            "Not found in Alpaca (may be purged or outside the query window)"
        ),
    }


def annotate_orders_with_alpaca_status(
    system: str,
    orders: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach live Alpaca status to audit order rows (display only)."""
    order_ids = [str(o.get("order_id") or "") for o in orders]
    live = _fetch_live_order_statuses(system, order_ids)

    annotated: List[Dict[str, Any]] = []
    for row in orders:
        oid = str(row.get("order_id") or "")
        snap = static_order_snapshot(oid)
        if snap is None and oid:
            snap = live.get(oid) or _unknown_order_snapshot()
        annotated.append({**row, **(snap or {})})
    return annotated


def enrich_order_placed_audit_events(
    events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Add live Alpaca status into order_placed audit payloads for the dashboard API."""
    ids_by_system: Dict[str, List[str]] = {}
    for ev in events:
        if ev.get("event_type") != "order_placed":
            continue
        system = ev.get("system") or "baseline"
        if system not in ("baseline", "internal"):
            continue
        payload = ev.get("payload") or {}
        oid = str(payload.get("order_id") or "")
        if oid:
            ids_by_system.setdefault(system, []).append(oid)

    live_by_system = {
        system: _fetch_live_order_statuses(system, ids)
        for system, ids in ids_by_system.items()
    }

    enriched: List[Dict[str, Any]] = []
    for ev in events:
        if ev.get("event_type") != "order_placed":
            enriched.append(ev)
            continue

        payload = dict(ev.get("payload") or {})
        oid = str(payload.get("order_id") or "")
        system = ev.get("system") or "baseline"
        snap = static_order_snapshot(oid)
        if snap is None and oid:
            snap = live_by_system.get(system, {}).get(oid) or _unknown_order_snapshot()
        if snap:
            payload.update(snap)
        enriched.append({**ev, "payload": payload})
    return enriched


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

    orders = annotate_orders_with_alpaca_status(system, orders[:limit])

    return {
        "success": True,
        "system": system,
        "since_hours": hours,
        "decisions": decisions[:limit],
        "orders": orders,
        "recent_jobs": job_list,
        "counts": {
            "decisions": len(decisions),
            "orders": len(orders),
            "jobs": len(job_list),
        },
    }


def get_data_discovery_trail(
    hours: int = 168,
    limit: int = 20,
) -> Dict[str, Any]:
    """Return the agentic data-discovery audit trail (probes, registry runs, approved sources).

    Use this when the user asks what datasets were discovered, probed, or approved today
    or recently. Covers the autonomous catalog-scan → probe → evaluate pipeline.

    Args:
        hours: Lookback window in hours (default 168 = 7 days).
        limit: Max probe events and daily runs to return.
    """
    _hydrate_discovery_artifacts()
    hours = max(1, min(int(hours), 336))
    limit = max(1, min(int(limit), 50))
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)

    events = load_events(limit=3000, system="discovery", since_hours=hours)
    audit_probes: List[Dict[str, Any]] = []
    for ev in events:
        if ev.get("event_type") != "discovery_probe":
            continue
        payload = ev.get("payload") or {}
        evaluations = payload.get("feature_evaluations") or []
        audit_probes.append({
            "timestamp": ev.get("timestamp"),
            "trace_id": ev.get("trace_id"),
            "dataset": payload.get("dataset"),
            "schema": payload.get("schema"),
            "status": payload.get("status"),
            "approved_count": payload.get("approved_count"),
            "rejected_count": payload.get("rejected_count"),
            "best_ic": payload.get("best_ic"),
            "feature_strategy": payload.get("feature_strategy"),
            "proposed_features": (payload.get("proposed_features") or [])[:5],
            "feature_evaluations": [_evaluation_summary(e) for e in evaluations],
            "action": ev.get("action"),
        })

    approved = load_approved_sources()
    registry = load_registry()

    daily_runs: List[Dict[str, Any]] = []
    for run in reversed(registry.get("daily_runs") or []):
        completed = _parse_ts(run.get("completed_at"))
        if completed and completed < cutoff:
            continue
        daily_runs.append({
            "completed_at": run.get("completed_at"),
            "strategy": run.get("strategy"),
            "probes_run": run.get("probes_run"),
            "catalog_size": run.get("catalog_size"),
            "approved_total": run.get("approved_total"),
            "probe_results": (run.get("probe_results") or [])[:limit],
        })
        if len(daily_runs) >= limit:
            break

    probes_today = list(approved.get("probes_today") or [])
    legacy_probe = _legacy_probe_from_cache(approved)
    if legacy_probe and not probes_today:
        probes_today = [legacy_probe]

    sources = approved.get("sources") or []
    ticker_features = approved.get("ticker_features") or {}
    cache = _cache_status(approved, hours)

    recent_discovery_actions: List[Dict[str, Any]] = []
    for ev in load_events(limit=100, system="discovery", since_hours=max(hours, 168)):
        if ev.get("event_type") not in ("agent_action", "discovery_probe"):
            continue
        recent_discovery_actions.append({
            "timestamp": ev.get("timestamp"),
            "event_type": ev.get("event_type"),
            "action": ev.get("action"),
            "payload": ev.get("payload"),
        })

    return {
        "success": True,
        "since_hours": hours,
        "cache_status": cache,
        "approval_criteria": _approval_criteria(),
        "latest_discovery_run": {
            "generated_at": approved.get("generated_at"),
            "mode": approved.get("mode"),
            "strategy_note": approved.get("strategy_note"),
            "summary": approved.get("summary"),
            "probes_today": [_probe_summary(p) for p in probes_today[:limit]],
        },
        "approved_sources": [_source_summary(s) for s in sources[:limit]],
        "approved_source_count": len(sources),
        "tickers_with_discovered_features": list(ticker_features.keys()),
        "registry_daily_runs": daily_runs,
        "audit_probe_events": audit_probes[:limit],
        "recent_discovery_activity": recent_discovery_actions[:limit],
        "counts": {
            "audit_probes": len(audit_probes),
            "registry_runs": len(daily_runs),
            "approved_sources": len(sources),
            "probes_today": len(probes_today),
        },
    }


def get_proprietary_data_usage(
    system: str = "internal",
    hours: int = 72,
    limit: int = 25,
) -> Dict[str, Any]:
    """Return which enrichment / proprietary data sources were used for recent trading.

    For internal: proprietary signal API predictions + discovered market features.
    For baseline: Alpaca OHLCV and technical indicators only.

    Args:
        system: ``baseline`` or ``internal``.
        hours: Lookback window in hours (default 72).
        limit: Max enrichment events per category.
    """
    if system not in ("baseline", "internal"):
        return {"success": False, "error": f"Invalid system: {system}"}

    hours = max(1, min(int(hours), 168))
    limit = max(1, min(int(limit), 50))

    if system == "baseline":
        return {
            "success": True,
            "system": system,
            "since_hours": hours,
            "data_sources": ["alpaca_ohlcv", "technical_indicators", "competition_leaderboard"],
            "proprietary_enrichment": False,
            "note": (
                "Baseline does not use proprietary signal APIs or discovered feature "
                "enrichment — only public market data and technicals."
            ),
        }

    _hydrate_discovery_artifacts()
    events = load_events(limit=3000, system="internal", since_hours=hours)

    signal_api_fetches: List[Dict[str, Any]] = []
    enrichment_events: List[Dict[str, Any]] = []
    seen_tickers: set[str] = set()

    for ev in events:
        payload = ev.get("payload") or {}
        action = ev.get("action") or ""

        if payload.get("confidence") is not None or "MC analysis" in action:
            ticker = payload.get("ticker")
            if ticker:
                seen_tickers.add(ticker)
            signal_api_fetches.append({
                "timestamp": ev.get("timestamp"),
                "ticker": ticker,
                "confidence": payload.get("confidence"),
                "target_delta_numeric": payload.get("target_delta_numeric"),
                "target_price": payload.get("target_price"),
                "current_price": payload.get("current_price"),
            })
        elif "Enriched" in action or payload.get("discovery"):
            enrichment_events.append({
                "timestamp": ev.get("timestamp"),
                "action": action,
                "approved_count": payload.get("approved_count"),
                "enriched_tickers": payload.get("enriched_tickers"),
                "discovery": payload.get("discovery"),
            })

    approved = load_approved_sources()
    ticker_features = approved.get("ticker_features") or {}

    return {
        "success": True,
        "system": system,
        "since_hours": hours,
        "data_sources": [
            "alpaca_ohlcv",
            "proprietary_signal_api",
            "discovered_market_features",
            "competition_leaderboard",
        ],
        "proprietary_enrichment": True,
        "proprietary_signal_api": {
            "tickers_fetched": sorted(seen_tickers),
            "recent_fetches": signal_api_fetches[:limit],
        },
        "discovered_market_features": {
            "generated_at": approved.get("generated_at"),
            "strategy_note": approved.get("strategy_note"),
            "approved_source_count": len(approved.get("sources") or []),
            "approved_sources": [
                _source_summary(s) for s in (approved.get("sources") or [])[:limit]
            ],
            "tickers_with_features": list(ticker_features.keys()),
            "sample_features_by_ticker": {
                t: list((feats or {}).keys())[:8]
                for t, feats in list(ticker_features.items())[:8]
            },
        },
        "recent_enrichment_events": enrichment_events[:limit],
        "counts": {
            "signal_api_fetches": len(signal_api_fetches),
            "enrichment_events": len(enrichment_events),
        },
    }


def _trader_agent_roster(system: str) -> List[Dict[str, Any]]:
    """Six specialist roles per competing trader (discovery is Internal-only MC Internal IP)."""
    data_agent = "BaselineDataAgent" if system == "baseline" else "InternalDataAgent"
    signal_agent = "BaselineSignalAgent" if system == "baseline" else "InternalSignalAgent"
    internal_only = system == "internal"
    return [
        {
            "role": "coordinator",
            "name": f"twin_ledger_{system}",
            "label": "Coordinator",
            "audit_agents": ["orchestrator"],
            "event_types": [
                "job_started", "job_completed", "sub_job_started", "sub_job_completed",
            ],
            "description": "Routes scheduler phrases to deterministic workflows; chat entry point.",
        },
        {
            "role": "data",
            "name": f"{system}_data",
            "label": "Data agent",
            "audit_agents": [data_agent],
            "event_types": ["agent_action", "mc_context_loaded"],
            "description": (
                "Fetches account, positions, OHLCV, technicals, leaderboard"
                + (" + proprietary signal API + discovered features." if internal_only else ".")
            ),
        },
        {
            "role": "signal",
            "name": f"{system}_signal",
            "label": "Signal agent",
            "audit_agents": [signal_agent, "SignalAgent"],
            "event_types": ["ledger_decision", "arena_decision", "agent_action"],
            "description": "Gemini structured BUY/SELL/HOLD/CLOSE with rationale.",
        },
        {
            "role": "risk",
            "name": "RiskAgent + RiskMonitor",
            "label": "Risk agent",
            "audit_agents": ["RiskAgent", "RiskMonitor"],
            "event_types": [
                "agent_action",
                "risk_positions_checked",
                "risk_stop_exit",
                "risk_eod_exit",
                "risk_held",
                "trailing_stop_planned",
            ],
            "description": (
                "Deterministic scheduler path to intraday risk check; Gemini trailing-stop planner each cycle."
                + (" MC Internal prediction gate on stop exits." if internal_only else "")
                + " Pre-trade: rule-based weight/position caps."
            ),
        },
        {
            "role": "execution",
            "name": "ExecutionAgent",
            "label": "Execution agent",
            "audit_agents": ["ExecutionAgent"],
            "event_types": [
                "order_placed",
                "order_chased",
                "order_skipped",
                "order_cancelled_duplicate",
                "agent_action",
            ],
            "description": "DAY limit orders, delta vs pending, post-open chase.",
        },
        {
            "role": "monitor",
            "name": "MonitorAgent",
            "label": "Monitor agent",
            "audit_agents": ["MonitorAgent"],
            "event_types": ["portfolio_snapshot", "agent_action"],
            "description": "Portfolio metrics snapshot after daily runs.",
        },
    ]


def _event_matches_role(ev: Dict[str, Any], role_cfg: Dict[str, Any]) -> bool:
    agent = ev.get("agent") or ""
    et = ev.get("event_type") or ""
    agents = role_cfg.get("audit_agents") or []
    types = role_cfg.get("event_types") or []
    agent_ok = agent in agents if agents else True
    type_ok = et in types if types else True
    if agents and types:
        return agent_ok and type_ok
    return agent_ok or type_ok


def _summarize_agent_event(ev: Dict[str, Any]) -> Dict[str, Any]:
    payload = ev.get("payload") or {}
    summary: Dict[str, Any] = {
        "timestamp": ev.get("timestamp"),
        "trace_id": ev.get("trace_id"),
        "event_type": ev.get("event_type"),
        "agent": ev.get("agent"),
        "action": ev.get("action"),
        "status": ev.get("status"),
    }
    for key in (
        "ticker", "side", "qty", "order_id", "returns", "reason",
        "rationale", "confidence", "orders_placed", "message",
        "discovery", "enriched_tickers", "approved_count", "predictions",
        "source", "tickers_fetched", "lookback_days", "positions",
        "validation_results", "total_weight",
    ):
        if key in payload:
            summary[key] = payload[key]
    if payload and len(summary) <= 8:
        summary["payload"] = payload
    return summary


def _internal_data_enrichment_context() -> Dict[str, Any]:
    """What the Internal data agent loads from the discovery pipeline (not Discovery agent itself)."""
    _hydrate_discovery_artifacts()
    approved = load_approved_sources()
    sources = approved.get("sources") or []
    ticker_features = approved.get("ticker_features") or {}
    return {
        "note": (
            "Discovery agent runs catalog probes separately. The Internal DATA agent "
            "consumes gate-approved vendor features from approved_datasources.json "
            "during enrich_with_databento() — look for 'Enriched N/M tickers' in events."
        ),
        "last_discovery_at": approved.get("generated_at"),
        "gate_approved_source_count": len(sources),
        "tickers_with_vendor_features": list(ticker_features.keys()),
        "approved_source_ids": [s.get("id") for s in sources[:12]],
    }


def _other_agents_footer(system: str, exclude_role: Optional[str] = None) -> List[str]:
    roles = [r["role"] for r in _trader_agent_roster(system)] + ["discovery"]
    if exclude_role:
        roles = [r for r in roles if r != exclude_role]
    return roles


def get_agent_activity(
    system: str = "baseline",
    agent_role: str = "all",
    hours: int = 24,
    limit: int = 30,
) -> Dict[str, Any]:
    """Return audit-trail activity for specialist agents on a trader crew.

    Use when the user asks what a specific agent did (risk, execution, signal, data,
    coordinator, monitor). Reads the GCS audit log — does NOT run live workflows.

    Args:
        system: ``baseline`` or ``internal``.
        agent_role: ``coordinator``, ``data``, ``signal``, ``risk``, ``execution``,
            ``monitor``, ``discovery``, or ``all``.
        hours: Lookback window (default 24).
        limit: Max events per role.
    """
    if system not in ("baseline", "internal"):
        return {"success": False, "error": f"Invalid system: {system}"}

    hours = max(1, min(int(hours), 168))
    limit = max(1, min(int(limit), 50))
    role_key = (agent_role or "all").strip().lower()
    roster = _trader_agent_roster(system)

    internal_discovery = {
        "role": "discovery",
        "name": "DiscoveryAgent",
        "label": "Discovery agent · Internal only · MC Internal IP",
        "description": (
            "Proprietary vendor catalog probes; runs before Internal overnight if stale. "
            "Baseline does not use discovery. Separate from Internal data enrichment."
        ),
    }
    crew_roster = [
        {k: r[k] for k in ("role", "name", "label", "description")}
        for r in roster
    ]

    if role_key == "discovery":
        events = load_events(limit=3000, system="discovery", since_hours=hours)
        discovery_events = [
            _summarize_agent_event(ev)
            for ev in events
            if ev.get("agent") == "DiscoveryAgent"
            or ev.get("event_type") in ("discovery_probe", "agent_action")
        ]
        return {
            "success": True,
            "system": system,
            "agent_role": "discovery",
            "since_hours": hours,
            "focused_agent": {
                **internal_discovery,
                "event_count": len(discovery_events),
                "events": discovery_events[:limit],
            },
            "hint": "For probe gate metrics use get_data_discovery_trail.",
            "other_agents_you_can_ask": _other_agents_footer(system, "discovery"),
        }

    events = load_events(limit=4000, system=system, since_hours=hours)
    roles_to_query = roster
    if role_key != "all":
        roles_to_query = [r for r in roster if r["role"] == role_key]
        if not roles_to_query:
            return {
                "success": False,
                "error": f"Unknown agent_role: {agent_role}",
                "valid_roles": [r["role"] for r in roster] + ["discovery", "all"],
            }

    by_role: Dict[str, Any] = {}
    for role_cfg in roles_to_query:
        matched = [ev for ev in events if _event_matches_role(ev, role_cfg)]
        entry: Dict[str, Any] = {
            "label": role_cfg["label"],
            "name": role_cfg["name"],
            "description": role_cfg["description"],
            "event_count": len(matched),
            "events": [_summarize_agent_event(ev) for ev in matched[:limit]],
        }
        if role_cfg["role"] == "data" and system == "internal":
            entry["vendor_features_from_discovery"] = _internal_data_enrichment_context()
            entry["enrichment_events"] = [
                _summarize_agent_event(ev)
                for ev in matched
                if "Enriched" in (ev.get("action") or "")
                or (ev.get("payload") or {}).get("discovery")
            ][:limit]
        by_role[role_cfg["role"]] = entry

    if role_key != "all":
        focused = by_role.get(role_key, {})
        return {
            "success": True,
            "system": system,
            "agent_role": role_key,
            "since_hours": hours,
            "focused_agent": focused,
            "other_agents_you_can_ask": _other_agents_footer(system, role_key),
            "note": (
                "Historical audit events only. Reply about THIS agent only — "
                "do not enumerate the full crew roster in the answer."
            ),
        }

    return {
        "success": True,
        "system": system,
        "agent_role": role_key,
        "since_hours": hours,
        "roster": crew_roster,
        "internal_only_agents": [internal_discovery],
        "available_agent_roles": [r["role"] for r in roster] + ["discovery"],
        "agents": by_role,
        "note": (
            "Historical audit events only. Intraday risk runs every 15m via scheduler; "
            "if no risk events appear, the market may have been closed or no stops triggered."
        ),
    }


def format_agent_activity_report(data: Dict[str, Any]) -> str:
    """Plain-text summary of get_agent_activity for dashboard display."""
    if not data.get("success"):
        return f"Could not load agent activity: {data.get('error', 'unknown error')}"

    system = data.get("system", "?")
    hours = data.get("since_hours", 24)
    role_key = data.get("agent_role", "all")

    if role_key == "all":
        agents = data.get("agents") or {}
        lines = [f"All agents ({system}) — last {hours}h", ""]
        for role, entry in agents.items():
            lines.append(f"• {entry.get('label', role)}: {entry.get('event_count', 0)} event(s)")
        internal_only = data.get("internal_only_agents") or []
        for entry in internal_only:
            lines.append(f"• {entry.get('label', entry.get('role'))}: see discovery audit (Internal only)")
        return "\n".join(lines)

    focused = data.get("focused_agent") or {}
    label = focused.get("label") or role_key
    description = focused.get("description") or ""
    events = focused.get("events") or []
    count = focused.get("event_count", len(events))

    lines = [
        f"{label} ({system}) — last {hours}h",
        description,
        f"{count} audit event(s)",
        "",
    ]

    if focused.get("vendor_features_from_discovery"):
        ctx = focused["vendor_features_from_discovery"]
        lines.append(
            f"Vendor features loaded: {ctx.get('gate_approved_source_count', 0)} approved source(s), "
            f"last discovery {ctx.get('last_discovery_at') or 'unknown'}"
        )
        lines.append("")

    if not events:
        lines.append("No events in this window.")
    else:
        for i, ev in enumerate(events[:20], 1):
            ts = (ev.get("timestamp") or "?")[:19].replace("T", " ")
            et = ev.get("event_type") or "event"
            action = (ev.get("action") or "").strip()
            ticker = ev.get("ticker")
            parts = [f"{i}. [{ts}] {et}"]
            if action:
                parts.append(action[:120])
            if ticker:
                parts.append(f"({ticker})")
            for key in ("side", "qty", "reason", "rationale", "confidence", "orders_placed"):
                if ev.get(key) is not None:
                    parts.append(f"{key}={ev[key]}")
            lines.append(" · ".join(parts))

    others = data.get("other_agents_you_can_ask") or []
    if others:
        lines.extend(["", f"Also ask about: {', '.join(others)}"])
    return "\n".join(lines)


def get_agent_learning(
    system: str = "baseline",
    agent_role: str = "all",
) -> Dict[str, Any]:
    """Return persisted learning memory for trader crew agents."""
    from src.learning.context import build_risk_learning_block, build_signal_learning_block
    from src.learning.roles import roles_for_system
    from src.learning.store import load_all_learning, load_learning

    system = (system or "baseline").strip().lower()
    if system not in ("baseline", "internal"):
        return {"success": False, "error": "system must be baseline or internal"}

    role_key = (agent_role or "all").strip().lower()
    valid_roles = roles_for_system(system)

    if role_key != "all":
        if role_key not in valid_roles:
            return {
                "success": False,
                "error": f"agent_role must be one of: {', '.join(valid_roles)}, all",
            }
        store_system = "internal" if role_key == "discovery" else system
        agent_state = load_learning(store_system, role_key)
        return {
            "success": True,
            "system": system,
            "agent_role": role_key,
            "agent": agent_state,
            "other_agents_you_can_ask": [r for r in valid_roles if r != role_key],
        }

    agents = load_all_learning(system)
    result: Dict[str, Any] = {
        "success": True,
        "system": system,
        "agents": agents,
        # Back-compat for dashboard panels that read signal/risk at top level
        "signal": agents.get("signal", {}),
        "risk": agents.get("risk", {}),
    }
    if config.LEARNING_ENABLED:
        result["signal_prompt_block"] = build_signal_learning_block(system)
        result["risk_prompt_block"] = build_risk_learning_block(system)
    return result



def get_performance_metrics(hours: int = 720, perspective: Optional[str] = None) -> Dict[str, Any]:
    """Return Twin Ledger quant head-to-head metrics (same data as the Performance dashboard).

    Use for questions about excess return, daily delta, Sharpe difference, drawdown
    difference, statistical significance (p-values), and projected days to significance.

    Args:
        hours: Alpaca equity history lookback (720 ≈ 30d, 2160 ≈ 90d, 4320 ≈ 180d).
        perspective: Optional ``baseline`` or ``internal`` — adds ``for_you`` where
            positive values favor that desk. Omit for neutral Internal − Baseline view.
    """
    try:
        from src.agents.competition_context import (
            PERFORMANCE_METRICS_METHODOLOGY,
            build_perspective_quant_view,
            fetch_quant_head_to_head,
            get_competition_snapshot,
        )

        qh = fetch_quant_head_to_head(since_hours=int(hours))
        live = get_competition_snapshot()
    except Exception as exc:
        logger.exception("get_performance_metrics failed")
        return {"success": False, "error": str(exc)}

    metrics = qh.get("metrics") or {}
    perspectives = qh.get("perspectives") or {}
    if not perspectives and metrics.get("comparison"):
        perspectives = {
            "baseline": build_perspective_quant_view(metrics, "baseline"),
            "internal": build_perspective_quant_view(metrics, "internal"),
        }

    for_you = perspectives.get(perspective) if perspective in ("baseline", "internal") else None

    return {
        "success": True,
        "since_hours": qh.get("since_hours", hours),
        "history_points": qh.get("history_points") or {},
        "data_quality": qh.get("data_quality") or {},
        "live": live,
        "metrics": metrics,
        "perspectives": perspectives,
        "for_you": for_you,
        "methodology": PERFORMANCE_METRICS_METHODOLOGY,
        "report": format_performance_metrics_report(
            metrics, live, perspective=perspective, for_you=for_you,
        ),
    }


def format_performance_metrics_report(
    metrics: Optional[Dict[str, Any]],
    live: Optional[Dict[str, Any]] = None,
    *,
    perspective: Optional[str] = None,
    for_you: Optional[Dict[str, Any]] = None,
) -> str:
    """Plain-text summary of quant metrics for coordinator chat."""
    if not metrics or not metrics.get("observation_days"):
        return "Quant metrics unavailable (insufficient paired Alpaca equity history)."

    cmp = metrics.get("comparison") or {}
    sig = cmp.get("significance") or {}
    b = metrics.get("baseline") or {}
    i = metrics.get("internal") or {}
    lb = (live or {}).get("leaderboard") or {}
    lines = [
        "Twin Ledger quant head-to-head (aligned paired days):",
        f"  Sign convention: {cmp.get('sign_convention', 'internal_minus_baseline')} "
        f"({cmp.get('formula', 'internal_value - baseline_value')})",
        f"  Paired days: {metrics.get('observation_days')} "
        f"(through {metrics.get('latest_date') or 'n/a'})",
    ]
    if lb.get("gap_usd") is not None:
        leader = lb.get("leader", "n/a")
        lines.append(
            f"  Live leaderboard: {leader} leads by ${lb.get('gap_usd'):,.2f}"
        )

    if perspective in ("baseline", "internal") and for_you:
        fy = for_you.get("for_you") or {}
        interp = for_you.get("interpretation") or {}
        lines.extend([
            f"  Perspective: {for_you.get('you_are', perspective)} "
            "(for_you: positive favors you)",
            f"    Your excess return vs competitor: {fy.get('excess_return_pct')}% "
            f"({interp.get('excess_return', 'n/a')})",
            f"    Your daily delta (latest day): {fy.get('daily_delta_pct')}% "
            f"({interp.get('daily_delta', 'n/a')})",
            f"    Your Sharpe difference: {fy.get('sharpe_difference')} "
            f"({interp.get('sharpe', 'n/a')})",
            f"    Your drawdown advantage: {fy.get('drawdown_advantage_pp')} pp "
            f"({interp.get('drawdown', 'n/a')})",
        ])

    def _sig_line(name: str, block: Dict[str, Any]) -> str:
        if not block:
            return f"  {name}: n/a"
        p = block.get("p_value")
        p_txt = f"p={p:.3f}" if p is not None else "p=n/a"
        sig = "significant" if block.get("significant_95") else "not significant"
        extra = ""
        if block.get("zero_effect"):
            extra = " (flat effect)"
        elif block.get("insufficient_data"):
            extra = " (insufficient paired days)"
        else:
            rem = block.get("days_remaining_95")
            if rem is not None and rem > 0 and not block.get("significant_95"):
                extra = f", ~{rem} more paired days at current pace"
        return f"  {name}: {sig} ({p_txt}{extra})"

    imb = cmp.get("internal_minus_baseline") or cmp
    tr_diff = imb.get("excess_return_pct") or cmp.get("total_return_diff_pct")
    lines.append("  Neutral Internal − Baseline comparison:")
    lines.extend([
        f"  Excess return (Internal − Baseline): {tr_diff:+.2f}%"
        if tr_diff is not None
        else "  Excess return: n/a",
        f"    Baseline {b.get('total_return_pct')}% · Internal {i.get('total_return_pct')}%",
        _sig_line("Excess return test", sig.get("total_return_diff")),
        f"  Daily delta (latest day): {cmp.get('daily_delta_pct')}% "
        f"(mean alpha {cmp.get('mean_daily_alpha_pct')}%)",
        _sig_line("Daily alpha test", sig.get("daily_alpha")),
        f"  Sharpe difference: {cmp.get('sharpe_diff')} "
        f"(Baseline {b.get('sharpe')} · Internal {i.get('sharpe')})",
        _sig_line("Sharpe test", sig.get("sharpe_diff")),
        f"  Drawdown difference: {cmp.get('max_drawdown_diff_pct')} pp "
        f"(Baseline max DD {b.get('max_drawdown_pct')}% · Internal {i.get('max_drawdown_pct')}%)",
        _sig_line("Drawdown test", sig.get("max_drawdown_diff")),
    ])
    return "\n".join(lines)
