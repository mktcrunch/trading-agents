"""Deterministic daily trading pipeline (discovery → data → signal → execute).

Used as the primary path for scheduler messages and as a fallback when the
coordinator LLM does not complete the overnight workflow.
"""
from __future__ import annotations

import json
from typing import Any, Dict

from src import config
from src.adk.tools.alpaca_tools import execute_trading_decisions, get_technical_indicators
from src.agents.signal_context import fetch_signal_news
from src.adk.tools.databento_tools import get_databento_features
from src.adk.tools.marketcrunch_tools import get_marketcrunch_predictions
from src.agents.competition_context import build_competition_context
from src.agents.signal_agent_baseline import BaselineSignalAgent
from src.agents.signal_agent_internal import InternalSignalAgent
from src.gcs.store import get_gcs_store
from src.logger import setup_logger
from src.agents.ledger_utils import SignalLedgerResult
from src.models.trading_decision import TradingDecision

logger = setup_logger(__name__)


async def _ensure_discovery_fresh() -> Dict[str, Any]:
    """Run agentic discovery if approved sources are missing or stale (>24h)."""
    from src.agents.discovery_agent import DiscoveryAgent
    from src.discovery.approved_sources import is_stale, load_approved_sources

    stale = is_stale()
    agent = DiscoveryAgent()
    try:
        result = await agent.ensure_fresh_sources(force=False)
        summary = result.get("summary") or {}
        return {
            "success": True,
            "refreshed": stale,
            "approved_count": summary.get("approved_count", 0),
            "tickers_with_features": summary.get("tickers_with_features", 0),
            "probes_run": summary.get("probes_run"),
            "generated_at": result.get("generated_at"),
        }
    except Exception as e:
        logger.warning(f"[daily_pipeline] Discovery failed: {e}")
        cached = load_approved_sources()
        return {
            "success": False,
            "refreshed": False,
            "error": str(e),
            "approved_count": (cached.get("summary") or {}).get("approved_count", 0),
            "generated_at": cached.get("generated_at"),
        }


async def run_daily_trading_pipeline(system: str) -> Dict[str, Any]:
    """Run full overnight workflow without relying on coordinator LLM chaining."""
    if system not in ("baseline", "internal"):
        return {"success": False, "error": f"Invalid system: {system}"}

    try:
        get_gcs_store().hydrate_audit_log()
    except Exception as e:
        logger.warning(f"GCS audit hydrate failed: {e}")

    from src.audit import end_trace, start_trace

    dry = config.is_dry_run()
    start_trace("daily", system=system)
    logger.info(
        f"[daily_pipeline] Starting deterministic workflow for {system}"
        + (" (DRY RUN — no orders)" if dry else "")
    )

    from src.market.calendar import check_overnight_trading_session

    session_ok, session_reason = check_overnight_trading_session(system=system)
    if not session_ok:
        logger.info(f"[daily_pipeline] Skipping overnight for {system}: {session_reason}")
        end_trace(
            "daily",
            system=system,
            success=True,
            summary={
                "skipped": True,
                "skip_reason": session_reason,
                "orders_placed": 0,
            },
        )
        return {
            "success": True,
            "skipped": True,
            "skip_reason": session_reason,
            "pipeline": "deterministic",
            "system": system,
            "orders_placed": 0,
            "message": f"Overnight skipped: {session_reason}",
        }

    if config.LEARNING_ENABLED:
        try:
            from src.learning.reflection import refresh_system_learning

            await refresh_system_learning(system)
        except Exception as e:
            logger.warning(f"[daily_pipeline] Learning refresh failed: {e}")

    discovery_meta: Dict[str, Any] | None = None
    if system == "internal":
        logger.info("[daily_pipeline] Ensuring data discovery sources are fresh")
        discovery_meta = await _ensure_discovery_fresh()

    tech_result = get_technical_indicators(system=system, lookback_days=90)
    technical_data = tech_result.get("technical_data") or {}
    if not technical_data:
        end_trace("daily", system=system, success=False, summary={"error": "No technical data available"})
        return {"success": False, "error": "No technical data available", "system": system}

    competition = build_competition_context(system)
    mc_predictions_json: str | None = None
    technical_data_json = json.dumps(tech_result, default=str)

    logger.info("[daily_pipeline] Fetching recent news for signal step")
    news_data = fetch_signal_news(list(config.TICKER_UNIVERSE)).get("news") or {}

    if system == "internal":
        mc_result = get_marketcrunch_predictions()
        mc_predictions_json = json.dumps(mc_result, default=str)
        mc_predictions = mc_result.get("predictions") or {}
        databento = get_databento_features()
        signal_agent = InternalSignalAgent()
        ledger = await signal_agent.make_trading_decisions(
            technical_data,
            mc_predictions,
            competition,
            databento_sources=databento.get("sources"),
            prefer_direct=True,
            news_data=news_data,
        )
    else:
        signal_agent = BaselineSignalAgent()
        ledger = await signal_agent.make_trading_decisions(
            technical_data,
            competition,
            prefer_direct=True,
            news_data=news_data,
        )

    decisions = ledger.decisions if isinstance(ledger, SignalLedgerResult) else ledger
    actionable = [d for d in decisions if d.action != "HOLD"]

    if not actionable:
        out: Dict[str, Any] = {
            "success": True,
            "pipeline": "deterministic",
            "system": system,
            "orders_placed": 0,
            "message": "No actionable decisions from signal step",
            "no_action_rationale": getattr(ledger, "no_action_rationale", "") or "",
        }
        if discovery_meta:
            out["discovery"] = discovery_meta
        end_trace(
            "daily",
            system=system,
            success=True,
            summary={
                "orders_placed": 0,
                "actionable_count": 0,
                "no_action_rationale": out["no_action_rationale"],
            },
        )
        return out

    decisions_json = json.dumps(
        [d.to_dict() for d in decisions if isinstance(d, TradingDecision)],
        default=str,
    )

    result = await execute_trading_decisions(
        system=system,
        decisions_json=decisions_json,
        mc_predictions_json=mc_predictions_json if system == "internal" else None,
        technical_data_json=technical_data_json,
    )

    from src.agents.monitor_agent import MonitorAgent

    monitor = MonitorAgent(system=system)
    metrics = await monitor.get_portfolio_metrics()
    if metrics:
        await monitor.log_daily_performance(metrics)

    out = {
        "pipeline": "deterministic",
        "system": system,
        "decisions_count": len(decisions),
        "actionable_count": len([d for d in decisions if d.action != "HOLD"]),
        **result,
    }
    if discovery_meta:
        out["discovery"] = discovery_meta
    return out
