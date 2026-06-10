"""Deterministic daily trading pipeline (data → signal → execute).

Used as the primary path for scheduler messages and as a fallback when the
coordinator LLM does not complete the overnight workflow.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from src import config
from src.adk.tools.alpaca_tools import execute_trading_decisions, get_technical_indicators
from src.adk.tools.databento_tools import get_databento_features
from src.adk.tools.marketcrunch_tools import get_marketcrunch_predictions
from src.agents.competition_context import build_competition_context
from src.agents.signal_agent_baseline import BaselineSignalAgent
from src.agents.signal_agent_internal import InternalSignalAgent
from src.gcs.store import get_gcs_store
from src.logger import setup_logger
from src.models.trading_decision import TradingDecision

logger = setup_logger(__name__)


async def run_daily_trading_pipeline(system: str) -> Dict[str, Any]:
    """Run full overnight workflow without relying on coordinator LLM chaining."""
    if system not in ("baseline", "internal"):
        return {"success": False, "error": f"Invalid system: {system}"}

    try:
        get_gcs_store().hydrate_audit_log()
    except Exception as e:
        logger.warning(f"GCS audit hydrate failed: {e}")

    logger.info(f"[daily_pipeline] Starting deterministic workflow for {system}")

    tech_result = get_technical_indicators(system=system, lookback_days=90)
    technical_data = tech_result.get("technical_data") or {}
    if not technical_data:
        return {"success": False, "error": "No technical data available", "system": system}

    competition = build_competition_context(system)
    mc_predictions_json: str | None = None
    technical_data_json = json.dumps(tech_result, default=str)

    if system == "internal":
        mc_result = get_marketcrunch_predictions()
        mc_predictions_json = json.dumps(mc_result, default=str)
        mc_predictions = mc_result.get("predictions") or {}
        databento = get_databento_features()
        signal_agent = InternalSignalAgent()
        decisions = await signal_agent.make_trading_decisions(
            technical_data,
            mc_predictions,
            competition,
            databento_sources=databento.get("sources"),
            prefer_direct=True,
        )
    else:
        signal_agent = BaselineSignalAgent()
        decisions = await signal_agent.make_trading_decisions(
            technical_data,
            competition,
            prefer_direct=True,
        )

    if not decisions:
        return {
            "success": True,
            "pipeline": "deterministic",
            "system": system,
            "orders_placed": 0,
            "message": "No decisions returned from signal step",
        }

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

    return {
        "pipeline": "deterministic",
        "system": system,
        "decisions_count": len(decisions),
        "actionable_count": len([d for d in decisions if d.action != "HOLD"]),
        **result,
    }
