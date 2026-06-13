"""ADK Workflow for Internal Twin Ledger daily pipeline."""
from __future__ import annotations

from typing import Any, Dict

from google.adk.apps.app import App
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.workflow import Workflow, node
from google.adk.workflow._base_node import START
from google.genai import types

from src import config
from src.adk.model import configure_genai_env
from src.adk.workflows.common import (
    fetch_news_for_universe,
    invoke_signal_agent,
    ledger_to_state,
    workflow_daily_setup,
)
from src.adk.workflows.daily_pipeline import _ensure_discovery_fresh
from src.agents.competition_context import build_competition_context
from src.agents.data_agent_internal import InternalDataAgent
from src.agents.execution_agent import ExecutionAgent
from src.agents.monitor_agent import MonitorAgent
from src.agents.risk_agent import RiskAgent, entry_sides_from_decisions
from src.agents.signal_agent_internal import InternalSignalAgent
from src.models.trading_decision import TradingDecision
from src.strategies.allocator import PositionAllocator
from src.strategies.order_manager import OrderManager
from src.logger import setup_logger

logger = setup_logger(__name__)


@node(name="internal_fetch_context", rerun_on_resume=True)
async def internal_fetch_context(ctx):
    """Fetch account, MC predictions, technicals, DataBento enrichment, and news."""
    discovery_meta = await _ensure_discovery_fresh()
    ctx.state["discovery"] = discovery_meta

    data_agent = InternalDataAgent()
    account_info = await data_agent.get_account_info()
    if not account_info:
        ctx.state["error"] = "Failed to get account info"
        ctx.state["workflow_result"] = {"success": False, "error": "Failed to get account info"}
        return {"success": False}

    competition = build_competition_context("internal")
    mc_predictions = await data_agent.fetch_mc_predictions(config.TICKER_UNIVERSE)
    price_data = await data_agent.fetch_price_data(config.TICKER_UNIVERSE)
    from src.strategies.signal_generator import SignalGenerator

    technical_data = SignalGenerator.build_technical_data(price_data)
    databento_sources = await data_agent.enrich_with_databento(config.TICKER_UNIVERSE)
    news_data = fetch_news_for_universe()

    ctx.state["account_info"] = account_info
    ctx.state["competition"] = competition
    ctx.state["mc_predictions"] = mc_predictions
    ctx.state["technical_data"] = technical_data
    ctx.state["databento_sources"] = databento_sources or {}
    ctx.state["news_data"] = news_data
    ctx.state["current_positions"] = await data_agent.get_current_positions()
    return {
        "tickers": len(technical_data),
        "mc": len(mc_predictions),
        "discovery": discovery_meta,
    }


@node(name="internal_signal_decisions", rerun_on_resume=True)
async def internal_signal_decisions(ctx):
    """Generate Twin Ledger decisions via ADK signal LlmAgent (ctx.run_node)."""
    technical_data = ctx.state.get("technical_data") or {}
    competition = ctx.state.get("competition") or {}
    mc_predictions = ctx.state.get("mc_predictions") or {}
    databento_sources = ctx.state.get("databento_sources") or {}
    news_data = ctx.state.get("news_data") or {}

    learning_block = ""
    if config.LEARNING_ENABLED:
        from src.learning.context import build_signal_learning_block

        learning_block = build_signal_learning_block("internal")

    ledger = await invoke_signal_agent(
        ctx,
        "internal",
        {
            "system": "internal",
            "competition": competition,
            "technical_data": technical_data,
            "mc_predictions": mc_predictions,
            "databento_sources": databento_sources,
            "news_data": news_data,
            "signal_learning": learning_block,
            "valid_tickers": list(config.TICKER_UNIVERSE),
        },
    )
    signal_agent = InternalSignalAgent()
    from src.agents.ledger_utils import emit_signal_ledger_audit

    emit_signal_ledger_audit(signal_agent, ledger, competition)
    ctx.state.update(ledger_to_state(ledger))
    return {
        "decisions": len(ledger.decisions),
        "no_action_rationale": ledger.no_action_rationale or "",
    }


@node(name="internal_risk_and_execute", rerun_on_resume=True)
async def internal_risk_and_execute(ctx):
    """Kelly allocation, risk validation, order placement."""
    decisions_raw = ctx.state.get("decisions") or []
    mc_predictions = ctx.state.get("mc_predictions") or {}
    technical_data = ctx.state.get("technical_data") or {}
    no_action_rationale = ctx.state.get("no_action_rationale") or ""
    decisions = [
        TradingDecision.from_dict(d) for d in decisions_raw
        if TradingDecision.from_dict(d)
    ]
    account_info = ctx.state.get("account_info") or {}
    current_positions = ctx.state.get("current_positions") or {}

    actionable = [d for d in decisions if d.action != "HOLD"]
    if not actionable:
        result = {
            "success": True,
            "pipeline": "adk_workflow",
            "system": "internal",
            "orders_placed": 0,
            "dry_run": config.is_dry_run(),
            "message": "No actionable decisions from signal step",
            "no_action_rationale": no_action_rationale,
            "discovery": ctx.state.get("discovery"),
        }
        ctx.state["workflow_result"] = result
        ctx.state["orders_placed"] = 0
        ctx.state["success"] = True
        return result

    signal_agent = InternalSignalAgent()
    signals = signal_agent.decisions_to_signals(decisions, technical_data, mc_predictions)

    risk_agent = RiskAgent(system="internal")
    buy_signals = {
        t: s for t, s in signals.items()
        if any(d.action == "BUY" and d.ticker == t for d in decisions)
    }
    short_decisions = [d for d in decisions if d.action == "SHORT"]
    entry_decisions = [d for d in decisions if d.action in ("BUY", "SHORT")]
    proposed_weights = PositionAllocator.internal_target_weights(buy_signals)
    proposed_weights.update(PositionAllocator.decision_target_weights(short_decisions))
    validation = await risk_agent.validate_positions(
        proposed_weights,
        float(account_info.get("portfolio_value", 0)),
        current_positions,
        entry_sides=entry_sides_from_decisions(entry_decisions),
    )
    valid_entries = {t for t, ok in validation.items() if ok}
    filtered = [
        d for d in decisions
        if d.action not in ("BUY", "SHORT") or d.ticker in valid_entries
    ]
    filtered_buy_signals = {t: s for t, s in buy_signals.items() if t in valid_entries}

    latest_prices = {
        t: technical_data[t].get("close", 0)
        for t in technical_data
        if technical_data[t].get("close")
    }
    position_changes = PositionAllocator.allocate_internal_from_decisions(
        filtered,
        filtered_buy_signals,
        float(account_info.get("portfolio_value", 0)),
        current_positions,
        latest_prices,
    )

    execution_agent = ExecutionAgent(system="internal")
    placed = 0
    if position_changes:
        OrderManager().build_overnight_orders(position_changes, latest_prices, spread_pct=0.5)
        order_ids = await execution_agent.place_overnight_orders(
            position_changes, latest_prices, current_positions
        )
        placed = len([o for o in order_ids.values() if o])

    result = {
        "success": True,
        "pipeline": "adk_workflow",
        "system": "internal",
        "orders_placed": placed,
        "dry_run": config.is_dry_run(),
        "decisions_count": len(decisions),
        "actionable_count": len([d for d in decisions if d.action != "HOLD"]),
        "validation_results": validation,
        "discovery": ctx.state.get("discovery"),
    }
    ctx.state["workflow_result"] = result
    ctx.state["orders_placed"] = placed
    ctx.state["success"] = True
    return result


@node(name="internal_monitor", rerun_on_resume=True)
async def internal_monitor(ctx):
    monitor = MonitorAgent(system="internal")
    metrics = await monitor.get_portfolio_metrics()
    if metrics:
        await monitor.log_daily_performance(metrics)

    result = dict(ctx.state.get("workflow_result") or {})
    result.setdefault("success", ctx.state.get("success", True))
    result.setdefault("system", "internal")
    result.setdefault("pipeline", "adk_workflow")
    result["metrics"] = metrics
    ctx.state["workflow_result"] = result

    from src.audit import end_trace

    end_trace(
        "daily",
        system="internal",
        success=bool(result.get("success")),
        summary={
            "orders_placed": result.get("orders_placed", 0),
            "dry_run": config.is_dry_run(),
            "pipeline": "adk_workflow",
        },
    )
    return result


def build_internal_daily_workflow() -> Workflow:
    return Workflow(
        name="internal_daily_workflow",
        edges=[
            (START, internal_fetch_context, internal_signal_decisions, internal_risk_and_execute, internal_monitor),
        ],
    )


async def run_internal_daily_adk(*, skip_calendar: bool = False) -> Dict[str, Any]:
    skipped = await workflow_daily_setup("internal", skip_calendar=skip_calendar)
    if skipped:
        return skipped

    configure_genai_env()
    workflow = build_internal_daily_workflow()
    app = App(name="twin_ledger_internal_workflow", root_agent=workflow)
    session_service = InMemorySessionService()
    runner = Runner(app=app, session_service=session_service, auto_create_session=True)
    session_id = "internal_daily_adk"

    logger.info("[ADK Workflow] Starting internal_daily_workflow")
    try:
        async for _event in runner.run_async(
            user_id="twin_ledger",
            session_id=session_id,
            new_message=types.Content(
                role="user",
                parts=[types.Part(text="Run internal Twin Ledger daily workflow.")],
            ),
        ):
            pass
    except Exception as e:
        logger.exception("[ADK Workflow] internal_daily_workflow failed")
        from src.audit import end_trace

        end_trace("daily", system="internal", success=False, summary={"error": str(e)})
        return {"success": False, "pipeline": "adk_workflow", "system": "internal", "error": str(e)}

    session = await session_service.get_session(
        app_name=app.name,
        user_id="twin_ledger",
        session_id=session_id,
    )
    state = (session.state if session else {}) or {}
    result = state.get("workflow_result") or {
        "success": state.get("success", True),
        "orders_placed": state.get("orders_placed", 0),
        "pipeline": "adk_workflow",
        "system": "internal",
        "dry_run": config.is_dry_run(),
    }
    logger.info(f"[ADK Workflow] internal_daily_workflow complete: {result}")
    return result
